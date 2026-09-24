"""Small learned part codec; supplied association, no analytic view transform.

Core adapted from Claude Opus 5.5 medium's public design round two. Persistence
belongs to WorldSession, not this module. Stored codes are model-version specific.
"""
import torch
from torch import nn
from torch.nn import functional as F


def to_parts(rgb):
    if rgb.ndim != 4 or rgb.shape[1:] != (3, 16, 16):
        raise ValueError("Expected RGB [B,3,16,16]")
    return rgb.reshape(-1, 3, 2, 8, 2, 8).permute(0, 2, 4, 1, 3, 5).reshape(-1, 4, 192)


def from_parts(parts):
    if parts.ndim != 3 or parts.shape[1:] != (4, 192):
        raise ValueError("Expected parts [B,4,192]")
    return parts.reshape(-1, 2, 2, 3, 8, 8).permute(0, 3, 1, 4, 2, 5).reshape(-1, 3, 16, 16)


def _presence(values, present, width):
    if values.ndim != 3 or values.shape[1:] != (4, width):
        raise ValueError(f"Expected values [B,4,{width}]")
    if present.shape != values.shape[:2] or present.dtype != torch.bool or present.device != values.device:
        raise ValueError("Presence must be boolean [B,4] on the values device")
    if not values.is_floating_point() or not torch.isfinite(values[present]).all():
        raise ValueError("Observed values must be finite floating point")


def replace_details(old, present, new, visible):
    """Replacement, not confidence accumulation. No implicit detach during training."""
    _presence(old, present, old.shape[-1])
    _presence(new, visible, old.shape[-1])
    if old.shape != new.shape or old.device != new.device:
        raise ValueError("Old and new codes must match")
    return torch.where(visible[..., None], new, old), present | visible


class FiLMBlock(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.linear, self.norm = nn.Linear(hidden, hidden), nn.LayerNorm(hidden)
        self.film = nn.Embedding(4, 2 * hidden)
        nn.init.zeros_(self.film.weight)

    def forward(self, x, pose):
        scale, bias = self.film(pose).chunk(2, -1)
        return x + F.gelu(self.norm(self.linear(x)) * (1 + scale) + bias)


class DetailCodec(nn.Module):
    def __init__(self, width=64, hidden=512, variant="film"):
        super().__init__()
        if variant not in {"film", "linear"}:
            raise ValueError("Unknown detail decoder")
        self.width, self.hidden, self.variant = width, hidden, variant
        e = min(256, hidden)
        self.encoder = nn.Sequential(nn.Linear(192, e), nn.GELU(), nn.Linear(e, e), nn.GELU(), nn.Linear(e, width))
        self.prior = nn.Parameter(torch.zeros(4, width))
        if variant == "film":
            self.input = nn.Linear(4 * width + 8, hidden)
            self.blocks = nn.ModuleList(FiLMBlock(hidden) for _ in range(3))
            self.mean, self.logvar = nn.Linear(hidden, 768), nn.Linear(hidden, 768)
        else:
            # Four learned maps, no geometric permutation or inverse renderer.
            # Compute all heads once instead of expanding weights per batch item.
            self.mean = nn.Linear(4 * width, 4 * 768)
            self.logvar = nn.Linear(4, 4 * 768)
            nn.init.zeros_(self.logvar.weight)
        nn.init.constant_(self.logvar.bias, -3.0)

    def encode(self, parts, visible):
        _presence(parts, visible, 192)
        # Erase unavailable pixels before any projection, including masked NaNs.
        x = parts.masked_fill(~visible[..., None], 0)
        return self.encoder(x).masked_fill(~visible[..., None], 0)

    def decode(self, codes, present, pose):
        _presence(codes, present, self.width)
        if pose.shape != (len(codes),) or pose.dtype != torch.long or pose.device != codes.device:
            raise ValueError("Pose must be int64 [B] on the code device")
        if ((pose < 0) | (pose > 3)).any():
            raise ValueError("Pose must be a quarter turn 0..3")
        z = torch.where(present[..., None], codes, self.prior.expand_as(codes))
        if self.variant == "linear":
            rows = torch.arange(len(z), device=z.device)
            mean = self.mean(z.flatten(1)).reshape(-1, 4, 768)[rows, pose]
            raw = self.logvar(present.to(z.dtype)).reshape(-1, 4, 768)[rows, pose]
            lv = (1e-4 + F.softplus(raw)).log()
            return mean.reshape(-1, 3, 16, 16), lv.reshape(-1, 3, 16, 16)
        x = torch.cat((z.flatten(1), present.to(z.dtype), F.one_hot(pose, 4).to(z.dtype)), -1)
        x = self.input(x)
        for block in self.blocks:
            x = block(x, pose)
        return self.mean(x).reshape(-1, 3, 16, 16), self.logvar(x).clamp(-9.21, 2).reshape(-1, 3, 16, 16)

    def forward(self, parts, visible, pose):
        return self.decode(self.encode(parts, visible), visible, pose)


def detail_loss(mean, logvar, target):
    """Variance-weighted Gaussian objective from the generic Claude core."""
    if mean.shape != logvar.shape or mean.shape != target.shape:
        raise ValueError("Mean, log variance and target shapes must agree")
    nll = 0.5 * (logvar + (target - mean).square() * torch.exp(-logvar))
    return (torch.exp(0.5 * logvar).detach() * nll).mean()


def decode_read(model, client, read, pose, *, version):
    """Checked consumer: source pins and the actual external codec must be current."""
    from pathwm.io import state_hash
    if state_hash(model) != version or model.training:
        raise ValueError("Codec weights/version or mode changed")
    client.validate(read)
    if set(read.names) != {f"part{i}" for i in range(4)}:
        raise ValueError("A detail read must explicitly request all four parts")
    reasons = dict(getattr(read, "omission_reasons", ()))
    if any(reason != "never_observed" for reason in reasons.values()):
        raise ValueError("Unavailable detail needs rederivation or a larger read budget")
    device = next(model.parameters()).device
    codes = torch.zeros(1, 4, model.width, device=device)
    present = torch.zeros(1, 4, dtype=torch.bool, device=device)
    for c in read.components:
        if c.model_version != version:
            raise ValueError("Stored detail belongs to a different codec")
        if c.data.get("availability") == "unknown":
            if c.values or c.evidence or c.parents or not c.data.get("reason"):
                raise ValueError("Explicit unknown must be an empty, reasoned declaration")
            continue
        if c.name not in {f"part{i}" for i in range(4)} or c.shape != (model.width,):
            raise ValueError("Expected one correctly shaped detail code per part")
        i = int(c.name[4:]); codes[0, i] = c.tensor(device=device); present[0, i] = True
    result = model.decode(codes, present, torch.tensor([pose], device=device))
    client.validate(read)  # emission also rechecks: decoding alone grants no output authority
    return result
