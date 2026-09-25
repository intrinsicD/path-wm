"""Slot consumer of the existing multiscale image pyramid.

`MultiScaleImageEncoder` produces the shared, unmodified `FeaturePyramid` source.
Slot attention reads both exported scales and deliberately compresses 320 source
tokens into K slots (declared bottleneck). A broadcast decoder reconstructs RGB and
per-slot alpha. Environment masks/attributes are training labels only.
"""

from dataclasses import dataclass, replace
from itertools import permutations

import torch
from torch import nn
from torch.nn import functional as F

from .modalities import Observation
from .multiscale import MultiScaleImageEncoder

ATTRIBUTES, VALUES = 4, 4
KIND_NAMES = ("background", "machine", "object")


@dataclass
class Percept:
    slots: torch.Tensor  # [B,K,D]
    alpha: torch.Tensor  # [B,K,H,W] mixture logits
    recon: torch.Tensor  # [B,3,H,W]
    attributes: torch.Tensor  # [B,K,4,4] logits
    lamp: torch.Tensor  # [B,K] logits
    kind: torch.Tensor  # [B,K,3] logits

    def detach(self):
        return Percept(*(v.detach() for v in vars(self).values()))

    def select(self, index):
        return Percept(*(v[index] for v in vars(self).values()))


class SlotAttention(nn.Module):
    """Locatello et al. slot attention with deterministic learned seeds."""

    def __init__(self, width, slots, iterations):
        super().__init__()
        self.iterations = iterations
        self.seeds = nn.Parameter(torch.randn(slots, width) * 0.5)
        self.input_norm, self.slot_norm, self.mlp_norm = (
            nn.LayerNorm(width) for _ in range(3)
        )
        self.q, self.k, self.v = (nn.Linear(width, width, bias=False) for _ in range(3))
        self.gru = nn.GRUCell(width, width)
        self.mlp = nn.Sequential(
            nn.Linear(width, 2 * width), nn.GELU(), nn.Linear(2 * width, width)
        )

    def forward(self, tokens, valid):
        b, _, d = tokens.shape
        x = self.input_norm(tokens.masked_fill(~valid[..., None], 0))
        k, v = self.k(x), self.v(x)
        slots = self.seeds.expand(b, -1, -1)
        for _ in range(self.iterations):
            previous = slots
            q = self.q(self.slot_norm(slots))
            logits = torch.einsum("bnd,bkd->bnk", k, q) / d**0.5
            # Epsilon only on valid rows: padding count must carry no mass, even
            # though LayerNorm biases make padded keys/values nonzero.
            mask = valid[..., None].to(logits.dtype)
            attention = (logits.softmax(-1) + 1e-8) * mask
            weights = attention / attention.sum(1, keepdim=True).clamp_min(1e-8)
            updates = torch.einsum("bnk,bnd->bkd", weights, v)
            slots = self.gru(
                updates.reshape(-1, d), previous.reshape(-1, d)
            ).reshape(b, -1, d)
            slots = slots + self.mlp(self.mlp_norm(slots))
        return slots


class Double(nn.Module):
    """Nearest 2x upsampling via expand/reshape (deterministic backward on CUDA)."""

    def forward(self, x):
        b, c, h, w = x.shape
        return x[:, :, :, None, :, None].expand(b, c, h, 2, w, 2).reshape(b, c, 2 * h, 2 * w)


class BroadcastDecoder(nn.Module):
    """Slot -> 8x8 broadcast -> three nearest upsamplings -> RGB + alpha logit.

    Optional (`enable_pyramid_connections`): the same frame's encoder pyramid also
    conditions every slot's decoding, coarse 8x8 scale before the first conv and fine
    16x16 scale before the second. Default decoders are unchanged.
    """

    def __init__(self, width, hidden, size=64):
        super().__init__()
        self.width, self.hidden = width, hidden
        self.base = size // 8
        self.position = nn.Linear(2, width)
        layers = [nn.Conv2d(width, hidden, 3, padding=1), nn.ReLU()]
        for _ in range(3):
            layers += [
                Double(),
                nn.Conv2d(hidden, hidden, 3, padding=1),
                nn.ReLU(),
            ]
        layers.append(nn.Conv2d(hidden, 4, 3, padding=1))
        self.network = nn.Sequential(*layers)

    @property
    def pyramid_enabled(self):
        return hasattr(self, "coarse")

    def enable_pyramid_connections(self):
        """Add zero-initialized pointwise projections of the coarse (width->width at 8x8)
        and fine (width->hidden at 16x16) pyramid scales. Existing weights and the
        global RNG are preserved; the initial output equals the slot-only decoder."""
        if self.pyramid_enabled:
            return self
        with torch.random.fork_rng(devices=[]):
            self.coarse = nn.Conv2d(self.width, self.width, 1)
            self.fine = nn.Conv2d(self.width, self.hidden, 1)
        for layer in (self.coarse, self.fine):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        device, dtype = self.position.weight.device, self.position.weight.dtype
        self.coarse.to(device, dtype).train(self.training)
        self.fine.to(device, dtype).train(self.training)
        return self

    def _maps(self, pyramid, slots):
        """Native frame-local pyramid -> fine [B,W,16,16], coarse [B,W,8,8] (new tensors)."""
        side, batch = self.base, len(slots)
        expected = ((1, 2 * side, 2 * side), (1, side, side))
        scales = getattr(pyramid, "scales", None)
        if scales is None or len(scales) != 2 or tuple(tuple(s.grid) for s in scales) != expected:
            raise ValueError(f"Decoder pyramid must have exactly the native scales {expected}")
        weight = self.coarse.weight
        maps = []
        for scale, (_, h, w) in zip(scales, expected):
            v, valid = scale.values, scale.valid
            if not (torch.is_tensor(v) and v.ndim == 3 and v.shape == (batch, h * w, self.width)):
                raise ValueError("Decoder pyramid values must be [B,N,width] matching the slots")
            if v.device != slots.device or v.dtype != slots.dtype or weight.device != v.device or weight.dtype != v.dtype:
                raise ValueError("Decoder pyramid values must share the slots' and decoder's device and dtype")
            if not (torch.is_tensor(valid) and valid.dtype == torch.bool and valid.shape == (batch, h * w)
                    and valid.device == v.device):
                raise ValueError("Decoder pyramid valid mask must be a bool [B,N] tensor on the values' device")
            if not valid.all() or not torch.isfinite(v).all():
                raise ValueError("Decoder pyramid must be fully valid and finite (no partial frames)")
            maps.append(v.transpose(1, 2).reshape(batch, self.width, h, w))
        return maps

    def enable_fine_subpixels(self):
        """Optional fine-grid residual with a distinct output for each 4x4 pixel phase.

        Zero initialization preserves the existing decoder output and global RNG.
        This uses the same native fine export; no new encoder or image code.
        """
        if not self.pyramid_enabled:
            raise ValueError("Fine subpixels require the existing pyramid connections")
        if hasattr(self, "fine_subpixel"):
            return self
        with torch.random.fork_rng(devices=[]):
            self.fine_subpixel = nn.Conv2d(self.width, 16 * self.hidden, 1)
        nn.init.zeros_(self.fine_subpixel.weight)
        nn.init.zeros_(self.fine_subpixel.bias)
        self.fine_subpixel.to(self.position.weight).train(self.training)
        return self

    def forward(self, slots, pyramid=None):
        b, k, d = slots.shape
        if self.pyramid_enabled and pyramid is None:
            raise ValueError("This decoder is pyramid-connected; pass the frame's pyramid (no slot-only fallback)")
        if not self.pyramid_enabled and pyramid is not None:
            raise ValueError("Pyramid given to a slot-only decoder; enable_pyramid_connections() first")
        axis = torch.linspace(-1, 1, self.base, device=slots.device, dtype=slots.dtype)
        grid = torch.stack(torch.meshgrid(axis, axis, indexing="ij"), -1)
        x = slots.reshape(b * k, d, 1, 1) + self.position(grid).permute(2, 0, 1)[None]
        if pyramid is None:
            out = self.network(x)
        else:
            fine, coarse = self._maps(pyramid, slots)
            x = x + self.coarse(coarse).repeat_interleave(k, 0)
            x = self.network[2](self.network[1](self.network[0](x)))  # conv 8x8, ReLU, double
            x = x + self.fine(fine).repeat_interleave(k, 0)
            if hasattr(self, "fine_subpixel"):
                x = self.network[3:-1](x)
                detail = F.pixel_shuffle(self.fine_subpixel(fine), 4)
                out = self.network[-1](x + detail.repeat_interleave(k, 0))
            else:
                out = self.network[3:](x)
        out = out.reshape(b, k, 4, *out.shape[-2:])
        return out[:, :, :3].sigmoid(), out[:, :, 3]


class SlotPerception(nn.Module):
    def __init__(
        self,
        width=64,
        slots=7,
        iterations=3,
        *,
        patch_size=4,
        levels=2,
        decoder_width=32,
    ):
        super().__init__()
        self.width, self.slot_count = width, slots
        self.encoder = MultiScaleImageEncoder(
            width, patch_size, levels=levels, depth=1
        )
        self.slot_attention = SlotAttention(width, slots, iterations)
        self.decoder = BroadcastDecoder(width, decoder_width)
        self.heads = nn.ModuleDict(
            dict(
                attributes=nn.Linear(width, ATTRIBUTES * VALUES),
                lamp=nn.Linear(width, 1),
                kind=nn.Linear(width, len(KIND_NAMES)),
            )
        )

    def pyramid(self, rgb):
        """The shared multiscale source; consumers must not mutate it."""
        times = torch.zeros(len(rgb), 1, device=rgb.device, dtype=rgb.dtype)
        return self.encoder(Observation(rgb[:, None], times))

    def frame_pyramid(self, rgb, times):
        """Frame-local encoding (t=0, as trained) stamped with capture times [B]."""
        return at_time(self.pyramid(rgb), times)

    def forward(self, rgb):
        if rgb.ndim != 4 or rgb.shape[1:] != (3, 64, 64):
            raise ValueError("SlotPerception expects RGB [B,3,64,64]")
        return self.from_pyramid(self.pyramid(rgb))

    def from_pyramid(self, pyramid):
        """Slot consumer of an already encoded pyramid (times are metadata only)."""
        tokens = pyramid.as_tokens()
        slots = self.slot_attention(tokens.values, tokens.valid)
        # A pyramid-connected decoder reads the same pyramid the slots came from.
        colors, alpha = self.decoder(slots, pyramid) if self.decoder.pyramid_enabled else self.decoder(slots)
        recon = (alpha.softmax(1)[:, :, None] * colors).sum(1)
        b, k = slots.shape[:2]
        return Percept(
            slots,
            alpha,
            recon,
            self.heads["attributes"](slots).reshape(b, k, ATTRIBUTES, VALUES),
            self.heads["lamp"](slots).squeeze(-1),
            self.heads["kind"](slots),
        )


class SymbolicSlots(nn.Module):
    """Diagnostic 'supplied perception': slot tokens from hidden render symbols.

    Exact masks and slot order (entity order). Used only by the structured-input
    diagnostic stage to localize failures; it never supplies an R1 pixel gate.
    """

    def __init__(self, width=64):
        super().__init__()
        self.width = width
        self.role = nn.Embedding(3, width)
        self.lamp_embedding = nn.Embedding(2, width)
        self.attribute = nn.Embedding(ATTRIBUTES * VALUES, width)

    def forward(self, scenes, lamps, entity):
        b = len(scenes)
        device = self.role.weight.device
        attrs = scenes.attrs.to(device)
        offsets = torch.arange(ATTRIBUTES, device=device) * VALUES
        objects = self.attribute(attrs + offsets).sum(2) + self.role.weight[2]
        machines = self.lamp_embedding(lamps.to(device)) + self.role.weight[1]
        background = self.role.weight[0].expand(b, 1, -1)
        slots = torch.cat((background, machines, objects), 1)
        onehot = F.one_hot(entity.to(device), 7).permute(0, 3, 1, 2).float()
        alpha = (onehot * 2 - 1) * 20
        kind = torch.tensor([0, 1, 1, 2, 2, 2, 2], device=device)
        kind_logits = (F.one_hot(kind, 3).float() * 40 - 20).expand(b, -1, -1)
        lamp_logits = torch.cat(
            (
                torch.zeros(b, 1, device=device),
                (lamps.to(device).float() * 2 - 1) * 20,
                torch.zeros(b, 4, device=device),
            ),
            1,
        )
        attributes = F.one_hot(
            torch.cat((torch.zeros(b, 3, 4, dtype=torch.long, device=device), attrs), 1),
            VALUES,
        ).float() * 40 - 20
        return Percept(slots, alpha, torch.zeros(b, 3, 64, 64, device=device), attributes, lamp_logits, kind_logits)


def at_time(pyramid, times):
    """Stamp a frame-local (t=0) pyramid with absolute capture times [B] as metadata.

    The image encoder adds an absolute time position to feature values and R1
    perception is trained at t=0, so values stay frame-local; only the metadata
    (read by the belief as event age) carries capture time. Values are shared.
    """
    first = pyramid.scales[0].times
    times = torch.as_tensor(times, device=first.device).reshape(len(first), 1)

    def stamp(scale):
        t = times.to(scale.times.dtype).expand_as(scale.times).masked_fill(~scale.valid, 0)
        content = None if scale.content_times is None else t.to(scale.content_times.dtype)
        return replace(scale, times=t, content_times=content)

    return replace(pyramid, scales=tuple(stamp(s) for s in pyramid.scales))


def pointer(alpha, xy):
    """Slot with maximal alpha at pixel xy [B,2] (x, y)."""
    h, w = alpha.shape[-2:]
    x = xy[:, 0].floor().long().clamp(0, w - 1).to(alpha.device)
    y = xy[:, 1].floor().long().clamp(0, h - 1).to(alpha.device)
    return alpha[torch.arange(len(alpha), device=alpha.device), :, y, x].argmax(-1)


def slot_coordinates(alpha):
    """[B,K,2] pixel (x, y) of each slot's maximal alpha among pixels it wins."""
    b, k, h, w = alpha.shape
    probability = alpha.softmax(1)
    winner = alpha.argmax(1, keepdim=True)
    owned = winner == torch.arange(k, device=alpha.device)[None, :, None, None]
    score = torch.where(owned, probability, probability - 2.0).flatten(2)
    flat = score.argmax(-1)
    return torch.stack(((flat % w).float(), (flat // w).float()), -1)


def cross_entropy(logits, target, dim=1):
    """Mean cross-entropy via one-hot (deterministic backward on CUDA)."""
    onehot = F.one_hot(target, logits.shape[dim]).movedim(-1, dim).to(logits.dtype)
    return -(onehot * logits.log_softmax(dim)).sum(dim).mean()


_PERMUTATIONS = {}


def match_slots(alpha, entity, entities=7):
    """Exact minimum-cost slot->entity assignment by enumerating permutations."""
    b, k = alpha.shape[:2]
    if k != entities:
        raise ValueError("Supervised matching requires one slot per scene entity")
    if k not in _PERMUTATIONS:
        _PERMUTATIONS[k] = torch.tensor(list(permutations(range(k))))
    perms = _PERMUTATIONS[k].to(alpha.device)
    with torch.no_grad():
        truth = F.one_hot(entity, entities).permute(0, 3, 1, 2).float().flatten(2)
        logp = alpha.log_softmax(1).flatten(2)
        cost = -torch.einsum("bkp,bep->bke", logp, truth) / truth.sum(-1)[:, None].clamp_min(1)
        total = cost[:, torch.arange(k, device=alpha.device)[None], perms].sum(-1)  # [B,P]
        return perms[total.argmin(-1)]  # [B,K] entity of each slot


def perception_loss(percept, rgb, entity, attrs, lamps, *, weights=0.5):
    """RGB reconstruction plus training-only matched mask/type/attribute/lamp labels."""
    assign = match_slots(percept.alpha, entity)  # [B,K]
    slot_of_entity = torch.argsort(assign, 1)
    target = slot_of_entity.gather(1, entity.flatten(1)).reshape(entity.shape)
    mask = cross_entropy(percept.alpha, target)
    reconstruction = F.mse_loss(percept.recon, rgb)
    kind_target = torch.tensor([0, 1, 1, 2, 2, 2, 2], device=assign.device)[assign]
    kind = cross_entropy(percept.kind.flatten(0, 1), kind_target.flatten())
    is_object = assign >= 3
    object_attrs = attrs.to(assign.device).gather(
        1, (assign - 3).clamp_min(0)[..., None].expand(-1, -1, ATTRIBUTES)
    )
    attribute = cross_entropy(
        percept.attributes[is_object].flatten(0, 1), object_attrs[is_object].flatten()
    )
    is_machine = (assign == 1) | (assign == 2)
    lamp_target = lamps.to(assign.device).gather(1, (assign - 1).clamp(0, 1))
    lamp = F.binary_cross_entropy_with_logits(
        percept.lamp[is_machine], lamp_target[is_machine].float()
    )
    total = reconstruction + weights * (mask + kind + attribute + lamp)
    with torch.no_grad():
        predicted_entity = assign.gather(1, percept.alpha.argmax(1).flatten(1)).reshape(entity.shape)
        metrics = dict(
            loss=float(total),
            reconstruction=float(reconstruction),
            mask=float(mask),
            kind=float(kind),
            attribute=float(attribute),
            lamp=float(lamp),
            pixel_accuracy=float((predicted_entity == entity).float().mean()),
            kind_accuracy=float((percept.kind.argmax(-1) == kind_target).float().mean()),
            attribute_accuracy=float(
                (percept.attributes[is_object].argmax(-1) == object_attrs[is_object]).float().mean()
            ),
            lamp_accuracy=float(
                ((percept.lamp[is_machine] > 0).long() == lamp_target[is_machine]).float().mean()
            ),
        )
    return total, metrics
