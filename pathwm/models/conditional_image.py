"""Optional conditional spatial generation; solver progress is not world time."""

import math
import torch
from torch import nn
from .features import validate


def flow_pair(target, noise, progress):
    """Independent endpoint coupling: noise at zero, data at one."""
    if target.keys() != noise.keys():
        raise ValueError("Flow endpoints require identical scales")
    if (
        progress.ndim != 1
        or not torch.isfinite(progress).all()
        or ((progress < 0) | (progress > 1)).any()
    ):
        raise ValueError("Flow progress must be a finite batch vector in [0,1]")
    if any(
        v.shape != noise[k].shape or len(v) != len(progress) for k, v in target.items()
    ):
        raise ValueError("Flow endpoint shapes must agree")
    t = progress[:, None, None, None]
    return (
        {k: (1 - t) * noise[k] + t * v for k, v in target.items()},
        {k: v - noise[k] for k, v in target.items()},
    )


def integrate(field, noise, steps):
    """Explicit Euler on [0,1]; caller noise is never modified in place."""
    if type(steps) is not int or steps < 1:
        raise ValueError("Positive integer integration steps required")
    x = {k: v.clone() for k, v in noise.items()}
    for i in range(steps):
        velocity = field(x, i / steps)
        if velocity.keys() != x.keys():
            raise ValueError("Field changed the feature scales")
        if any(velocity[k].shape != v.shape for k, v in x.items()):
            raise ValueError("Field changed a feature shape")
        x = {k: v + velocity[k] / steps for k, v in x.items()}
    if not all(torch.isfinite(v).all() for v in x.values()):
        raise ValueError("Nonfinite integrated features")
    return x


class OutputBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.norms = nn.ModuleList([nn.LayerNorm(width) for _ in range(3)])
        self.self_attention = nn.MultiheadAttention(width, 4, batch_first=True)
        self.context_attention = nn.MultiheadAttention(width, 4, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(width, 2 * width), nn.GELU(), nn.Linear(2 * width, width)
        )

    def forward(self, x, context, valid):
        z = self.norms[0](x)
        x = x + self.self_attention(z, z, z, need_weights=False)[0]
        x = (
            x
            + self.context_attention(
                self.norms[1](x),
                context,
                context,
                key_padding_mask=~valid,
                need_weights=False,
            )[0]
        )
        return x + self.mlp(self.norms[2](x))


class ConditionalFeatureGenerator(nn.Module):
    """Generate every codec scale from context; targets are absent from inference.

    Direct and flow variants have identical parameter layouts. Defaults are saved
    with export settings. Explicit sample IDs make noise independent of batching;
    IDs are used only by the sampler and are never encoded as model context.
    """

    def __init__(
        self,
        width,
        feature_spec,
        head,
        *,
        depth=1,
        fusion_depth=1,
        objective="flow",
        steps=8,
        seed=13,
        hidden_width=None,
    ):
        super().__init__()
        if width < 4 or width % 4 or depth < 1 or fusion_depth < 1 or not feature_spec:
            raise ValueError("Valid width, scales and positive depths required")
        if objective not in ("flow", "direct") or type(steps) is not int or steps < 1:
            raise ValueError("Unknown objective or invalid step count")
        self.context_width = width
        hidden_width = width if hidden_width is None else hidden_width
        if type(hidden_width) is not int or hidden_width < 4 or hidden_width % 4:
            raise ValueError("Hidden width must be positive and divisible by four")
        self.width, self.feature_spec, self.head = (
            hidden_width,
            dict(feature_spec),
            head.requires_grad_(False),
        )
        self.objective, self.steps, self.seed = objective, steps, seed
        self.input, self.output, self.blocks = (
            nn.ModuleDict(),
            nn.ModuleDict(),
            nn.ModuleDict(),
        )
        self.positions = nn.ParameterDict()
        self.context_norm = nn.LayerNorm(width)
        self.context_projection = (
            nn.Identity() if hidden_width == width else nn.Linear(width, hidden_width)
        )
        width = hidden_width
        self.progress = nn.Sequential(
            nn.Linear(4, width), nn.SiLU(), nn.Linear(width, width)
        )
        for i, (k, s) in enumerate(self.feature_spec.items()):
            self.input[k] = nn.Linear(s.channels, width)
            self.output[k] = nn.Linear(width, s.channels)
            self.positions[k] = nn.Parameter(
                torch.randn(1, math.prod(s.size), width) * 0.02
            )
            self.blocks[k] = nn.ModuleList([OutputBlock(width) for _ in range(depth)])
            self.register_buffer(f"mean_{i}", torch.zeros(1, s.channels, 1, 1))
            self.register_buffer(f"scale_{i}", torch.ones(1, s.channels, 1, 1))
        self.fusion = nn.ModuleList([OutputBlock(width) for _ in range(fusion_depth)])

    @torch.no_grad()
    def calibrate(self, features):
        validate(features, self.feature_spec)
        if not all(torch.isfinite(features[k]).all() for k in self.feature_spec):
            raise ValueError("Nonfinite calibration targets")
        for i, k in enumerate(self.feature_spec):
            getattr(self, f"mean_{i}").copy_(features[k].mean((0, 2, 3), keepdim=True))
            getattr(self, f"scale_{i}").copy_(
                features[k].std((0, 2, 3), correction=0, keepdim=True).clamp_min(0.01)
            )

    def standardize(self, features):
        validate(features, self.feature_spec)
        return {
            k: (features[k] - getattr(self, f"mean_{i}")) / getattr(self, f"scale_{i}")
            for i, k in enumerate(self.feature_spec)
        }

    def unstandardize(self, features):
        return {
            k: features[k] * getattr(self, f"scale_{i}") + getattr(self, f"mean_{i}")
            for i, k in enumerate(self.feature_spec)
        }

    def field(self, features, progress, context, valid=None):
        validate(features, self.feature_spec)
        if (
            context.ndim != 3
            or context.shape[-1] != self.context_width
            or context.shape[1] < 1
        ):
            raise ValueError("Invalid context shape")
        if valid is None:
            valid = torch.ones(
                context.shape[:2], dtype=torch.bool, device=context.device
            )
        if (
            valid.dtype != torch.bool
            or valid.shape != context.shape[:2]
            or not valid.any(1).all()
        ):
            raise ValueError("Each context needs valid tokens")
        context = context.masked_fill(~valid[..., None], 0)
        if not torch.isfinite(context).all():
            raise ValueError("Nonfinite valid context")
        context = self.context_projection(self.context_norm(context))
        b = len(context)
        t = torch.as_tensor(
            progress, device=context.device, dtype=context.dtype
        ).expand(b)
        if not torch.isfinite(t).all() or (t < 0).any() or (t > 1).any():
            raise ValueError("Progress must lie in [0,1]")
        time = self.progress(
            torch.stack([t, 1 - t, (math.pi * t).sin(), (math.pi * t).cos()], -1)
        )[:, None]
        scales = []
        for k, s in self.feature_spec.items():
            if len(features[k]) != b or not torch.isfinite(features[k]).all():
                raise ValueError("Invalid feature batch or nonfinite features")
            x = (
                self.input[k](features[k].flatten(2).transpose(1, 2))
                + self.positions[k]
                + time
            )
            for block in self.blocks[k]:
                x = block(x, context, valid)
            scales.append(x)
        x = torch.cat(scales, 1)
        for block in self.fusion:
            x = block(x, context, valid)
        out = {}
        offset = 0
        for k, s in self.feature_spec.items():
            count = math.prod(s.size)
            out[k] = (
                self.output[k](x[:, offset : offset + count])
                .transpose(1, 2)
                .reshape(b, s.channels, *s.size)
            )
            offset += count
        return out

    def features(
        self, context, trace=None, *, seed=None, sample_ids=None, steps=None, valid=None
    ):
        if trace is not None:
            raise ValueError("Generator attention trace is not implemented")
        count = len(context)
        if sample_ids is None:
            sample_ids = range(count)
        sample_ids = list(sample_ids)
        if len(sample_ids) != count or any(
            type(i) is not int or i < 0 for i in sample_ids
        ):
            raise ValueError("One nonnegative integer sample ID per context required")
        if self.objective == "direct":
            x = {
                k: context.new_zeros(count, s.channels, *s.size)
                for k, s in self.feature_spec.items()
            }
            result = self.field(x, 0.0, context, valid)
        else:
            streams = [
                torch.Generator().manual_seed((self.seed if seed is None else seed) + i)
                for i in sample_ids
            ]
            x = {
                k: torch.stack(
                    [torch.randn(s.channels, *s.size, generator=g) for g in streams]
                ).to(context)
                for k, s in self.feature_spec.items()
            }
            result = integrate(
                lambda x, t: self.field(x, t, context, valid),
                x,
                self.steps if steps is None else steps,
            )
        return self.unstandardize(result)

    def forward(self, context, **kwargs):
        return self.head(self.features(context, **kwargs))


def configure_generator(model, settings):
    """Replace only the output producer, retaining the existing image head."""
    previous = model.agent.decoders["image"]
    model.requires_grad_(False).eval()
    new = ConditionalFeatureGenerator(
        model.agent.width, previous.feature_spec, previous.head, **settings
    )
    new.to(next(model.parameters()).device)
    model.agent.decoders["image"] = new
    return new
