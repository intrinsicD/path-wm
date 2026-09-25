"""Optional conditional spatial generation; solver progress is not world time."""

import math
import torch
from torch import nn
from torch.nn import functional as F
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

    def forward(self, x, context, valid, kv=None):
        """`kv` is this block's `_context_kv(context)`; it replaces only the
        context key/value projections, with the same weights, query and output."""
        z = self.norms[0](x)
        x = x + self.self_attention(z, z, z, need_weights=False)[0]
        z = self.norms[1](x)
        if kv is None:
            z = self.context_attention(
                z, context, context, key_padding_mask=~valid, need_weights=False
            )[0]
        else:
            a = self.context_attention
            b, n, w = z.shape
            q = F.linear(z, a.in_proj_weight[:w], a.in_proj_bias[:w])
            q = q.view(b, n, a.num_heads, -1).transpose(1, 2)
            z = F.scaled_dot_product_attention(q, *kv, attn_mask=valid[:, None, None])
            z = a.out_proj(z.transpose(1, 2).reshape(b, n, w))
        x = x + z
        return x + self.mlp(self.norms[2](x))

    def _context_kv(self, context):
        """Fused key/value projection of fixed context, as [B,heads,N,W/heads]."""
        a = self.context_attention
        b, n, w = context.shape
        kv = F.linear(context, a.in_proj_weight[w:], a.in_proj_bias[w:])
        return kv.view(b, n, 2, a.num_heads, -1).permute(2, 0, 3, 1, 4).unbind(0)


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
        return self._field(features, progress, *self._context(context, valid))

    def _context(self, context, valid):
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
        return self.context_projection(self.context_norm(context)), valid

    def _field(self, features, progress, context, valid, kv=None):
        kv = kv or {}
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
                x = block(x, context, valid, kv.get(block))
            scales.append(x)
        x = torch.cat(scales, 1)
        for block in self.fusion:
            x = block(x, context, valid, kv.get(block))
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
        self,
        context,
        trace=None,
        *,
        seed=None,
        sample_ids=None,
        steps=None,
        valid=None,
        cache_context=None,
    ):
        """`cache_context` (flow only): prepare the context and each block's context
        keys/values once per call and reuse them for every solver step. Default
        None enables it only in eval mode without autograd. It is always disabled
        in training mode or under autograd, including explicit True, so gradients
        use the unchanged path; False forces that path. The cache is a local of
        this call, never stored, so changed context, mask, weights or dtype are
        picked up by the next call. Outputs agree to float rounding, not bitwise."""
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
            cached = cache_context is not False and not (
                self.training or torch.is_grad_enabled()
            )
            if cached:
                prepared = self._context(context, valid)
                kv = {
                    m: m._context_kv(prepared[0])
                    for group in [*self.blocks.values(), self.fusion]
                    for m in group
                }

                def field(x, t):
                    validate(x, self.feature_spec)
                    return self._field(x, t, *prepared, kv)

            else:
                field = lambda x, t: self.field(x, t, context, valid)
            result = integrate(field, x, self.steps if steps is None else steps)
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


class ComponentRequest(nn.Module):
    """Supplied structured edit request -> generator context tokens.

    Context = source fine tokens plus one request token built from a bound memory
    component (e.g. a stored instance `visual_slot`) and a desired discrete state.
    The caller selects the entity; this module does not learn request selection.
    `null=True` removes both binding and state (request-erased control).
    """

    def __init__(self, width=64, states=2):
        super().__init__()
        if width < 1 or states < 2:
            raise ValueError("Positive width and at least two states required")
        self.width, self.states = width, states
        self.value = nn.Linear(width, width)
        self.state = nn.Embedding(states, width)
        self.kind = nn.Parameter(torch.randn(2, width) * 0.02)  # 0 source token, 1 request token

    def context(self, source, values, state, *, null=False):
        b = len(source)
        if (source.ndim != 3 or source.shape[-1] != self.width or b < 1
                or values.shape != (b, self.width) or state.shape != (b,)
                or state.dtype != torch.long or ((state < 0) | (state >= self.states)).any()
                or not torch.isfinite(source).all() or not torch.isfinite(values).all()):
            raise ValueError("Request needs source [B,N,W], values [B,W] and long states in range")
        request = self.kind[1].expand(b, -1)
        if not null:
            request = request + self.value(values) + self.state(state)
        return torch.cat((source + self.kind[0], request[:, None]), 1)


def edit_code(mode, source, *, generator=None, context=None, sample_ids=None, steps=None):
    """`reconstruct`: return `source` itself; the generator, normalization and RNG are
    never touched. `edit`: source fine tokens [B,N,C] plus the generator's sampled
    residual for its single feature scale (unstandardized). Creation is not an edit."""
    if mode == "reconstruct":
        return source
    if mode != "edit":
        raise ValueError("Mode must be reconstruct or edit")
    if len(generator.feature_spec) != 1:
        raise ValueError("Residual editing needs a single feature scale")
    (name, spec), = generator.feature_spec.items()
    if source.shape[1:] != (math.prod(spec.size), spec.channels):
        raise ValueError("Source tokens do not match the generator feature scale")
    delta = generator.features(context, sample_ids=sample_ids, steps=steps)[name]
    return source + delta.flatten(2).transpose(1, 2)
