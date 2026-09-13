"""Independent output modules; the recipe chooses their inputs and gradients."""

import torch
from torch import nn
from torch.nn import functional as F
from .blocks import SpatialResidual
from .features import validate
from .modalities import Attend


class PatchDetailHead(nn.Module):
    """Combine base patch means and independently supplied intra-patch detail.

    Both inputs must come from the same producer. No encoder or source image is
    owned by this head. The identity detail initialization is not a learned codec.
    """

    def __init__(self, base, patch_size=4):
        super().__init__()
        if patch_size < 1:
            raise ValueError("Patch size must be positive")
        self.base, self.patch_size = base, patch_size
        channels = 3 * patch_size**2
        self.projection = nn.Conv2d(channels, channels, 1, bias=False)
        with torch.no_grad():
            self.projection.weight.copy_(torch.eye(channels)[:, :, None, None])

    def forward(self, features):
        if "detail" not in features:
            raise ValueError("Decoder requires explicit detail features")
        base, detail, p = self.base(features), features["detail"], self.patch_size
        if (
            base.ndim != 4
            or base.shape[1] != 3
            or base.shape[-2] % p
            or base.shape[-1] % p
            or detail.shape
            != (len(base), 3 * p**2, base.shape[-2] // p, base.shape[-1] // p)
        ):
            raise ValueError("Base output and detail feature shapes disagree")
        residual = F.pixel_shuffle(self.projection(detail), p)
        residual = residual - F.interpolate(
            F.avg_pool2d(residual, p), scale_factor=p, mode="nearest"
        )
        means = F.interpolate(F.avg_pool2d(base, p), scale_factor=p, mode="nearest")
        return (means + residual).clamp(0, 1)


class StateFeatureDecoder(nn.Module):
    """Produce a spatial decoder's complete input from model state tokens.

    Calibration uses training features only; buffers travel with checkpoints.
    Freezing the head leaves the path to these predicted features differentiable.
    This deterministic producer is not a trained general generative prior.
    """

    def __init__(self, width, feature_spec, head):
        super().__init__()
        if not feature_spec:
            raise ValueError("A state decoder requires spatial feature specifications")
        self.width, self.feature_spec, self.head = width, dict(feature_spec), head
        self.queries = nn.ParameterDict()
        self.reader, self.projections = nn.ModuleDict(), nn.ModuleDict()
        self.refinements = nn.ModuleDict()
        for i, (name, spec) in enumerate(self.feature_spec.items()):
            self.queries[name] = nn.Parameter(
                torch.randn(spec.size[0] * spec.size[1], width) * 0.02
            )
            self.reader[name] = Attend(width)
            self.projections[name] = nn.Linear(width, spec.channels)
            self.register_buffer(f"mean_{i}", torch.zeros(1, spec.channels, 1, 1))
            self.register_buffer(f"scale_{i}", torch.ones(1, spec.channels, 1, 1))

    def enable_refinement(self):
        """Add one initially inactive pointwise residual MLP per feature scale.

        Existing weights and RNG streams are preserved. Only the final linear
        receives a gradient on the first step; earlier layers learn afterward.
        Trainability is chosen by the caller, alongside the rest of the producer.
        """
        if self.refinements:
            return self
        with torch.random.fork_rng(devices=[]):
            for name in self.feature_spec:
                layer = nn.Sequential(
                    nn.LayerNorm(self.width),
                    nn.Linear(self.width, 2 * self.width),
                    nn.GELU(),
                    nn.Linear(2 * self.width, self.width),
                )
                nn.init.zeros_(layer[-1].weight)
                nn.init.zeros_(layer[-1].bias)
                self.refinements[name] = layer.to(self.queries[name]).train(
                    self.training
                )
        return self

    @torch.no_grad()
    def calibrate(self, training_features):
        validate(training_features, self.feature_spec)
        if not all(
            torch.isfinite(training_features[k]).all() for k in self.feature_spec
        ):
            raise ValueError("Calibration requires finite training features")
        for i, name in enumerate(self.feature_spec):
            value = training_features[name]
            getattr(self, f"mean_{i}").copy_(value.mean((0, 2, 3), keepdim=True))
            getattr(self, f"scale_{i}").copy_(
                value.std((0, 2, 3), correction=0, keepdim=True).clamp_min(0.01)
            )

    def features(self, tokens, trace=None):
        if tokens.ndim != 3 or tokens.shape[-1] != self.width or tokens.shape[1] < 1:
            raise ValueError("State decoder requires nonempty [B,N,width] tokens")
        result = {}
        for i, (name, spec) in enumerate(self.feature_spec.items()):
            query = self.queries[name].expand(len(tokens), -1, -1)
            x = self.reader[name](
                query, tokens, trace=trace, name=f"decode.{name}.attention"
            )
            if self.refinements:
                x = x + self.refinements[name](x)
            x = (
                self.projections[name](x)
                .transpose(1, 2)
                .reshape(len(tokens), spec.channels, *spec.size)
            )
            result[name] = x * getattr(self, f"scale_{i}") + getattr(self, f"mean_{i}")
        return result

    def latent_loss(self, generated, target):
        validate(generated, self.feature_spec)
        validate(target, self.feature_spec)
        return sum(
            ((generated[k] - target[k]) / getattr(self, f"scale_{i}")).square().mean()
            for i, k in enumerate(self.feature_spec)
        ) / len(self.feature_spec)

    def forward(self, tokens, trace=None):
        return self.head(self.features(tokens, trace=trace))


class ReconstructionDecoder(nn.Module):
    """Small reference RGB decoder; compatible with equal-width fine/coarse maps."""

    def __init__(self, spec):
        super().__init__()
        if (
            not {"fine", "coarse"} <= spec.keys()
            or spec["fine"].channels != spec["coarse"].channels
        ):
            raise ValueError(
                "ReconstructionDecoder needs equal-width fine/coarse features; use DenseHead otherwise"
            )
        self.spec = {k: spec[k] for k in ("fine", "coarse")}
        width = spec["fine"].channels
        if width % 4:
            raise ValueError("Reference reconstruction width must be divisible by four")
        self.conv1 = nn.Conv2d(2 * width, width, 3, padding=1)
        self.conv2 = nn.Conv2d(width, width // 2, 3, padding=1)
        self.conv3 = nn.Conv2d(width // 2, width // 4, 3, padding=1)
        self.output_projection = nn.Conv2d(width // 4, 3, 1)

    def forward(self, features):
        validate(features, self.spec)
        f, c = features["fine"], features["coarse"]
        x = torch.cat((f, F.interpolate(c, f.shape[-2:], mode="nearest")), 1)
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(F.interpolate(x, scale_factor=2, mode="nearest")))
        x = F.gelu(self.conv3(F.interpolate(x, scale_factor=2, mode="nearest")))
        return self.output_projection(x).sigmoid()


class SpatialInput(nn.Module):
    """Select named levels, normalize/project each, align at the largest grid."""

    def __init__(
        self, spec, levels=("fine", "coarse"), width=128, retain_statistics=False
    ):
        super().__init__()
        if (
            not levels
            or any(k not in spec for k in levels)
            or len(set(levels)) != len(levels)
        ):
            raise ValueError(f"Choose distinct available levels: {tuple(spec)}")
        self.spec = {k: spec[k] for k in levels}
        self.retain_statistics = retain_statistics
        self.size = tuple(max(s.size[i] for s in self.spec.values()) for i in (0, 1))
        self.norms = nn.ModuleDict(
            {
                k: nn.Identity() if retain_statistics else nn.LayerNorm(s.channels)
                for k, s in self.spec.items()
            }
        )
        self.projections = nn.ModuleDict(
            {
                k: nn.Linear(s.channels + (2 if retain_statistics else 0), width)
                for k, s in self.spec.items()
            }
        )

    def forward(self, features):
        validate(features, self.spec)
        maps = []
        for k in self.spec:
            x = features[k].permute(0, 2, 3, 1)
            if self.retain_statistics:
                mean = x.mean(-1, keepdim=True)
                scale = (x.var(-1, unbiased=False, keepdim=True) + 1e-5).sqrt()
                x = torch.cat(((x - mean) / scale, mean, scale), -1)
            x = self.projections[k](self.norms[k](x)).permute(0, 3, 1, 2)
            maps.append(F.interpolate(x, self.size, mode="nearest"))
        return torch.cat(maps, 1)


class DenseHead(nn.Module):
    """Reusable mask/RGB/dense-map head; activation is an explicit output choice."""

    def __init__(
        self,
        spec,
        channels=1,
        levels=("fine", "coarse"),
        output_size=(64, 64),
        activation=None,
        retain_statistics=False,
    ):
        super().__init__()
        if activation not in (None, "sigmoid"):
            raise ValueError("Supported activations: None or sigmoid")
        self.input = SpatialInput(spec, levels, retain_statistics=retain_statistics)
        self.output_size, self.activation = output_size, activation
        self.trunk = nn.Sequential(
            nn.Conv2d(128 * len(levels), 128, 3, padding=1),
            SpatialResidual(128),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv2d(32, channels, 1),
        )

    def forward(self, features):
        x = self.trunk(self.input(features))
        if x.shape[-2:] != self.output_size:
            x = F.interpolate(x, self.output_size, mode="bilinear", align_corners=False)
        return x.sigmoid() if self.activation == "sigmoid" else x


def spatial_expectation(logits):
    h, w = logits.shape[-2:]
    y, x = torch.meshgrid(
        torch.linspace(0, 1, h, device=logits.device, dtype=logits.dtype),
        torch.linspace(0, 1, w, device=logits.device, dtype=logits.dtype),
        indexing="ij",
    )
    p = logits.flatten(2).softmax(-1)
    return torch.stack(
        ((p * x.flatten()).sum(-1), (p * y.flatten()).sum(-1)), -1
    ).flatten(1)


class PushTPoseHead(nn.Module):
    """Optional task readout: pusher XY, object XY, sin(theta), cos(theta).

    These labels constrain this head, not every encoder's representation.
    """

    def __init__(self, spec, levels=("fine", "coarse")):
        super().__init__()
        self.input = SpatialInput(spec, levels)
        self.trunk = nn.Sequential(
            nn.Conv2d(128 * len(levels), 128, 3, padding=1),
            SpatialResidual(128),
            SpatialResidual(128),
            nn.GroupNorm(8, 128),
            nn.GELU(),
        )
        self.locations = nn.Conv2d(128, 2, 1)
        self.orientation_attention = nn.Conv2d(128, 1, 1)
        self.orientation = nn.Sequential(
            nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 2)
        )

    def forward(self, features, return_maps=False):
        x = self.trunk(self.input(features))
        logits = self.locations(x)
        weights = self.orientation_attention(x).flatten(2).softmax(-1)
        pose = torch.cat(
            (
                spatial_expectation(logits),
                self.orientation((x.flatten(2) * weights).sum(-1)),
            ),
            1,
        )
        if return_maps:
            return (
                pose,
                logits.flatten(2).softmax(-1).reshape_as(logits),
                weights.reshape(-1, 1, *x.shape[-2:]),
            )
        return pose
