"""Independent output modules; the recipe chooses their inputs and gradients."""

import torch
from torch import nn
from torch.nn import functional as F
from .blocks import SpatialResidual
from .features import validate


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
