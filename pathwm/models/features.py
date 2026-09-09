"""Named spatial features. Shapes belong to a component, not the whole system."""

from dataclasses import dataclass
import torch
from torch import nn


@dataclass(frozen=True)
class FeatureSpec:
    channels: int
    size: tuple[int, int]
    source: str


def validate(features, spec):
    for name, item in spec.items():
        if name not in features:
            raise ValueError(f"Missing feature {name!r}; available: {tuple(features)}")
        x = features[name]
        expected = (item.channels, *item.size)
        if x.ndim != 4 or tuple(x.shape[1:]) != expected:
            raise ValueError(f"{name}: expected [B,{expected}], got {tuple(x.shape)}")
    if len({features[k].shape[0] for k in spec}) != 1:
        raise ValueError("Selected features must have the same batch size")


class ProjectFeatures(nn.Module):
    """Explicit adapter when a temporal model requires a shared token width."""

    def __init__(self, spec, width=64):
        super().__init__()
        self.input_spec = dict(spec)
        self.feature_spec = {
            k: FeatureSpec(width, v.size, f"projection({v.source})")
            for k, v in spec.items()
        }
        self.projections = nn.ModuleDict(
            {k: nn.Conv2d(v.channels, width, 1) for k, v in spec.items()}
        )

    def forward(self, features):
        validate(features, self.input_spec)
        return {k: p(features[k]) for k, p in self.projections.items()}


def tokens(features, spec):
    validate(features, spec)
    return torch.cat([features[k].flatten(2).transpose(1, 2) for k in spec], 1)


def grids(value, spec):
    result, offset = {}, 0
    for name, item in spec.items():
        count = item.size[0] * item.size[1]
        result[name] = (
            value[:, offset : offset + count]
            .transpose(1, 2)
            .reshape(-1, item.channels, *item.size)
        )
        offset += count
    if value.shape[1] != offset:
        raise ValueError("Token count differs from declared spatial features")
    return result
