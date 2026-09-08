"""Independent typed readouts on explicitly named fine and coarse feature grids.

The backbone remains frozen. Native source width is normalized per token; each
consumer owns its projection so one task cannot silently bottleneck another.
"""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F


class Residual(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.net = nn.Sequential(nn.GroupNorm(8, width), nn.GELU(),
                                 nn.Conv2d(width, width, 3, padding=1),
                                 nn.GroupNorm(8, width), nn.GELU(),
                                 nn.Conv2d(width, width, 3, padding=1))

    def forward(self, x):
        return x + self.net(x)


class SpatialInput(nn.Module):
    def __init__(self, source_width, width=128, access='both'):
        super().__init__()
        if access not in ('both', 'fine', 'coarse'):
            raise ValueError('unknown scale access')
        self.access = access
        self.fine_norm = nn.LayerNorm(source_width)
        self.coarse_norm = nn.LayerNorm(source_width)
        self.fine_projection = nn.Linear(source_width, width)
        self.coarse_projection = nn.Linear(source_width, width)

    def forward(self, fine, coarse):
        if fine.shape[1] != 256 or coarse.shape[1] != 64:
            raise ValueError('expected explicit 16x16 and 8x8 grids')
        f = self.fine_projection(self.fine_norm(fine)).transpose(1, 2).reshape(-1, 128, 16, 16)
        c = self.coarse_projection(self.coarse_norm(coarse)).transpose(1, 2).reshape(-1, 128, 8, 8)
        c = F.interpolate(c, (16, 16), mode='nearest')
        # Zero the inaccessible branch *after* projection, preventing learned bias
        # from acting as an unintended extra spatial input.
        if self.access == 'coarse':
            f = torch.zeros_like(f)
        if self.access == 'fine':
            c = torch.zeros_like(c)
        return torch.cat((f, c), dim=1)


def spatial_expectation(logits):
    """Two ordered x,y estimates on a learned output basis spanning [0,1]."""
    h, w = logits.shape[-2:]
    y, x = torch.meshgrid(torch.linspace(0, 1, h, device=logits.device, dtype=logits.dtype),
                          torch.linspace(0, 1, w, device=logits.device, dtype=logits.dtype), indexing='ij')
    probabilities = logits.flatten(2).softmax(-1)
    return torch.stack(((probabilities * x.flatten()).sum(-1),
                        (probabilities * y.flatten()).sum(-1)), -1).flatten(1)


class DenseHead(nn.Module):
    def __init__(self, source_width, channels, access='both'):
        super().__init__()
        self.input = SpatialInput(source_width, access=access)
        self.trunk = nn.Sequential(nn.Conv2d(256, 128, 3, padding=1), Residual(128),
            nn.GroupNorm(8, 128), nn.GELU(), nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.GELU(),
            nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv2d(64, 32, 3, padding=1),
            nn.GroupNorm(8, 32), nn.GELU(), nn.Conv2d(32, channels, 1))
        self.rgb = channels == 3

    def forward(self, fine, coarse):
        output = self.trunk(self.input(fine, coarse))
        return output.sigmoid() if self.rgb else output


class PoseHead(nn.Module):
    def __init__(self, source_width, access='both'):
        super().__init__()
        self.input = SpatialInput(source_width, access=access)
        self.trunk = nn.Sequential(nn.Conv2d(256, 128, 3, padding=1), Residual(128),
                                   Residual(128), nn.GroupNorm(8, 128), nn.GELU())
        self.locations = nn.Conv2d(128, 2, 1)
        self.orientation_attention = nn.Conv2d(128, 1, 1)
        self.orientation = nn.Sequential(nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 2))

    def forward(self, fine, coarse, return_maps=False):
        x = self.trunk(self.input(fine, coarse))
        logits = self.locations(x)
        weights = self.orientation_attention(x).flatten(2).softmax(-1)
        pooled = (x.flatten(2) * weights).sum(-1)
        pose = torch.cat((spatial_expectation(logits), self.orientation(pooled)), 1)
        if return_maps:
            return pose, logits.flatten(2).softmax(-1).reshape_as(logits), weights.reshape(-1, 1, 16, 16)
        return pose


def make_heads(source_width, seed, device='cpu', access='both'):
    """Pair all shape-compatible trunk draws; isolate input-width-dependent RNG."""
    result = {}
    for index, (name, factory) in enumerate((('rgb', lambda: DenseHead(64, 3, access)),
                                            ('mask', lambda: DenseHead(64, 1, access)),
                                            ('pose', lambda: PoseHead(64, access)))):
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed + 1009 * index)
            head = factory()
            torch.random.default_generator.manual_seed(seed + 100003 + 1009 * index)
            head.input = SpatialInput(source_width, access=access)
        result[name] = head.to(device)
    return result


def feature_grids(tokens, encoder):
    if encoder == 'cnn':
        if tokens.shape[1:] != (320, 64):
            raise ValueError('CNN feature contract')
        return tokens[:, :256], tokens[:, 256:]
    if encoder == 'vit':
        if tokens.shape[1:] != (256, 384):
            raise ValueError('ViT feature contract')
        pooled = F.avg_pool2d(tokens.transpose(1, 2).reshape(-1, 384, 16, 16), 2)
        return tokens, pooled.flatten(2).transpose(1, 2)
    raise ValueError('unknown encoder package')
