"""Attention and residual blocks extracted from the tested CNN/E-U-P references."""

import math
from torch import nn
from torch.nn import functional as F


class Attention(nn.Module):
    def __init__(self, width=64, heads=4, query_width=None):
        super().__init__()
        if width % heads:
            raise ValueError("Attention width must be divisible by heads")
        self.width, self.heads = width, heads
        self.query_projection = nn.Linear(query_width or width, width)
        self.key_projection = nn.Linear(width, width)
        self.value_projection = nn.Linear(width, width)
        self.output_projection = nn.Linear(width, width)

    def forward(self, query, key, value):
        def split(x):
            return x.reshape(
                x.shape[0], x.shape[1], self.heads, self.width // self.heads
            ).transpose(1, 2)

        q, k, v = (
            split(self.query_projection(query)),
            split(self.key_projection(key)),
            split(self.value_projection(value)),
        )
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        return self.output_projection(
            x.transpose(1, 2).reshape(query.shape[0], query.shape[1], self.width)
        )


def mlp(width):
    return nn.Sequential(
        nn.Linear(width, 2 * width), nn.GELU(), nn.Linear(2 * width, width)
    )


class TransformerBlock(nn.Module):
    def __init__(self, width=64, heads=4):
        super().__init__()
        self.attention_norm = nn.LayerNorm(width, eps=1e-5)
        self.attention = Attention(width, heads)
        self.mlp_norm = nn.LayerNorm(width, eps=1e-5)
        self.mlp = mlp(width)

    def forward(self, x):
        z = self.attention_norm(x)
        x = x + self.attention(z, z, z)
        return x + self.mlp(self.mlp_norm(x))


class SpatialResidual(nn.Module):
    def __init__(self, width, bias=True):
        super().__init__()
        groups = math.gcd(8, width)
        self.net = nn.Sequential(
            nn.GroupNorm(groups, width),
            nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1, bias=bias),
            nn.GroupNorm(groups, width),
            nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1, bias=bias),
        )

    def forward(self, x):
        return x + self.net(x)
