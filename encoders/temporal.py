"""Shared temporal-token utilities for ABI-v2 evidence encoders.

What: continuous Fourier coordinates and a padding-aware pre-LN token transformer.
How: positions are encoded into native tokens before the ABI-v2 evidence projection; padding is masked
in attention and zeroed on output.
Why: video and audio share time but not spatial geometry.  The common utility handles time without
claiming that video space and audio frequency are canonical world-state coordinates.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


def fourier_features(coordinates: torch.Tensor, bands: int = 4) -> torch.Tensor:
    """Return [..., axes * (1 + 2*bands)] continuous coordinate features."""
    if not torch.is_floating_point(coordinates):
        coordinates = coordinates.float()
    frequencies = (2.0 ** torch.arange(bands, device=coordinates.device, dtype=torch.float32)) * math.pi
    angles = coordinates.float().unsqueeze(-1) * frequencies
    periodic = torch.cat((angles.sin(), angles.cos()), dim=-1)
    return torch.cat((coordinates.float().unsqueeze(-1), periodic), dim=-1).flatten(-2)


class CoordinateEmbedding(nn.Module):
    def __init__(self, axes: int, dim: int, bands: int = 4) -> None:
        super().__init__()
        self.axes = axes
        self.bands = bands
        self.projection = nn.Linear(axes * (1 + 2 * bands), dim)

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        if coordinates.shape[-1] != self.axes:
            raise ValueError(
                f"coordinate width {coordinates.shape[-1]} != configured axes {self.axes}"
            )
        return self.projection(fourier_features(coordinates, self.bands))


class TokenTransformer(nn.Module):
    """Small batch-independent Transformer over one modality's native tokens."""

    def __init__(self, dim: int, layers: int, heads: int, mlp_ratio: int) -> None:
        super().__init__()
        if layers < 1 or heads < 1 or dim % heads:
            raise ValueError(f"need layers>=1, heads>=1 and dim divisible by heads; got {dim=}, {layers=}, {heads=}")
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * mlp_ratio,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)

    def forward(self, tokens: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        if valid_mask.dtype != torch.bool or tuple(valid_mask.shape) != tuple(tokens.shape[:2]):
            raise ValueError("token valid_mask must be bool with shape (B,N)")
        if not valid_mask.any(dim=1).all():
            raise ValueError("every encoded observation row needs at least one valid token")
        transformed = self.blocks(tokens, src_key_padding_mask=~valid_mask)
        transformed = self.norm(transformed)
        return transformed.masked_fill(~valid_mask[..., None], 0.0)
