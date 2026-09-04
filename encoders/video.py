"""Tubelet video evidence encoder for the common ABI-v2 base.

What: raw uint8 clips become variable-length native space-time tokens with one timestamp per tubelet.
How: a non-overlapping Conv3d patch embed, continuous (time,y,x) Fourier coordinates, and a small
padding-aware Transformer.  Geometry is native evidence metadata, never ABI world-state topology.
Why: representation stages R0/R1 need a temporal video frontend; flattening independently encoded
frames would recreate the single-frame hidden-velocity failure measured in E1-a.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from contracts import EvidenceTokens, TemporalObservation
from encoders.temporal import CoordinateEmbedding, TokenTransformer


class TubeletVideoEncoder(nn.Module):
    modality = "video"

    def __init__(
        self,
        input_channels: int,
        dim: int,
        layers: int,
        heads: int,
        mlp_ratio: int,
        tubelet: int,
        patch: int,
    ) -> None:
        super().__init__()
        if tubelet < 1 or patch < 1:
            raise ValueError("video tubelet and patch sizes must be positive")
        self.input_channels = input_channels
        self.output_dim = dim
        self.tubelet = tubelet
        self.patch = patch
        self.patch_embed = nn.Conv3d(
            input_channels,
            dim,
            kernel_size=(tubelet, patch, patch),
            stride=(tubelet, patch, patch),
        )
        self.position = CoordinateEmbedding(3, dim)
        self.transformer = TokenTransformer(dim, layers, heads, mlp_ratio)

    def encode_observation(self, observation: TemporalObservation) -> EvidenceTokens:
        values, timestamps, valid = observation.values, observation.timestamps, observation.valid_mask
        if values.ndim != 5 or values.dtype != torch.uint8 or values.shape[2] != self.input_channels:
            raise ValueError(
                f"video must be uint8 (B,T,{self.input_channels},H,W), got {tuple(values.shape)} {values.dtype}"
            )
        batch, frames, _, height, width = values.shape
        if tuple(timestamps.shape) != (batch, frames) or not torch.is_floating_point(timestamps):
            raise ValueError("video timestamps must be floating (B,T)")
        if tuple(valid.shape) != (batch, frames) or valid.dtype != torch.bool:
            raise ValueError("video valid_mask must be bool (B,T)")
        if frames % self.tubelet or height % self.patch or width % self.patch:
            raise ValueError(
                f"video T/H/W {(frames, height, width)} must be divisible by "
                f"tubelet/patch {(self.tubelet, self.patch, self.patch)}"
            )

        # Conversion and normalization allocate new storage; caller-owned uint8 clips are untouched.
        x = (values.float() / 255.0 - 0.5) / 0.5
        x = self.patch_embed(x.permute(0, 2, 1, 3, 4))
        _, _, time_tokens, rows, columns = x.shape
        tokens = x.flatten(2).transpose(1, 2)

        tube_valid = valid.reshape(batch, time_tokens, self.tubelet).all(dim=-1)
        token_valid = tube_valid[:, :, None].expand(-1, -1, rows * columns).reshape(batch, -1)
        tube_times = timestamps.reshape(batch, time_tokens, self.tubelet).mean(dim=-1)
        token_times = tube_times[:, :, None].expand(-1, -1, rows * columns).reshape(batch, -1)

        y = torch.linspace(-1.0, 1.0, rows, device=values.device)
        x_coord = torch.linspace(-1.0, 1.0, columns, device=values.device)
        yy, xx = torch.meshgrid(y, x_coord, indexing="ij")
        yy = yy.flatten()[None, None].expand(batch, time_tokens, -1)
        xx = xx.flatten()[None, None].expand(batch, time_tokens, -1)
        coords = torch.stack((token_times.reshape(batch, time_tokens, -1), yy, xx), dim=-1).reshape(
            batch, -1, 3
        )
        tokens = tokens + self.position(coords)
        tokens = self.transformer(tokens, token_valid)
        return EvidenceTokens(tokens, token_times, token_valid, self.modality)

    def forward(self, observation: TemporalObservation) -> EvidenceTokens:
        return self.encode_observation(observation)
