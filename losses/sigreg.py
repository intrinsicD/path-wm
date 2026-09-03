"""Sliced isotropic-Gaussian regularization for ABI state tokens (§6.1).

What: an Epps-Pulley-style empirical characteristic-function test over fixed random 1D projections,
with the configured knots on [0, 3] and a Gaussian integration window.
How: projected characteristic functions are compared with exp(-t^2/2), in fp32 and projection chunks;
`per_token` treats each token position as its own batch test, while false pools all token positions.
Why: E1 needs a collapse-excluding distributional term without an EMA target; fixed directions make
runs and checkpoint continuation reproducible while preserving the random-slicing construction.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SIGReg(nn.Module):
    def __init__(
        self,
        dim: int,
        projections: int,
        knots: int,
        per_token: bool,
        projection_chunk: int = 64,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if projections < 1 or knots < 2 or projection_chunk < 1:
            raise ValueError("SIGReg needs positive projections/chunks and at least two knots")
        generator = torch.Generator().manual_seed(seed)
        directions = F.normalize(torch.randn(projections, dim, generator=generator), dim=-1)
        self.register_buffer("directions", directions)
        self.register_buffer("knots", torch.linspace(0.0, 3.0, knots))
        self.per_token = per_token
        self.projection_chunk = projection_chunk

    def forward(self, W: torch.Tensor) -> torch.Tensor:
        if W.ndim != 3 or W.shape[-1] != self.directions.shape[-1]:
            raise ValueError(
                f"SIGReg expects (samples, tokens, {self.directions.shape[-1]}), got {tuple(W.shape)}"
            )
        # Explicitly disable an enclosing training autocast: the ABI fixes regularizer statistics to fp32.
        with torch.autocast(device_type=W.device.type, enabled=False):
            x = W.float()
            if not self.per_token:
                x = x.reshape(-1, 1, x.shape[-1])
            samples = x.shape[0]
            target_real = torch.exp(-0.5 * self.knots.square())
            window = torch.exp(-0.5 * self.knots.square())
            error_sum = x.new_zeros(())
            error_count = 0
            for start in range(0, self.directions.shape[0], self.projection_chunk):
                directions = self.directions[start : start + self.projection_chunk]
                projected = torch.einsum("btd,pd->btp", x, directions)
                phase = projected[..., None] * self.knots
                empirical_real = phase.cos().mean(dim=0)
                empirical_imag = phase.sin().mean(dim=0)
                error = window * ((empirical_real - target_real).square() + empirical_imag.square())
                error_sum = error_sum + error.sum()
                error_count += error.numel()
        # Epps-Pulley statistics scale by the number of samples; averaging tests keeps the loss
        # independent of token and projection counts while retaining that sample-size calibration.
            return samples * error_sum / error_count
