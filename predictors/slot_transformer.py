"""Modality-neutral slot predictor for ABI v2.

What: fixed persistent belief slots plus timestamped action tokens and a continuous horizon token map to
the next belief state.
How: learned slot identities replace ABI-v1's visual 2-D RoPE; a padding-aware Transformer produces a
small normalized residual. Scratch registers are created and discarded per call.
Why: the dynamics consumer must depend on the canonical belief, not on whether evidence came from a
video grid, audio spectrogram, or a future sensor.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from contracts import ActionTokens
from encoders.temporal import CoordinateEmbedding, TokenTransformer
from world_state.abi_v2 import ABIv2


class SlotTransformerPredictor(nn.Module):
    def __init__(
        self,
        abi: ABIv2,
        dim: int,
        layers: int,
        heads: int,
        mlp_ratio: int,
        n_registers: int,
        readout_init_scale: float,
    ) -> None:
        super().__init__()
        if n_registers not in abi.register_options:
            raise ValueError(f"register count {n_registers} not in {abi.register_options}")
        if not math.isfinite(readout_init_scale) or readout_init_scale < 0:
            raise ValueError("predictor.readout_init_scale must be finite and non-negative")
        self.abi = abi
        self.dim = dim
        self.n_registers = n_registers
        self.state_input = nn.Identity() if dim == abi.dim else nn.Linear(abi.dim, dim)
        self.action_input = nn.Identity() if dim == abi.action_dim else nn.Linear(abi.action_dim, dim)
        self.state_positions = nn.Parameter(torch.empty(1, abi.n_tokens, dim))
        self.action_type = nn.Parameter(torch.empty(1, 1, dim))
        self.delta_type = nn.Parameter(torch.empty(1, 1, dim))
        self.delta_time = CoordinateEmbedding(1, dim)
        self.register_tokens = nn.Parameter(torch.empty(1, n_registers, dim))
        self.transformer = TokenTransformer(dim, layers, heads, mlp_ratio)
        self.readout = nn.Linear(dim, abi.dim)

        nn.init.trunc_normal_(self.state_positions, std=0.02)
        nn.init.trunc_normal_(self.action_type, std=0.02)
        nn.init.trunc_normal_(self.delta_type, std=0.02)
        nn.init.trunc_normal_(self.register_tokens, std=0.02)
        with torch.no_grad():
            self.readout.weight.mul_(readout_init_scale)
            self.readout.bias.mul_(readout_init_scale)

    def predict_state(
        self,
        W: torch.Tensor,
        actions: ActionTokens,
        delta_t: torch.Tensor,
    ) -> torch.Tensor:
        self.abi.check_state(W)
        self.abi.check_action_tokens(
            actions.tokens, actions.timestamps, actions.valid_mask, actions.observed_mask
        )
        batch = W.shape[0]
        if actions.tokens.shape[0] != batch:
            raise ValueError("state and action batches differ")
        if tuple(delta_t.shape) != (batch,) or not torch.is_floating_point(delta_t):
            raise ValueError(f"delta_t must be floating (B,), got {tuple(delta_t.shape)} {delta_t.dtype}")
        if not torch.isfinite(delta_t).all() or (delta_t < 0).any():
            raise ValueError("delta_t must be finite and non-negative")

        state = self.state_input(W.float()) + self.state_positions
        action = self.action_input(actions.tokens.float()) + self.action_type
        delta = self.delta_time(delta_t[:, None, None].float()) + self.delta_type
        registers = self.register_tokens.expand(batch, -1, -1)
        tokens = torch.cat((state, action, delta, registers), dim=1)
        valid = torch.cat(
            (
                torch.ones(batch, self.abi.n_tokens, dtype=torch.bool, device=W.device),
                actions.valid_mask.to(W.device),
                torch.ones(batch, 1 + self.n_registers, dtype=torch.bool, device=W.device),
            ),
            dim=1,
        )
        transformed = self.transformer(tokens, valid)
        residual = self.readout(transformed[:, : self.abi.n_tokens])
        predicted = F.layer_norm(W.float() + residual, (self.abi.dim,))
        return predicted.to(self.abi.dtype)

    def forward(
        self,
        W: torch.Tensor,
        actions: ActionTokens,
        delta_t: torch.Tensor,
    ) -> torch.Tensor:
        return self.predict_state(W, actions, delta_t)
