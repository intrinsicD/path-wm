"""Markov transformer predictor P_phi: (W, action chunk, delta_t) -> W_hat (§5.6).

What: state tokens, one token per action, one delta-t token and fresh learned register tokens pass
through full-attention pre-LN blocks. The 65 state positions are read as a delta and returned as
non-affine LN(W + delta) in the ABI activation dtype.
How: actions use an MLP plus their within-chunk time embedding; delta_t uses a learned embedding;
consumer-side 2D RoPE is applied to grid-token queries and keys in ABI row-major order. Registers are
expanded from their learned initialization inside each call and discarded. All compute is fp32.
Why: E1_reference trains the frozen dynamics consumer used by the E2 H1 stitching gate. Markov state,
fresh registers, action conditioning and the normalized residual readout enforce Invariant 4 and the
transition-compatible ABI while keeping free-running rollouts stable.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from world_state.abi import ABI


def _rotate_axis(x: torch.Tensor, coordinate: torch.Tensor) -> torch.Tensor:
    """Apply one-axis rotary positions to x (..., tokens, axis_dim), pairing adjacent channels."""
    axis_dim = x.shape[-1]
    frequencies = torch.arange(0, axis_dim, 2, device=x.device, dtype=torch.float32) / axis_dim
    angles = coordinate.to(device=x.device, dtype=torch.float32)[:, None] * (10_000.0 ** -frequencies)
    pairs = x.reshape(*x.shape[:-1], axis_dim // 2, 2)
    even, odd = pairs[..., 0], pairs[..., 1]
    cos = angles.cos()[None, None, :, :]
    sin = angles.sin()[None, None, :, :]
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)


class Attention(nn.Module):
    """Pre-LN full self-attention with fixed 2D RoPE on the ABI grid-token positions."""

    def __init__(self, dim: int, heads: int, coordinates: torch.Tensor) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError(f"predictor dim {dim} not divisible by heads {heads}")
        head_dim = dim // heads
        if head_dim % 4:
            raise ValueError(f"2D RoPE needs predictor head dim divisible by 4, got {head_dim}")
        self.heads = heads
        self.head_dim = head_dim
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.to_out = nn.Linear(dim, dim)
        self.register_buffer("coordinates", coordinates, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, dim = x.shape
        qkv = self.to_qkv(self.norm(x)).reshape(batch, tokens, 3, self.heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)

        # Half of every head rotates by grid row, half by grid column. Global, action, delta-t and
        # register tokens have coordinate zero; action order is carried by its explicit time embedding.
        split = self.head_dim // 2
        coordinates = self.coordinates[:tokens]
        q = torch.cat(
            (_rotate_axis(q[..., :split], coordinates[:, 0]), _rotate_axis(q[..., split:], coordinates[:, 1])),
            dim=-1,
        )
        k = torch.cat(
            (_rotate_axis(k[..., :split], coordinates[:, 0]), _rotate_axis(k[..., split:], coordinates[:, 1])),
            dim=-1,
        )
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        return self.to_out(attended.permute(0, 2, 1, 3).reshape(batch, tokens, dim))


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: int, coordinates: torch.Tensor) -> None:
        super().__init__()
        self.attention = Attention(dim, heads, coordinates)
        self.feed_forward = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_ratio * dim),
            nn.GELU(),
            nn.Linear(mlp_ratio * dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(x)
        return x + self.feed_forward(x)


class TransformerPredictor(nn.Module):
    """The E1 transformer, with no context or mutable state beyond its learned parameters."""

    def __init__(
        self,
        abi: ABI,
        dim: int,
        layers: int,
        heads: int,
        mlp_ratio: int,
        n_registers: int,
        delta_t_conditioned: bool,
    ) -> None:
        super().__init__()
        abi.check_registers(n_registers)
        if layers < 1:
            raise ValueError(f"predictor.layers must be positive, got {layers}")
        if mlp_ratio < 1:
            raise ValueError(f"predictor.mlp_ratio must be positive, got {mlp_ratio}")

        self.abi = abi
        self.dim = dim
        self.n_registers = n_registers  # plain int: part of contracts.Predictor
        self.delta_t_conditioned = delta_t_conditioned
        self.state_input = nn.Identity() if dim == abi.dim else nn.Linear(abi.dim, dim)
        self.action_tokenizer = nn.Sequential(
            nn.Linear(abi.action_dims, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.action_time = nn.Parameter(torch.empty(abi.max_chunk, dim))
        self.delta_time = nn.Embedding(abi.max_chunk + 1, dim) if delta_t_conditioned else None
        self.delta_token = nn.Parameter(torch.empty(1, dim)) if not delta_t_conditioned else None
        self.register_tokens = nn.Parameter(torch.empty(n_registers, dim))

        max_tokens = abi.n_tokens + abi.max_chunk + 1 + n_registers
        coordinates = torch.zeros(max_tokens, 2, dtype=torch.float32)
        side = math.isqrt(abi.grid_tokens)
        if side * side != abi.grid_tokens:
            raise ValueError(f"2D RoPE needs a square ABI grid, got {abi.grid_tokens} tokens")
        grid = torch.arange(abi.grid_tokens)
        coordinates[abi.global_tokens : abi.n_tokens, 0] = grid // side
        coordinates[abi.global_tokens : abi.n_tokens, 1] = grid % side
        self.blocks = nn.ModuleList(Block(dim, heads, mlp_ratio, coordinates) for _ in range(layers))
        self.readout = nn.Linear(dim, abi.dim)

        nn.init.trunc_normal_(self.action_time, std=0.02)
        nn.init.trunc_normal_(self.register_tokens, std=0.02)
        if self.delta_token is not None:
            nn.init.trunc_normal_(self.delta_token, std=0.02)

    def _check_inputs(self, W: torch.Tensor, actions: torch.Tensor, delta_t: int) -> int:
        self.abi.check_state(W)
        if actions.ndim != 3:
            raise ValueError(f"actions must have shape (B, k, {self.abi.action_dims}), got {tuple(actions.shape)}")
        if actions.shape[0] != W.shape[0] or actions.shape[-1] != self.abi.action_dims:
            raise ValueError(
                f"actions shape {tuple(actions.shape)} incompatible with W batch {W.shape[0]} "
                f"and action width {self.abi.action_dims}"
            )
        chunk = actions.shape[1]
        if not isinstance(delta_t, int) or isinstance(delta_t, bool) or not 1 <= delta_t <= chunk <= self.abi.max_chunk:
            raise ValueError(
                f"need integer 1 <= delta_t <= chunk <= {self.abi.max_chunk}, got delta_t={delta_t!r}, chunk={chunk}"
            )
        if not torch.is_floating_point(actions):
            raise ValueError(f"actions must be floating point, got {actions.dtype}")
        lo, hi = self.abi.action_range
        if not torch.isfinite(actions).all() or (actions.numel() and (actions.min() < lo or actions.max() > hi)):
            raise ValueError(f"actions must be finite and within [{lo}, {hi}]")
        return chunk

    def predict(self, W: torch.Tensor, actions: torch.Tensor, delta_t: int) -> torch.Tensor:
        chunk = self._check_inputs(W, actions, delta_t)
        batch = W.shape[0]
        state = self.state_input(W.float())
        action_tokens = self.action_tokenizer(actions.float()) + self.action_time[:chunk]
        if self.delta_time is not None:
            dt_index = torch.full((batch,), delta_t, dtype=torch.long, device=W.device)
            delta_token = self.delta_time(dt_index)[:, None, :]
        else:
            delta_token = self.delta_token.expand(batch, 1, -1)
        registers = self.register_tokens.expand(batch, -1, -1)
        x = torch.cat((state, action_tokens, delta_token, registers), dim=1)
        for block in self.blocks:
            x = block(x)

        # Non-affine output LN is the ABI boundary. W.float() and every op above allocate, so neither
        # reservoir input can be modified and every returned state owns storage independent of later calls.
        delta = self.readout(x[:, : self.abi.n_tokens])
        predicted = F.layer_norm(W.float() + delta, (self.abi.dim,))
        return predicted.to(self.abi.dtype)

    def forward(self, W: torch.Tensor, actions: torch.Tensor, delta_t: int) -> torch.Tensor:
        return self.predict(W, actions, delta_t)
