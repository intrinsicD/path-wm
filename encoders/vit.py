"""ViT-S/8 encoder E_A: o -> z, written here for E0 (§5.3, DDR §1 and §2).

What: a small pre-LN vision transformer over 64 px RGB. encode(obs uint8 (B, 3, 64, 64)) -> z float32
(B, 65, dim): token 0 is the cls token, tokens 1..64 the 8 x 8 patches in row-major order, which is
exactly abi_v1.yaml's token_order [global, grid], so the linear adapter maps it token by token.
How: preprocessing inside encode (uint8 -> float / 255, then (x - 0.5) / 0.5, never in place: the obs
belongs to the caller, DDR §18); an 8 x 8 stride-8 conv patch embed; learned cls token and learned
position embedding (encoder-internal — the ABI positions of §5.2 are the consumer's, applied by the
predictor, so this one is free); `layers` pre-LN blocks of full attention + GELU MLP; a final LayerNorm.
Why: E_A of E1_reference (H1, §4): the native encoder the predictor is trained with, and the reference
the foreign encoders of E2 are stitched against. Written here rather than reused because LeWM's ViT
comes from the stable_pretraining dependency (README); the Block / Attention / FeedForward below are
adapted from LeWM's module.py with einops replaced by reshape / permute (no einops dependency).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# adapted from lucas-maes/le-wm@8edfeb3:module.py (Attention, FeedForward, Block; einops removed, no dropout,
# one affine pre-LN per sublayer instead of LeWM's non-affine block norm followed by the sublayer's own affine norm)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Attention(nn.Module):
    """Full (non-causal) multi-head self-attention with a pre-LN; heads split by reshape / permute."""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        if dim % heads:
            raise ValueError(f"encoder dim {dim} not divisible by heads {heads}")
        self.heads = heads
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        qkv = self.to_qkv(self.norm(x)).reshape(B, T, 3, self.heads, D // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # each (B, heads, T, dim_head)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        return self.to_out(out.permute(0, 2, 1, 3).reshape(B, T, D))


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_dim: int):
        super().__init__()
        self.attn = Attention(dim, heads)
        self.mlp = FeedForward(dim, mlp_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x)
        return x + self.mlp(x)


class ViT(nn.Module):
    """ViT-S/8 shape (patch 8, mlp ratio 4); dim / layers / heads come from cfg so the dev spec can shrink it."""

    def __init__(self, dim: int, layers: int, heads: int, resolution: int, patch: int = 8, mlp_ratio: int = 4):
        super().__init__()
        if resolution % patch:
            raise ValueError(f"resolution {resolution} not a multiple of patch {patch}")
        self.grid = resolution // patch                 # 8 for E0: 64 grid tokens (abi_v1.yaml grid_tokens)
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch, stride=patch)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.grid * self.grid + 1, dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.blocks = nn.ModuleList(Block(dim, heads, mlp_ratio * dim) for _ in range(layers))
        self.norm = nn.LayerNorm(dim)  # final LN: O(1) tokens, which the adapter's LN sweep assumes (test_adapter.py)

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.dtype != torch.uint8 or obs.ndim != 4 or obs.shape[1] != 3:
            raise ValueError(f"obs must be (B, 3, H, W) uint8, got {tuple(obs.shape)} {obs.dtype}")
        # Own preprocessing, out of place (contracts.py Encoder): a fresh float tensor, obs untouched.
        x = (obs.to(torch.float32) / 255.0 - 0.5) / 0.5
        x = self.patch_embed(x)                          # (B, dim, 8, 8)
        x = x.flatten(2).transpose(1, 2)                 # (B, 64, dim), row-major over the grid = ABI token_order
        x = torch.cat([self.cls_token.expand(x.shape[0], -1, -1), x], dim=1) + self.pos_embed
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.encode(obs)
