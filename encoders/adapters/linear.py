"""Linear adapter A_m: z -> W in the ABI v1 layout (§5.4; DDR §3, §13.8).

What: adapt(z float (B, n, z_dim)) -> W bf16 (B, 65, 192): one Linear(z_dim -> dim) applied per token,
then a non-affine LayerNorm over dim, then the ABI activation dtype.
How: n == 65 means the encoder already emits [global, grid] (the ViT's cls + 8 x 8 patches) and every
token is projected in place; n == 64 means grid only, and the global token is the projection of the
token mean (a function of z, so it is not constant at init: LN of a constant is the zero vector).
Compute is fp32 (z.float()), the cast to bf16 is explicit at the end (producers emit bf16, DDR §13).
Why: the ABI is what makes "foreign" well-defined for H1 (§5.2); the adapter is the object under
training in E2, so it must be the cheapest map that lands in the layout, with no per-producer scale or
shift at the boundary (abi_v1.yaml per_token_affine: false) — an affine LN would let a foreign adapter
escape through scale (§14). First experiment to run it: E1_reference.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearAdapter(nn.Module):
    def __init__(self, z_shape: tuple[int, ...], n_tokens: int, dim: int, dtype: torch.dtype):
        super().__init__()
        if len(z_shape) != 2 or z_shape[0] not in (n_tokens, n_tokens - 1):
            raise ValueError(
                f"linear adapter needs z of per-sample shape ({n_tokens} or {n_tokens - 1}, z_dim), got {z_shape}; "
                "non-token encoders (conv feature maps) arrive with the encoder-zoo slice (DDR §2)"
            )
        self.has_global = z_shape[0] == n_tokens
        self.dim, self.dtype = dim, dtype
        self.proj = nn.Linear(int(z_shape[1]), dim)

    def adapt(self, z: torch.Tensor) -> torch.Tensor:
        z = z.float()                                                    # fp32 compute; never modifies z
        if not self.has_global:
            z = torch.cat([z.mean(dim=1, keepdim=True), z], dim=1)       # global token from the grid mean
        W = self.proj(z)                                                 # (B, 65, dim), token_order [global, grid]
        # ABI boundary LayerNorm: non-affine (abi_v1.yaml per_token_affine: false), per token over dim.
        W = F.layer_norm(W, (self.dim,))
        return W.to(self.dtype)                                          # producers emit the ABI dtype explicitly

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.adapt(z)
