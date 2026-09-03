"""Inverse-dynamics head I_omega: (W_t, W_{t+1}) -> a_hat_t (§5.7).

What: the first slice's head, `mlp_pooled_pair`, and its builder `build_inverse(cfg)` (DDR §13.6).
How: mean-pool the tokens of each state in fp32, concatenate [w, w', w' - w], run an MLP
3*dim -> hidden -> action_dims with GELU. `raw_action` is the unbounded pre-squash output the E1
loss consumes; `infer_action` clamps it into the ABI action range (DDR §13.7).
Why: L_inverse is the mandatory anti-collapse term of E1 (§6.1) and the head is the interface anchor
of E2 (§5.7). The output feeds Environment.step and planner inverse proposals (§7.3), so it must be an
action in the ABI action space (contracts.InverseDynamics); the loss is taken before the clamp so the
gradient does not die at the bounds. Labels never enter: the head sees ABI states only (Invariant 11).
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from world_state.abi import ABI, load_abi

ROOT = Path(__file__).resolve().parents[1]


class MLPPooledPair(nn.Module):
    def __init__(self, abi: ABI, hidden: int) -> None:
        super().__init__()
        self.abi = abi
        self.mlp = nn.Sequential(
            nn.Linear(3 * abi.dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, abi.action_dims),
        )

    def raw_action(self, W: torch.Tensor, W_next: torch.Tensor) -> torch.Tensor:
        """(B, 65, 192) x2 -> (B, action_dims) float32, unbounded: the E1 L_inverse target (DDR §13.7)."""
        self.abi.check_state(W)
        self.abi.check_state(W_next)
        # Consumers upcast the bf16 W inside; .float() on bf16 allocates, so the inputs stay untouched.
        w = W.float().mean(dim=1)
        w_next = W_next.float().mean(dim=1)
        return self.mlp(torch.cat([w, w_next, w_next - w], dim=-1))

    def infer_action(self, W: torch.Tensor, W_next: torch.Tensor) -> torch.Tensor:
        """The bounded contract output: clamp into the ABI action range (contracts.InverseDynamics)."""
        lo, hi = self.abi.action_range
        return self.raw_action(W, W_next).clamp(lo, hi)  # clamp allocates: no aliasing of internal state


def build_inverse(cfg: dict) -> nn.Module:
    """cfg is the whole parsed spec; reads inverse_dynamics.kind (and optional hidden, default 256) plus abi."""
    section = cfg["inverse_dynamics"]
    abi = load_abi(ROOT / cfg["abi"])
    heads = {"mlp_pooled_pair": MLPPooledPair}
    kind = section["kind"]
    if kind not in heads:
        raise KeyError(f"inverse_dynamics.kind {kind!r} not in {sorted(heads)}")
    return heads[kind](abi, hidden=int(section.get("hidden", 256)))
