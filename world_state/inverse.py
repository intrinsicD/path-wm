"""Inverse-dynamics head I_omega: (W_t, W_{t+1}) -> a_hat_t (§5.7).

What: the Step 3 `mlp_pooled_pair` fallback, the Step 4 `mlp_token_pair`, and their plain builder.
How: both form aligned [w, w', w' - w] features in fp32. The fallback averages states before its
MLP; the tokenwise head transforms every aligned pair first, then concatenates mean and max pools so
localized transition evidence cannot cancel before reaching the inverse loss. `raw_action` is the
unbounded pre-squash output the loss consumes; `infer_action` clamps it into the ABI action range.
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


class MLPTokenPair(nn.Module):
    """Token-correspondence inverse head used by the first Step 4 iteration.

    The per-token transform precedes aggregation. Mean pooling retains scene-wide evidence while max
    pooling gives a localized motion feature an undiluted route to the action head; both operate within
    each row, so independently batched worlds cannot interact.
    """

    def __init__(self, abi: ABI, hidden: int) -> None:
        super().__init__()
        self.abi = abi
        self.token_mlp = nn.Sequential(
            nn.Linear(3 * abi.dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.action_mlp = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, abi.action_dims),
        )

    def raw_action(self, W: torch.Tensor, W_next: torch.Tensor) -> torch.Tensor:
        self.abi.check_state(W)
        self.abi.check_state(W_next)
        current, following = W.float(), W_next.float()
        tokens = self.token_mlp(torch.cat((current, following, following - current), dim=-1))
        pooled = torch.cat((tokens.mean(dim=1), tokens.amax(dim=1)), dim=-1)
        return self.action_mlp(pooled)

    def infer_action(self, W: torch.Tensor, W_next: torch.Tensor) -> torch.Tensor:
        lo, hi = self.abi.action_range
        return self.raw_action(W, W_next).clamp(lo, hi)


def build_inverse(cfg: dict) -> nn.Module:
    """cfg is the whole parsed spec; reads inverse_dynamics.kind (and optional hidden, default 256) plus abi."""
    section = cfg["inverse_dynamics"]
    abi = load_abi(ROOT / cfg["abi"])
    heads = {"mlp_pooled_pair": MLPPooledPair, "mlp_token_pair": MLPTokenPair}
    kind = section["kind"]
    if kind not in heads:
        raise KeyError(f"inverse_dynamics.kind {kind!r} not in {sorted(heads)}")
    return heads[kind](abi, hidden=int(section.get("hidden", 256)))
