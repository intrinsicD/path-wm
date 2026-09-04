"""Embodiment-specific raw action adapter for ABI v2.

What: continuous raw actions become fixed-width, timestamped action tokens.
How: an MLP handles observed values; a distinct learned token handles unknown/passive actions; invalid
padding is masked and zeroed after non-affine LayerNorm.
Why: treating an absent action as numerical zero confounds passive video with a real no-op and can make
causal dynamics appear correct for the wrong reason.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from contracts import ActionSequence, ActionTokens
from encoders.temporal import CoordinateEmbedding
from world_state.abi_v2 import ABIv2, ROOT, load_abi_v2


class ContinuousActionAdapter(nn.Module):
    def __init__(self, input_dim: int, abi: ABIv2) -> None:
        super().__init__()
        if input_dim < 1:
            raise ValueError("action input_dim must be positive")
        self.input_dim = input_dim
        self.abi = abi
        self.value = nn.Sequential(
            nn.Linear(input_dim, abi.action_dim),
            nn.GELU(),
            nn.Linear(abi.action_dim, abi.action_dim),
        )
        self.unknown = nn.Parameter(torch.empty(1, 1, abi.action_dim))
        self.time_embedding = CoordinateEmbedding(1, abi.action_dim)
        nn.init.trunc_normal_(self.unknown, std=0.02)

    def adapt_actions(self, actions: ActionSequence) -> ActionTokens:
        values = actions.values
        if values.ndim != 3 or values.shape[-1] != self.input_dim or not torch.is_floating_point(values):
            raise ValueError(f"raw actions must be floating (B,K,{self.input_dim})")
        shape = tuple(values.shape[:2])
        masks = (actions.valid_mask, actions.observed_mask)
        if any(tuple(mask.shape) != shape or mask.dtype != torch.bool for mask in masks):
            raise ValueError("action valid_mask and observed_mask must be bool (B,K)")
        if tuple(actions.timestamps.shape) != shape or not torch.is_floating_point(actions.timestamps):
            raise ValueError("action timestamps must be floating (B,K)")
        if values.shape[1] > self.abi.max_action_tokens:
            raise ValueError(
                f"action sequence length {values.shape[1]} exceeds ABI max {self.abi.max_action_tokens}"
            )
        if not torch.isfinite(values).all() or not torch.isfinite(actions.timestamps).all():
            raise ValueError("raw actions and timestamps must be finite")

        observed = self.value(values.float())
        unknown = self.unknown.expand(values.shape[0], values.shape[1], -1)
        tokens = torch.where(actions.observed_mask[..., None], observed, unknown)
        tokens = tokens + self.time_embedding(actions.timestamps[..., None])
        tokens = F.layer_norm(tokens, (self.abi.action_dim,))
        tokens = tokens.masked_fill(~actions.valid_mask[..., None], 0.0).to(self.abi.dtype)
        result = ActionTokens(
            tokens,
            actions.timestamps.clone(),
            actions.valid_mask.clone(),
            actions.observed_mask.clone(),
        )
        self.abi.check_action_tokens(
            result.tokens, result.timestamps, result.valid_mask, result.observed_mask
        )
        return result

    def forward(self, actions: ActionSequence) -> ActionTokens:
        return self.adapt_actions(actions)


def build_action_adapter(cfg: dict) -> nn.Module:
    section = cfg["action_adapter"]
    if section["kind"] != "continuous_mlp":
        raise ValueError("the common-base slice implements action_adapter.kind='continuous_mlp'")
    return ContinuousActionAdapter(int(section["input_dim"]), load_abi_v2(ROOT / cfg["abi"]))
