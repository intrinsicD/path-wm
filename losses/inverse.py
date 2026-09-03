"""Ground-truth inverse-dynamics anchoring loss for E1 (§6.3)."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def inverse_loss(inverse, states: torch.Tensor, actions: torch.Tensor, horizon: int) -> torch.Tensor:
    """MSE on the unbounded head output, before the contract clamp (DDR §13.14)."""
    current = states[:, :horizon].reshape(-1, *states.shape[2:])
    following = states[:, 1 : horizon + 1].reshape(-1, *states.shape[2:])
    targets = actions[:, :horizon].reshape(-1, actions.shape[-1]).float()
    return F.mse_loss(inverse.raw_action(current, following), targets)
