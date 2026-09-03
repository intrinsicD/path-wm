"""Action-conditioned and free-running latent rollout losses for E1 (§6.2, DDR §13.12).

All target states are detached at the comparison line. Inputs and predictions remain differentiable;
distances are fp32 per-token MSE averaged over batch and token positions.
"""
from __future__ import annotations

import torch


def token_mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean of per-token channel MSE; target is always stop-gradient (§6.1)."""
    target = target.detach()  # Invariant 1/§6.1: no prediction loss gradient enters a target branch.
    return (prediction.float() - target.float()).square().mean(dim=-1).mean()


def rollout_losses(
    predictor,
    states: torch.Tensor,
    actions: torch.Tensor,
    *,
    gamma: float,
    horizon: int,
    action_weight: float,
    chunk_delta_t: int | None,
) -> dict[str, torch.Tensor]:
    """Compute k=1 action loss, k>=2 free-running loss, and one VLWM chunk target."""
    if states.ndim != 4 or actions.ndim != 3 or states.shape[:2] != (actions.shape[0], actions.shape[1] + 1):
        raise ValueError("states/actions must be (B, H+1, tokens, dim) and (B, H, action_dims)")
    if not 1 <= horizon <= actions.shape[1]:
        raise ValueError(f"horizon {horizon} outside available action length {actions.shape[1]}")
    zero = torch.zeros((), device=states.device, dtype=torch.float32)
    action_loss = zero
    free_running_loss = zero
    predicted = states[:, 0]
    for k in range(1, horizon + 1):
        predicted = predictor.predict(predicted, actions[:, k - 1 : k], 1)
        term = (gamma**k) * token_mse(predicted, states[:, k])
        if k == 1:
            action_loss = term
        else:
            free_running_loss = free_running_loss + term

    chunk_loss = zero
    if chunk_delta_t is not None:
        if not 1 <= chunk_delta_t <= horizon:
            raise ValueError(f"chunk_delta_t {chunk_delta_t} outside trained horizon {horizon}")
        chunk_prediction = predictor.predict(states[:, 0], actions[:, :chunk_delta_t], chunk_delta_t)
        chunk_loss = (gamma**chunk_delta_t) * token_mse(chunk_prediction, states[:, chunk_delta_t])
    return {
        "action": action_loss,
        "rollout": free_running_loss,
        "chunk": chunk_loss,
        "total": action_weight * action_loss + free_running_loss + chunk_loss,
    }
