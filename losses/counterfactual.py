"""Paired-intervention InfoNCE action anchor for E1 and E2 (§6.4).

What: each prediction P(W0, a_i) classifies its matching next state among K branches sharing X0 and
reports its absolute diagonal MSE so the staged objective can explicitly balance ranking and fidelity.
How: the K x K logits are negative fp32 token MSE divided by kappa; diagonal indices are positives,
and every encoded next-state target is detached at the distance boundary.
Why: unlike opposite-action separation, InfoNCE requires the predictor's action response to have the
correct semantics, while the optional positive term prevents relative ranking by absolute overshoot.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def counterfactual_loss(
    predictor,
    initial: torch.Tensor,
    following: torch.Tensor,
    actions: torch.Tensor,
    *,
    kappa: float,
) -> dict[str, torch.Tensor]:
    """Return scalar InfoNCE loss and diagonal discrimination accuracy."""
    if initial.ndim != 3 or following.ndim != 4 or actions.ndim != 3:
        raise ValueError("counterfactual tensors must be (B,T,D), (B,K,T,D), and (B,K,A)")
    batch, branches = actions.shape[:2]
    if branches < 2 or following.shape[:2] != (batch, branches) or initial.shape[0] != batch:
        raise ValueError("counterfactual batch/branch dimensions do not align or K < 2")
    if following.shape[2:] != initial.shape[1:]:
        raise ValueError("counterfactual initial and next-state token layouts differ")
    if not math.isfinite(kappa) or kappa <= 0.0:
        raise ValueError(f"counterfactual kappa must be finite and positive, got {kappa}")

    repeated = initial[:, None].expand(-1, branches, -1, -1).reshape(
        batch * branches, *initial.shape[1:]
    )
    predicted = predictor.predict(
        repeated,
        actions.reshape(batch * branches, 1, actions.shape[-1]),
        1,
    ).reshape(batch, branches, *initial.shape[1:])
    targets = following.detach()  # Invariant 1/§6.1: encoded prediction targets never receive gradients.
    distances = (
        predicted[:, :, None].float() - targets[:, None].float()
    ).square().mean(dim=(-1, -2))
    logits = -distances / kappa
    labels = torch.arange(branches, device=actions.device)[None].expand(batch, -1)
    loss = F.cross_entropy(logits.reshape(batch * branches, branches), labels.reshape(-1))
    positive_mse = distances.diagonal(dim1=1, dim2=2).mean()
    accuracy = (distances.argmin(dim=-1) == labels).float().mean()
    return {"loss": loss, "positive_mse": positive_mse, "accuracy": accuracy}
