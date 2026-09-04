"""Counterfactual InfoNCE semantics and its stop-gradient target boundary (§6.4).

What: each predicted action branch must identify its matching next-state branch, while a collapsed set
scores at chance.
How: scalar latent transitions make the complete K x K distance matrix analytic; a learned action scale
checks that gradients reach the predictor but never the encoded target states.
Why: a diagonal-only loss would not contrast interventions, and target gradients could let both sides
move together instead of anchoring the predictor's action semantics (Invariant 1).
"""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from losses.counterfactual import counterfactual_loss


class ScaledAdditivePredictor(nn.Module):
    def __init__(self, scale: float) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale))

    def predict(self, W: torch.Tensor, actions: torch.Tensor, delta_t: int) -> torch.Tensor:
        assert delta_t == 1 and actions.shape[1] == 1
        return W + self.scale * actions[:, 0, 0, None, None]


def test_counterfactual_loss_matches_actions_and_stops_target_gradients():
    predictor = ScaledAdditivePredictor(scale=1.0)
    initial = torch.tensor([[[0.2]], [[-0.1]]])
    action_x = torch.tensor([[-0.8, -0.1, 0.6], [-0.5, 0.2, 0.9]])
    actions = torch.stack((action_x, torch.zeros_like(action_x)), dim=-1)
    targets = (initial[:, None] + action_x[:, :, None, None]).detach().requires_grad_()

    values = counterfactual_loss(predictor, initial, targets, actions, kappa=0.1)
    values["loss"].backward()

    assert values["accuracy"] == pytest.approx(1.0)
    assert values["loss"] < 0.02
    assert predictor.scale.grad is not None and torch.isfinite(predictor.scale.grad)
    assert targets.grad is None or torch.count_nonzero(targets.grad) == 0


def test_collapsed_counterfactuals_have_log_k_loss_and_chance_accuracy():
    predictor = ScaledAdditivePredictor(scale=0.0)
    initial = torch.zeros(2, 1, 1)
    actions = torch.zeros(2, 4, 2)
    targets = torch.zeros(2, 4, 1, 1)

    values = counterfactual_loss(predictor, initial, targets, actions, kappa=0.1)

    assert float(values["loss"].detach()) == pytest.approx(math.log(4.0))
    assert float(values["accuracy"]) == pytest.approx(0.25)
