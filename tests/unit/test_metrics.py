"""Action-correctness controls for the first-slice instrument panel.

What: compare correct-action prediction against identity, zero-action and a deterministic trajectory
shuffle on the same one-step pairs.
How: a scalar oracle world makes each expected MSE analytic, including use of every probe time step and
the exact cross-trajectory permutation.
Why: action sensitivity can rise even when a predictor gives the wrong action semantics (DDR §13.18);
an unequal cohort or nondeterministic shuffle could reverse that diagnosis between evaluations.
"""
from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from evaluation.learning_curve import representation_diagnostics
from evaluation.metrics import evaluate_models
from evaluation.probe_set import ProbeSet
from training.data import PairedInterventionStore


class ScalarEncoder:
    def encode(self, observations: torch.Tensor) -> torch.Tensor:
        return observations.float().flatten(1) / 10.0


class TokenAdapter:
    def adapt(self, encoded: torch.Tensor) -> torch.Tensor:
        return encoded.unsqueeze(1)


class AdditivePredictor:
    def predict(self, W: torch.Tensor, actions: torch.Tensor, delta_t: int) -> torch.Tensor:
        assert delta_t == 1 and actions.shape[1] == 1
        return W + actions[:, 0, 0, None, None]


@dataclass
class OracleModels:
    encoder: ScalarEncoder = ScalarEncoder()
    adapter: TokenAdapter = TokenAdapter()
    predictor: AdditivePredictor = AdditivePredictor()

    def eval(self) -> None:
        pass


def test_action_correctness_controls_share_all_pairs_and_use_fixed_trajectory_shuffle():
    # Encoded scalar states advance by the first action coordinate. Actions stay within ABI range.
    action_x = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    actions = torch.stack((action_x, torch.zeros_like(action_x)), dim=-1)
    observations = torch.zeros(3, 3, 1, 1, 1, dtype=torch.uint8)
    observations[:, 0, 0, 0, 0] = torch.tensor([10, 30, 50], dtype=torch.uint8)
    observations[:, 1:, 0, 0, 0] = torch.tensor([[11, 13], [33, 37], [55, 61]], dtype=torch.uint8)

    metrics = evaluate_models(OracleModels(), ProbeSet(observations, actions), torch.device("cpu"))

    assert metrics["transition_error_one_step"] == pytest.approx(0.0, abs=1e-12)
    assert metrics["transition_error"] == pytest.approx(0.0, abs=1e-12)
    assert metrics["transition_error_identity"] == pytest.approx(91.0 / 600.0)
    assert metrics["transition_error_zero_action"] == pytest.approx(91.0 / 600.0)
    # torch.roll(..., dims=0) maps action trajectories 2->0, 0->1, 1->2: deltas .4, -.2, -.2.
    assert metrics["transition_error_shuffled_action"] == pytest.approx(0.08)


def test_representation_diagnostics_separate_scene_variation_and_transition_scale():
    initial = torch.tensor([10, 30], dtype=torch.uint8).reshape(2, 1, 1, 1)
    following = torch.tensor([11, 9, 33, 27], dtype=torch.uint8).reshape(2, 2, 1, 1, 1)
    action_x = torch.tensor([[0.1, -0.1], [0.3, -0.3]])
    actions = torch.stack((action_x, torch.zeros_like(action_x)), dim=-1)
    paired = PairedInterventionStore(initial, following, actions)

    diagnostics = representation_diagnostics(OracleModels(), paired, torch.device("cpu"), kappa=0.1)

    assert diagnostics["state_rms"] == pytest.approx((5.0**0.5))
    assert diagnostics["state_across_example_variance"] == pytest.approx(1.0)
    assert diagnostics["paired_positive_mse"] == pytest.approx(0.0, abs=1e-12)
    assert diagnostics["predicted_branch_separation"] == pytest.approx(
        diagnostics["target_branch_separation"]
    )
