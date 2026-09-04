"""Essential E1 loss behavior: stop-gradient targets, anti-collapse pressure, curriculum and masking."""
from __future__ import annotations

import copy

import pytest
import torch

from losses import build_objective
from losses.e1 import curriculum_horizon, curriculum_stage_start
from losses.rollout import rollout_losses
from losses.sigreg import SIGReg


class AdditivePredictor:
    def predict(self, W, actions, delta_t):
        delta = actions[:, :delta_t].sum(dim=(1, 2))[:, None, None]
        return W + delta


def test_rollout_targets_are_stop_gradient():
    source = torch.randn(2, 3, 4, requires_grad=True)
    target1 = torch.randn(2, 3, 4, requires_grad=True)
    target2 = torch.randn(2, 3, 4, requires_grad=True)
    states = torch.stack((source, target1, target2), dim=1)
    actions = torch.randn(2, 2, 2)
    values = rollout_losses(
        AdditivePredictor(),
        states,
        actions,
        gamma=0.9,
        horizon=2,
        action_weight=1.0,
        chunk_delta_t=2,
    )
    values["total"].backward()
    assert source.grad is not None and torch.isfinite(source.grad).all()
    # torch.stack's backward may materialize explicit zeros for unused slices; either form means
    # the detached target branches received no gradient.
    for target in (target1, target2):
        assert target.grad is None or torch.count_nonzero(target.grad) == 0


def test_sigreg_penalizes_a_collapsed_batch_more_than_gaussian():
    regularizer = SIGReg(dim=16, projections=64, knots=17, per_token=True, projection_chunk=16)
    gaussian = torch.randn(256, 3, 16, generator=torch.Generator().manual_seed(4))
    collapsed = torch.zeros_like(gaussian)
    assert regularizer(collapsed) > regularizer(gaussian)


def test_curriculum_reaches_full_horizon_at_end_of_growth():
    assert curriculum_horizon(0, 1000, 4, 0.1, 0.4) == 1
    assert curriculum_horizon(99, 1000, 4, 0.1, 0.4) == 1
    assert curriculum_horizon(299, 1000, 4, 0.1, 0.4) == 2
    assert curriculum_horizon(499, 1000, 4, 0.1, 0.4) == 4


def test_curriculum_stage_boundaries_are_explicit():
    assert curriculum_stage_start(0, 2000, 0.1, 0.4) == 0
    assert curriculum_stage_start(1, 2000, 0.1, 0.4) == 200
    assert curriculum_stage_start(2, 2000, 0.1, 0.4) == 1000
    with pytest.raises(ValueError, match="stage must be 0, 1, or 2"):
        curriculum_stage_start(3, 2000, 0.1, 0.4)


def test_weight_zero_inverse_is_not_computed(cfg, abi):
    spec = copy.deepcopy(cfg)
    spec["losses"]["reg"]["weight"] = 0.0
    spec["losses"]["inverse"]["weight"] = 0.0
    objective = build_objective(spec, abi)

    class MustNotRun:
        def raw_action(self, *args):
            raise AssertionError("weight-zero inverse term was computed")

    states = torch.randn(2, 2, abi.n_tokens, abi.dim).to(abi.dtype)
    actions = torch.zeros(2, 1, abi.action_dims)
    values = objective(AdditivePredictor(), MustNotRun(), states, actions, step=0, total_steps=10)
    assert torch.isfinite(values["total"])
    assert values["inverse"].item() == 0.0


def test_objective_does_not_reschedule_an_already_sized_window(cfg, abi):
    spec = copy.deepcopy(cfg)
    spec["losses"]["reg"]["weight"] = 0.0
    spec["losses"]["inverse"]["weight"] = 0.0
    objective = build_objective(spec, abi)
    expected = curriculum_horizon(949, 2000, 4, 0.1, 0.4)
    assert expected == 3
    states = torch.randn(2, expected + 1, abi.n_tokens, abi.dim).to(abi.dtype)
    actions = torch.zeros(2, expected, abi.action_dims)
    values = objective(AdditivePredictor(), object(), states, actions, step=949, total_steps=2000)
    assert values["horizon"] == expected


def test_counterfactual_term_starts_only_at_stage_two(cfg, abi):
    spec = copy.deepcopy(cfg)
    spec["losses"]["reg"]["weight"] = 0.0
    spec["losses"]["inverse"]["weight"] = 0.0
    spec["losses"]["rollout"].update(h_train=1, delta_t_max=1)
    spec["losses"]["counterfactual"].update(weight=1.0, k=3, kappa=0.1, batch_size=2)
    objective = build_objective(spec, abi)
    states = torch.randn(2, 2, abi.n_tokens, abi.dim).to(abi.dtype)
    actions = torch.zeros(2, 1, abi.action_dims)

    before = objective(AdditivePredictor(), object(), states, actions, step=999, total_steps=2000)
    assert before["counterfactual"].item() == 0.0
    assert before["counterfactual_positive"].item() == 0.0
    assert before["counterfactual_accuracy"] is None

    paired_states = torch.randn(2, 4, abi.n_tokens, abi.dim).to(abi.dtype)
    paired_actions = torch.zeros(2, 3, abi.action_dims)
    after = objective(
        AdditivePredictor(),
        object(),
        states,
        actions,
        step=1000,
        total_steps=2000,
        counterfactual_states=paired_states,
        counterfactual_actions=paired_actions,
    )
    assert torch.isfinite(after["counterfactual"])
    assert torch.isfinite(after["counterfactual_positive"])
    assert after["counterfactual_accuracy"] is not None


def test_counterfactual_term_can_start_with_joint_training(cfg, abi):
    spec = copy.deepcopy(cfg)
    spec["losses"]["reg"]["weight"] = 0.0
    spec["losses"]["inverse"]["weight"] = 0.0
    spec["losses"]["rollout"].update(h_train=1, delta_t_max=1)
    spec["losses"]["counterfactual"].update(weight=1.0, k=3, kappa=0.1, batch_size=2)
    spec["curriculum"]["counterfactual_from_stage"] = 0
    objective = build_objective(spec, abi)
    states = torch.randn(2, 2, abi.n_tokens, abi.dim).to(abi.dtype)
    actions = torch.zeros(2, 1, abi.action_dims)
    paired_states = torch.randn(2, 4, abi.n_tokens, abi.dim).to(abi.dtype)
    paired_actions = torch.zeros(2, 3, abi.action_dims)

    values = objective(
        AdditivePredictor(),
        object(),
        states,
        actions,
        step=0,
        total_steps=2000,
        counterfactual_states=paired_states,
        counterfactual_actions=paired_actions,
    )

    assert torch.isfinite(values["counterfactual"])
    assert values["counterfactual_accuracy"] is not None


def test_representation_first_stage_does_not_call_predictor(cfg, abi):
    spec = copy.deepcopy(cfg)
    spec["losses"]["reg"]["weight"] = 0.0
    spec["losses"]["inverse"]["weight"] = 0.0
    spec["curriculum"]["dynamics_from_stage"] = 1
    objective = build_objective(spec, abi)
    states = torch.randn(2, 2, abi.n_tokens, abi.dim).to(abi.dtype)
    actions = torch.zeros(2, 1, abi.action_dims)

    class MustNotRun:
        def predict(self, *args):
            raise AssertionError("predictor ran during representation-only stage")

    values = objective(MustNotRun(), object(), states, actions, step=0, total_steps=2000)
    assert values["action"].item() == 0.0
    assert values["rollout"].item() == 0.0
    assert values["chunk"].item() == 0.0


def test_paired_positive_weight_adds_matching_branch_error(cfg, abi):
    spec = copy.deepcopy(cfg)
    spec["losses"]["reg"]["weight"] = 0.0
    spec["losses"]["inverse"]["weight"] = 0.0
    spec["losses"]["rollout"].update(h_train=1, delta_t_max=1)
    spec["losses"]["counterfactual"].update(
        weight=0.0, positive_weight=2.0, k=3, kappa=0.1, batch_size=2
    )
    objective = build_objective(spec, abi)
    states = torch.randn(2, 2, abi.n_tokens, abi.dim).to(abi.dtype)
    actions = torch.zeros(2, 1, abi.action_dims)
    paired_states = torch.randn(2, 4, abi.n_tokens, abi.dim).to(abi.dtype)
    paired_actions = torch.zeros(2, 3, abi.action_dims)

    values = objective(
        AdditivePredictor(),
        object(),
        states,
        actions,
        step=1000,
        total_steps=2000,
        counterfactual_states=paired_states,
        counterfactual_actions=paired_actions,
    )

    ordinary = values["action"] + values["rollout"] + values["chunk"]
    torch.testing.assert_close(values["total"], ordinary + 2.0 * values["counterfactual_positive"])
