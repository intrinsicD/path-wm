"""PushT-specific contracts: absolute actions, circular goals, and primitive CEM."""
import copy
import math
import random

import numpy as np
import pytest
import torch
from torch import nn

from world_model.paddle.types import ObservationLatent, PlanningState
from world_model.pusht.models import (
    Encoder, Decoder, MemoryUpdater, Predictor, PoseReadout, StateReadout,
    initial_previous_action, rollout,
)
from world_model.pusht.planner import CEMPlanner, PlanningFailure, pose_cost
from world_model.pusht.env import PushTEnv


def latent(pose=(.2, .3, .4, .5, 0., 1.), batch=1):
    fine = torch.zeros(batch, 256, 64)
    fine[:, 0, :6] = torch.tensor(pose)
    return ObservationLatent(fine, torch.zeros(batch, 64, 64))


class PoseHead(nn.Module):
    def forward(self, observation):
        return observation.fine[:, 0, :6]


class TargetPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, observation, memory, action):
        self.calls.append((observation.clone(), memory.clone(), action.clone()))
        prediction = observation.clone()
        prediction.fine[:, 0, :2] = action
        prediction.fine[:, 0, 2:4] = action
        return prediction


class RecordingUpdater(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, memory, observation, previous_action):
        self.calls.append((memory.clone(), observation.clone(), previous_action.clone()))
        return memory + observation.fine[:, 0, :1] + previous_action[:, :1]


def test_task_dimensions_start_marker_and_zero_predictor_copy():
    torch.manual_seed(9130)
    encoder, decoder, updater = Encoder(), Decoder(), MemoryUpdater()
    predictor, pose_head, state_head = Predictor(), PoseReadout(), StateReadout()
    observation = encoder(torch.rand(2, 3, 64, 64))
    marker = initial_previous_action(2)
    assert torch.equal(marker, torch.full((2, 2), -1.))
    memory = updater(torch.zeros(2, 128), observation, marker)
    assert updater.gru.input_size == 66
    assert memory.shape == (2, 128)
    assert pose_head(observation).shape == (2, 6)
    assert state_head(memory).shape == (2, 11)
    assert decoder(observation).shape == (2, 3, 64, 64)
    assert predictor.action_projection.in_features == 2
    prediction = predictor(observation, memory, torch.tensor([[0., 0.], [.2, .8]]))
    assert torch.equal(prediction.fine, observation.fine)
    assert torch.equal(prediction.coarse, observation.coarse)
    with pytest.raises(ValueError, match="executable"):
        predictor(observation, memory, marker)
    with pytest.raises(ValueError, match="previous"):
        updater(memory, observation, torch.tensor([[-1., 0.], [.2, .8]]))


def test_push_rollout_is_p_then_u_with_previous_candidate_and_branch_memory():
    predictor, updater = TargetPredictor(), RecordingUpdater()
    actual = PlanningState(latent(), torch.full((1, 128), .3))
    before = actual.clone()
    actions = torch.tensor([[[.1, .2], [.8, .9]]])
    states = rollout(actual, actions, predictor, updater)
    assert len(states) == 2
    assert torch.equal(predictor.calls[0][1], before.memory)
    assert torch.equal(updater.calls[0][1].fine, states[0].observation.fine)
    assert torch.equal(updater.calls[0][2], actions[:, 0])
    assert torch.equal(predictor.calls[1][1], states[0].memory)
    assert torch.equal(updater.calls[1][2], actions[:, 1])
    assert torch.equal(actual.memory, before.memory)
    assert torch.equal(actual.observation.fine, before.observation.fine)
    with pytest.raises(ValueError, match="executable"):
        rollout(actual, torch.full((1, 1, 2), -1.), predictor, updater)


def test_frozen_push_observer_transmits_future_prediction_gradient():
    torch.manual_seed(9131)
    updater = MemoryUpdater().requires_grad_(False)
    fine = torch.randn(1, 256, 64, requires_grad=True)
    observation = ObservationLatent(fine, torch.randn(1, 64, 64, requires_grad=True))
    memory = updater(torch.zeros(1, 128), observation, torch.tensor([[.2, .3]]))
    memory.square().sum().backward()
    assert fine.grad is not None and fine.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in updater.parameters())


def test_goal_cost_wraps_angles_normalizes_pairs_and_rejects_degenerate_orientation():
    angle = .01
    goal = torch.tensor([[.2, .3, .4, .5, math.sin(angle), math.cos(angle)]], dtype=torch.float64)
    predicted = goal.repeat(4, 1)
    predicted[0, 4:] = 5 * torch.tensor([math.sin(2*math.pi-angle), math.cos(2*math.pi-angle)])
    predicted[1, 0] += 20/512
    predicted[2, 2] += 20/512
    predicted[3, 4:] = 0
    cost = pose_cost(predicted, goal)
    assert cost[0] == pytest.approx((.02/(math.pi/9))**2, rel=1e-6)
    assert cost[1] == pytest.approx(0.)  # Goal cost ignores final pusher location.
    assert cost[2] == pytest.approx(1.)
    assert torch.isinf(cost[3])
    with pytest.raises(PlanningFailure, match="goal"):
        pose_cost(predicted, torch.zeros(1, 6))


def test_cem_keeps_actual_memory_private_rng_bounded_actions_and_scores_final_mean():
    predictor, updater = TargetPredictor(), RecordingUpdater()
    actual = PlanningState(latent(), torch.zeros(1, 128))
    before = actual.clone()
    goal = latent((.7, .8, .7, .8, 0., 1.))
    python_rng, numpy_rng = random.getstate(), copy.deepcopy(np.random.get_state())
    torch_rng = torch.get_rng_state().clone()
    planner = CEMPlanner(predictor, updater, PoseHead(), candidates=16, iterations=2, elites=4, seed=9140)
    prepared = planner.prepare_goal(goal)
    result = planner.plan(actual, prepared)
    assert result.action.shape == (2,)
    assert result.sequence.shape == (5, 2)
    assert torch.equal(result.action, result.sequence[0])
    assert torch.all((result.sequence >= 0) & (result.sequence <= 1))
    assert result.stats['predictor_transitions'] == (16*2+1)*5
    assert result.stats['final_mean_evaluated'] is True
    assert len(predictor.calls) == 15
    for index in range(5):
        assert torch.equal(predictor.calls[index][2][0], before.observation.fine[0, 0, :2].clamp(0, 1))
    assert predictor.calls[-1][2].shape == (1, 2)
    assert torch.equal(predictor.calls[-1][2][0], result.sequence[-1])
    # Each iteration and final-mean evaluation restart every candidate's memory.
    for index in (0, 5, 10):
        assert torch.equal(predictor.calls[index][1], torch.zeros_like(predictor.calls[index][1]))
    assert torch.equal(actual.memory, before.memory)
    assert torch.equal(actual.observation.fine, before.observation.fine)
    assert random.getstate() == python_rng
    after_numpy = np.random.get_state()
    assert numpy_rng[0] == after_numpy[0] and numpy_rng[2:] == after_numpy[2:]
    np.testing.assert_array_equal(numpy_rng[1], after_numpy[1])
    assert torch.equal(torch.get_rng_state(), torch_rng)
    matched = CEMPlanner(TargetPredictor(), RecordingUpdater(), PoseHead(), candidates=16, iterations=2, elites=4, seed=9140)
    repeat = matched.plan(actual, matched.prepare_goal(goal))
    assert torch.equal(result.sequence, repeat.sequence)
    # The goal must be an encoded image or its explicitly prepared H readout.
    with pytest.raises((TypeError, ValueError)):
        planner.plan(actual, np.array([100, 100, 200, 200, 0]))


def test_cem_explicitly_rejects_invalid_final_mean_and_all_invalid_predictions():
    class InvalidPredictor(TargetPredictor):
        def forward(self, observation, memory, action):
            result = super().forward(observation, memory, action)
            result.fine[:, 0, 4:] = float('nan')
            return result

    planner = CEMPlanner(InvalidPredictor(), RecordingUpdater(), PoseHead(), candidates=4, iterations=1, elites=2)
    with pytest.raises(PlanningFailure, match="invalid"):
        planner.plan(PlanningState(latent(), torch.zeros(1, 128)), latent())

    class MeanInvalidPredictor(TargetPredictor):
        def forward(self, observation, memory, action):
            result = super().forward(observation, memory, action)
            if action.shape[0] == 1:
                result.fine[:, 0, 4:] = 0
            return result

    planner = CEMPlanner(MeanInvalidPredictor(), RecordingUpdater(), PoseHead(), candidates=4, iterations=1, elites=2)
    with pytest.raises(PlanningFailure, match="final mean"):
        planner.plan(PlanningState(latent(), torch.zeros(1, 128)), latent())


def test_wrapper_fresh_rgb_absolute_scaling_one_primitive_and_privileged_hold(monkeypatch):
    from world_model.pusht.data import canonical_frame
    env = PushTEnv()
    initial = np.array([100., 100., 256., 256., .3])
    goal = np.array([400., 400., 270., 250., .4])
    frame, info = env.reset(initial, goal, seed=9150)
    assert frame.shape == (64, 64, 3) and frame.dtype == np.uint8
    np.testing.assert_array_equal(frame, canonical_frame(env.simulator.render()))
    assert info['pose_world'].shape == (5,)
    np.testing.assert_array_equal(env.hold_action, info['pose_world'][:2]/512)
    assert np.any(env.hold_action != 0)
    assert env.simulator.relative is False and env.simulator.render_size == 96
    original_step = env.simulator.step
    commands = []
    def step(command):
        commands.append(np.array(command, copy=True))
        return original_step(command)
    monkeypatch.setattr(env.simulator, 'step', step)
    result = env.step(np.array([.25, .75]))
    assert len(commands) == 1
    np.testing.assert_array_equal(commands[0], [128., 384.])
    assert len(result) == 5
    assert result[4]['primitive_steps'] == 1
    assert result[4]['elapsed_seconds'] == .1
    assert result[0].shape == (64, 64, 3)
    for bad in ([-1., -1.], [1.1, .5], [.5, float('nan')], [1, 2, 3]):
        with pytest.raises(ValueError, match="action"):
            env.step(bad)
    assert len(commands) == 1
    env.close()


def test_wrapper_block_success_ignores_pusher_and_pose_property_is_copy():
    env = PushTEnv()
    initial = np.array([100., 100., 256., 256., .3])
    goal = np.array([450., 450., 256., 256., .3])
    _, info = env.reset(initial, goal, seed=9151)
    assert info['success'] is True
    assert not bool(env.simulator.eval_state(env.simulator.goal_state, env.simulator._get_obs())[0])
    pose = env.pose
    pose[:] = 0
    assert np.any(env.pose != 0)
    before = env.render().copy()
    changed_goal = goal.copy()
    changed_goal[4] += math.pi/9 + .01
    env.set_goal(changed_goal)
    np.testing.assert_array_equal(env.render(), before)  # Numeric goal doesn't move green drawing.
    _, _, terminated, _, info = env.step(env.hold_action)
    assert terminated is False and info['success'] is False
    assert info['block_distance'] < 1e-6
    assert info['angle_error'] > math.pi/9
    env.close()


def test_cem_uses_recorded_per_axis_proposal_scale_with_private_draws():
    predictor = TargetPredictor()
    actual = PlanningState(latent(), torch.zeros(1, 128))
    planner = CEMPlanner(predictor, RecordingUpdater(), PoseHead(), candidates=4, iterations=1,
                         elites=2, seed=9160, initial_std=(.02, .04))
    result = planner.plan(actual, latent())
    noise = torch.randn((4, 5, 2), generator=torch.Generator().manual_seed(9160))
    expected = (torch.tensor([.2, .3]) + noise*torch.tensor([.02, .04])).clamp(0, 1)
    expected[0] = torch.tensor([.2, .3])
    for index in range(5):
        assert torch.equal(predictor.calls[index][2], expected[:, index])
    assert result.stats['initial_std'] == [.02, .04]


def test_desired_goal_does_not_change_reset_physics_or_observation():
    initial = np.array([100., 100., 256., 256., .3])
    goals = (initial, np.array([256., 256., 256., 256., .3]))
    snapshots = []
    for goal in goals:
        env = PushTEnv()
        try:
            frame, _ = env.reset(initial, goal, seed=123)
            snapshots.append((frame, env.pose,
                np.array([*env.simulator.agent.velocity, env.simulator.agent.angular_velocity,
                          *env.simulator.block.velocity, env.simulator.block.angular_velocity])))
        finally:
            env.close()
    for first, second in zip(*snapshots):
        np.testing.assert_array_equal(first, second)
