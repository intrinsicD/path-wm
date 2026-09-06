"""TwoRoom closed-loop evaluation interface: fast CPU checks against the pinned simulator.

Targets (not yet vendored/implemented; informative import failure is expected):
  third_party.swm.tworoom.TwoRoomEnv   vendored SWM two_room/env.py@6f1e499e, using
                                       third_party.swm.spaces instead of stable_worldmodel.spaces
  world_model.eval_tworoom.run_actions(state, target, actions, *, budget, seed=42)
                                       model-free control: executes the given (N,2) unit-box
                                       actions from `state` toward `target`, stops at success
                                       (distance < 16 px after a step) or `budget` steps, returns
                                       dict(success, initial_success, steps, state_distance)
  world_model.eval_tworoom.evaluate_case(model, source_frame, goal_frame, state, target, stats,
                                         *, seed, samples, iterations, elites, budget, frameskip)
                                       same contract as eval_pusht.evaluate_case: the policy sees
                                       only the injected source frame, the goal frame and the
                                       simulator's own renders; returns the same result keys.

Simulator facts these tests rely on (defaults of the pinned env): speed 5 px/step, agent
radius 7, vertical wall at x=112 with half-thickness 5, one door centred at y=49 with
half-extent 14 (+1.75 margin), border clamp to [21, 203], actions clamped to [-1, 1].
"""
import numpy as np
import pytest
import torch
from torch import nn

from third_party.swm.tworoom import TwoRoomEnv
from world_model.data import preprocess_pixels
from world_model.eval_tworoom import evaluate_case, run_actions

LEFT, RIGHT = [60.0, 112.0], [164.0, 112.0]
# Up to the door row, straight through the doorway, back down to the target row.
DOOR_PATH = np.array([[0, -1]] * 13 + [[1, 0]] * 21 + [[0, 1]] * 13, dtype=np.float32)


def fresh(state, target, seed=0):
    env = TwoRoomEnv()
    env.reset(seed=seed, options={'state': state, 'target_state': target})
    return env


def xs_after(env, action, steps):
    out = []
    for _ in range(steps):
        _, _, _, _, info = env.step(action)
        out.append(float(info['state'][0]))
    return out


def test_simulator_clips_actions_and_resolves_wall_collisions_one_step_late():
    # Outside the door the agent first touches the wall zone (x=100) and is only
    # pushed back to 99.5 on the following step; it never crosses.
    assert xs_after(fresh([95.0, 112.0], RIGHT), [1.0, 0.0], 3) == [100.0, 99.5, 99.5]
    # In the door row the same motion passes straight through the wall.
    assert xs_after(fresh([95.0, 49.0], RIGHT), [1.0, 0.0], 5) == [100.0, 105.0, 110.0, 115.0, 120.0]
    # Actions are clamped to the unit box before scaling by speed.
    assert xs_after(fresh(LEFT, RIGHT), [3.0, 0.0], 1) == [65.0]
    # Border clamp accounts for the agent radius.
    assert xs_after(fresh([200.0, 112.0], RIGHT), [1.0, 0.0], 2) == [203.0, 203.0]


def test_replay_reaches_goal_it_generated_while_stationary_and_short_budget_do_not():
    env = fresh(LEFT, RIGHT)
    for action in DOOR_PATH:
        _, _, _, _, info = env.step(action)
    goal = info['state'].astype(float).copy()
    assert goal[0] > 112, 'scripted path must end in the other room'

    replay = run_actions(LEFT, goal, DOOR_PATH, budget=len(DOOR_PATH))
    assert replay['success'] and not replay['initial_success']
    assert 0 < replay['steps'] < len(DOOR_PATH), 'stops at first step within 16 px, not at the end'
    assert replay['state_distance'] < 16

    stationary = run_actions(LEFT, goal, np.zeros_like(DOOR_PATH), budget=len(DOOR_PATH))
    assert not stationary['success'] and stationary['steps'] == len(DOOR_PATH)
    assert stationary['state_distance'] == pytest.approx(np.linalg.norm(goal - np.array(LEFT)))

    cut = run_actions(LEFT, goal, DOOR_PATH, budget=20)
    assert not cut['success'] and cut['steps'] == 20


class _RecordingModel(nn.Module):
    """Constant world model that records every observation batch it is asked to encode."""

    def __init__(self, dim=4):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))
        self.seen = []
        self.dim = dim

    def encode(self, pixels):
        self.seen.append(pixels.detach().clone())
        b, t = pixels.shape[:2]
        return torch.zeros(b, t, self.dim)

    def rollout(self, context, actions):
        return torch.zeros(len(actions), actions.shape[1] - context.shape[1] + 1, self.dim)


def _as_input(frame):
    return preprocess_pixels(torch.from_numpy(np.asarray(frame).copy()).permute(2, 0, 1)[None, None])


def test_evaluate_case_sees_only_injected_source_goal_and_simulator_renders_until_budget():
    true_source = fresh(LEFT, RIGHT).render()
    injected = (255 - true_source).astype(np.uint8)  # deliberately not the simulator render
    goal_frame = np.zeros_like(true_source)
    model = _RecordingModel()
    stats = dict(mean=np.zeros(2, dtype=np.float32), std=np.ones(2, dtype=np.float32))

    result = evaluate_case(model, injected, goal_frame, LEFT, RIGHT, stats, seed=1,
                           samples=6, iterations=2, elites=2, budget=30, frameskip=5)

    assert result['steps'] == 30 and not result['success'] and not result['initial_success']
    assert result['actions'].shape == (30, 2) and len(result['plans']) == 2
    assert result['predictor_steps'] == sum(p['predictor_steps'] for p in result['plans'])

    def seen(frame):
        ref = _as_input(frame)
        return any(s.shape == ref.shape and torch.equal(s, ref) for s in model.seen)

    assert seen(injected) and seen(goal_frame)
    assert not seen(true_source), 'the initial context must be the injected frame, not a re-render'
    # The only other observation is the simulator's own render after the first 25 executed actions.
    replay = fresh(LEFT, RIGHT)
    for action in result['actions'][:25]:
        replay.step(action)
    assert seen(replay.render())
    assert len(model.seen) == 3
