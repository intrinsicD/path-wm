"""Scientific parity for state-only simulator planning; timings are diagnostics."""

import copy
import random

import numpy as np
import pytest
import torch

from world_model.paddle import evaluation
from world_model.paddle.env import PaddleEnv
from world_model.paddle.planner import all_action_sequences, sequence_score


def assert_environment_equal(first, second):
    assert first.__dict__.keys() == second.__dict__.keys()
    for key, value in first.__dict__.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(value, second.__dict__[key])
        else:
            assert value == second.__dict__[key], key


def original_candidates(env):
    """Independent full-sequence simulation, public step, canonical scalar score."""
    records = []
    for sequence in all_action_sequences():
        branch = env.clone()
        positions = []
        for action in sequence:
            if not branch.terminated and not branch.truncated:
                branch.step(action)
            positions.append(branch.state[[0, 1, 4]].tolist())
        score = sequence_score(torch.tensor(positions, dtype=torch.float64), sequence)
        records.append((score, sequence))
    return records


CASES = [
    ([2.25, 10, -6, 3, 6], 0),  # side wall and clipped paddle
    ([62, 2, 6, -3, 58], 0),  # simultaneous wall/ceiling
    ([32, 52, 2, 3, 32], 0),  # catch opportunities depend on candidate action
    ([52, 52, 4, 3, 6], 0),  # missed contact plane
    ([40, 60, 4, 3, 6], 0),  # terminal prefix ends on the first step
    ([40, 61, 0, 0, 32], 20),  # already terminal input
    ([32, 10, 0, 2, 32], 0),  # symmetry and zero-movement tie
    ([32, 50, 0, 3, 32], 198),  # catch immediately before truncation
    ([32, 10, 0, 2, 32], 199),  # time limit must not become a miss
    ([40, 60, 4, 3, 6], 199),  # loss at time limit takes precedence
]
CASES += [(PaddleEnv(seed=seed).state.tolist(), 2) for seed in (9100, 9101, 9102)]


@pytest.mark.parametrize("state,step_index", CASES)
def test_all_243_privileged_scores_match_original_exhaustive_reference(state, step_index):
    env = PaddleEnv(state=state)
    env.step_index = step_index
    before = env.clone()
    expected = original_candidates(env)
    actual = evaluation._privileged_candidate_scores(env)
    assert len(actual) == 243
    assert actual == expected  # Exact floats: no approximate assertion for ties.
    score, sequence = min(expected)
    assert evaluation.privileged_plan(env) == (sequence[0], sequence, score)
    assert_environment_equal(env, before)


@pytest.mark.parametrize("state,step_index", [case for case in CASES if case[0][1] < 61])
def test_advance_preserves_public_step_state_events_counters_and_frames(state, step_index):
    initial = PaddleEnv(state=state)
    initial.step_index = step_index
    for sequence in ((1, 0, 2, 1, 0), (2, 2, 0, 0, 1)):
        rendered, raw = initial.clone(), initial.clone()
        for action in sequence:
            if rendered.terminated or rendered.truncated:
                with pytest.raises(RuntimeError, match="finished"):
                    raw.advance(action)
                break
            frame, terminated, truncated, info = rendered.step(action)
            assert raw.advance(action) == (terminated, truncated, info)
            assert_environment_equal(rendered, raw)
            np.testing.assert_array_equal(frame, raw.render())


def test_privileged_planning_and_advance_do_not_render_or_consume_rng(monkeypatch):
    env = PaddleEnv(seed=9110)
    env.step(1)
    before = env.clone()
    python_rng = random.getstate()
    numpy_rng = copy.deepcopy(np.random.get_state())
    torch_rng = torch.get_rng_state().clone()

    def forbidden_render(self):
        raise AssertionError("privileged candidate transitions must not render images")

    monkeypatch.setattr(PaddleEnv, "render", forbidden_render)
    evaluation.privileged_plan(env)
    env.clone().advance(1)
    assert_environment_equal(env, before)
    assert random.getstate() == python_rng
    after_numpy = np.random.get_state()
    assert numpy_rng[0] == after_numpy[0]
    np.testing.assert_array_equal(numpy_rng[1], after_numpy[1])
    assert numpy_rng[2:] == after_numpy[2:]
    assert torch.equal(torch.get_rng_state(), torch_rng)


@pytest.mark.parametrize("action", [True, -1, 3, 1.0, "1"])
def test_advance_retains_executable_action_validation(action):
    with pytest.raises(ValueError, match="executable integer"):
        PaddleEnv(seed=9100).advance(action)
