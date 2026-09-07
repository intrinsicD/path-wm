"""Behavioral contracts for exact paddle dynamics and replayable populations."""

import itertools
import json

import numpy as np
import pytest

from world_model.paddle.env import PaddleEnv
from world_model.paddle.data import (
    EpisodeDataset,
    generate_dataset,
    history_pairs,
    previous_action_vectors,
    verify_dataset,
)


def test_seed_replay_clone_and_executable_actions():
    env = PaddleEnv(seed=19)
    initial = env.state.copy()
    replay = PaddleEnv(state=initial)
    actions = [1, 1, 0, 2, 2, 1, 0, 0]
    for action in actions:
        frame, terminal, truncated, info = env.step(action)
        other = replay.step(action)
        np.testing.assert_array_equal(frame, other[0])
        np.testing.assert_array_equal(env.state, replay.state)
        assert (terminal, truncated, info) == (other[1], other[2], other[3])
    clone = env.clone()
    np.testing.assert_array_equal(clone.step(0)[0], env.step(0)[0])
    clone.state[0] += 0.25
    assert clone.state[0] != env.state[0]
    for action, displacement in [(0, -4), (1, 0), (2, 4)]:
        moving = PaddleEnv(state=[32, 20, 2, 2, 32])
        moving.step(action)
        assert moving.state[4] == 32 + displacement
    clipped = PaddleEnv(state=[32, 20, 2, 2, 57])
    clipped.step(2)
    assert clipped.state[4] == 58
    with pytest.raises(ValueError, match="action"):
        clipped.step(3)


def test_reflection_simultaneous_events_and_immediate_wall():
    env = PaddleEnv(state=[60, 4, 2, -2, 32])
    env.step(1)
    np.testing.assert_allclose(env.state, [62, 2, -2, 2, 32], atol=1e-12)
    env.step(1)
    np.testing.assert_allclose(env.state, [60, 4, -2, 2, 32], atol=1e-12)
    env = PaddleEnv(state=[62, 30, 6, 2, 58])
    env.step(2)
    np.testing.assert_allclose(env.state, [56, 32, -6, 2, 58], atol=1e-12)


def test_contact_uses_paddle_position_at_event_not_endpoint():
    caught = PaddleEnv(state=[31.5, 51.75, 6, 3, 32])
    _, _, _, info = caught.step(0)
    np.testing.assert_allclose(caught.state, [37.5, 53.25, 6, -3, 28])
    assert caught.hit_count == 1
    hit = next(event for event in info["events"] if event["type"] == "paddle_hit")
    assert hit["fractional_time"] == 0.75
    assert (hit["ball_x"], hit["paddle_x"]) == (36, 29)
    # At the endpoint these objects overlap; at contact they were 9 apart.
    missed = PaddleEnv(state=[43.5, 53.25, -6, 3, 32])
    _, _, _, info = missed.step(2)
    assert missed.hit_count == 0 and missed.state[3] == 3
    assert [e["type"] for e in info["events"]] == ["paddle_miss"]
    missed.step(2)
    _, terminal, _, info = missed.step(2)
    assert terminal and missed.hit_count == 0
    assert not any(e["type"] == "paddle_miss" for e in info["events"])


def test_plane_boundary_already_passed_terminal_visible_and_time_limit():
    env = PaddleEnv(state=[32, 54, 2, 3, 32])
    env.step(1)
    assert env.hit_count == 0 and env.state[1] == 57
    env = PaddleEnv(state=[32, 60, 6, 3, 32])
    frame, terminated, truncated, _ = env.step(2)
    assert terminated and not truncated
    np.testing.assert_allclose(env.state, [34, 61, 0, 0, 32 + 4 / 3])
    assert np.any(frame[59:63, :, 0]) and not np.any(frame[63, :, 0])
    with pytest.raises(RuntimeError, match="reset"):
        env.step(1)
    env = PaddleEnv(state=[32, 20, 2, 2, 32])
    env.step_index = 199
    assert env.step(1)[1:3] == (False, True)
    env = PaddleEnv(state=[32, 60, 2, 3, 32])
    env.step_index = 199
    assert env.step(1)[1:3] == (True, False)


def test_exact_subpixel_coverage_and_ball_precedence():
    env = PaddleEnv(state=[26.25, 57.5, 2, 3, 32.5])
    frame = env.render()
    assert frame.dtype == np.uint8 and frame.shape == (64, 64, 3)
    # Pixel (28,57): ball covers .25, paddle covers 1; overlap .25.
    np.testing.assert_array_equal(frame[57, 28], [64, 255, 255])
    # Pixel (24,55) is .75 x .5 covered only by the ball.
    np.testing.assert_array_equal(frame[55, 24], [96, 96, 96])
    assert np.count_nonzero(frame[:50]) == 0


def test_dataset_alignment_split_replay_and_resume(tmp_path):
    config = {"dataset": {"train": 2, "validation": 1, "test": 1}}
    manifest = generate_dataset(config, tmp_path)
    first_hash = (tmp_path / "manifest.json").read_bytes()
    second = generate_dataset(config, tmp_path)
    assert manifest == second and first_hash == (tmp_path / "manifest.json").read_bytes()
    report = verify_dataset(tmp_path)
    assert report["passed"] and report["episodes"] == 4
    seeds = []
    for split, count in [("train", 2), ("validation", 1), ("test", 1)]:
        dataset = EpisodeDataset(tmp_path, split)
        assert len(dataset) == count and dataset.fingerprint
        for episode in dataset:
            steps = len(episode["actions"])
            assert len(episode["frames"]) == steps + 1
            assert len(episode["states"]) == steps + 1
            np.testing.assert_array_equal(episode["actions"][:2], [1, 1])
            np.testing.assert_array_equal(episode["timestamps"], np.arange(steps + 1) * 0.05)
            previous = previous_action_vectors(episode["actions"])
            np.testing.assert_array_equal(previous[0], [0, 0, 0])
            np.testing.assert_array_equal(previous[1:3], [[0, 1, 0], [0, 1, 0]])
            assert episode["terminated"][-1] != episode["truncated"][-1]
            assert not episode["terminated"][:-1].any()
            assert not episode["truncated"][:-1].any()
            meta = json.loads(str(episode["metadata_json"]))
            seeds.append(meta["seed"])
    assert seeds == [0, 1, 5000, 5500]
    with pytest.raises(ValueError, match="config|fingerprint"):
        generate_dataset({"dataset": {"train": 3, "validation": 1, "test": 1}}, tmp_path)


def test_resume_rejects_corrupted_episode_metadata(tmp_path):
    config = {"dataset": {"train": 1, "validation": 0, "test": 0}}
    generate_dataset(config, tmp_path)
    path = next(tmp_path.rglob("*.npz"))
    with np.load(path, allow_pickle=False) as stored:
        episode = dict(stored)
    metadata = json.loads(str(episode["metadata_json"]))
    metadata["environment_fingerprint"] = "corrupted"
    episode["metadata_json"] = np.array(json.dumps(metadata))
    np.savez_compressed(path, **episode)
    with pytest.raises(ValueError, match="fingerprint|metadata|checksum"):
        generate_dataset(config, tmp_path)


def test_paired_histories_identical_image_and_essential_first_action():
    for pair in history_pairs(count=3, seed=7000):
        negative, positive = pair["members"]
        np.testing.assert_array_equal(negative["frames"][-1], positive["frames"][-1])
        assert not np.array_equal(negative["frames"][0], positive["frames"][0])
        for member in pair["members"]:
            possible_first_actions = set()
            for sequence in itertools.product(range(3), repeat=3):
                env = PaddleEnv(state=member["control_state"])
                for action in sequence:
                    env.step(action)
                if env.hit_count:
                    possible_first_actions.add(sequence[0])
            assert possible_first_actions == {member["correct_action"]}
            assert member["correct_action"] == (0 if member["direction"] < 0 else 2)
