"""Prospective observer-start intervention: source identity, timing and selection."""

import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from world_model.paddle import history_training as history
from world_model.paddle import training
from world_model.paddle.types import ObservationLatent


def episode(length=9, *, truncated=False):
    index = np.arange(length)
    frames = np.broadcast_to(index.astype(np.uint8)[:, None, None, None], (length, 64, 64, 3)).copy()
    states = np.column_stack([index, index+10, index+1, -(index+2), index+20]).astype(np.float64)
    if not truncated:
        states[-1, 2:4] = 0
    terminal = np.zeros(length, np.bool_)
    terminal[-1] = True
    return {
        "frames": frames, "states": states, "actions": (index[:-1] % 3).astype(np.int64),
        "timestamps": index.astype(np.float64) * .05,
        "terminated": terminal.copy() if not truncated else np.zeros(length, np.bool_),
        "truncated": terminal.copy() if truncated else np.zeros(length, np.bool_),
        "hit_counts": (index // 3).astype(np.int64),
        "metadata": {"seed": 19, "split": "train", "source_tag": ["original"]},
        "metadata_json": np.array('{"seed":19,"split":"train"}'),
    }


@pytest.mark.parametrize("truncated", [False, True])
def test_suffix_preserves_exact_source_rows_terminal_and_independent_arrays(truncated):
    source = episode(truncated=truncated)
    before = copy.deepcopy(source)
    suffix = history.slice_episode(source, 6)  # The final eligible three-frame suffix.
    for key in ("frames", "states", "timestamps", "terminated", "truncated", "hit_counts"):
        np.testing.assert_array_equal(suffix[key], source[key][6:])
        assert not np.shares_memory(suffix[key], source[key])
    np.testing.assert_array_equal(suffix["actions"], source["actions"][6:])
    assert len(suffix["actions"]) == len(suffix["frames"])-1 == 2
    assert not np.shares_memory(suffix["actions"], source["actions"])
    assert bool(suffix["terminated"][-1]) != bool(suffix["truncated"][-1])
    if not truncated:
        np.testing.assert_array_equal(suffix["states"][-1, 2:4], [0, 0])
    suffix["frames"][0] = 255
    suffix["states"][0] = -999
    suffix["actions"][0] = 2
    suffix["metadata"]["source_tag"].append("changed")
    for key, original in before.items():
        if isinstance(original, np.ndarray): np.testing.assert_array_equal(source[key], original)
        else: assert source[key] == original
    full = history.slice_episode(source, 0)
    np.testing.assert_array_equal(full["frames"], source["frames"])
    np.testing.assert_array_equal(full["actions"], source["actions"])
    for invalid in (-1, len(source["frames"])-2, len(source["frames"]), 1.5):
        with pytest.raises(ValueError, match="start|suffix|integer|three|3"):
            history.slice_episode(source, invalid)


def test_sampler_matches_independent_uniform_global_suffix_reference_and_membership():
    # Episode 0 contributes no augmented start, episode 1 contributes one, episode 2 four.
    lengths = [3, 4, 7]
    eligible = [(e, s) for e, length in enumerate(lengths) for s in range(1, length-2)]
    sampler = history.HistoryStartSampler(lengths, seed=1701, suffix_seed=38802)
    full_rng, suffix_rng = np.random.default_rng(1701), np.random.default_rng(38802)
    seen = set()
    for _ in range(16):
        full = full_rng.integers(0, len(lengths), size=4)
        suffix_indices = suffix_rng.integers(0, len(eligible), size=4)
        expected = [row for f, s in zip(full, suffix_indices) for row in [(int(f), 0), eligible[int(s)]]]
        actual = [tuple(map(int, row)) for row in sampler.sample(batch_size=8)]
        assert actual == expected
        assert all(start == 0 for _, start in actual[::2])
        assert all(1 <= start <= lengths[e]-3 for e, start in actual[1::2])
        assert all(0 <= e < len(lengths) for e, _ in actual)
        seen.update(actual[1::2])
    assert seen == set(eligible)
    for invalid in (0, 3, -2, 2.5, True):
        with pytest.raises(ValueError, match="batch|even|integer|positive"):
            sampler.sample(batch_size=invalid)
    with pytest.raises(ValueError, match="suffix|eligible|start"):
        history.HistoryStartSampler([3, 3])


def test_sampler_has_separate_private_streams_and_exact_state_resume():
    lengths = [4, 8, 11]
    global_before = np.random.get_state()
    first = history.HistoryStartSampler(lengths, seed=1701, suffix_seed=38802)
    other_full = history.HistoryStartSampler(lengths, seed=1702, suffix_seed=38802)
    for _ in range(3):
        a, b = first.sample(), other_full.sample()
        assert a[1::2] == b[1::2]  # Changing full-episode draws cannot perturb suffix draws.
    saved = copy.deepcopy(first.state_dict())
    expected = [first.sample() for _ in range(5)]
    # Validation computations and global NumPy draws cannot advance either private stream.
    history.joint_validation_score({"squared_error_sum": 8., "supervised_scalar_count": 11},
                                   {"squared_error_sum": 7., "supervised_scalar_count": 26})
    restored = history.HistoryStartSampler(lengths, seed=1701, suffix_seed=38802)
    restored.load_state_dict(saved)
    assert [restored.sample() for _ in range(5)] == expected
    global_after = np.random.get_state()
    assert global_before[0] == global_after[0]
    np.testing.assert_array_equal(global_before[1], global_after[1])
    assert global_before[2:] == global_after[2:]
    for kwargs in ({"lengths": [4, 8, 12]}, {"lengths": lengths, "seed": 1702},
                   {"lengths": lengths, "suffix_seed": 38803}):
        incompatible = history.HistoryStartSampler(**kwargs)
        with pytest.raises(ValueError, match="identity|length|seed|population|sampler"):
            incompatible.load_state_dict(saved)


def test_suffix_memory_integration_masks_relative_velocities_and_preserves_padding():
    class Encoder(nn.Module):
        def forward(self, x):
            marker = (x[:, 0, 0, 0] * 255).round()[:, None, None]
            return ObservationLatent(marker.expand(-1, 256, 64), marker.expand(-1, 64, 64))
    class Observer(nn.Module):
        def __init__(self):
            super().__init__()
            self.inputs = []
        def forward(self, memory, observation, previous):
            self.inputs.append((memory.detach().clone(), observation.fine[:, 0, 0].clone(), previous.clone()))
            action = (previous * torch.tensor([10., 20., 30.])).sum(-1, keepdim=True)
            return memory + observation.fine[:, 0, :1] + action
    class Readout(nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = nn.Parameter(torch.tensor([.2, .3, .4, .5, .6]))
            self.outputs = []
        def forward(self, memory):
            output = self.bias[None] + memory[:, :5] / 1000
            output.retain_grad()
            self.outputs.append(output)
            return output
    source = episode()
    suffixes = [history.slice_episode(source, start) for start in (4, 6)]
    samples = SimpleNamespace(episode=lambda index: suffixes[int(index)])
    observer, readout = Observer(), Readout()
    loss, metrics = training.memory_batch({"E": Encoder(), "U": observer, "R": readout}, samples, [0, 1], "cpu")
    assert metrics["observations"] == 8
    assert metrics["post_warmup_observations"] == 4
    assert metrics["supervised_scalar_count"] == (5*5-4) + (5*3-4) == 32
    torch.testing.assert_close(observer.inputs[0][0], torch.zeros(2, 128))
    torch.testing.assert_close(observer.inputs[0][1], torch.tensor([4., 6.]))
    torch.testing.assert_close(observer.inputs[0][2], torch.zeros(2, 3))
    for relative in (1, 2):
        expected = torch.nn.functional.one_hot(torch.tensor([source["actions"][s+relative-1] for s in (4, 6)]), 3).float()
        torch.testing.assert_close(observer.inputs[relative][2], expected)
    # The short suffix has ended: padding neither changes its carried state nor contributes loss.
    torch.testing.assert_close(observer.inputs[3][0][1], observer.inputs[4][0][1])
    denominator = 0
    numerator = torch.zeros(())
    scale = torch.tensor([64., 64., 6., 6., 64.])
    for b, suffix in enumerate(suffixes):
        for relative, state in enumerate(suffix["states"]):
            mask = torch.ones(5, dtype=torch.bool)
            if relative < 2: mask[2:4] = False
            target = torch.as_tensor(state, dtype=torch.float32) / scale
            numerator += (readout.outputs[relative][b].detach()-target).square()[mask].sum()
            denominator += int(mask.sum())
    torch.testing.assert_close(loss, numerator/denominator)
    loss.backward()
    for relative, output in enumerate(readout.outputs):
        for b, suffix in enumerate(suffixes):
            if relative >= len(suffix["frames"]):
                torch.testing.assert_close(output.grad[b], torch.zeros(5))
            elif relative < 2:
                torch.testing.assert_close(output.grad[b, 2:4], torch.zeros(2))
                assert torch.all(output.grad[b, [0, 1, 4]] != 0)
            else:
                assert torch.all(output.grad[b, 2:4] != 0)


def test_joint_validation_normalizes_each_population_before_equal_weighting():
    ordinary = {"squared_error_sum": 18., "supervised_scalar_count": 6}
    suffix = {"squared_error_sum": 4., "supervised_scalar_count": 16}
    assert history.joint_validation_score(ordinary, suffix) == 1.625
    assert history.joint_validation_score(ordinary, suffix) != (18+4)/(6+16)
    assert history.joint_validation_score({**ordinary, "paired_state_mse": 1e12},
                                         {**suffix, "paired_state_mse": -1e12}) == 1.625
    assert ordinary == {"squared_error_sum": 18., "supervised_scalar_count": 6}
    for invalid in ({"squared_error_sum": 1., "supervised_scalar_count": 0},
                    {"squared_error_sum": -1., "supervised_scalar_count": 6},
                    {"squared_error_sum": float("nan"), "supervised_scalar_count": 6}):
        with pytest.raises(ValueError, match="finite|count|positive|negative|error"):
            history.joint_validation_score(invalid, suffix)
