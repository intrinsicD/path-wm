"""Prospective observer-start intervention: source identity, timing and selection."""

import copy
import hashlib
import json
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


def _tiny_history_stage(tmp_path, monkeypatch):
    """Real masked memory_batch with tiny modules; no simulation or GPU work."""
    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.gain = nn.Parameter(torch.ones(()))
        def forward(self, x):
            marker = self.gain * x[:, 0, 0, 0][:, None, None]
            return ObservationLatent(marker.expand(-1, 256, 64), marker.expand(-1, 64, 64))
    class Observer(nn.Module):
        def __init__(self):
            super().__init__()
            self.gain = nn.Parameter(torch.tensor(.05))
        def forward(self, memory, observation, previous):
            signal = observation.fine[:, 0, :1] + (previous*torch.tensor([.1, .2, .3])).sum(-1, keepdim=True)
            return memory + self.gain * signal
    class Readout(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(128, 5)
        def forward(self, memory):
            return self.linear(memory)
    class Samples:
        def __init__(self, data, split):
            self.lengths = [5, 7, 8]
            self.dataset = SimpleNamespace(fingerprint="tiny-history-source", split=split,
                entries=[{"seed": i+(5000 if split=="validation" else 0), "frames": n} for i, n in enumerate(self.lengths)])
            self.raw_cache = None
            self.episodes = [episode(n) for n in self.lengths]
            for i, record in enumerate(self.episodes):
                record["frames"] += np.uint8(10*i+(20 if split=="validation" else 0))
        def episode(self, index):
            return self.episodes[int(index)]
    frozen = {"E": Encoder(), "D": nn.Linear(1, 1), "H": nn.Linear(1, 1)}
    weights = {k: copy.deepcopy(m.state_dict()) for k, m in frozen.items()}
    perception = tmp_path / "perception.pt"
    history.atomic_checkpoint(perception, {
        "schema_version": history.SCHEMA_VERSION, "tensor_schema": history.TENSOR_SCHEMA,
        "stage": "perception", "dataset_fingerprint": "tiny-history-source", "models": weights,
        "model_fingerprint": history.fingerprint_modules(frozen),
    })
    def load(*args):
        result = {"E": Encoder(), "D": nn.Linear(1, 1), "H": nn.Linear(1, 1)}
        for key, model in result.items(): model.load_state_dict(weights[key])
        return result
    monkeypatch.setattr(history, "Samples", Samples)
    monkeypatch.setattr(history, "load_observer", load)
    monkeypatch.setattr(history, "MemoryUpdater", Observer)
    monkeypatch.setattr(history, "StateReadout", Readout)
    config = {"seed": 1701, "device": "cpu", "cpu_threads": 1, "smoke": True,
        "history_starts": {"schema_version": 1, "suffix_seed": 38802, "validation_suffix_seed": 38920,
                           "normalization": [64, 64, 6, 6, 64], "ordinary_validation_indices": [0, 1]},
        "training": {"learning_rate": .01, "weight_decay": .0001, "grad_clip": 1.,
                     "validate_every": 1, "encoder_batch_size": 16,
                     "memory": {"updates": 4, "batch_size": 4, "validation_examples": 2, "validation_pairs": 0}}}
    return config, perception


def _assert_exact(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left: _assert_exact(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right): _assert_exact(a, b)
    else:
        assert left == right


@pytest.mark.parametrize("crash", ["after_initial_last", "before_update2_last", "after_update2_last"])
def test_history_stage_exact_resume_preserves_mixed_draws_optimizer_rng_and_counters(tmp_path, monkeypatch, crash):
    config, perception = _tiny_history_stage(tmp_path, monkeypatch)
    original_source = hashlib.sha256(perception.read_bytes()).hexdigest()
    full, resumed = tmp_path / "full", tmp_path / "resumed"
    history.train_history_memory(config, "unused", perception, full)
    atomic = history.atomic_checkpoint
    def interrupt(path, value):
        target = 0 if crash == "after_initial_last" else 2
        selected = path.name == "last.pt" and value["global_update"] == target
        if selected and crash == "before_update2_last":
            raise KeyboardInterrupt("planned uncommitted snapshot interruption")
        atomic(path, value)
        if selected:
            raise KeyboardInterrupt("planned committed-last interruption")
    monkeypatch.setattr(history, "atomic_checkpoint", interrupt)
    with pytest.raises(KeyboardInterrupt, match="planned"):
        history.train_history_memory(config, "unused", perception, resumed)
    monkeypatch.setattr(history, "atomic_checkpoint", atomic)
    history.train_history_memory(config, "unused", perception, resumed, resume=True)
    a, b = [history.read_checkpoint(path / "last.pt") for path in (full, resumed)]
    assert a["global_update"] == b["global_update"] == 4
    for key in ("models", "optimizer", "rng", "history_sampler", "history_counters", "seen_source_episodes",
                "seen_source_starts", "metrics", "best_validation", "history_identity"):
        _assert_exact(a[key], b[key])
    rows = [json.loads(line) for line in (resumed / "training.jsonl").read_text().splitlines()]
    validation = [json.loads(line) for line in (resumed / "validation.jsonl").read_text().splitlines()]
    assert [r["step"] for r in rows] == [1, 2, 3, 4]
    assert [r["step"] for r in validation] == [0, 1, 2, 3, 4]
    lengths = [5, 7, 8]
    eligible = [(e, s) for e, n in enumerate(lengths) for s in range(1, n-2)]
    full_rng, suffix_rng = np.random.default_rng(1701), np.random.default_rng(38802)
    observed = scalars = 0
    for row in rows:
        pairs = [p for f, s in zip(full_rng.integers(3, size=2), suffix_rng.integers(len(eligible), size=2))
                 for p in [[int(f), 0], list(eligible[int(s)])]]
        assert row["source_starts"] == pairs
        observed += sum(lengths[e]-s for e, s in pairs)
        scalars += sum(5*(lengths[e]-s)-4 for e, s in pairs)
    counters = b["history_counters"]
    assert counters["sequences"] == 16
    assert counters["full_sequences"] == counters["suffix_sequences"] == 8
    assert counters["observations"] == observed and counters["supervised_scalar_count"] == scalars
    _assert_exact(b["history_sampler"]["full_rng"], full_rng.bit_generator.state)
    _assert_exact(b["history_sampler"]["suffix_rng"], suffix_rng.bit_generator.state)
    assert (resumed / "best.pt").read_bytes() == (resumed / b["best_checkpoint"]).read_bytes()
    if crash == "before_update2_last":
        assert list((resumed / "checkpoints").glob("orphan_*_update_00000002.pt"))
    assert hashlib.sha256(perception.read_bytes()).hexdigest() == original_source
    untouched = {name: (resumed / name).read_bytes() for name in ("training.jsonl", "validation.jsonl", "last.pt", "best.pt")}
    history.train_history_memory(config, "unused", perception, resumed, resume=True)
    assert untouched == {name: (resumed / name).read_bytes() for name in untouched}


def test_history_selection_keeps_step0_tie_and_latches_its_paired_metrics(tmp_path, monkeypatch):
    config, perception = _tiny_history_stage(tmp_path, monkeypatch)
    validation_calls = 0
    paired_calls = 0
    reverse = False
    def validate(*args):
        nonlocal validation_calls
        update = validation_calls // 2
        count = 6 if validation_calls % 2 == 0 else 16
        validation_calls += 1
        score = 1. + max(0, update-1)  # First update ties initialization; later ones worsen.
        return {"squared_error_sum": score*count, "supervised_scalar_count": count, "loss": score,
                "post_warmup": {"mae": [score]*5}}, []
    def paired(*args):
        nonlocal paired_calls
        value = (4-paired_calls if reverse else paired_calls) * 1000.
        paired_calls += 1
        return {"paired_validation_r_mae": [value]*5}, []
    monkeypatch.setattr(history, "_validate_population", validate)
    monkeypatch.setattr(history, "_paired_diagnostics", paired)
    first = history.train_history_memory(config, "unused", perception, tmp_path / "increasing_pairs")
    validation_calls = paired_calls = 0
    reverse = True
    second = history.train_history_memory(config, "unused", perception, tmp_path / "decreasing_pairs")
    assert first["selected_update"] == second["selected_update"] == 0
    assert first["metrics"]["loss"] == second["metrics"]["loss"] == 1
    assert first["metrics"]["paired"]["paired_validation_r_mae"] == [0.]*5
    assert first["final_metrics"]["paired"]["paired_validation_r_mae"] == [4000.]*5
    assert second["metrics"]["paired"]["paired_validation_r_mae"] == [4000.]*5
    assert second["final_metrics"]["paired"]["paired_validation_r_mae"] == [0.]*5
    a = history.read_checkpoint(tmp_path / "increasing_pairs" / "last.pt")
    b = history.read_checkpoint(tmp_path / "decreasing_pairs" / "last.pt")
    for key in ("models", "optimizer", "rng", "history_sampler", "history_counters"):
        _assert_exact(a[key], b[key])


@pytest.mark.parametrize("command", ["run-all", "train-memory"])
def test_history_config_cannot_silently_dispatch_to_generic_observer_training(monkeypatch, capsys, command):
    from world_model.paddle import cli
    class WrongTrainerInvoked(Exception):
        pass
    calls = []
    def wrong(**kwargs):
        calls.append(kwargs)
        raise WrongTrainerInvoked("generic observer trainer was invoked for a history-start config")
    monkeypatch.setattr(cli, "load_config", lambda path: {"history_starts": {"schema_version": 1}})
    monkeypatch.setattr(cli, "run_all", wrong)
    monkeypatch.setattr(training, "train_memory", wrong)
    args = [command, "--config", "unused.yaml", "--data", "unused-data", "--run", "unused-run"]
    if command == "train-memory": args += ["--perception", "unused-perception.pt"]
    with pytest.raises(SystemExit) as stopped:
        cli.main(args)
    assert stopped.value.code == 1
    assert not calls
    assert "train-history-memory" in capsys.readouterr().err
