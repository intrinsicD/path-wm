"""Independent review checks for stage timing, aggregation, and reproducible resume."""

import json
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from world_model.paddle import checkpoints, models, training
from world_model.paddle.types import ObservationLatent


class PixelEncoder(nn.Module):
    def forward(self, image):
        marker = (image[:, 0, 0, 0] * 255).round()[:, None, None]
        return ObservationLatent(marker.expand(-1, 256, 64), marker.expand(-1, 64, 64))


class SumObserver(nn.Module):
    def forward(self, memory, observation, previous_action):
        return memory + observation.fine[:, 0, :1]


class EpisodeSamples:
    def __init__(self, lengths=(3, 6), offsets=(0, 10), split="train"):
        self.lengths = list(lengths)
        self.dataset = SimpleNamespace(fingerprint="whole-dataset-fingerprint", split=split)
        self.episodes = []
        for length, offset in zip(lengths, offsets):
            marker = np.arange(length, dtype=np.uint8) + offset
            frames = np.broadcast_to(marker[:, None, None, None], (length, 64, 64, 3)).copy()
            states = np.zeros((length, 5), np.float64)
            states[:, [0, 1, 4]] = marker[:, None]
            actions = np.resize(np.array([1, 1, 0, 2, 1, 2, 0], dtype=np.int64), length-1)
            self.episodes.append({"frames": frames, "states": states, "actions": actions})

    def episode(self, index):
        return self.episodes[int(index)]


def test_observer_cache_does_not_reuse_another_split_with_matching_shapes(tmp_path):
    system = {"E": PixelEncoder(), "U": SumObserver()}
    train = EpisodeSamples((3,), (0,), "train")
    validation = EpisodeSamples((3,), (10,), "validation")
    dependencies = {"perception": "E-same", "memory": "U-same"}
    training.observer_cache(system, train, tmp_path, dependencies, "cpu")
    val_memories = training.observer_cache(system, validation, tmp_path, dependencies, "cpu")
    np.testing.assert_array_equal(val_memories[0][:, 0], [10, 21, 33])


def test_memory_validation_aggregation_is_independent_of_episode_batching():
    samples = EpisodeSamples()
    readout = nn.Linear(128, 5)
    with torch.no_grad():
        readout.weight.zero_()
        readout.bias.zero_()
    system = {"E": PixelEncoder(), "U": SumObserver(), "R": readout}
    combined_loss, combined = training.memory_batch(system, samples, [0, 1], "cpu")
    a_loss, a = training.memory_batch(system, samples, [0], "cpu")
    b_loss, b = training.memory_batch(system, samples, [1], "cpu")
    combined_metrics = training._mean_metrics([(1, a), (1, b)])
    np.testing.assert_allclose(combined_metrics["r_mae"], combined["r_mae"], rtol=1e-6)
    # Validation must aggregate the loss using supervised scalar counts, too.
    scalar_counts = [5*n-4 for n in samples.lengths]
    expected = (a_loss*scalar_counts[0]+b_loss*scalar_counts[1])/sum(scalar_counts)
    torch.testing.assert_close(combined_loss, expected)


def test_five_step_training_requires_one_step_initialization(monkeypatch):
    monkeypatch.setattr(training, "_stage", lambda *args, **kwargs: "stage-entered")
    with pytest.raises(ValueError, match="one-step|initialize"):
        training.train_predictor({}, "data", "perception", "memory", 5, "run")


def test_source_windows_include_terminal_target_and_exclude_missing_future():
    samples = object.__new__(training.Samples)
    samples.lengths = [3, 4, 8, 10]
    assert samples.windows(5) == [(2, 2), (3, 2), (3, 3), (3, 4)]
    assert (1, 2) in samples.windows(1)
    assert all(source >= 2 and source+1 < samples.lengths[episode] for episode, source in samples.windows(1))


def test_predictor_batch_preserves_source_action_target_order_and_future_isolation():
    class IncrementPredictor(nn.Module):
        def __init__(self):
            super().__init__()
            self.seen = []

        def forward(self, observation, memory, action):
            self.seen.append((observation.fine[:, 0, 0].clone(), memory[:, 0].clone(), action.argmax(-1)))
            return ObservationLatent(observation.fine+1, observation.coarse+1)

    class MarkerReadout(nn.Module):
        def forward(self, observation):
            return observation.fine[:, 0, :3]/64

    predictor = IncrementPredictor()
    system = {"E": PixelEncoder(), "U": SumObserver(), "P": predictor, "H": MarkerReadout()}
    samples = EpisodeSamples((9, 10), (0, 20))
    memories = [np.broadcast_to((np.arange(n)+100*e)[:, None], (n, 128)).copy().astype(np.float32)
                for e, n in enumerate(samples.lengths)]
    windows = [(0, 2), (1, 3)]
    loss, metrics = training.predictor_batch(system, samples, memories, windows, [1, 0], 5,
                                             {"v_fine": 1., "v_coarse": 1.}, "cpu")
    assert float(loss) == 0
    assert metrics["copy_loss"] == 11
    torch.testing.assert_close(predictor.seen[0][0], torch.tensor([23., 2.]))
    torch.testing.assert_close(predictor.seen[0][1], torch.tensor([103., 2.]))
    for step, (source, memory, action) in enumerate(predictor.seen):
        torch.testing.assert_close(source, torch.tensor([23.+step, 2.+step]))
        assert action.tolist() == [int(samples.episodes[1]["actions"][3+step]), int(samples.episodes[0]["actions"][2+step])]
    before = [(a.clone(), b.clone(), c.clone()) for a, b, c in predictor.seen]
    samples.episodes[0]["frames"][3:8] += 10
    samples.episodes[1]["frames"][4:9] += 10
    predictor.seen.clear()
    changed_loss, _ = training.predictor_batch(system, samples, memories, windows, [1, 0], 5,
                                                {"v_fine": 1., "v_coarse": 1.}, "cpu")
    assert changed_loss > loss
    for previous, current in zip(before, predictor.seen):
        for a, b in zip(previous, current):
            torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_rng_checkpoint_roundtrip_reproduces_every_sampler_stream(tmp_path):
    training.seed_all(173)
    sampler = np.random.default_rng(492)
    state = checkpoints.rng_state(sampler)
    checkpoints.atomic_checkpoint(tmp_path/"rng.pt", state)
    def draw():
        return (random.random(), np.random.uniform(), sampler.integers(100, size=10), torch.rand(7))
    expected = draw()
    checkpoints.restore_rng(torch.load(tmp_path/"rng.pt", weights_only=False), sampler)
    actual = draw()
    assert actual[:2] == expected[:2]
    np.testing.assert_array_equal(actual[2], expected[2])
    torch.testing.assert_close(actual[3], expected[3], rtol=0, atol=0)


def test_checkpoint_detects_weight_tampering_and_tensor_schema_change(tmp_path):
    weights = {"E": nn.Linear(3, 2).state_dict()}
    value = {"schema_version": checkpoints.SCHEMA_VERSION, "tensor_schema": checkpoints.TENSOR_SCHEMA,
             "models": weights, "model_fingerprint": checkpoints.fingerprint_modules(weights)}
    path = tmp_path/"weights.pt"
    checkpoints.atomic_checkpoint(path, value)
    checkpoints.read_checkpoint(path)
    value["models"]["E"]["weight"][0, 0] += 1
    checkpoints.atomic_checkpoint(path, value)
    with pytest.raises(ValueError, match="fingerprint"):
        checkpoints.read_checkpoint(path)
    value["tensor_schema"] = "coarse-before-fine"
    checkpoints.atomic_checkpoint(path, value)
    with pytest.raises(ValueError, match="ordering"):
        checkpoints.read_checkpoint(path)


def test_resume_matches_optimizer_rng_and_keeps_one_ledger_row_per_update(tmp_path, monkeypatch):
    """Four scalar-model updates are a lifecycle check, not a training experiment."""
    class TinySamples:
        def __init__(self, data, split):
            self.dataset = SimpleNamespace(fingerprint="tiny-sample-fixture")
            self.frames = list(range(7))

        def frame_batch(self, indices, device):
            x = torch.tensor(indices, dtype=torch.float32, device=device)[:, None]/7
            return x, x.expand(-1, 5)

    monkeypatch.setattr(training, "Samples", TinySamples)
    monkeypatch.setattr(models, "Encoder", lambda: nn.Linear(1, 1))
    monkeypatch.setattr(models, "Decoder", lambda: nn.Linear(1, 1))
    monkeypatch.setattr(models, "PositionReadout", lambda: nn.Linear(1, 3))
    # This lifecycle fixture uses scalar observations, so its visualization is
    # independent of the RGB renderer verified by the end-to-end smoke run.
    monkeypatch.setattr(training, "perception_debug", lambda *args, **kwargs: None)
    config = {"seed": 4, "device": "cpu", "cpu_threads": 2, "smoke": True,
              "training": {"learning_rate": .01, "weight_decay": 1e-4, "grad_clip": 1.,
                           "validate_every": 2, "early_stopping": False,
                           "perception": {"batch_size": 2, "validation_examples": 5, "updates": 4}}}
    uninterrupted = tmp_path/"full"
    resumed = tmp_path/"resumed"
    training.train_perception(config, "unused", uninterrupted)
    original_optimizer_for = training.optimizer_for

    def interrupted_optimizer(*args, **kwargs):
        optimizer = original_optimizer_for(*args, **kwargs)
        original_step = optimizer.step
        calls = 0
        def step(*a, **kw):
            nonlocal calls
            calls += 1
            if calls == 4:
                raise RuntimeError("intentional interruption after uncheckpointed update 3")
            return original_step(*a, **kw)
        optimizer.step = step
        return optimizer

    monkeypatch.setattr(training, "optimizer_for", interrupted_optimizer)
    with pytest.raises(RuntimeError, match="intentional interruption"):
        training.train_perception(config, "unused", resumed)
    monkeypatch.setattr(training, "optimizer_for", original_optimizer_for)
    training.train_perception(config, "unused", resumed, resume=True)
    a, b = checkpoints.read_checkpoint(uninterrupted/"last.pt"), checkpoints.read_checkpoint(resumed/"last.pt")
    assert a["model_fingerprint"] == b["model_fingerprint"]
    assert a["examples_processed"] == b["examples_processed"] == 8
    for key, value in a["optimizer"]["state"].items():
        for field, expected in value.items():
            torch.testing.assert_close(b["optimizer"]["state"][key][field], expected, rtol=0, atol=0)
    torch.testing.assert_close(a["rng"]["torch"], b["rng"]["torch"], rtol=0, atol=0)
    assert a["rng"]["sampler"] == b["rng"]["sampler"]
    rows = [json.loads(line) for line in (resumed/"training.jsonl").read_text().splitlines()]
    assert [row["step"] for row in rows] == [1, 2, 3, 4]
