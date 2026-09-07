"""Independent probes of silent source, selection, and recovery failures."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from world_model.paddle.types import ObservationLatent
from world_model.pusht import checkpoints, models, training


class PixelEncoder(nn.Module):
    def forward(self, frames):
        value = (frames[:, 0, 0, 0] * 255).round()[:, None, None]
        return ObservationLatent(value.expand(-1, 256, 64), value.expand(-1, 64, 64))


class EpisodeSamples:
    def __init__(self, split="train", markers=(2, 4, 7, 11)):
        self.dataset = SimpleNamespace(fingerprint="whole-source", split=split,
                                       manifest={"normalization": {"motion_scales": [1.]*5}})
        self.lengths = [len(markers)]
        self.frames = [(0, t) for t in range(len(markers))]
        self.record = {
            "frames": np.broadcast_to(np.asarray(markers, np.uint8)[:, None, None, None], (len(markers), 64, 64, 3)).copy(),
            "actions": np.asarray([[.1, .2], [.3, .4], [.5, .6]][:len(markers)-1], np.float32),
            "pose_targets": np.zeros((len(markers), 6), np.float32),
            "motion_targets": np.zeros((len(markers), 5), np.float32),
            "motion_mask": np.arange(len(markers))[:, None].repeat(5, 1) >= 2,
        }
        self.record["pose_targets"][:, 5] = 1

    def episode(self, index):
        return self.record


def test_latent_statistics_refuse_heldout_samples_instead_of_stamping_train(tmp_path):
    samples = EpisodeSamples(split="validation")
    with pytest.raises(ValueError, match="train|held.?out|split"):
        training.compute_statistics(PixelEncoder(), samples, tmp_path, "encoder", batch_size=2)
    assert not (tmp_path / "statistics.json").exists()


def test_observer_cache_uses_complete_previous_action_history_and_is_split_specific(tmp_path):
    class Observer(nn.Module):
        def forward(self, memory, observation, previous):
            # Weight action coordinates differently to expose timing/axis errors.
            return memory + observation.fine[:, 0, :1] + previous[:, :1] + 10 * previous[:, 1:2]

    system = {"E": PixelEncoder(), "U": Observer()}
    dependencies = {"perception": "encoder", "memory": "observer"}
    samples = EpisodeSamples()
    cached = training.observer_cache(system, samples, tmp_path, dependencies, "cpu")
    # S0+start marker; S1+a0; S2+a1; S3+a2, with no resets at window starts.
    np.testing.assert_allclose(cached[0][:, 0], [-9, -2.9, 8.4, 25.9], atol=1e-6)
    other = EpisodeSamples(split="validation", markers=(20, 40, 70, 110))
    changed = training.observer_cache(system, other, tmp_path, dependencies, "cpu")
    np.testing.assert_allclose(changed[0][:, 0], [9, 51.1, 125.4, 241.9], atol=2e-5)


def test_predictor_batch_retains_frozen_observer_derivatives_and_future_isolation():
    class Observer(nn.Module):
        def __init__(self):
            super().__init__()
            self.gain = nn.Parameter(torch.tensor(2.), requires_grad=False)
        def forward(self, memory, observation, previous):
            return (self.gain * observation.fine[:, 0, :1]).expand(-1, 128)
    class Predictor(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(.5))
            self.seen = []
        def forward(self, observation, memory, action):
            self.seen.append((observation.fine[:, 0, 0].detach().clone(), memory[:, 0].detach().clone()))
            change = self.weight * memory[:, :1, None]
            return ObservationLatent(observation.fine + change, observation.coarse + change)
    class Pose(nn.Module):
        def forward(self, observation):
            pose = torch.zeros(len(observation.fine), 6)
            pose[:, 5] = 1
            return pose
    samples = EpisodeSamples(markers=(0, 1, 2, 3, 4))
    samples.record["actions"] = np.full((4, 2), .5, np.float32)
    observer, predictor = Observer(), Predictor()
    system = {"E": PixelEncoder(), "U": observer, "P": predictor, "H": Pose()}
    memories = [np.ones((5, 128), np.float32)]
    loss, _ = training.predictor_batch(system, samples, memories, [(0, 2)], [0], 2,
                                       {"v_fine": 1., "v_coarse": 1.}, "cpu")
    assert loss.item() == pytest.approx(.625)
    loss.backward()
    # s1=2+w, m1=2*s1, s2=s1+w*m1; gradient is 6.5, versus 5.5 if U is detached.
    assert predictor.weight.grad.item() == pytest.approx(6.5)
    assert observer.gain.grad is None
    before = predictor.seen.copy()
    samples.record["frames"][3:] += 20
    predictor.seen.clear()
    changed_loss, _ = training.predictor_batch(system, samples, memories, [(0, 2)], [0], 2,
                                               {"v_fine": 1., "v_coarse": 1.}, "cpu")
    assert changed_loss > loss
    for prior, current in zip(before, predictor.seen):
        for a, b in zip(prior, current):
            torch.testing.assert_close(a, b, rtol=0, atol=0)


class Scalar(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(()))


def _tiny_perception(monkeypatch):
    class TinySamples:
        def __init__(self, data, split, **kwargs):
            self.dataset = SimpleNamespace(fingerprint="whole-source", split=split,
                                           manifest={"normalization": {"motion_scales": [1.]*5}})
            self.frames = [(0, t) for t in range(4)]
            self.lengths = [4]

    monkeypatch.setattr(training, "Samples", TinySamples)
    for name in ("Encoder", "Decoder", "PoseReadout"):
        monkeypatch.setattr(models, name, Scalar)
    monkeypatch.setattr(training, "perception_debug", lambda *args: None)
    def batch(system, samples, indices, device):
        target = 1. if samples.dataset.split == "train" else 0.
        loss = (system["E"].weight-target).square()
        return loss, {"image_mse": float(loss.detach())}
    monkeypatch.setattr(training, "perception_batch", batch)
    return {"seed": 3, "device": "cpu", "cpu_threads": 1, "smoke": True,
            "training": {"learning_rate": .1, "weight_decay": 0., "grad_clip": 1.,
                         "validate_every": 1, "early_stopping": False,
                         "perception": {"updates": 2, "batch_size": 1, "validation_examples": 2}}}


def test_initial_validation_is_eligible_for_minimum_validation_selection(tmp_path, monkeypatch):
    config = _tiny_perception(monkeypatch)
    result = training.train_perception(config, "unused", tmp_path / "run")
    rows = [json.loads(line) for line in (tmp_path / "run" / "validation.jsonl").read_text().splitlines()]
    assert rows[0]["step"] == 0 and rows[0]["loss"] == 0
    assert all(row["loss"] > 0 for row in rows[1:])
    assert result["selected_update"] == 0
    assert result["metrics"]["loss"] == min(row["loss"] for row in rows)


@pytest.mark.parametrize("selected_passed", [False, True])
def test_resume_gate_follows_selected_checkpoint_despite_completed_result_status(tmp_path, monkeypatch, selected_passed):
    """Reject the old result-write crash gap, while respecting a passing best."""
    class TinySamples:
        def __init__(self, data, split, **kwargs):
            self.dataset = SimpleNamespace(fingerprint="whole-source", split=split,
                                           manifest={"normalization": {"motion_scales": [1.]*5}})
            self.frames = [(0, t) for t in range(4)]
            self.lengths = [4]

    monkeypatch.setattr(training, "Samples", TinySamples)
    monkeypatch.setattr(models, "Predictor", Scalar)
    observer = {key: Scalar() for key in ("E", "D", "H", "U", "R")}
    monkeypatch.setattr(training, "load_observer", lambda *args: observer)
    config = {"seed": 3, "device": "cpu", "cpu_threads": 1, "training": {
        "learning_rate": .1, "weight_decay": 0., "grad_clip": 1., "validate_every": 1,
        "predictor_1": {"updates": 1, "batch_size": 1, "validation_examples": 1}}}
    predictor = Scalar()
    saved = {
        "config": config, "horizon": 1, "models": {"P": predictor.state_dict()},
        "optimizer": training.optimizer_for({"P": predictor}, config["training"]).state_dict(),
        "global_update": 1, "best_validation": 2., "examples_processed": 1,
        "elapsed_seconds": 1., "training_complete": True, "statistics": {},
        "quality_gate": {"passed": False},
        "best_checkpoint": "checkpoints/best.pt",
    }
    selected = {**saved, "quality_gate": {"passed": selected_passed}}
    def read(path, **kwargs):
        if str(path) == "perception": return {"model_fingerprint": "encoder"}
        if str(path) == "memory": return {"model_fingerprint": "observer"}
        if Path(path).name == "best.pt": return selected
        return saved
    monkeypatch.setattr(training, "read_checkpoint", read)
    run = tmp_path / "run"
    run.mkdir()
    (run / "checkpoints").mkdir()
    for path in (run / "checkpoints" / "best.pt", run / "best.pt"):
        path.write_bytes(b"identical committed selected snapshot and alias")
    (run / "last.pt").write_bytes(b"committed checkpoint represented by trusted fixture")
    (run / "pusht_result.json").write_text(json.dumps({"status": "completed", "global_update": 1,
                                                       "quality_gate": {"passed": selected_passed}}))
    if selected_passed:
        result = training.train_predictor(config, "unused", "perception", "memory", 1, run, resume=True)
        assert result["status"] == "completed" and result["quality_gate"]["passed"]
    else:
        with pytest.raises(RuntimeError, match="gate|failed"):
            training.train_predictor(config, "unused", "perception", "memory", 1, run, resume=True)
