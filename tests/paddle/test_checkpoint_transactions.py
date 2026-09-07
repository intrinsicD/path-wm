"""Tiny scalar fixtures exercise interruption boundaries, not model training."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from world_model.paddle import checkpoints, models, training


DATASET_FINGERPRINT = "checkpoint-transaction-scalar-fixture"


class TinySamples:
    def __init__(self, data, split):
        self.dataset = SimpleNamespace(fingerprint=DATASET_FINGERPRINT, split=split)
        self.frames = list(range(8))
        self.lengths = [8]

    def frame_batch(self, indices, device):
        count = len(indices)
        return torch.zeros(count, 1, device=device), torch.zeros(count, 5, device=device)

    def windows(self, horizon):
        return [(0, source) for source in range(2, 8-horizon)]


def scalar_layer(outputs=1):
    layer = nn.Linear(1, outputs)
    with torch.no_grad():
        layer.weight.zero_()
        layer.bias.fill_(1)
    return layer


def tiny_config(updates=2, validate_every=1):
    return {
        "seed": 4, "device": "cpu", "cpu_threads": 2, "smoke": True,
        "training": {
            "learning_rate": .01, "weight_decay": 1e-4, "grad_clip": 1.,
            "validate_every": validate_every, "early_stopping": False,
            "perception": {"batch_size": 1, "validation_examples": 1, "updates": updates},
            "predictor_1": {"batch_size": 1, "validation_examples": 1, "updates": updates},
            "predictor_5": {"batch_size": 1, "validation_examples": 1, "updates": updates},
        },
    }


@pytest.fixture
def scalar_stage(monkeypatch):
    monkeypatch.setattr(training, "Samples", TinySamples)
    monkeypatch.setattr(models, "Encoder", scalar_layer)
    monkeypatch.setattr(models, "Decoder", scalar_layer)
    monkeypatch.setattr(models, "PositionReadout", lambda: scalar_layer(3))
    monkeypatch.setattr(models, "Predictor", scalar_layer)
    monkeypatch.setattr(training, "perception_debug", lambda *args, **kwargs: None)


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_resume_after_last_replacement_preserves_new_best_weights(tmp_path, monkeypatch, scalar_stage):
    """A crash between checkpoint file replacements cannot discard a new best."""
    run = tmp_path / "interrupted-best"
    config = tiny_config()
    original_save = training.atomic_checkpoint

    def fail_after_last(path, value):
        original_save(path, value)
        if Path(path).name == "last.pt" and value["global_update"] == 2:
            raise RuntimeError("intentional interruption immediately after last.pt replacement")

    monkeypatch.setattr(training, "atomic_checkpoint", fail_after_last)
    with pytest.raises(RuntimeError, match="intentional interruption"):
        training.train_perception(config, "unused", run)
    committed = checkpoints.read_checkpoint(run / "last.pt")
    validation = read_rows(run / "validation.jsonl")
    assert validation[-1]["loss"] < validation[-2]["loss"]
    assert committed["metrics"]["loss"] == validation[-1]["loss"]
    expected_identity = committed["model_fingerprint"]
    monkeypatch.setattr(training, "atomic_checkpoint", original_save)
    result = training.train_perception(config, "unused", run, resume=True)
    selected = checkpoints.read_checkpoint(run / "best.pt")
    assert selected["model_fingerprint"] == expected_identity
    assert selected["metrics"]["loss"] == committed["metrics"]["loss"]
    assert result["selected_update"] == 2


def test_fresh_restart_before_first_checkpoint_does_not_duplicate_ledger_rows(tmp_path, monkeypatch, scalar_stage):
    run = tmp_path / "interrupted-before-checkpoint"
    config = tiny_config(validate_every=2)
    original_optimizer = training.optimizer_for

    def fail_second_update(*args, **kwargs):
        optimizer = original_optimizer(*args, **kwargs)
        original_step = optimizer.step
        calls = 0

        def step(*step_args, **step_kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("intentional interruption before first checkpoint")
            return original_step(*step_args, **step_kwargs)

        optimizer.step = step
        return optimizer

    monkeypatch.setattr(training, "optimizer_for", fail_second_update)
    with pytest.raises(RuntimeError, match="intentional interruption"):
        training.train_perception(config, "unused", run)
    assert not (run / "last.pt").exists()
    assert [row["step"] for row in read_rows(run / "training.jsonl")] == [1]
    monkeypatch.setattr(training, "optimizer_for", original_optimizer)
    training.train_perception(config, "unused", run)
    assert [row["step"] for row in read_rows(run / "training.jsonl")] == [1, 2]
    assert [row["step"] for row in read_rows(run / "validation.jsonl")] == [0, 2]
    training_archives = list(run.glob("training_interrupted_before_checkpoint_*.jsonl"))
    validation_archives = list(run.glob("validation_interrupted_before_checkpoint_*.jsonl"))
    assert len(training_archives) == len(validation_archives) == 1
    assert [row["step"] for row in read_rows(training_archives[0])] == [1]
    assert [row["step"] for row in read_rows(validation_archives[0])] == [0]


def _dependency_checkpoint(path, stage, named_models, **extra):
    value = {
        "schema_version": checkpoints.SCHEMA_VERSION,
        "tensor_schema": checkpoints.TENSOR_SCHEMA, "stage": stage,
        "models": {name: model.state_dict() for name, model in named_models.items()},
        "model_fingerprint": checkpoints.fingerprint_modules(named_models),
        "dataset_fingerprint": DATASET_FINGERPRINT, **extra,
    }
    checkpoints.atomic_checkpoint(path, value)
    return value


@pytest.fixture
def predictor_stage(tmp_path, monkeypatch, scalar_stage):
    observer = {"E": scalar_layer(), "D": scalar_layer(), "H": scalar_layer(3),
                "U": scalar_layer(), "R": scalar_layer(5)}
    perception = tmp_path / "perception.pt"
    memory = tmp_path / "memory.pt"
    initial = tmp_path / "one-step.pt"
    a = _dependency_checkpoint(perception, "perception", {k: observer[k] for k in ("E", "D", "H")})
    b = _dependency_checkpoint(memory, "memory", {k: observer[k] for k in ("U", "R")},
                               dependencies={"perception": a["model_fingerprint"]})
    statistics = {
        "encoder_fingerprint": a["model_fingerprint"], "dataset_fingerprint": DATASET_FINGERPRINT,
        "tensor_schema": checkpoints.TENSOR_SCHEMA, "precision": "float32", "count": 8,
        "v_fine": .25, "v_coarse": .5,
    }
    one_step = _dependency_checkpoint(initial, "predictor", {"P": scalar_layer()}, horizon=1,
                                      dependencies={"perception": a["model_fingerprint"], "memory": b["model_fingerprint"]},
                                      statistics=statistics, quality_gate={"passed": True})
    monkeypatch.setattr(training, "load_observer", lambda *args, **kwargs: copy.deepcopy(observer))
    monkeypatch.setattr(training, "observer_cache", lambda *args, **kwargs: [np.zeros((8, 128), np.float32)])
    recomputed = {**statistics, "v_fine": 9., "v_coarse": 13.}
    monkeypatch.setattr(training, "compute_statistics", lambda *args, **kwargs: copy.deepcopy(recomputed))

    def scalar_predictor_batch(system, samples, memories, windows, indices, horizon, stats, device):
        loss = sum(parameter.square().mean() for parameter in system["P"].parameters())
        return loss, {"copy_loss": 100., "h_mae": [[.5, .5, .5]] * horizon,
                      "copy_h_mae": [[1., 1., 1.]] * horizon, "windows": len(indices)}

    monkeypatch.setattr(training, "predictor_batch", scalar_predictor_batch)
    return SimpleNamespace(perception=perception, memory=memory, initial=initial,
                           one_step=one_step, statistics=statistics)


def test_five_step_checkpoint_records_initial_predictor_fingerprint(tmp_path, predictor_stage):
    fixture = predictor_stage
    run = tmp_path / "five-step-provenance"
    training.train_predictor(tiny_config(updates=1), "unused", fixture.perception, fixture.memory, 5,
                             run, initialize_from=fixture.initial)
    saved = checkpoints.read_checkpoint(run / "last.pt")
    assert saved["dependencies"].get("initial_predictor") == fixture.one_step["model_fingerprint"]


def test_five_step_reuses_exact_one_step_scale_statistics(tmp_path, predictor_stage):
    fixture = predictor_stage
    run = tmp_path / "five-step-statistics"
    training.train_predictor(tiny_config(updates=1), "unused", fixture.perception, fixture.memory, 5,
                             run, initialize_from=fixture.initial)
    saved = checkpoints.read_checkpoint(run / "last.pt")
    assert saved["statistics"] == fixture.statistics
    assert json.loads((run / "statistics.json").read_text()) == fixture.statistics


@pytest.mark.parametrize("status,stage", [("completed", "perception"), ("failed_quality_gate", "predictor")])
@pytest.mark.parametrize("receipt_exists", [True, False])
def test_resuming_completed_early_stop_or_gate_failure_never_advances_optimizer(
    tmp_path, monkeypatch, predictor_stage, status, stage, receipt_exists
):
    """Synthesize a terminal checkpoint below budget, avoiding 1,000 mock updates."""
    fixture = predictor_stage
    run = tmp_path / f"terminal-{status}"
    config = tiny_config(updates=1)
    if stage == "perception":
        training.train_perception(config, "unused", run)
    else:
        training.train_predictor(config, "unused", fixture.perception, fixture.memory, 1, run)
    # All numeric states/RNG/optimizer slots come from an actual scalar update.
    # Only lifecycle metadata is synthesized to represent early stopping below
    # the same fixed budget, or a completed failed-gate stage.
    config["training"]["perception"]["updates"] = 4
    config["training"]["predictor_1"]["updates"] = 4
    config["training"]["early_stopping"] = True
    config["smoke"] = False
    checkpoint_names = {"last.pt", "best.pt"}
    snapshot = checkpoints.read_checkpoint(run / "last.pt").get("best_checkpoint")
    if snapshot:
        checkpoint_names.add(snapshot)
    for name in checkpoint_names:
        value = checkpoints.read_checkpoint(run / name)
        value["config"] = copy.deepcopy(config)
        value["training_complete"] = True
        value["stop_reason"] = "early_stopping"
        value["no_improvement"] = 8
        value["status"] = status
        if status == "failed_quality_gate":
            value["quality_gate"] = {"passed": False, "smoke_bypass": False}
        checkpoints.atomic_checkpoint(run / name, value)
    result = json.loads((run / "paddle_result.json").read_text())
    result.update(status=status, smoke=False, stop_reason="early_stopping")
    if status == "failed_quality_gate":
        result["quality_gate"] = {"passed": False, "smoke_bypass": False}
    checkpoints.json_atomic(run / "paddle_result.json", result)
    checkpoints.json_atomic(run / "status.json", result)
    if not receipt_exists:
        # Checkpoint completion must survive a crash before its JSON receipts.
        (run / "paddle_result.json").unlink()
        (run / "status.json").unlink()
    before = checkpoints.read_checkpoint(run / "last.pt")
    before_rows = (run / "training.jsonl").read_bytes()
    original_optimizer = training.optimizer_for

    def optimizer_with_forbidden_step(*args, **kwargs):
        optimizer = original_optimizer(*args, **kwargs)

        def step(*step_args, **step_kwargs):
            pytest.fail("A completed early-stop or failed-gate stage advanced its optimizer")

        optimizer.step = step
        return optimizer

    monkeypatch.setattr(training, "optimizer_for", optimizer_with_forbidden_step)
    try:
        if stage == "perception":
            resumed = training.train_perception(config, "unused", run, resume=True)
        else:
            resumed = training.train_predictor(config, "unused", fixture.perception, fixture.memory, 1, run, resume=True)
    except RuntimeError as error:
        assert status == "failed_quality_gate" and ("gate" in str(error).lower() or "copy" in str(error).lower())
    else:
        assert resumed["status"] == status
    after = checkpoints.read_checkpoint(run / "last.pt")
    assert after["global_update"] == before["global_update"] == 1
    assert after["model_fingerprint"] == before["model_fingerprint"]
    assert (run / "training.jsonl").read_bytes() == before_rows
