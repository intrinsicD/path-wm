"""A budget-only child preserves its parent and the exact optimizer trajectory."""

import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from world_model.paddle import checkpoints, continuation, training
from world_model.paddle.continuation import fork_predictor

from test_checkpoint_transactions import predictor_stage, scalar_stage, tiny_config


def assert_identical(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_identical(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert type(a) is type(b) and len(a) == len(b)
        for av, bv in zip(a, b):
            assert_identical(av, bv)
    else:
        assert a == b


def hash_tree(root):
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.rglob("*") if path.is_file()}


@pytest.fixture
def selected_parent(tmp_path, predictor_stage):
    fixture = predictor_stage
    parent = tmp_path / "parent"
    config = tiny_config(updates=2)
    training.train_predictor(config, "unused", fixture.perception, fixture.memory, 1, parent)
    return parent, config, fixture


def extended(config):
    child = copy.deepcopy(config)
    child["training"]["predictor_1"]["updates"] += 2
    return child


def test_fork_preserves_parent_bytes_all_numeric_state_and_audited_counters(tmp_path, selected_parent):
    parent, config, _ = selected_parent
    parent_checkpoint = parent / "best.pt"
    source = checkpoints.read_checkpoint(parent_checkpoint)
    before = hash_tree(parent)
    child = tmp_path / "child"
    new_config = extended(config)
    result = fork_predictor(parent_checkpoint, new_config, child)
    assert hash_tree(parent) == before
    saved = checkpoints.read_checkpoint(child / "last.pt")
    for field in ("models", "optimizer", "rng", "statistics", "metrics", "best_validation",
                  "best_validation_criterion", "global_update", "examples_processed", "elapsed_seconds",
                  "dependencies", "model_fingerprint", "dataset_fingerprint", "quality_gate"):
        assert_identical(source[field], saved[field])
    assert saved["config"] == new_config and source["config"] == config
    assert saved["training_complete"] is False and saved["stop_reason"] is None
    assert saved["best_checkpoint"] == f"checkpoints/best_{source['global_update']:08d}.pt"
    snapshot = checkpoints.read_checkpoint(child / saved["best_checkpoint"])
    assert_identical(saved, snapshot)
    assert_identical(saved, checkpoints.read_checkpoint(child / "best.pt"))
    provenance = saved["continuation"]
    assert provenance["parent_checkpoint"] == str(parent_checkpoint.resolve())
    assert provenance["parent_sha256"] == before["best.pt"]
    assert provenance["parent_model_fingerprint"] == source["model_fingerprint"]
    assert provenance["parent_global_update"] == source["global_update"]
    assert provenance["parent_examples_processed"] == source["examples_processed"]
    assert provenance["parent_elapsed_seconds"] == source["elapsed_seconds"]
    assert provenance["parent_config"] == config
    assert provenance["added_budget_updates"] == 2
    assert provenance["remaining_updates_from_parent_checkpoint"] == 2
    assert "cumulative" in provenance["counter_semantics"]
    assert json.loads((child / "statistics.json").read_text()) == source["statistics"]
    assert (child / "training.jsonl").read_bytes() == b""
    rows = [json.loads(line) for line in (child / "validation.jsonl").read_text().splitlines()]
    assert rows == [{"step": source["global_update"], **source["metrics"]}]
    assert not (child / "paddle_result.json").exists()
    assert result["status"] == "ready_to_resume"
    assert json.loads((child / "status.json").read_text())["status"] == "ready_to_resume"


@pytest.mark.parametrize("field,value", [
    (("seed",), 99),
    (("smoke",), False),
    (("training", "learning_rate"), .02),
    (("training", "predictor_1", "batch_size"), 2),
    (("training", "predictor_1", "validation_examples"), 2),
    (("training", "predictor_5", "updates"), 77),
    (("training", "predictor_1", "updates"), 2),
    (("training", "predictor_1", "updates"), True),
])
def test_only_a_strict_predictor_one_budget_increase_is_allowed(tmp_path, selected_parent, field, value):
    parent, config, _ = selected_parent
    new_config = extended(config)
    current = new_config
    for key in field[:-1]:
        current = current[key]
    current[field[-1]] = value
    target = tmp_path / "must-not-be-created"
    before = hash_tree(parent)
    with pytest.raises(ValueError, match="config|budget|updates|only"):
        fork_predictor(parent / "best.pt", new_config, target)
    assert not target.exists() and hash_tree(parent) == before


@pytest.mark.parametrize("stage,horizon,nonselected", [("memory", 1, False), ("predictor", 5, False), ("predictor", 1, True)])
def test_only_selected_one_step_predictors_can_be_forked(tmp_path, selected_parent, stage, horizon, nonselected):
    parent, config, _ = selected_parent
    value = checkpoints.read_checkpoint(parent / "last.pt")
    value.update(stage=stage, horizon=horizon)
    if nonselected:
        value["metrics"]["loss"] = value["best_validation"] + .1
    unsuitable = parent / "unsuitable.pt"
    checkpoints.atomic_checkpoint(unsuitable, value)
    target = tmp_path / "rejected"
    with pytest.raises(ValueError, match="predictor|one-step|selected|best.pt"):
        fork_predictor(unsuitable, extended(config), target)
    assert not target.exists()


def test_rejects_parent_descendants_existing_output_and_checkpoint_path(tmp_path, selected_parent):
    parent, config, _ = selected_parent
    checkpoint = parent / "best.pt"
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "unrelated.txt").write_text("preserve")
    before_parent, before_occupied = hash_tree(parent), hash_tree(occupied)
    for target in (parent, parent / "nested-child", checkpoint, occupied):
        with pytest.raises(ValueError, match="parent|exist|empty|directory|checkpoint"):
            fork_predictor(checkpoint, extended(config), target)
    # A checkpoint stored under checkpoints/ still belongs to its whole parent run.
    snapshot = parent / checkpoints.read_checkpoint(checkpoint)["best_checkpoint"]
    with pytest.raises(ValueError, match="parent"):
        fork_predictor(snapshot, extended(config), parent / "another-child")
    assert hash_tree(parent) == before_parent and hash_tree(occupied) == before_occupied


def test_interrupted_preparation_exposes_no_partial_child(tmp_path, monkeypatch, selected_parent):
    parent, config, _ = selected_parent
    before = hash_tree(parent)
    child = tmp_path / "atomic-child"

    def interrupt_alias(*args, **kwargs):
        raise OSError("intentional interruption while preparing child aliases")

    monkeypatch.setattr(continuation, "atomic_checkpoint_copy", interrupt_alias)
    with pytest.raises(OSError, match="intentional interruption"):
        fork_predictor(parent / "best.pt", extended(config), child)
    assert not child.exists()
    assert not list(tmp_path.glob(".atomic-child.fork-*"))
    assert hash_tree(parent) == before


def test_fork_then_resume_matches_same_uninterrupted_scalar_trajectory(tmp_path, selected_parent):
    parent, config, fixture = selected_parent
    new_config = extended(config)
    uninterrupted, child = tmp_path / "uninterrupted", tmp_path / "child"
    training.train_predictor(new_config, "unused", fixture.perception, fixture.memory, 1, uninterrupted)
    before = hash_tree(parent)
    fork_predictor(parent / "best.pt", new_config, child)
    provenance = checkpoints.read_checkpoint(child / "last.pt")["continuation"]
    training.train_predictor(new_config, "unused", fixture.perception, fixture.memory, 1, child, resume=True)
    full, resumed = checkpoints.read_checkpoint(uninterrupted / "last.pt"), checkpoints.read_checkpoint(child / "last.pt")
    for field in ("models", "optimizer", "rng", "statistics", "metrics", "model_fingerprint",
                  "global_update", "examples_processed", "best_validation"):
        assert_identical(full[field], resumed[field])
    assert resumed["global_update"] == 4
    assert hash_tree(parent) == before
    training_rows = [json.loads(line) for line in (child / "training.jsonl").read_text().splitlines()]
    assert [row["step"] for row in training_rows] == [3, 4]
    validation_rows = [json.loads(line) for line in (child / "validation.jsonl").read_text().splitlines()]
    assert [row["step"] for row in validation_rows] == [2, 3, 4]
    manifest = json.loads((child / "paddle_manifest.json").read_text())
    assert manifest["continuation"] == provenance
    artifacts = [resumed, checkpoints.read_checkpoint(child / "best.pt"),
                 json.loads((child / "paddle_result.json").read_text())]
    for artifact in artifacts:
        assert artifact["continuation"] == provenance
        assert artifact["additional_updates"] == 2
        assert artifact["additional_examples_processed"] == 2
        assert math.isfinite(artifact["additional_elapsed_seconds"])
        assert artifact["additional_elapsed_seconds"] >= 0
