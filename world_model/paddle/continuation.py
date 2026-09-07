"""Audited budget-only forks of selected one-step predictor checkpoints.

The supplied checkpoint must itself be selected: its validation loss must equal
its recorded best. A nonselected ``last.pt`` is refused; supply ``best.pt``
instead. The sole configurable intervention is a larger predictor_1 update cap.
The parent remains immutable, and a fully prepared child directory is published
atomically for ``train_predictor(..., resume=True)``. No optimization runs here.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

from .checkpoints import atomic_checkpoint, atomic_checkpoint_copy, json_atomic, read_checkpoint


def _canonical(value) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("configuration and provenance must contain finite JSON values") from error


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _parent_run(checkpoint: Path, saved: dict) -> Path:
    # Standard immutable best snapshots belong to the enclosing run, not to a
    # separate run rooted at checkpoints/. Resolve symlinks before comparison.
    relative = Path(saved.get("best_checkpoint") or "")
    if relative.parts and relative.parts[0] == "checkpoints" and checkpoint.parent.name == "checkpoints":
        return checkpoint.parent.parent
    return checkpoint.parent


def _validate(checkpoint: Path, config: dict, run: Path, parent: dict):
    if parent.get("stage") != "predictor" or parent.get("horizon") != 1:
        raise ValueError("Only a selected one-step predictor checkpoint can be forked")
    required = ("models", "model_fingerprint", "optimizer", "rng", "statistics", "config", "metrics",
                "best_validation", "best_validation_criterion", "global_update", "examples_processed",
                "elapsed_seconds", "dependencies", "dataset_fingerprint", "quality_gate")
    missing = [key for key in required if key not in parent]
    if missing:
        raise ValueError(f"Predictor checkpoint is missing resume state: {missing}")
    if set(parent["models"]) != {"P"}:
        raise ValueError("One-step predictor checkpoint must contain exactly the predictor model P")
    try:
        loss, best = float(parent["metrics"]["loss"]), float(parent["best_validation"])
        old_budget = parent["config"]["training"]["predictor_1"]["updates"]
        new_budget = config["training"]["predictor_1"]["updates"]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Predictor checkpoint/config must record validation loss and predictor_1.updates") from error
    if not math.isfinite(loss) or not math.isfinite(best) or loss != best:
        raise ValueError("Checkpoint is not the selected best validation state; supply the parent's selected best.pt")
    if type(old_budget) is not int or type(new_budget) is not int or old_budget <= 0 or new_budget <= old_budget:
        raise ValueError("Only a strict integer increase of training.predictor_1.updates is allowed")
    expected = copy.deepcopy(parent["config"])
    expected["training"]["predictor_1"]["updates"] = new_budget
    if _canonical(config) != _canonical(expected):
        raise ValueError("Only training.predictor_1.updates may change; every other config field must match the parent exactly")
    update = parent["global_update"]
    examples = parent["examples_processed"]
    elapsed = parent["elapsed_seconds"]
    if type(update) is not int or not 0 < update <= old_budget:
        raise ValueError("Parent global update must be positive and within its recorded budget")
    if type(examples) is not int or examples < 0 or not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError("Parent cumulative example/time counters are invalid")
    if not all(key in parent["rng"] for key in ("python", "numpy", "sampler", "torch", "cuda")):
        raise ValueError("Parent checkpoint lacks a complete RNG/sampler state")
    statistics = parent["statistics"]
    if not isinstance(statistics, dict) or statistics.get("encoder_fingerprint") != parent["dependencies"].get("perception") or statistics.get("dataset_fingerprint") != parent["dataset_fingerprint"]:
        raise ValueError("Parent predictor statistics do not match its encoder/dataset identity")
    _canonical(statistics)
    parent_directory = _parent_run(checkpoint, parent)
    if run == checkpoint or run == parent_directory or parent_directory in run.parents:
        raise ValueError("Child directory must be separate from the parent checkpoint and parent run directory")
    if run.exists() and (not run.is_dir() or any(run.iterdir())):
        raise ValueError("Child output directory must be new or empty; existing content is preserved")
    return old_budget, new_budget


def fork_predictor(checkpoint, config, run) -> dict:
    """Prepare a new child preserving exact model/Adam/RNG/statistics state.

    Global update, examples and elapsed seconds remain cumulative counters from
    the supplied selected checkpoint. Added budget is the increase over the
    parent's configured cap; remaining work also includes any gap between that
    cap and the selected checkpoint's update. All are recorded separately.
    Validation/config/path checks finish before creating directories or files.
    """
    checkpoint = Path(checkpoint).resolve(strict=True)
    requested_run = Path(run)
    if requested_run.is_symlink():
        raise ValueError("Child output directory cannot be an existing symlink")
    run = requested_run.resolve()
    parent_hash = _sha256(checkpoint)
    parent = read_checkpoint(checkpoint)
    if _sha256(checkpoint) != parent_hash:
        raise ValueError("Parent checkpoint changed while it was being read; fork a completed immutable parent")
    old_budget, new_budget = _validate(checkpoint, config, run, parent)
    provenance = {
        "schema_version": "paddle-predictor-continuation-v1",
        "kind": "selected_one_step_predictor_budget_extension",
        "parent_checkpoint": str(checkpoint),
        "parent_sha256": parent_hash,
        "parent_model_fingerprint": parent["model_fingerprint"],
        "parent_global_update": parent["global_update"],
        "parent_examples_processed": parent["examples_processed"],
        "parent_elapsed_seconds": parent["elapsed_seconds"],
        "parent_config": copy.deepcopy(parent["config"]),
        "parent_budget_updates": old_budget,
        "child_budget_updates": new_budget,
        "added_budget_updates": new_budget - old_budget,
        "remaining_updates_from_parent_checkpoint": new_budget - parent["global_update"],
        "counter_semantics": "global_update, examples_processed, elapsed_seconds are cumulative from the supplied parent checkpoint; additional_* counts only this child continuation",
    }
    if parent.get("continuation"):
        provenance["parent_continuation"] = copy.deepcopy(parent["continuation"])
    _canonical(provenance)
    child = copy.deepcopy(parent)
    child.update({
        "config": copy.deepcopy(config), "training_complete": False, "stop_reason": None,
        "best_checkpoint": f"checkpoints/best_{parent['global_update']:08d}.pt",
        "continuation": provenance,
        "additional_updates": 0, "additional_examples_processed": 0,
        "additional_elapsed_seconds": 0.0,
    })
    # A predecessor's terminal status is not the lifecycle status of its child.
    child.pop("status", None)
    manifest = {
        "stage": "predictor", "horizon": 1, "status": "ready_to_resume",
        "config": copy.deepcopy(config), "dependencies": copy.deepcopy(parent["dependencies"]),
        "dataset_fingerprint": parent["dataset_fingerprint"], "tensor_schema": parent["tensor_schema"],
        "continuation": provenance,
    }
    status = {
        "stage": "predictor", "horizon": 1, "status": "ready_to_resume",
        "global_update": parent["global_update"], "best_validation": parent["best_validation"],
        "metrics": parent["metrics"], "quality_gate": parent["quality_gate"],
        "continuation": provenance, "additional_updates": 0,
        "additional_examples_processed": 0, "additional_elapsed_seconds": 0.0,
    }
    _canonical(manifest)
    _canonical(status)
    # All validation is complete. Temporary work is a sibling, so directory
    # rename is on one filesystem and no partial run is exposed to a trainer.
    run.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{run.name}.fork-", dir=run.parent))
    try:
        snapshot = temporary / child["best_checkpoint"]
        atomic_checkpoint(snapshot, child)
        atomic_checkpoint_copy(snapshot, temporary / "last.pt")
        atomic_checkpoint_copy(snapshot, temporary / "best.pt")
        json_atomic(temporary / "statistics.json", parent["statistics"])
        json_atomic(temporary / "resolved_config.json", config)
        json_atomic(temporary / "continuation.json", provenance)
        json_atomic(temporary / "paddle_manifest.json", manifest)
        json_atomic(temporary / "status.json", status)
        (temporary / "training.jsonl").write_text("")
        (temporary / "validation.jsonl").write_text(
            _canonical({"step": parent["global_update"], **parent["metrics"]}) + "\n"
        )
        if _sha256(checkpoint) != parent_hash:
            raise ValueError("Parent checkpoint changed before publication; no child was published")
        if run.exists() and (not run.is_dir() or any(run.iterdir())):
            raise ValueError("Child output directory acquired existing content before publication")
        os.rename(temporary, run)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return {"status": "ready_to_resume", "run": str(run),
            "checkpoint": str(run / "last.pt"), "continuation": provenance}
