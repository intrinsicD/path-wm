"""Checkpoint-backed ABI thresholds for the fixed first-slice probe set.

The fixture always regenerates and records s(w), H-step/one-step transition error, the identity,
zero-action and shuffled-action controls, and paired-intervention discrimination accuracy. Null ABI
thresholds produce an explicit `threshold_unset` skip only after values are persisted; the action
controls and counterfactual accuracy are diagnostics, not thresholds.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from evaluation import evaluate_checkpoint
from world_state.abi import ROOT


@pytest.fixture(scope="module")
def threshold_result(request):
    configured = Path(request.config.getoption("--run-dir"))
    run_dir = configured if configured.is_absolute() else ROOT / configured
    checkpoint = run_dir / "checkpoint.pt"
    if not checkpoint.exists():
        pytest.fail(f"missing {checkpoint}; run `python run.py configs/dev/first_slice.yaml` first")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metrics = evaluate_checkpoint(checkpoint, run_dir, device)
    # Threshold values live under conformance in YAML and are intentionally absent from the ABI
    # layout dataclass; read them directly so null remains distinguishable from an unset default.
    import yaml

    abi_yaml = yaml.safe_load((ROOT / payload["cfg"]["abi"]).read_text())
    thresholds = abi_yaml["conformance"]
    record = {"metrics": metrics, "thresholds": thresholds, "checkpoint": str(checkpoint)}
    (run_dir / "threshold_record.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metrics, thresholds


@pytest.mark.threshold
def test_action_sensitivity_threshold(threshold_result):
    metrics, thresholds = threshold_result
    threshold = thresholds["action_sensitivity_min"]
    if threshold is None:
        pytest.skip("threshold_unset")
    assert metrics["action_sensitivity_ratio"] >= threshold


@pytest.mark.threshold
def test_transition_error_threshold(threshold_result):
    metrics, thresholds = threshold_result
    threshold = thresholds["transition_error_max"]
    if threshold is None:
        pytest.skip("threshold_unset")
    assert metrics["transition_error"] <= threshold
