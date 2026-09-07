"""Evaluation must bind physical metrics to the trained source and scales."""

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from torch import nn

from world_model.pusht import checkpoints, data, evaluation


class SimulationStarted(RuntimeError):
    pass


@pytest.fixture
def context(tmp_path, monkeypatch):
    normalization = {
        "schema": "pusht-train-rms-motion-and-action-offset-v1", "fit_split": "train",
        "source_episode_ids": [0], "source_sha256": "source-hdf5",
        "motion_scales": [8., 8., 2., 2., .03], "motion_count": 4,
        "action_offset_rms": [.04, .05], "action_count": 5,
    }
    config = {"seed": 3107, "device": "cpu", "cpu_threads": 1,
              "evaluation": {"horizon": 5, "split": "test", "control_cases": 1, "prediction_windows": 1}}
    value = {"schema_version": checkpoints.SCHEMA_VERSION, "tensor_schema": checkpoints.TENSOR_SCHEMA,
             "action_schema": checkpoints.ACTION_SCHEMA_VERSION, "dataset_fingerprint": "requested-dataset",
             "normalization": normalization, "config": config, "global_update": 1,
             "horizon": 1, "dependencies": {}}
    a = {**copy.deepcopy(value), "stage": "perception", "models": {k: nn.Linear(1, 1).state_dict() for k in ("E", "D", "H")}}
    a["model_fingerprint"] = checkpoints.fingerprint_modules(a["models"])
    b = {**copy.deepcopy(value), "stage": "memory", "models": {k: nn.Linear(1, 1).state_dict() for k in ("U", "R")},
         "dependencies": {"perception": a["model_fingerprint"]}}
    b["model_fingerprint"] = checkpoints.fingerprint_modules(b["models"])
    statistics = {"encoder_fingerprint": a["model_fingerprint"], "dataset_fingerprint": "requested-dataset",
                  "tensor_schema": checkpoints.TENSOR_SCHEMA, "precision": "float32", "split": "train",
                  "lengths": [6], "count": 6, "v_fine": 1., "v_coarse": 2.}
    c = {**copy.deepcopy(value), "stage": "predictor", "horizon": 5,
         "models": {"P": nn.Linear(1, 1).state_dict()}, "statistics": statistics,
         "dependencies": {"perception": a["model_fingerprint"], "memory": b["model_fingerprint"], "initial_predictor": "K1"}}
    c["model_fingerprint"] = checkpoints.fingerprint_modules(c["models"])
    paths = [tmp_path / f"{name}.pt" for name in ("perception", "memory", "predictor")]
    calls = {"construct": 0, "simulation": 0}
    def construct():
        calls["construct"] += 1
        return nn.Linear(1, 1)
    monkeypatch.setattr(checkpoints, "_construct", lambda: {key: construct for key in ("E", "D", "H", "U", "R", "P")})
    dataset = SimpleNamespace(fingerprint="requested-dataset", split="test", manifest={"normalization": normalization})
    monkeypatch.setattr(data, "EpisodeDataset", lambda *args: dataset)
    def simulation(*args, **kwargs):
        calls["simulation"] += 1
        raise SimulationStarted("simulation was reached")
    monkeypatch.setattr(evaluation, "prepare_cases", simulation)
    def run():
        for path, checkpoint in zip(paths, (a, b, c)):
            checkpoints.atomic_checkpoint(path, checkpoint)
        return evaluation.evaluate(config, "unused", *paths, tmp_path / "evaluation")
    return SimpleNamespace(a=a, b=b, c=c, dataset=dataset, calls=calls, config=config,
                           output=tmp_path / "evaluation", run=run)


@pytest.mark.parametrize("mismatch", [
    "dataset", "normalization", "statistics_encoder", "statistics_dataset",
    "statistics_split", "statistics_schema", "statistics_nonfinite",
    "statistics_negative", "statistics_count", "predictor_horizon",
])
def test_incompatible_evaluation_inputs_are_rejected_before_models_or_simulation(context, mismatch):
    if mismatch == "dataset":
        for checkpoint in (context.a, context.b, context.c):
            checkpoint["dataset_fingerprint"] = "another-mutually-consistent-dataset"
        context.c["statistics"]["dataset_fingerprint"] = "another-mutually-consistent-dataset"
    elif mismatch == "normalization":
        for checkpoint in (context.a, context.b, context.c):
            checkpoint["normalization"]["motion_scales"][0] = 999.
    elif mismatch == "statistics_encoder": context.c["statistics"]["encoder_fingerprint"] = "another-encoder"
    elif mismatch == "statistics_dataset": context.c["statistics"]["dataset_fingerprint"] = "another-dataset"
    elif mismatch == "statistics_split": context.c["statistics"]["split"] = "validation"
    elif mismatch == "statistics_schema": context.c["statistics"]["tensor_schema"] = "paddle"
    elif mismatch == "statistics_nonfinite": context.c["statistics"]["v_fine"] = float("nan")
    elif mismatch == "statistics_negative": context.c["statistics"]["v_coarse"] = -1.
    elif mismatch == "statistics_count": context.c["statistics"]["count"] = 0
    elif mismatch == "predictor_horizon": context.c["horizon"] = 1
    with pytest.raises(ValueError, match="dataset|normalization|statistic|encoder|schema|variance|count|horizon|five|K5"):
        context.run()
    assert context.calls == {"construct": 0, "simulation": 0}
    assert not (context.output / "goals").exists()
    assert not (context.output / "case_manifest.json").exists()
    if (context.output / "metrics.json").exists():
        assert json.loads((context.output / "metrics.json").read_text())["status"] not in ("completed", "report_pending")


def test_matching_dependencies_reach_simulation_and_record_native_preprocessing_code(context):
    with pytest.raises(SimulationStarted):
        context.run()
    assert context.calls == {"construct": 6, "simulation": 1}
    protocol = json.loads((context.output / "protocol.json").read_text())
    assert protocol["normalization"] == context.dataset.manifest["normalization"]
    root = Path(__file__).resolve().parents[2]
    fingerprints = protocol["code"]
    for relative in ("world_model/pusht/data.py", "third_party/swm/pusht.py"):
        expected = hashlib.sha256((root / relative).read_bytes()).hexdigest()
        assert expected in fingerprints.values(), f"Missing behavior dependency {relative} in evaluation identity"
