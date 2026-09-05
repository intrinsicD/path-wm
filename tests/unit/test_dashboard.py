"""The viewer faithfully reconciles run ledgers and exposes every result family."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml
import pytest

from viewer.dashboard import (
    DashboardDataError,
    build_dashboard_artifact,
    collect_run_results,
    write_experiment_dashboard,
)


def _write_run(runs_root: Path, name: str, *, accuracy: float, total: float) -> Path:
    run_dir = runs_root / "dev" / name / "0"
    run_dir.mkdir(parents=True)
    metrics = {
        "action_sensitivity_ratio": accuracy / 10,
        "counterfactual_accuracy": accuracy,
        "counterfactual_branches": 4,
        "counterfactual_probe_count": 8,
        "probe_count": 8,
        "probe_horizon": 2,
        "transition_error": total / 10,
        "transition_error_identity": total / 20,
        "transition_error_one_step": total / 30,
        "transition_error_shuffled_action": total / 25,
        "transition_error_zero_action": total / 28,
    }
    final = {
        "action": total / 4,
        "counterfactual": total / 3,
        "counterfactual_accuracy": accuracy,
        "delta_t": 1,
        "gradient_norm": total * 2,
        "horizon": 2,
        "inverse": total / 5,
        "reg": total / 6,
        "rollout": total / 7,
        "step": 100,
        "total": total,
        "chunk": total / 8,
    }
    record = {
        "status": "development",
        "seed": 0,
        "step": 100,
        "checkpoint": str(run_dir / "checkpoint.pt"),
        "parameter_counts": {"encoder": 10, "predictor": 20},
        "metrics": metrics,
    }
    summary = {"device": "cpu", "final_training": final, "metrics": metrics, "run_dir": str(run_dir)}
    spec = {
        "experiment": "E1_reference",
        "status": "dev",
        "inverse_dynamics": {"kind": "mlp_token_pair"},
        "losses": {"counterfactual": {"weight": 1.0, "kappa": 0.001}},
        "train": {"steps": 100},
    }
    threshold = {
        "metrics": metrics,
        "thresholds": {"action_sensitivity_min": None, "transition_error_max": None},
    }
    training = [
        {**final, "step": 1, "counterfactual_accuracy": -1},
        final,
    ]
    (run_dir / "metrics.json").write_text(json.dumps(record), encoding="utf-8")
    (run_dir / "run_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (run_dir / "spec.yaml").write_text(yaml.safe_dump(spec), encoding="utf-8")
    (run_dir / "threshold_record.json").write_text(json.dumps(threshold), encoding="utf-8")
    (run_dir / "training.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in training), encoding="utf-8"
    )
    return run_dir


def test_dashboard_contains_all_completed_runs_and_result_families(tmp_path):
    runs_root = tmp_path / "runs"
    _write_run(runs_root, "control", accuracy=0.25, total=1.0)
    _write_run(runs_root, "candidate", accuracy=0.75, total=0.5)

    runs, notices = collect_run_results(runs_root)
    artifact = build_dashboard_artifact(runs, notices)

    assert {run.run_id for run in runs} == {"dev/candidate/0", "dev/control/0"}
    assert artifact["surface"] == "dashboard"
    assert artifact["snapshot"]["status"] == "ready"
    assert len(artifact["snapshot"]["datasets"]["run_overview"]) == 2
    sections = {row["section"] for row in artifact["snapshot"]["datasets"]["result_detail"]}
    assert sections == {"Evaluation", "Final training", "Parameters", "Threshold"}
    chart_ids = {chart["id"] for chart in artifact["manifest"]["charts"]}
    assert {
        "action_sensitivity_comparison",
        "counterfactual_accuracy_comparison",
        "transition_error_comparison",
        "selected_action_controls",
        "training_primary_objectives",
        "training_auxiliary_objectives",
        "training_counterfactual_accuracy",
        "training_gradient_norm",
        "training_curriculum",
        "parameter_counts",
    } <= chart_ids
    charts = {chart["id"]: chart for chart in artifact["manifest"]["charts"]}
    assert charts["action_sensitivity_comparison"]["title"].endswith("↑ higher is better")
    assert charts["counterfactual_accuracy_comparison"]["title"].endswith("↑ higher is better")
    assert charts["transition_error_comparison"]["title"].endswith("↓ lower is better")
    assert charts["selected_action_controls"]["title"].endswith("↓ correct / ↑ controls")
    assert charts["parameter_counts"]["title"].endswith("↔ context only")
    assert all(
        any(indicator in chart["title"] for indicator in ("↑", "↓", "↔"))
        for chart in charts.values()
        if chart["type"] == "bar"
    )
    assert artifact["manifest"]["filters"][0]["defaultValue"].startswith("DEV")
    selected = artifact["snapshot"]["datasets"]["run_overview"][0]
    assert selected["transition_error_milli"] == selected["transition_error"] * 1_000
    inventory = artifact["snapshot"]["datasets"]["run_inventory"]
    assert {row["kappa_display"] for row in inventory} == {"=0.001"}
    assert all(
        isinstance(row["value"], str)
        for row in artifact["snapshot"]["datasets"]["result_detail"]
    )
    assert all(card.get("sourceId") for card in artifact["manifest"]["cards"])
    assert all(chart.get("sourceId") for chart in artifact["manifest"]["charts"])
    assert all(table.get("sourceId") for table in artifact["manifest"]["tables"])


def test_dashboard_writer_publishes_only_after_builder_success(tmp_path, monkeypatch):
    runs_root = tmp_path / "runs"
    _write_run(runs_root, "candidate", accuracy=0.75, total=0.5)
    builder = tmp_path / "deliver_portable_artifact.mjs"
    builder.write_text("// test double", encoding="utf-8")
    called = {}

    def fake_deliver(artifact_path, html_path, builder_path):
        called.update(artifact=artifact_path, html=html_path, builder=builder_path)
        html_path.write_text("<!doctype html><title>PATH-WM Experiment Dashboard</title>", encoding="utf-8")
        return {"ok": True, "stages": {"verification": "passed"}}

    monkeypatch.setattr("viewer.dashboard._deliver_portable_artifact", fake_deliver)
    artifact_path, html_path, receipt = write_experiment_dashboard(runs_root, builder_path=builder)

    assert json.loads(artifact_path.read_text())["manifest"]["title"] == "PATH-WM Experiment Dashboard"
    assert html_path.read_text().startswith("<!doctype html>")
    assert called["builder"] == builder.resolve()
    assert receipt["stages"]["verification"] == "passed"


def test_dashboard_refuses_disagreeing_metric_copies(tmp_path):
    runs_root = tmp_path / "runs"
    run_dir = _write_run(runs_root, "candidate", accuracy=0.75, total=0.5)
    summary_path = run_dir / "run_summary.json"
    summary = json.loads(summary_path.read_text())
    summary["metrics"]["counterfactual_accuracy"] = 0.25
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(DashboardDataError, match="metric copies disagree"):
        collect_run_results(runs_root)


def test_runner_refreshes_dashboard_after_writing_summary(tmp_path, monkeypatch):
    import run as runner

    spec_path = tmp_path / "configs" / "dev" / "dashboard_smoke.yaml"
    spec_path.parent.mkdir(parents=True)
    spec = {
        "experiment": "E1_reference",
        "status": "dev",
        "probe_set": {"seed": 0},
        "losses": {"counterfactual": {"weight": 0.0}},
        "seeds": [0],
    }
    spec_path.write_text(yaml.safe_dump(spec), encoding="utf-8")
    dashboard_calls = []

    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "ensure_episode_store", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        runner,
        "train_e1",
        lambda _cfg, _store, run_dir, *_args, **_kwargs: (run_dir / "checkpoint.pt", {"step": 1}),
    )
    monkeypatch.setattr(runner, "evaluate_checkpoint", lambda *_args, **_kwargs: {"score": 0.5})

    def fake_dashboard(runs_root):
        summary_path = runs_root / "dev" / "dashboard_smoke" / "0" / "run_summary.json"
        assert summary_path.is_file()
        dashboard_calls.append(runs_root)
        return runs_root / "artifact.json", runs_root / "dashboard.html", {
            "stages": {"verification": "passed"}
        }

    monkeypatch.setattr(runner, "write_experiment_dashboard", fake_dashboard)
    monkeypatch.setattr(sys, "argv", ["run.py", str(spec_path), "--device", "cpu"])

    runner.main()

    assert dashboard_calls == [tmp_path / "runs"]


@pytest.mark.parametrize("r0_initialization", [None, {"checkpoint": "source.pt", "sha256": "a" * 64, "seed": 3}])
def test_runner_dispatches_common_base_without_touching_the_e1_collector(tmp_path, monkeypatch, r0_initialization):
    from types import SimpleNamespace

    import run as runner

    spec_path = tmp_path / "configs" / "dev" / "common_base.yaml"
    spec_path.parent.mkdir(parents=True)
    spec = {
        "experiment": "E1_common_base",
        "status": "dev",
        "seeds": [3],
        "train": {"stage": "representation_av" if r0_initialization else "representation_unimodal", "max_steps": 2},
    }
    spec_path.write_text(yaml.safe_dump(spec), encoding="utf-8")
    data = object()
    training_calls = []

    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "build_representation_data", lambda cfg, root: data)

    def fake_train(cfg, received_data, run_dir, seed, device, *, resume=False):
        assert not resume
        training_calls.append((cfg, received_data, run_dir, seed, device))
        return SimpleNamespace(
            checkpoint=run_dir / "checkpoint.pt",
            final_training={"step": 2, "total": 0.25},
            metrics={"video_feature_std": 0.5, "gate_passed": 0},
            parameter_counts={"common_base_total": 10},
            r0_initialization=r0_initialization,
            gate=SimpleNamespace(gate="unimodal_representation_ready", passed=False, failures=("not ready",)),
        )

    monkeypatch.setattr(runner, "train_common_base", fake_train)
    monkeypatch.setattr(
        runner,
        "ensure_episode_store",
        lambda *_args, **_kwargs: pytest.fail("common-base dispatch must not collect E0 episodes"),
    )
    monkeypatch.setattr(
        runner,
        "write_experiment_dashboard",
        lambda root: (root / "artifact.json", root / "dashboard.html", {"stages": {"verification": "passed"}}),
    )
    monkeypatch.setattr(sys, "argv", ["run.py", str(spec_path), "--device", "cpu"])

    runner.main()

    run_dir = tmp_path / "runs" / "dev" / "common_base" / "3"
    record = json.loads((run_dir / "metrics.json").read_text())
    summary = json.loads((run_dir / "run_summary.json").read_text())
    assert len(training_calls) == 1 and training_calls[0][1] is data
    assert record["metrics"] == summary["metrics"]
    assert summary["gate"]["passed"] is False

    assert record.get("r0_initialization") == summary.get("r0_initialization") == r0_initialization
    if r0_initialization is None:
        assert "r0_initialization" not in record and "r0_initialization" not in summary


@pytest.mark.parametrize("audiovisual", [False, True])
def test_mixed_dashboard_shows_latest_representation_metrics_without_inventing_action_values(tmp_path, audiovisual):
    runs_root = tmp_path / "runs"
    _write_run(runs_root, "action", accuracy=0.5, total=0.5)
    latest = _write_run(runs_root, "representation", accuracy=0.5, total=0.5)
    metrics = {"video_effective_rank_fraction": 0.12, "audio_effective_rank_fraction": 0.06,
               "video_future_prediction_advantage": -0.3, "audio_future_prediction_advantage": -0.00015,
               "gate_passed": 0, "gate_failure_count": 4}
    av_metrics = {"video_to_audio_retrieval_margin": 0.02, "audio_to_video_retrieval_margin": 0.03,
                  "synchrony_accuracy_above_chance": 0.04}
    if audiovisual:
        metrics.update(av_metrics)
    for filename in ("metrics.json", "run_summary.json", "threshold_record.json"):
        path = latest / filename
        value = json.loads(path.read_text())
        value["metrics"] = metrics
        path.write_text(json.dumps(value))
    spec = {"experiment": "E1_common_base", "status": "dev",
            "train": {"stage": "representation_av" if audiovisual else "representation_unimodal", "max_steps": 10000, "batch_size": 64},
            "data": {"source": {"subset": "development"}},
            "representation": {"weights": {"covariance": 0.002}}}
    (latest / "spec.yaml").write_text(yaml.safe_dump(spec))
    runs, notices = collect_run_results(runs_root)
    artifact = build_dashboard_artifact(runs, notices)
    datasets = artifact["snapshot"]["datasets"]
    cards = [c for c in artifact["manifest"]["cards"] if c["id"].startswith("selected_")]
    assert cards and all(c["dataset"] == "latest_run" for c in cards)
    assert all(datasets["latest_run"][0].get(c["metrics"][0]["field"]) is not None for c in cards)
    card_ids = {c["id"] for c in cards}
    assert all((f"selected_{name}" in card_ids) == audiovisual for name in av_metrics)
    assert not any("action_sensitivity" in c["id"] for c in cards)
    assert {row["value"] for row in datasets["representation_rank"]} == {0.12, 0.06}
    assert {row["value"] for row in datasets["representation_future"]} == {-0.3, -0.00015}
    row = next(row for row in datasets["run_inventory"] if "representation" in row["run"])
    assert row["train_steps"] == 10000
    assert row["source_subset"] == "development" and row["covariance_weight"] == 0.002
    assert datasets["latest_run"][0]["audio_future_prediction_advantage_milli"] == -0.15
    assert "action_sensitivity_comparison" in {c["id"] for c in artifact["manifest"]["charts"]}
