"""Prevent false completion and misleading evidence in the experiment harness."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from run import run_experiment
from viewer.ledger import DashboardDataError, collect_run_results


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_stopped_run_keeps_validation_and_saved_checkpoint_steps(tmp_path):
    run = tmp_path / "runs" / "stopped"
    write_json(run / "manifest.json", {"config": {"seed": 4}})
    write_rows(run / "metrics.jsonl", [
        {"kind": "validation", "step": 0, "pred_mse": 0.5},
        {"kind": "train", "step": 140, "loss": 0.1},
        {"kind": "stopped_by_user", "last_logged_step": 140, "saved_checkpoint_step": 0},
    ])
    write_json(run / "status.json", {"kind": "stopped_by_user", "last_logged_step": 140,
                                     "saved_checkpoint_step": 0})
    results, notices = collect_run_results(tmp_path / "runs")
    assert len(results) == 1
    result = results[0]
    assert result.status == "stopped_by_user"
    assert result.step == 140
    assert result.validation[-1]["step"] == 0
    assert result.context["saved_checkpoint_step"] == 0
    assert "pred_mse" not in result.metrics  # No stale validation on step 140.


def test_control_count_disagreement_blocks_dashboard(tmp_path):
    run = tmp_path / "runs" / "control"
    write_json(run / "summary.json", {"successes": 1, "cases": 2})
    write_rows(run / "cases.jsonl", [{"success": False}, {"success": False}])
    with pytest.raises(DashboardDataError, match="success"):
        collect_run_results(tmp_path / "runs")


def test_unfinished_control_is_visible_but_not_complete(tmp_path):
    run = tmp_path / "runs" / "interrupted"
    write_rows(run / "cases.jsonl", [{"success": True}])
    results, notices = collect_run_results(tmp_path / "runs")
    assert not results
    assert any("interrupted" in notice and "summary" in notice for notice in notices)


def test_nonfinite_metric_is_rejected(tmp_path):
    run = tmp_path / "runs" / "bad"
    write_json(run / "summary.json", {"successes": 0, "cases": 1, "mean_steps": float("nan")})
    write_rows(run / "cases.jsonl", [{"success": False}])
    with pytest.raises(DashboardDataError, match="non-finite"):
        collect_run_results(tmp_path / "runs")


def test_failed_experiment_still_refreshes_and_preserves_failure():
    calls = []
    def execute(command):
        calls.append(command)
        return SimpleNamespace(returncode=7)
    def refresh():
        calls.append("refresh")
        return Path("artifact.json"), Path("dashboard.html"), {"stages": {"verification": "passed"}}
    assert run_experiment(["-m", "example", "--seed", "3"], execute=execute, refresh=refresh) == 7
    assert calls[0][1:] == ["-m", "example", "--seed", "3"]
    assert calls[-1] == "refresh"


def test_html_failure_prevents_workflow_success(capsys):
    def refresh():
        raise RuntimeError("HTML verification failed")
    code = run_experiment(["example.py"], execute=lambda _: SimpleNamespace(returncode=0), refresh=refresh)
    assert code != 0
    assert "HTML verification failed" in capsys.readouterr().err


def test_case_navigation_and_sql_snapshot_preserve_counts(tmp_path):
    from viewer.dashboard import build_dashboard_artifact
    for name, successes in (("learned", [False, False]), ("released", [True, False])):
        run = tmp_path / "runs" / name
        write_json(run / "summary.json", {"successes": sum(successes), "cases": 2})
        write_rows(run / "cases.jsonl", [{"episode": i, "start": 5, "success": success}
                                         for i, success in enumerate(successes)])
    results, notices = collect_run_results(tmp_path / "runs")
    artifact = build_dashboard_artifact(results, notices)
    rows = artifact["snapshot"]["datasets"]["control"]
    assert len({row["case_set"] for row in rows}) == 1
    assert {row["run"]: (row["successes"], row["cases"], row["success_rate"])
            for row in rows} == {"learned": (0, 2, 0.0), "released": (1, 2, 0.5)}
    details = artifact["snapshot"]["datasets"]["metrics"]
    assert next(row["value"] for row in details if row["run"] == "released" and row["metric"] == "success_rate") == "0.5"


def test_structural_only_does_not_claim_visual_verification(capsys):
    def refresh():
        return Path("artifact.json"), Path("dashboard.html"), {"stages": {"verification": "structural_only"}}
    assert run_experiment(["example.py"], execute=lambda _: SimpleNamespace(returncode=0), refresh=refresh) != 0
    assert "visual verification remains pending" in capsys.readouterr().err


def test_packaging_failure_keeps_last_html(tmp_path, monkeypatch):
    from viewer import dashboard
    html = tmp_path / "runs" / "experiment_dashboard.html"
    html.parent.mkdir()
    html.write_text("previous verified dashboard")
    def fail_builder(artifact, target, builder):
        raise dashboard.DashboardBuildError("browser QA failed before publication")
    monkeypatch.setattr(dashboard, "_deliver_portable_artifact", fail_builder)
    with pytest.raises(dashboard.DashboardBuildError):
        dashboard.write_experiment_dashboard(html.parent, builder_path=tmp_path / "builder.mjs")
    assert html.read_text() == "previous verified dashboard"
