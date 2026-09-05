"""Restore the offline experiment panel independently of implementation goals.

Chart construction, bounded sampling, atomic JSON and portable packaging are
recovered from `eca742a:viewer/dashboard.py`. The adapter reads current ledgers;
the canonical builder owns HTML and browser QA. No raw evidence is modified.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sqlite3
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from viewer.ledger import DashboardDataError, RunResult, collect_run_results

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS_ROOT = ROOT / "runs"
DEFAULT_ARTIFACT = DEFAULT_RUNS_ROOT / "experiment_dashboard.artifact.json"
DEFAULT_HTML = DEFAULT_RUNS_ROOT / "experiment_dashboard.html"
SOURCE_ID = "path_wm_run_ledger"
MAX_DATASET_ROWS = 2000
DIRECTION_HINTS = {"higher": "higher is better", "lower": "lower is better", "context": "context"}

class DashboardBuildError(RuntimeError):
    """The canonical artifact could not be packaged as verified HTML."""

def _downsample(rows: tuple[dict[str, int | float], ...], limit: int) -> list[dict[str, int | float]]:
    if limit <= 0 or len(rows) <= limit:
        return [dict(row) for row in rows]
    if limit == 1:
        return [dict(rows[-1])]
    indices = {round(index * (len(rows) - 1) / (limit - 1)) for index in range(limit)}
    return [dict(rows[index]) for index in sorted(indices)]


def _chart(
    chart_id: str,
    title: str,
    subtitle: str,
    dataset: str,
    chart_type: str,
    x: dict[str, Any],
    y: dict[str, Any],
    *,
    color: dict[str, Any] | None = None,
    layout: str = "half",
    tooltip: list[dict[str, Any]] | None = None,
    reference_lines: list[dict[str, Any]] | None = None,
    group_mode: str | None = None,
    orientation: str | None = None,
    direction: str | None = None,
) -> dict[str, Any]:
    encodings: dict[str, Any] = {"x": x, "y": y}
    if color is not None:
        encodings["color"] = color
    if tooltip:
        encodings["tooltip"] = tooltip
    if direction is not None:
        title = f"{title} · {DIRECTION_HINTS[direction]}"
    chart: dict[str, Any] = {
        "id": chart_id,
        "title": title,
        "subtitle": subtitle,
        "intent": "trend" if chart_type == "line" else "comparison",
        "type": chart_type,
        "dataset": dataset,
        "sourceId": SOURCE_ID,
        "encodings": encodings,
        "layout": layout,
        "palette": {"kind": "categorical" if color else "identity", "name": "PATH-WM blue-orange"},
        "labels": {"values": "auto" if chart_type == "bar" else "endpoints"},
        "surface": {"interactiveLegend": color is not None, "viewMode": "both"},
    }
    if color is not None:
        chart["legend"] = {"position": "bottom", "sort": "spec"}
    if reference_lines:
        chart["referenceLines"] = reference_lines
    if chart_type == "line":
        settings: dict[str, Any] = {"showPoints": "never"}
    else:
        settings = {"showValues": True, "sort": "none"}
        if group_mode:
            settings["groupMode"] = group_mode
        if orientation:
            settings.update({"orientation": orientation, "categoryLabelPolicy": "wrap"})
    chart["settings"] = settings
    return chart


def _text(value):
    return format(value, ".17g") if isinstance(value, float) else str(value)


def build_dashboard_artifact(run_results: list[RunResult], notices: list[str]) -> dict:
    generated = datetime.now(UTC).isoformat()
    datasets = {name: [] for name in ("inventory", "metrics", "context", "training_runs", "training_loss",
                                    "validation", "embedding", "control", "prediction_runs", "prediction")}
    training_runs = [r for r in run_results if r.kind == "training"]
    controls = [r for r in run_results if r.kind == "control"]
    predictions = [r for r in run_results if r.kind == "prediction"]
    records = [{**asdict(run), "sampled_training": _downsample(run.training, 50),
                "sampled_validation": _downsample(run.validation, 50)} for run in run_results]
    source_sql = (ROOT / "viewer/experiment_results.sql").read_text()
    with sqlite3.connect(":memory:") as connection:
        connection.row_factory = sqlite3.Row
        sections = re.split(r"-- dataset: ([a-z_]+)\n", source_sql)
        for name, query in zip(sections[1::2], sections[2::2]):
            datasets[name] = [dict(row) for row in connection.execute(query, {"reconciled_runs": json.dumps(records)})]
    for name in ("metrics", "context"):
        for row in datasets[name]:
            row["value"] = _text(row["value"])
    for name, rows in datasets.items():
        if len(rows) > MAX_DATASET_ROWS:
            raise DashboardDataError(f"{name} exceeds {MAX_DATASET_ROWS} rows; select a smaller --runs-root")
    source_paths = sorted({p for run in run_results for p in run.source_paths})
    source = {"id": SOURCE_ID, "label": "PATH-WM local experiment ledgers", "path": "viewer/experiment_results.sql",
              "query": {"engine": "sqlite", "language": "sql", "executed_at": generated,
                        "sql": source_sql,
                        "description": "viewer.ledger reconciles raw JSON/JSONL. SQLite executes this query bundle over those records (:reconciled_runs) to produce the chart datasets and coverage counts. Exact context/value tables use the same queries and Python text formatting. No source files are modified.",
                        "tables_used": source_paths,
                        "transformation": "viewer/ledger.py validates and reconciles the bound RunResult records; viewer/dashboard.py samples chart trajectories and formats exact numeric text.",
                        "filters": ["Training metrics.jsonl; supported control summary.json with case evidence; action_baselines.json; prediction.json",
                                    "Up to 50 deterministic evenly spaced points per training/validation trajectory; exact ledger values remain in sources",
                                    "Control grouping uses ordered case identities only, not a claim of equivalent protocols. No cross-run pooling.",
                                    "Ranking uses complete matched candidate sets; five-block curves select one model, case and action sequence."],
                        "metric_definitions": {"position_error": "Euclidean distance in combined agent/block XY positions after all 25 candidate actions; pixels. Success also requires circular angle error below pi/9.",
                                               "predicted_cost": "Sum of squared terminal latent differences to the source goal, within a selected checkpoint and case.",
                                               "rollout_mse": "Mean squared latent error of an autoregressive prediction against simulator-rendered observations at each five-action block.",
                                               "success_rate": "Count of boolean successful cases divided by the recorded case count; summary counts/rates must agree. Includes initially successful cases when recorded.",
                                               "pred_mse": "Recorded one-step latent prediction mean squared error for the run's validation sample and precision.",
                                               "identity_mse": "Recorded latent MSE for copying the current embedding, on matched validation windows.",
                                               "shuffled_action_mse": "Recorded latent prediction MSE after the evaluator's action permutation.",
                                               "zero_action_mse": "Recorded latent prediction MSE with zero actions.",
                                               "embedding_std": "Recorded embedding standard deviation; context for changing latent scale, not a success gate.",
                                               "loss": "Recorded total training objective. pred_loss and sigreg_loss are raw components; weights are in run context.",
                                               "completed_training": "Training records with status.kind equal to complete. Does not imply prediction or control success."}}}
    cards = [{"id": field, "dataset": "coverage", "sourceId": SOURCE_ID, "description": description,
              "metrics": [{"label": label, "field": field, "format": "number"}]}
             for field, label, description in (
                 ("completed_training", "Completed training runs", "Completed training loops; scientific gates are recorded separately."),
                 ("control_evaluations", "Control result records", "Separately reported model, evaluator and action-baseline results; case sets may differ."),
                 ("prediction_evaluations", "Standalone prediction checks", "Saved prediction.json evaluations; in-training validation appears in curves."))]
    charts, filters = [], []
    number = lambda field: {"field": field, "type": "quantitative"}
    category = lambda field: {"field": field, "type": "nominal"}
    if controls:
        charts.append(_chart("control", "Control success on selected case identities",
                             "Case identities match within the selection; weights, normalization and protocols can differ. Counts and context are below.",
                             "control", "bar", category("label"), number("success_rate"), layout="full",
                             tooltip=[category("run"), number("successes"), number("cases"), category("protocol")],
                             orientation="horizontal", direction="higher"))
        filters.append({"id": "case_set", "label": "Control case identities", "dataset": "control", "field": "case_set",
                        "defaultValue": controls[-1].context["case_set"], "includeAll": False,
                        "targets": [{"dataset": "control", "field": "case_set"}]})
    if training_runs:
        for name, title, subtitle in (
            ("training_loss", "Training objective components", "Raw components have different weights; consult run context. At most 50 logged points per run."),
            ("validation", "Validation predictions and matched controls", "Latent scale changes during learning; compare controls at the same step, not absolute errors across encoders.")):
            if datasets[name]:
                charts.append(_chart(name, title, subtitle, name, "line", number("step"), number("value"), color=category("metric")))
        if datasets["embedding"]:
            charts.append(_chart("embedding", "Validation embedding scale", "Changing latent scale helps interpret the prediction curves; it does not establish useful control.",
                                 "embedding", "line", number("step"), number("value")))
        filters.append({"id": "training_run", "label": "Training run", "dataset": "training_runs", "field": "run",
                        "defaultValue": training_runs[-1].label, "includeAll": False,
                        "targets": [{"dataset": name, "field": "run"} for name in ("training_loss", "validation", "embedding")]})
    if predictions:
        charts.append(_chart("prediction", "Standalone prediction controls", "Each selection uses its own encoder, validation windows and precision; values are comparable within that selection.",
                             "prediction", "bar", category("metric"), number("value"), direction="lower"))
        filters.append({"id": "prediction_run", "label": "Prediction check", "dataset": "prediction_runs", "field": "run",
                        "defaultValue": predictions[-1].label, "includeAll": False,
                        "targets": [{"dataset": "prediction", "field": "run"}]})
    if datasets.get("ranking"):
        charts.append(_chart("ranking", "Predicted goal cost versus simulator position error",
                             "20 identical raw action sequences per model/case. Lower values are better; angle and terminal success remain separate.",
                             "ranking", "scatter", number("predicted_cost"), number("position_error"),
                             color=category("family"), layout="full",
                             tooltip=[category("candidate"), number("angle_error"), category("success_terminal")]))
        for dataset, title, subtitle in (
            ("rollout_error", "Multi-step prediction and copy error",
             "MSE against simulator-rendered observations, within the selected model's latent space."),
            ("rollout_goal", "Predicted and measured latent goal distance",
             "Squared latent distance to the source goal at all five planning blocks. This is not physical distance.")):
            charts.append(_chart(dataset, title, subtitle, dataset, "line", number("environment_step"),
                                 number("value"), color=category("metric")))
        filters.append({"id": "ranking_run", "label": "Ranking model and case", "dataset": "ranking_runs", "field": "run",
                        "defaultValue": datasets["ranking_runs"][0]["run"], "includeAll": False,
                        "targets": [{"dataset": name, "field": "run"} for name in
                                    ("ranking", "rollout_error", "rollout_goal")]})
        filters.append({"id": "rollout_candidate", "label": "Rollout action sequence", "dataset": "rollout_error", "field": "candidate",
                        "defaultValue": "pilot_plan", "includeAll": False,
                        "targets": [{"dataset": name, "field": "candidate"} for name in ("rollout_error", "rollout_goal")]})
    def table(name, title, columns, sort, subtitle):
        return {"id": name, "title": title, "subtitle": subtitle, "dataset": name, "sourceId": SOURCE_ID,
                "defaultSort": {"field": sort, "direction": "asc"}, "density": "dense", "layout": "full",
                "columns": [{"field": field, "label": label, "type": "text"} for field, label in columns]}
    tables = [table("inventory", "Run inventory and recorded gate status",
                    [("run", "Run"), ("kind", "Kind"), ("status", "Status"), ("step", "Logged step"), ("gate", "Recorded gate"), ("sources", "Source files")],
                    "run", "Training completion is independent of the scientific gate. Missing gates remain unassessed.")]
    if controls:
        # Exact counts are strings in a separate dataset, so the chart retains numeric rates.
        datasets["control_detail"] = [{k: _text(v) for k, v in row.items()} for row in datasets["control"]]
        tables.append(table("control_detail", "All control outcomes",
                            [("run", "Run"), ("successes", "Successes"), ("cases", "Cases"), ("success_rate", "Exact success fraction"), ("case_set", "Case identities"), ("protocol", "Protocol")],
                            "case_set", "All recorded populations, without pooling. This inventory is independent of the case filter."))
    tables.extend([table("metrics", "Selected record: exact measured values",
                         [("section", "Measurement"), ("metric", "Metric"), ("value", "Exact value")], "section", "Validation values retain their actual recorded step. No thresholds or passing gates are inferred."),
                   table("context", "Selected record: protocol and configuration",
                         [("field", "Field"), ("value", "Recorded value")], "field", "Consult the source manifest for complete case lists, normalization arrays and checkpoint identity.")])
    if datasets.get("ranking"):
        tables.append(table("ranking", "Selected model/case: exact candidate outcomes",
                            [("candidate", "Candidate"), ("predicted_cost", "Predicted cost"),
                             ("position_error", "Position error (px)"), ("angle_error", "Angle error (rad)"),
                             ("success_terminal", "Terminal success"), ("actual_latent_cost", "Measured latent cost")],
                            "candidate", "The scatter and this table use the same 20 raw candidate sequences."))
    if run_results:
        filters.append({"id": "detail_run", "label": "Exact record details", "dataset": "inventory", "field": "run",
                        "defaultValue": run_results[-1].label, "includeAll": False,
                        "targets": [{"dataset": name, "field": "run"} for name in ("metrics", "context")]})
    blocks = [{"id": "intro", "type": "markdown", "body": "# PATH-WM Experiment Dashboard\n\nOffline snapshot of raw experiment evidence. Use the independent selectors for control cases, training curves, prediction checks and exact record details. Completion and scientific success are separate."},
              {"id": "coverage", "type": "metric-strip", "cardIds": [card["id"] for card in cards]}]
    blocks += [{"id": f"chart_{chart['id']}", "type": "chart", "chartId": chart["id"], "layout": chart["layout"]} for chart in charts]
    blocks += [{"id": f"table_{item['id']}", "type": "table", "tableId": item["id"], "layout": "full"} for item in tables]
    notes = ["No experimental pass threshold is inferred by the viewer. Recorded gate annotations remain visible.",
             "Control groups reflect ordered case identities only; inspect protocol context before comparing results."] + notices
    if not run_results:
        notes.append("No supported run ledgers found yet. This is an empty instrument panel.")
    blocks.append({"id": "notices", "type": "markdown", "sourceId": SOURCE_ID,
                   "body": "## Evidence notes\n\n" + "\n".join(f"- {note}" for note in notes)})
    return {"surface": "dashboard", "manifest": {"version": 1, "surface": "dashboard", "title": "PATH-WM Experiment Dashboard",
            "description": "Offline visual inspection of training, prediction and control ledgers.", "generatedAt": generated,
            "filters": filters, "cards": cards, "charts": charts, "tables": tables, "sources": [source], "blocks": blocks},
            "snapshot": {"version": 1, "generatedAt": generated, "status": "ready", "datasets": datasets}, "sources": [source]}


def find_portable_artifact_builder() -> Path:
    """Locate the canonical builder, with an explicit override for non-Codex environments."""

    configured = os.environ.get("PATH_WM_ARTIFACT_BUILDER")
    if configured:
        candidate = Path(configured).expanduser().resolve()
        if candidate.is_file():
            return candidate
        raise DashboardBuildError(f"PATH_WM_ARTIFACT_BUILDER does not name a file: {candidate}")

    cache_root = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex")) / "plugins" / "cache"
    candidates = list(
        cache_root.glob(
            "openai-curated-remote/data-analytics/*/skills/build-report/scripts/deliver_portable_artifact.mjs"
        )
    )
    if not candidates:
        raise DashboardBuildError(
            "Data Analytics portable-artifact builder was not found. Set PATH_WM_ARTIFACT_BUILDER "
            "to deliver_portable_artifact.mjs."
        )

    def version_key(path: Path) -> tuple[int, int, int, int]:
        match = re.match(r"(\d+)\.(\d+)\.(\d+)", path.parents[3].name)
        version = tuple(int(part) for part in match.groups()) if match else (0, 0, 0)
        return (*version, path.stat().st_mtime_ns)

    return max(candidates, key=version_key)


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _deliver_portable_artifact(artifact_path: Path, html_path: Path, builder_path: Path) -> dict[str, Any]:
    node = shutil.which("node")
    if node is None:
        raise DashboardBuildError("Node.js is required to package the self-contained experiment dashboard")
    completed = subprocess.run(
        [node, str(Path(__file__).with_name("deliver_dashboard.mjs")),
         "--builder", str(builder_path), "--input", str(artifact_path), "--output", str(html_path)],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "CHROMIUM_EXECUTABLE_PATH": os.environ.get(
            "CHROMIUM_EXECUTABLE_PATH", str(Path(__file__).with_name("chromium_transport.mjs")))},
        timeout=120,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "portable builder failed"
        raise DashboardBuildError(detail)
    try:
        receipt = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise DashboardBuildError("portable builder returned an unreadable verification receipt") from error
    if not receipt.get("ok") or not html_path.is_file():
        raise DashboardBuildError(f"portable builder did not publish {html_path}")
    return receipt


def write_experiment_dashboard(
    runs_root: Path = DEFAULT_RUNS_ROOT,
    artifact_path: Path | None = None,
    html_path: Path | None = None,
    builder_path: Path | None = None,
) -> tuple[Path, Path, dict[str, Any]]:
    """Rebuild the aggregate artifact and verified, self-contained HTML dashboard."""

    runs_root = runs_root.resolve()
    artifact_path = (artifact_path or runs_root / DEFAULT_ARTIFACT.name).resolve()
    html_path = (html_path or runs_root / DEFAULT_HTML.name).resolve()
    run_results, notices = collect_run_results(runs_root)
    artifact = build_dashboard_artifact(run_results, notices)
    _write_json_atomic(artifact_path, artifact)
    receipt = _deliver_portable_artifact(
        artifact_path,
        html_path,
        (builder_path or find_portable_artifact_builder()).resolve(),
    )
    _write_json_atomic(html_path.with_suffix(".receipt.json"), receipt)
    return artifact_path, html_path, receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--builder", type=Path)
    args = parser.parse_args()
    artifact_path, html_path, receipt = write_experiment_dashboard(
        args.runs_root,
        artifact_path=args.artifact,
        html_path=args.output,
        builder_path=args.builder,
    )
    verification = receipt.get("stages", {}).get("verification", "unknown")
    print(f"Dashboard: {html_path}")
    print(f"Artifact: {artifact_path}")
    print(f"Verification: {verification}")


if __name__ == "__main__":
    main()
