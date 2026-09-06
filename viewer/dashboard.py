"""Restore the offline experiment panel independently of implementation goals.

Chart construction, bounded sampling, atomic JSON and portable packaging are
recovered from `eca742a:viewer/dashboard.py`. The adapter reads current ledgers;
the canonical builder owns HTML and browser QA. No raw evidence is modified.
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import os
import re
import shutil
import subprocess
import sqlite3
import tempfile
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
TRAINING_PANELS = {
    "pred_loss": ("Prediction loss on training batches", "Mean squared error between predicted and actual next embeddings. The latent scale also changes, so read it with the validation ratios."),
    "sigreg_loss": ("SIGReg regularizer on training batches", "Epps–Pulley statistic of the embeddings against an isotropic Gaussian; entered into the objective multiplied by sigreg_weight (run context)."),
    "loss": ("Total training objective", "Prediction loss plus weighted SIGReg, as optimized."),
    "grad_norm": ("Gradient norm before clipping", "Global L2 norm of all parameter gradients before grad_clip (run context); spikes indicate instability."),
    "lr": ("Learning rate (×10⁻⁶)", "Warmup then cosine decay, as applied at the logged step; values are the recorded learning rate multiplied by one million so the axis is legible."),
}
INTERNALS_PANELS = {
    "internals_rank": ("effective_rank", "rankme", "participation_ratio"),
    "internals_gaussianity": ("shapiro_w_mean", "shapiro_fraction_below_0_95", "excess_kurtosis_mean", "skewness_mean_abs"),
    "internals_action_use": ("gate_msa_mean", "gate_mlp_mean", "sensitivity_action_over_state"),
    "internals_attention": ("encoder_attention_entropy_last", "encoder_attention_entropy_mean", "predictor_attention_entropy_mean"),
    "internals_param_norm": tuple(f"param_norm_{g}" for g in ("encoder", "projector", "predictor", "action_encoder", "pred_proj")),
    "internals_grad_norm": tuple(f"grad_norm_{g}" for g in ("encoder", "projector", "predictor", "action_encoder", "pred_proj")),
}
INTERNALS_TITLES = {
    "internals_rank": ("Representation rank over training", "Effective rank and RankMe of the validation embedding covariance; the isotropic Gaussian target of SIGReg scores the full dimension (run context)."),
    "internals_gaussianity": ("Gaussianity of embedding dimensions", "Per-dimension Shapiro–Wilk W (1 = Gaussian; below 0.95 flagged), excess kurtosis and skewness of validation embeddings."),
    "internals_action_use": ("Predictor action use", "Mean |AdaLN gate| for attention and MLP branches (zero at initialization) and the ratio of action to state sensitivity of the next-state prediction."),
    "internals_attention": ("Attention entropy", "Normalized entropy of attention rows (1 = uniform, 0 = one-hot) for the encoder's last and mean layer and the predictor over its history."),
    "internals_param_norm": ("Parameter norm by module", "L2 norm of all parameters in each module group of the saved checkpoint."),
    "internals_grad_norm": ("Gradient norm by module (log10)", "log10 of the L2 norm of the full-objective gradient on one validation batch, eval mode, no update; shows where the objective pushes. Modules differ by orders of magnitude."),
}

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


def _panel_blocks(internals: list[RunResult]) -> list[dict[str, Any]]:
    """Embed rendered checkpoint panels as small multiples; images are static evidence, not charts."""
    ordered: dict[str, list[tuple[RunResult, dict]]] = {}
    for run in internals:
        for panel in run.internals.get("panels", []):
            ordered.setdefault(panel["id"], []).append((run, panel))
    blocks = []
    for panel_id, items in ordered.items():
        figures = []
        for run, panel in items:
            encoded = base64.b64encode(Path(panel["path"]).read_bytes()).decode("ascii")
            step = f"step {run.step}" if run.step is not None else "no training step"
            figures.append(
                f'<figure style="margin:0;flex:0 1 calc(50% - 8px);min-width:320px;max-width:100%">'
                f'<img alt="{_escape(panel["title"])} for {_escape(run.label)}" src="data:image/png;base64,{encoded}" style="width:100%;height:auto">'
                f'<figcaption style="font-size:12px;color:GrayText"><strong>{_escape(run.label)}</strong> · {step} · {_escape(panel.get("caption", ""))}</figcaption></figure>')
        title = items[0][1]["title"]
        body = (f'<h3 style="margin:0 0 8px;font-size:16px">{_escape(title)}</h3>'
                f'<div style="display:flex;flex-wrap:wrap;gap:16px">{"".join(figures)}</div>'
                f'<p style="font-size:12px;color:GrayText;margin:8px 0 16px">Rendered from raw inspection outputs; sources: '
                f'{_escape(", ".join(sorted({Path(panel["path"]).name for _, panel in items})))}. LeWM has no image decoder; images show real inputs and measured internals.</p>')
        blocks.append({"id": f"panel_{panel_id}", "type": "html", "body": body, "layout": "full"})
    return blocks


def _escape(text: Any) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def build_dashboard_artifact(run_results: list[RunResult], notices: list[str], focus: str | None = None) -> dict:
    generated = datetime.now(UTC).isoformat()
    datasets = {name: [] for name in ("inventory", "metrics", "context", "training_runs", "training_loss",
                                    "validation", "embedding", "control", "prediction_runs", "prediction",
                                    "training_scalar", "validation_ratio", "internals_scalar", "internals_summary",
                                    "internals_spectrum", "internals_horizon", "internals_probe", "training_internals")}
    training_runs = [r for r in run_results if r.kind == "training"]
    controls = [r for r in run_results if r.kind == "control"]
    predictions = [r for r in run_results if r.kind == "prediction"]
    records = [{**asdict(run), "sampled_training": _downsample(run.training, 50),
                "sampled_validation": _downsample(run.validation, 50),
                "sampled_internals": _downsample(tuple(run.internals.get("training_rows", ())), 50)} for run in run_results]
    source_sql = (ROOT / "viewer/experiment_results.sql").read_text()
    with sqlite3.connect(":memory:") as connection:
        connection.row_factory = sqlite3.Row
        sections = re.split(r"-- dataset: ([a-z_]+)\n", source_sql)
        for name, query in zip(sections[1::2], sections[2::2]):
            datasets[name] = [dict(row) for row in connection.execute(query, {"reconciled_runs": json.dumps(records)})]
    for name in ("metrics", "context", "internals_summary"):
        for row in datasets[name]:
            row["value"] = _text(row["value"])
    # One panel per training scalar so components with different scales stay readable.
    for metric in TRAINING_PANELS:
        datasets[f"train_{metric}"] = [row for row in datasets["training_scalar"] if row["metric"] == metric]
    for row in datasets["train_lr"]:
        row["value"] = row["value"] * 1e6
    # Copy error is near zero at initialization, so plain ratios explode; the chart uses log10.
    for row in datasets["validation_ratio"]:
        row["log10_ratio"] = math.log10(row["value"]) if row["value"] > 0 else None
    for row in datasets["internals_spectrum"]:
        row["log10_eigenvalue"] = math.log10(row["eigenvalue"]) if row["eigenvalue"] > 0 else None
    # The reader draws x categories in row order, so every step/component axis is sorted here.
    for name in ("internals_scalar", "internals_probe", "training_internals"):
        datasets[name].sort(key=lambda row: (row["metric"] if "metric" in row else row["target"],
                                             row["step"] is None, row["step"] or 0))
    datasets["internals_spectrum"].sort(key=lambda row: (row["run"], row["component"]))
    datasets["internals_horizon"].sort(key=lambda row: (row["run"], row["metric"], row["horizon"]))
    for name, metrics in INTERNALS_PANELS.items():
        datasets[name] = [row for row in datasets["internals_scalar"] if row["metric"] in metrics]
        datasets[f"training_{name}"] = [row for row in datasets["training_internals"] if row["metric"] in metrics]
    for name in ("internals_grad_norm", "training_internals_grad_norm"):
        for row in datasets[name]:
            row["log10_value"] = math.log10(row["value"]) if row["value"] > 0 else None
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
                                               "noninitial_success_rate": "Boolean successful cases with initial_success false divided by all cases with initial_success false; unavailable without complete initial-state evidence or with zero eligible cases.",
                                               "embedding_std": "Recorded embedding standard deviation; context for changing latent scale, not a success gate.",
                                               "loss": "Recorded total training objective. pred_loss and sigreg_loss are raw components; weights are in run context.",
                                               "completed_training": "Training records with status.kind equal to complete. Does not imply prediction or control success.",
                                               "validation_ratio": "pred_mse divided by identity_mse, shuffled_action_mse or zero_action_mse at the same validation step; rollout_mse over identity_mse; action_effect over pred_mse. 1.0 equals the control.",
                                               "effective_rank": "exp(entropy) of normalized covariance eigenvalues of validation embeddings; rankme uses singular values; participation_ratio is (sum e)^2 / sum e^2.",
                                               "shapiro_fraction_below_0_95": "Fraction of embedding dimensions whose Shapiro–Wilk statistic on up to 500 validation samples is below 0.95.",
                                               "gate_msa_mean": "Mean absolute AdaLN gate of the predictor's attention branches over validation inputs; gate_mlp_mean likewise for MLP branches. Both are zero at initialization.",
                                               "sensitivity_action_over_state": "Mean finite-difference directional derivative norm of the next-state prediction with respect to the action input divided by the same with respect to the state input.",
                                               "probe_r2": "Held-out R² of a ridge regression from the first-frame latent to a physical target, fitted on training-window latents.",
                                               "grad_norm_by_module": "L2 norm of the full-objective gradient for one validation batch in eval mode, per module group, without an optimizer step."}}}
    cards = [{"id": field, "dataset": "coverage", "sourceId": SOURCE_ID, "description": description,
              "metrics": [{"label": label, "field": field, "format": "number"}]}
             for field, label, description in (
                 ("completed_training", "Completed training runs", "Completed training loops; scientific gates are recorded separately."),
                 ("control_evaluations", "Control result records", "Separately reported model, evaluator and action-baseline results; case sets may differ."),
                 ("prediction_evaluations", "Standalone prediction checks", "Saved prediction.json evaluations; in-training validation appears in curves."))]
    charts = []
    number = lambda field: {"field": field, "type": "quantitative"}
    category = lambda field: {"field": field, "type": "nominal"}
    # The canonical reader applies only filters that target every dataset, so per-section
    # selectors are inert there. Selections are therefore fixed here, named in titles, and
    # complete evidence stays in the tables. Rebuild with --focus to change the training run.
    focus = focus or (training_runs[-1].label if training_runs else None)
    if focus and focus not in {r.label for r in training_runs}:
        raise DashboardDataError(f"focus run {focus!r} is not an indexed training run")
    selections = {"training_run": focus}
    keep = lambda name, predicate: datasets.__setitem__(name, [row for row in datasets[name] if predicate(row)])
    for name in ("training_loss", "validation", "embedding", "validation_ratio", *(f"train_{m}" for m in TRAINING_PANELS)):
        keep(name, lambda row: row["run"] == focus)
    if controls:
        case_set = controls[-1].context["case_set"]
        selections["control_case_set"] = case_set
        datasets["control_detail"] = [{k: _text(v) for k, v in row.items()} for row in datasets["control"]]
        keep("control", lambda row: row["case_set"] == case_set)
        charts.append(_chart("control", f"Control success on case identities {case_set}",
                             "Same source and goal identities; weights, normalization and budgets can differ. Raw success includes initially satisfied goals; their counts and conditional rates are below.",
                             "control", "bar", category("label"), number("success_rate"), layout="full",
                             tooltip=[category("run"), number("successes"), number("cases"), category("protocol")],
                             orientation="horizontal", direction="higher"))
        datasets["control_noninitial"] = [row for row in datasets["control"]
                                           if row.get("noninitial_cases") and row.get("noninitial_success_rate") is not None]
        if datasets["control_noninitial"]:
            charts.append(_chart("control_noninitial", f"Reaching initially unsolved goals · {case_set}",
                                 "Successful cases that were not initially within the goal threshold, divided by all initially unsolved cases. Results without initial-state evidence are omitted from this chart.",
                                 "control_noninitial", "bar", category("label"), number("noninitial_success_rate"), layout="full",
                                 tooltip=[category("run"), number("noninitial_successes"), number("noninitial_cases"), number("initial_successes")],
                                 orientation="horizontal", direction="higher"))
    if training_runs:
        for metric, (title, subtitle) in TRAINING_PANELS.items():
            name = f"train_{metric}"
            if datasets[name]:
                charts.append(_chart(name, f"{title} · {focus}", subtitle, name, "line", number("step"), number("value"),
                                     color=category("run"), direction="context" if metric == "lr" else "lower"))
        if datasets["validation_ratio"]:
            charts.append(_chart("validation_ratio", f"Validation error relative to matched controls (log10) · {focus}",
                                 "log10 of prediction MSE over the copy, shuffled-action and zero-action control MSE at the same step; rollout over copy; action effect over prediction. 0 = equal to the control, −1 = ten times smaller. Exact plain ratios are in the record table.",
                                 "validation_ratio", "line", number("step"), number("log10_ratio"), color=category("metric"),
                                 reference_lines=[{"axis": "y", "value": 0, "label": "equal to control", "lineStyle": "dashed"}],
                                 direction="lower"))
        for name, title, subtitle in (
            ("training_loss", "Training objective components together", "Raw components on one axis for reference; the per-component panels above are readable. At most 50 logged points per run."),
            ("validation", "Validation predictions and matched controls (absolute)", "Latent scale changes during learning; compare controls at the same step, not absolute errors across encoders.")):
            if datasets[name]:
                charts.append(_chart(name, f"{title} · {focus}", subtitle, name, "line", number("step"), number("value"), color=category("metric")))
        if datasets["embedding"]:
            charts.append(_chart("embedding", f"Validation embedding scale · {focus}", "Changing latent scale helps interpret the prediction curves; it does not establish useful control.",
                                 "embedding", "line", number("step"), number("value"), color=category("run")))
    if training_runs:
        for name, (title, subtitle) in INTERNALS_TITLES.items():
            captured = f"training_{name}"
            keep(captured, lambda row: row["run"] == focus)
            if datasets[captured]:
                charts.append(_chart(captured, f"{title}, captured during training · {focus}",
                                     subtitle + " Measured on the first validation batch at each validation step (opt-in introspect).",
                                     captured, "line", number("step"), number("value"), color=category("metric")))
    if predictions:
        selections["prediction_run"] = predictions[-1].label
        keep("prediction", lambda row: row["run"] == predictions[-1].label)
        charts.append(_chart("prediction", f"Standalone prediction controls · {predictions[-1].label}",
                             "This selection uses its own encoder, validation windows and precision; values are comparable within it. Other prediction checks are in the exact tables.",
                             "prediction", "bar", category("metric"), number("value"), direction="lower"))
    internals = [r for r in run_results if r.kind == "internals"]
    if internals:
        # Step curves belong to the focus run's checkpoints; reference weights appear in the spectrum,
        # horizon curves, panels and the summary table, never on the step axis.
        for name in INTERNALS_PANELS:
            keep(name, lambda row: row["training_run"] == focus)
        keep("internals_probe", lambda row: row["training_run"] == focus)
        for name, (title, subtitle) in INTERNALS_TITLES.items():
            if datasets[name]:
                y = "log10_value" if name == "internals_grad_norm" else "value"
                charts.append(_chart(name, f"{title} · {focus}", subtitle, name, "line", number("step"), number(y),
                                     color=category("metric"), tooltip=[category("run"), category("family")]))
        if datasets["internals_probe"]:
            charts.append(_chart("internals_probe", f"Linear readout of physical state from the latent · {focus}",
                                 "Validation R² of a ridge probe from the encoder's latent to each physical target; 1 = perfectly linearly readable, 0 = no better than the mean.",
                                 "internals_probe", "line", number("step"), number("r2"), color=category("target"),
                                 tooltip=[category("run"), category("family")], direction="higher"))
        if datasets["internals_spectrum"]:
            charts.append(_chart("internals_spectrum", "Embedding covariance spectrum, all inspected checkpoints",
                                 "log10 eigenvalues of the validation embedding covariance, sorted; a flat curve is isotropic, a cliff is dimensional collapse. Absolute levels follow each model's latent scale.",
                                 "internals_spectrum", "line", number("component"), number("log10_eigenvalue"),
                                 color=category("run"), layout="full", direction="context"))
        by_run = {}
        for row in datasets["internals_horizon"]:
            by_run.setdefault((row["run"], row["horizon"]), {})[row["metric"]] = row["value"]
        datasets["internals_horizon_ratio"] = [
            {"run": run, "horizon": horizon, "metric": label, "value": values[a] / values[b],
             "log10_ratio": math.log10(values[a] / values[b]) if values[a] > 0 else None}
            for (run, horizon), values in sorted(by_run.items())
            for label, a, b in (("autoregressive / copy first state", "prediction_mse", "copy_mse"),
                                ("one step / copy previous state", "one_step_mse", "previous_mse"))
            if a in values and b in values and values[b] > 0]
        for label in ("autoregressive / copy first state", "one step / copy previous state"):
            rows = [row for row in datasets["internals_horizon_ratio"] if row["metric"] == label]
            if rows:
                name = "internals_horizon_autoregressive" if label.startswith("auto") else "internals_horizon_one_step"
                datasets[name] = rows
                charts.append(_chart(name, f"Latent error versus horizon (log10): {label}",
                                     "log10 of the model's error over the matching copy baseline at each horizon on its recorded evaluation population; 0 equals copying, −1 is ten times better. Scale-free within each checkpoint.",
                                     name, "line", number("horizon"), number("log10_ratio"), color=category("run"),
                                     reference_lines=[{"axis": "y", "value": 0, "label": "equal to copying", "lineStyle": "dashed"}],
                                     direction="lower"))
    if datasets.get("ranking"):
        ranking_run = datasets["ranking_runs"][0]["run"]
        selections["ranking_run"], selections["rollout_candidate"] = ranking_run, "pilot_plan"
        datasets["ranking_selected"] = [row for row in datasets["ranking"] if row["run"] == ranking_run]
        charts.append(_chart("ranking", f"Predicted goal cost versus simulator position error · {ranking_run}",
                             "20 identical raw action sequences for this model/case. Lower values are better; angle and terminal success remain separate. All model/case records are in the table.",
                             "ranking_selected", "scatter", number("predicted_cost"), number("position_error"),
                             color=category("family"), layout="full",
                             tooltip=[category("candidate"), number("angle_error"), category("success_terminal")]))
        for dataset, title, subtitle in (
            ("rollout_error", "Multi-step prediction and copy error",
             "MSE against simulator-rendered observations, within the selected model's latent space."),
            ("rollout_goal", "Predicted and measured latent goal distance",
             "Squared latent distance to the source goal at all five planning blocks. This is not physical distance.")):
            keep(dataset, lambda row: row["run"] == ranking_run and row["candidate"] == "pilot_plan")
            charts.append(_chart(dataset, f"{title} · {ranking_run} · pilot_plan", subtitle, dataset, "line", number("environment_step"),
                                 number("value"), color=category("metric")))
    def table(name, title, columns, sort, subtitle):
        return {"id": name, "title": title, "subtitle": subtitle, "dataset": name, "sourceId": SOURCE_ID,
                "defaultSort": {"field": sort, "direction": "asc"}, "density": "dense", "layout": "full",
                "columns": [{"field": field, "label": label, "type": "text"} for field, label in columns]}
    tables = [table("inventory", "Run inventory and recorded gate status",
                    [("run", "Run"), ("kind", "Kind"), ("status", "Status"), ("step", "Logged step"), ("gate", "Recorded gate"), ("sources", "Source files")],
                    "run", "Training completion is independent of the scientific gate. Missing gates remain unassessed.")]
    if controls:
        tables.append(table("control_detail", "All control outcomes",
                            [("run", "Run"), ("successes", "Successes"), ("cases", "Cases"), ("success_rate", "Raw success fraction"), ("initial_successes", "Initially satisfied"), ("noninitial_cases", "Initially unsolved"), ("noninitial_successes", "Newly reached"), ("noninitial_success_rate", "Success among initially unsolved"), ("case_set", "Goal identities"), ("protocol", "Protocol")],
                            "case_set", "All recorded populations, without pooling; sort by case identities to compare like with like."))
    tables.extend([table("metrics", "Exact measured values for every record",
                         [("run", "Record"), ("section", "Measurement"), ("metric", "Metric"), ("value", "Exact value")], "run", "Validation values retain their actual recorded step. No thresholds or passing gates are inferred. Sort or page by record."),
                   table("context", "Protocol and configuration for every record",
                         [("run", "Record"), ("field", "Field"), ("value", "Recorded value")], "run", "Consult the source manifest for complete case lists, normalization arrays and checkpoint identity.")])
    if internals:
        tables.append(table("internals_summary", "All inspected checkpoints: exact internals",
                            [("run", "Inspected checkpoint"), ("family", "Family"), ("step", "Step"), ("metric", "Metric"), ("value", "Exact value")],
                            "run", "Every scalar for every inspected checkpoint, including reference weights without a training step. Latent-scale quantities are not comparable across separately trained encoders."))
    if datasets.get("ranking"):
        tables.append(table("ranking", "All model/case candidate outcomes",
                            [("run", "Model and case"), ("candidate", "Candidate"), ("predicted_cost", "Predicted cost"),
                             ("position_error", "Position error (px)"), ("angle_error", "Angle error (rad)"),
                             ("success_terminal", "Terminal success"), ("actual_latent_cost", "Measured latent cost")],
                            "run", "The scatter above shows one model/case; this table holds the same 20 raw candidate sequences for every model/case."))
    guide = ["# PATH-WM Experiment Dashboard", "",
             "Offline snapshot of raw experiment evidence. Charts show one named selection each; tables hold every record.",
             "Completion and scientific success are separate. Rebuild with `python -m viewer.dashboard --focus <training run>` to chart another run.", "",
             "Selections in this snapshot:"] + [f"- {k.replace('_', ' ')}: `{v}`" for k, v in selections.items() if v]
    blocks = [{"id": "intro", "type": "markdown", "body": "\n".join(guide)},
              {"id": "coverage", "type": "metric-strip", "cardIds": [card["id"] for card in cards]}]
    blocks += [{"id": f"chart_{chart['id']}", "type": "chart", "chartId": chart["id"], "layout": chart["layout"]} for chart in charts]
    # PNG evidence dominates the portable payload. Bound only the image panels;
    # every inspection remains in the exact tables and quantitative series.
    focused_panels = sorted([r for r in internals if focus is not None and r.internals.get("training_run") == focus],
                            key=lambda r: (r.step if r.step is not None else -1, r.modified_at))
    panel_runs = []
    if focused_panels:
        panel_runs = [focused_panels[0]]
        if focused_panels[-1].label != focused_panels[0].label:
            panel_runs.append(focused_panels[-1])
        source_manifest = focused_panels[-1].context.get("source_run_manifest")
        matched = [r for r in internals if r.internals.get("family") == "released" and source_manifest
                   and r.context.get("source_run_manifest") == source_manifest]
        if matched:
            panel_runs.append(max(matched, key=lambda r: r.modified_at))
    blocks += _panel_blocks(panel_runs)
    blocks += [{"id": f"table_{item['id']}", "type": "table", "tableId": item["id"], "layout": "full"} for item in tables]
    notes = ["No experimental pass threshold is inferred by the viewer. Recorded gate annotations remain visible.",
             "Control groups reflect ordered case identities only; inspect protocol context before comparing results.",
             "The reader ignores per-section selectors, so each chart's selection is fixed at build time and named in its title."] + notices
    if internals:
        notes.append("Image panels show the focus run's earliest and latest inspected checkpoints plus a released inspection "
                     "only when it uses the same source-run manifest. All inspection scalars, spectra, horizon curves and raw panel "
                     "paths remain in the exact tables, charts and source inventory. Selected panels: "
                     + (", ".join(r.label for r in panel_runs) or "none for this focus run"))
    if not run_results:
        notes.append("No supported run ledgers found yet. This is an empty instrument panel.")
    blocks.append({"id": "notices", "type": "markdown", "sourceId": SOURCE_ID,
                   "body": "## Evidence notes\n\n" + "\n".join(f"- {note}" for note in notes)})
    filters = []
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
    focus: str | None = None,
) -> tuple[Path, Path, dict[str, Any]]:
    """Rebuild the aggregate artifact and verified, self-contained HTML dashboard."""

    runs_root = runs_root.resolve()
    artifact_path = (artifact_path or runs_root / DEFAULT_ARTIFACT.name).resolve()
    html_path = (html_path or runs_root / DEFAULT_HTML.name).resolve()
    run_results, notices = collect_run_results(runs_root)
    artifact = build_dashboard_artifact(run_results, notices, focus)
    # Keep the last verified companion intact when canonical publication fails.
    # A unique staging directory also avoids sharing temporary inputs with a
    # concurrent builder. The canonical builder publishes HTML only after QA.
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".dashboard-", dir=artifact_path.parent) as temporary:
        staged_artifact = Path(temporary) / artifact_path.name
        _write_json_atomic(staged_artifact, artifact)
        receipt = _deliver_portable_artifact(
            staged_artifact,
            html_path,
            (builder_path or find_portable_artifact_builder()).resolve(),
        )
        staged_artifact.replace(artifact_path)
        _write_json_atomic(html_path.with_suffix(".receipt.json"), receipt)
    return artifact_path, html_path, receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--builder", type=Path)
    parser.add_argument("--focus", help="Training run label to chart; default is the most recently modified training run")
    args = parser.parse_args()
    artifact_path, html_path, receipt = write_experiment_dashboard(
        args.runs_root,
        artifact_path=args.artifact,
        html_path=args.output,
        builder_path=args.builder,
        focus=args.focus,
    )
    verification = receipt.get("stages", {}).get("verification", "unknown")
    print(f"Dashboard: {html_path}")
    print(f"Artifact: {artifact_path}")
    print(f"Verification: {verification}")


if __name__ == "__main__":
    main()
