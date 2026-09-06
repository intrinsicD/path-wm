"""Reconcile current experiment ledgers without importing model or training code.

Adapted from the read-only ledger contract and JSON validation in
`eca742a:viewer/dashboard.py`. Raw files remain authoritative. Separate records
retain training status, validation step and control population so a completed
training loop cannot imply a successful scientific gate.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class DashboardDataError(RuntimeError):
    """Recorded evidence is inconsistent or unreadable."""


def _check_finite(value: Any, source: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise DashboardDataError(f"{source} contains a non-finite value")
    if isinstance(value, dict):
        for key, item in value.items():
            _check_finite(item, f"{source}:{key}")
    elif isinstance(value, list):
        for item in value:
            _check_finite(item, source)


def _object(value: Any, source: str) -> dict:
    if not isinstance(value, dict):
        raise DashboardDataError(f"{source} must contain a JSON object")
    _check_finite(value, source)
    return value


def read_json(path: Path) -> dict:
    try:
        return _object(json.loads(path.read_text()), str(path))
    except (OSError, json.JSONDecodeError) as error:
        raise DashboardDataError(f"cannot read {path}: {error}") from error


def read_jsonl(path: Path) -> tuple[dict, ...]:
    try:
        return tuple(_object(json.loads(line), f"{path}:{i}")
                     for i, line in enumerate(path.read_text().splitlines(), 1) if line.strip())
    except (OSError, json.JSONDecodeError) as error:
        raise DashboardDataError(f"cannot read {path}: {error}") from error


def numeric(value: dict) -> dict:
    return {key: item for key, item in value.items()
            if isinstance(item, (int, float)) and not isinstance(item, bool)}


@dataclass(frozen=True)
class RunResult:
    label: str
    kind: str
    status: str
    step: int | None
    metrics: dict
    context: dict
    source_paths: tuple[str, ...]
    modified_at: float
    training: tuple[dict, ...] = field(default_factory=tuple)
    validation: tuple[dict, ...] = field(default_factory=tuple)
    ranking: tuple[dict, ...] = field(default_factory=tuple)
    internals: dict = field(default_factory=dict)


def _context(manifest: dict) -> dict:
    config = manifest.get("config", {})
    context = {key: value for key, value in config.items()
               if isinstance(value, (str, int, float, bool)) or value is None}
    for key in ("protocol", "population", "normalization", "precision", "batchnorm",
                "seed", "sampling_seed", "checkpoint", "checkpoint_sha256", "released",
                "budget", "horizon", "samples", "iterations", "goal_offset", "code_commit",
                "environment_commit", "train_windows", "val_windows", "total_steps", "source_run_manifest"):
        if key in manifest:
            context[key] = manifest[key]
    dataset = manifest.get("dataset", {})
    if isinstance(dataset, dict):
        for key in ("name", "path", "revision", "sha256", "subset_sha256", "subset_protocol"):
            if key in dataset:
                context[f"dataset.{key}"] = dataset[key]
    for key in ("train_episodes", "val_episodes", "validation_window_indices"):
        if isinstance(manifest.get(key), list):
            context[f"{key}_count"] = len(manifest[key])
    # A fork changes its directory but can retain every inspection sample.
    # Partial identities cannot establish a match with a fully recorded population.
    sampling = ("dataset", "data_protocol", "validation_window_indices",
                "probe_window_indices_sha256", "rollout_window_indices_sha256",
                "validation_windows", "probe_windows", "rollout_windows", "rollout_horizon",
                "history", "image_size", "seeds", "precision")
    if all(manifest.get(key) is not None for key in sampling):
        identity = {key: manifest[key] for key in sampling}
        identity.update({key: manifest.get(key) for key in
                         ("panel_frames", "predictor_batch", "gradient_batch", "sigreg", "probe_targets")})
        context["inspection_population_sha256"] = hashlib.sha256(
            json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return context


def _counts(summary: dict, records: list | tuple, source: Path,
            success_key: str = "success", prefix: str = "") -> dict:
    if not records or any(not isinstance(row.get(success_key), bool) for row in records):
        raise DashboardDataError(f"{source}: missing boolean {success_key} case evidence")
    count, successes = len(records), sum(row[success_key] for row in records)
    expected_count = summary.get("cases", summary.get("episodes"))
    if expected_count is not None and expected_count != count:
        raise DashboardDataError(f"{source}: case count disagrees with recorded cases")
    if prefix + "successes" in summary and summary[prefix + "successes"] != successes:
        raise DashboardDataError(f"{source}: success count disagrees with recorded cases")
    rate = successes / count
    if "success_rate" in summary and not math.isclose(summary["success_rate"], rate, abs_tol=1e-12):
        raise DashboardDataError(f"{source}: success rate disagrees with recorded cases")
    if "initial_successes" in summary:
        initial = [row.get("initial_success") for row in records]
        if any(not isinstance(item, bool) for item in initial) or sum(initial) != summary["initial_successes"]:
            raise DashboardDataError(f"{source}: initial success count disagrees with recorded cases")
    result = {"successes": successes, "cases": count, "success_rate": rate}
    if all(isinstance(row.get("initial_success"), bool) for row in records):
        eligible = [row for row in records if not row["initial_success"]]
        reached = sum(row[success_key] for row in eligible)
        result.update(initial_successes=count-len(eligible), noninitial_cases=len(eligible),
                      noninitial_successes=reached)
        if eligible:
            result["noninitial_success_rate"] = reached / len(eligible)
        for key in ("noninitial_cases", "noninitial_successes", "noninitial_success_rate"):
            if key in summary and summary[key] != result.get(key):
                raise DashboardDataError(f"{source}: {key} disagrees with case evidence")
    return result


def _case_set(records, source, manifest=None):
    """Same identities permit navigation together, not protocol equivalence."""
    keys = ("episode", "source_episode", "start", "row", "source_row")
    identities = [{k: row[k] for k in keys if k in row} for row in records]
    manifest = manifest or {}
    dataset = manifest.get("dataset", {})
    # Episode/row numbers only identify goals within a specific source and offset.
    source_identity = {k: dataset[k] for k in ("name", "path", "revision") if k in dataset}
    value = dict(cases=identities if all(identities) else str(source),
                 source=source_identity, goal_offset=manifest.get("goal_offset"))
    digest = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:10]
    return f"{len(records)} cases · {digest}"


def collect_run_results(runs_root: Path) -> tuple[list[RunResult], list[str]]:
    runs_root = runs_root.resolve()
    results, notices = [], []

    def add(directory, kind, status, metrics, sources, context=None, step=None,
            training=(), validation=(), ranking=(), suffix="", internals=None):
        label = directory.relative_to(runs_root).as_posix() + suffix
        results.append(RunResult(label, kind, status, step, metrics, context or {},
                                 tuple(p.relative_to(runs_root.parent).as_posix() for p in sources),
                                 max(p.stat().st_mtime for p in sources), training, validation, ranking,
                                 internals or {}))

    for path in sorted(runs_root.rglob("metrics.jsonl")):
        directory = path.parent
        manifest_path, status_path = directory / "manifest.json", directory / "status.json"
        rows = read_jsonl(path)
        sources = [path]
        manifest = read_json(manifest_path) if manifest_path.exists() else {}
        if manifest_path.exists():
            sources.append(manifest_path)
        else:
            notices.append(f"{directory.relative_to(runs_root)}: training manifest missing")
        status = read_json(status_path) if status_path.exists() else {"kind": "incomplete"}
        if status_path.exists():
            sources.append(status_path)
        kind = status.get("kind", "unknown")
        # Gate annotations can be updated after evaluation; completion steps must agree.
        if kind != "incomplete":
            terminal = [r for r in rows if r.get("kind") == kind]
            if not terminal or any(terminal[-1].get(k) != status[k]
                                   for k in ("step", "last_logged_step", "saved_checkpoint_step") if k in status):
                raise DashboardDataError(f"{directory}: status disagrees with training completion record")
        context = _context(manifest)
        context.update(status)
        step = status.get("step", status.get("last_logged_step"))
        training = tuple(numeric(r) for r in rows if r.get("kind") == "train")
        validation = tuple(numeric(r) for r in rows if r.get("kind") == "validation")
        captured = tuple(numeric(r) for r in rows if r.get("kind") == "internals")
        add(directory, "training", kind, numeric(status), sources, context, step, training, validation,
            internals={"training_rows": captured} if captured else None)
        if kind != "complete":
            notices.append(f"{directory.relative_to(runs_root)}: {kind}; validation retains its recorded step")

    for path in sorted(runs_root.rglob("summary.json")):
        summary = read_json(path)
        if not any(key in summary for key in ("successes", "success_rate", "native_successes")):
            notices.append(f"Unindexed summary schema: {path.relative_to(runs_root)}")
            continue
        directory = path.parent
        sources = [path]
        manifest_path = directory / "manifest.json"
        manifest = read_json(manifest_path) if manifest_path.exists() else {}
        if manifest_path.exists():
            sources.append(manifest_path)
        context = _context(manifest)
        if "protocol" in summary:
            context["protocol"] = summary["protocol"]
        record_path = next((directory / name for name in ("cases.jsonl", "episodes.jsonl")
                            if (directory / name).exists()), None)
        if record_path:
            records = read_jsonl(record_path)
            sources.append(record_path)
        elif "episode_successes" in summary:
            records = [{"success": value} for value in summary["episode_successes"]]
        else:
            notices.append(f"Skipped {directory.relative_to(runs_root)}: no case evidence for summary")
            continue
        declared_cases = manifest.get("cases")
        if isinstance(declared_cases, list) and len(declared_cases) != len(records):
            raise DashboardDataError(f"{directory}: manifest case count disagrees with results")
        context["case_set"] = _case_set(records, path, manifest)
        variants = (("native_success", "native_"), ("local_success", "local_")) if "native_successes" in summary else (("success", ""),)
        for success_key, prefix in variants:
            metrics = {**numeric(summary), **_counts(summary, records, path, success_key, prefix)}
            if prefix:
                context_variant = {**context, "evaluator": prefix.rstrip("_")}
            else:
                context_variant = context
            state = "checkpoint_changed" if summary.get("checkpoint_unchanged") is False else "evaluated"
            add(directory, "control", state, metrics, sources, context_variant,
                manifest.get("step"), suffix=f" / {prefix[:-1]}" if prefix else "")

    for path in sorted(runs_root.rglob("action_baselines.json")):
        value = read_json(path)
        manifest_path = path.parent / "manifest.json"
        manifest = read_json(manifest_path) if manifest_path.exists() else {}
        sources = [path] + ([manifest_path] if manifest_path.exists() else [])
        for name, summary in value.get("summary", {}).items():
            records = [row for row in value.get("records", []) if row.get("kind") == name]
            context = {**_context(manifest), "policy": name,
                       "protocol": manifest.get("protocol", "Recorded action baseline; inspect case identities in source"),
                       "case_set": _case_set(records, path, manifest)}
            add(path.parent, "control", "evaluated", {**numeric(summary), **_counts(summary, records, path)},
                sources, context, suffix=f" / {name}")

    for path in sorted(runs_root.rglob("prediction.json")):
        manifest_path = path.parent / "manifest.json"
        manifest = read_json(manifest_path) if manifest_path.exists() else {}
        sources = [path] + ([manifest_path] if manifest_path.exists() else [])
        add(path.parent, "prediction", "evaluated", numeric(read_json(path)), sources,
            _context(manifest), manifest.get("step"), suffix=" / prediction")

    for path in sorted(runs_root.rglob("ranking.json")):
        value = read_json(path)
        manifest_path = path.parent / "manifest.json"
        record_path = path.parent / "ranking_records.jsonl"
        manifest, records = read_json(manifest_path), read_jsonl(record_path)
        summaries = value.get("case_summaries", [])
        expected = manifest.get("candidates_per_case")
        if value.get("records") != len(records) or not expected:
            raise DashboardDataError(f"{path}: ranking record count disagrees")
        keys = {(r["case_index"], r["model"]) for r in records}
        if keys != {(r["case_index"], r["model"]) for r in summaries} or len(keys) != len(summaries):
            raise DashboardDataError(f"{path}: ranking model/case summaries disagree")
        models = manifest.get("models", ["pilot", "released"])
        if (not isinstance(models, list) or not models or len(set(models)) != len(models)
                or not all(isinstance(model, str) and model for model in models)
                or keys != {(i, model) for i in range(len(manifest["cases"])) for model in models}):
            raise DashboardDataError(f"{path}: ranking case/model population incomplete")
        matched = {}
        for record in records:
            key = (record["case_index"], record["candidate_index"])
            outcome = {k: record.get(k) for k in ("candidate", "position_error", "angle_error",
                       "state_distance", "success_terminal", "success_any", "initial_success")}
            if key in matched and matched[key] != outcome:
                raise DashboardDataError(f"{path}: ranking simulator outcomes are not matched")
            matched[key] = outcome
        for summary in summaries:
            selected_rows = sorted((r for r in records if (r["case_index"], r["model"]) ==
                                    (summary["case_index"], summary["model"])), key=lambda r: r["candidate_index"])
            if len(selected_rows) != expected or [r["candidate_index"] for r in selected_rows] != list(range(expected)):
                raise DashboardDataError(f"{path}: ranking candidates incomplete or duplicated")
            best = min(selected_rows, key=lambda r: r["predicted_cost"])
            distance = best["position_error"]
            minimum = min(r["position_error"] for r in selected_rows)
            checks = {"candidates": expected, "selected_index": best["candidate_index"],
                      "selected_distance": distance, "best_distance": minimum, "regret": distance-minimum}
            if any(summary.get(k) != v for k, v in checks.items()):
                raise DashboardDataError(f"{path}: ranking selection summary disagrees with candidates")
            context = {**_context(manifest), "population": summary["population"], "model": summary["model"],
                       "case_index": summary["case_index"], "selected_candidate": best["candidate"],
                       "checkpoint_sha256": manifest.get("checkpoint_sha256s", {}).get(summary["model"])}
            add(path.parent, "ranking", "evaluated" if value.get("checkpoint_unchanged") else "checkpoint_changed",
                numeric(summary), [path, manifest_path, record_path], context, ranking=tuple(selected_rows),
                suffix=f" / {summary['population']} case {summary['case_index']} {summary['model']}")

    for path in sorted(runs_root.rglob("real_batch_parity*.json")):
        value = read_json(path)
        for index, record in enumerate(value.get("results", [])):
            metrics = numeric(record)
            for side in ("native", "local"):
                metrics.update({f"{side}.{k}": v for k, v in numeric(record.get(side, {})).items()})
            context = {k: value[k] for k in ("checkpoint_sha256", "batch_size", "limitation", "source") if k in value}
            context.update(precision=record["precision"], variant=record.get("variant", "local_recomputed"),
                           protocol="Discarded real-data clones; numerical comparison, not training completion")
            add(path.parent, "diagnostic", "measured" if value.get("checkpoint_unchanged") else "checkpoint_changed",
                metrics, [path], context, suffix=f" / {path.stem} {index} {record['precision']}")
    for path in sorted(runs_root.rglob("reproduction_memory.json")):
        value = read_json(path)
        add(path.parent, "diagnostic", value.get("status", "unknown"), numeric(value), [path],
            {k: value[k] for k in ("checkpoint_sha256", "limitation", "precision", "all_gradients_finite") if k in value},
            suffix=" / full-batch memory probe")

    for path in sorted(runs_root.rglob("gradient_audit.json")):
        value = read_json(path)
        manifest_path = path.parent / 'manifest.json'
        manifest = read_json(manifest_path)
        panel_path = path.parent / 'gradient_geometry.png'
        if not panel_path.is_file():
            raise DashboardDataError(f'{path}: gradient panel is missing')
        metrics = {}
        def flatten(item, prefix=''):
            if isinstance(item, dict):
                for key, child in item.items(): flatten(child, f'{prefix}.{key}' if prefix else key)
            elif isinstance(item, list):
                for index, child in enumerate(item): flatten(child, f'{prefix}.{index}')
            elif isinstance(item, (int, float)) and not isinstance(item, bool): metrics[prefix] = item
        flatten(value)
        context = {**_context(manifest), 'protocol': 'Training mode, restored buffers, no updates; batch-coupled objective. Four descriptive replicates, no critical batch estimate.'}
        add(path.parent, 'gradient_audit', 'measured' if value.get('checkpoint_unchanged') and value.get('model_state_unchanged') else 'checkpoint_changed',
            metrics, [path, manifest_path, panel_path], context, value.get('step'),
            internals={'panels': [dict(id='training_gradient_geometry', title='Sample-efficiency investigation: training gradients',
                caption='Fixed training windows, bf16, original weights and buffers preserved. Exact values are in the record table.', path=str(panel_path))]})

    for path in sorted(runs_root.rglob("internals.json")):
        # Saved-checkpoint internals: scalars over step, series and rendered panels.
        value = read_json(path)
        manifest_path = path.parent / "manifest.json"
        manifest = read_json(manifest_path)
        sources, panels = [path, manifest_path], []
        for panel in value.get("panels", []):
            panel_path = path.parent / panel.get("path", "")
            if not panel_path.is_file():
                raise DashboardDataError(f"{path}: panel {panel.get('path')} is missing")
            sources.append(panel_path)
            panels.append({**panel, "path": str(panel_path)})
        context = {**_context(manifest), "family": value.get("family", "unknown"),
                   "checkpoint_sha256": value.get("checkpoint_sha256"),
                   "protocol": "Read-only checkpoint inspection in eval mode; see manifest for windows and seeds"}
        status = "inspected" if value.get("checkpoint_unchanged") else "checkpoint_changed"
        add(path.parent, "internals", status, numeric(value.get("scalars", {})), sources, context, value.get("step"),
            internals={"series": value.get("series", {}), "panels": panels, "family": context["family"],
                       "training_run": value.get("training_run")})

    for name in ("cases.jsonl", "episodes.jsonl"):
        for path in sorted(runs_root.rglob(name)):
            if not (path.parent / "summary.json").exists():
                notices.append(f"{path.parent.relative_to(runs_root)}: incomplete evaluation; summary missing")
    omitted = list(runs_root.rglob("diagnostics.json"))
    if omitted:
        notices.append(f"{len(omitted)} diagnostics.json files contain auxiliary mode/clone checks; "
                       "consult the experiment reports and raw files for these separate diagnostics.")
    return sorted(results, key=lambda r: (r.modified_at, r.label)), notices
