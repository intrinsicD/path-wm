"""Prevent false completion and misleading evidence in the experiment harness."""
import json
import math
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
    artifact = html.with_suffix('.artifact.json')
    # The actual default companion name is experiment_dashboard.artifact.json.
    artifact.write_text('{"previous": true}')
    receipt = html.with_suffix('.receipt.json')
    receipt.write_text('{"ok": true, "generation": "previous"}')
    def fail_builder(artifact, target, builder):
        raise dashboard.DashboardBuildError("browser QA failed before publication")
    monkeypatch.setattr(dashboard, "_deliver_portable_artifact", fail_builder)
    with pytest.raises(dashboard.DashboardBuildError):
        dashboard.write_experiment_dashboard(html.parent, builder_path=tmp_path / "builder.mjs")
    assert html.read_text() == "previous verified dashboard"
    assert artifact.read_text() == '{"previous": true}'
    assert receipt.read_text() == '{"ok": true, "generation": "previous"}'


def test_ranking_ledger_requires_complete_matched_candidates(tmp_path):
    run=tmp_path/'runs'/'ranking'
    write_json(run/'manifest.json', {'candidates_per_case':2,'cases':[{'population':'training'}]})
    write_json(run/'ranking.json', {'records':4,'checkpoint_unchanged':True,
        'case_summaries':[{'case_index':0,'model':m,'population':'training','candidates':2,
                           'selected_index':0,'selected_distance':4.,'best_distance':4.,'regret':0.}
                          for m in ('pilot','released')]})
    rows=[{'case_index':0,'model':m,'candidate_index':i,'candidate':kind,
           'population':'training','predicted_cost':float(i+1),'position_error':float(4+i)}
          for m in ('pilot','released') for i,kind in enumerate(('replay','stationary'))]
    write_rows(run/'ranking_records.jsonl',rows)
    results,_=collect_run_results(tmp_path/'runs')
    assert len(results)==2 and all(r.kind=='ranking' for r in results)
    assert all(len(r.ranking)==2 for r in results)
    from viewer.dashboard import build_dashboard_artifact
    artifact=build_dashboard_artifact(results,[])
    assert len(artifact['snapshot']['datasets']['ranking'])==4
    assert any(c['id']=='ranking' for c in artifact['manifest']['charts'])
    write_rows(run/'ranking_records.jsonl',rows[:-1])
    with pytest.raises(DashboardDataError,match='ranking'):
        collect_run_results(tmp_path/'runs')


def test_training_scalars_get_own_panels_and_validation_ratios(tmp_path):
    from viewer.dashboard import build_dashboard_artifact
    run = tmp_path / "runs" / "learn"
    write_json(run / "manifest.json", {"config": {"seed": 1, "sigreg_weight": 0.09}})
    write_rows(run / "metrics.jsonl", [
        {"kind": "validation", "step": 0, "pred_mse": 0.5, "identity_mse": 1.0, "shuffled_action_mse": 0.25,
         "zero_action_mse": 0.5, "rollout_mse": 2.0, "action_effect": 0.1, "embedding_std": 0.3},
        {"kind": "train", "step": 1, "loss": 1.0, "pred_loss": 0.2, "sigreg_loss": 9.0, "grad_norm": 3.0, "lr": 1e-5},
        {"kind": "train", "step": 2, "loss": 0.9, "pred_loss": 0.19, "sigreg_loss": 8.0, "grad_norm": 2.0, "lr": 2e-5},
        {"kind": "complete", "step": 2, "total_steps": 2},
    ])
    write_json(run / "status.json", {"kind": "complete", "step": 2, "total_steps": 2})
    results, notices = collect_run_results(tmp_path / "runs")
    artifact = build_dashboard_artifact(results, notices)
    datasets = artifact["snapshot"]["datasets"]
    for name in ("train_pred_loss", "train_sigreg_loss", "train_loss", "train_grad_norm", "train_lr"):
        assert [row["step"] for row in datasets[name]] == [1, 2], name
        assert any(chart["dataset"] == name for chart in artifact["manifest"]["charts"]), name
    ratios = {row["metric"]: row["value"] for row in datasets["validation_ratio"]}
    assert ratios == pytest.approx({"prediction / copy": 0.5, "prediction / shuffled actions": 2.0,
                                    "prediction / zero actions": 1.0, "rollout / copy": 2.0,
                                    "action effect / prediction": 0.2})
    logs = {row["metric"]: row["log10_ratio"] for row in datasets["validation_ratio"]}
    assert logs["prediction / copy"] == pytest.approx(math.log10(0.5))
    ratio_chart = next(chart for chart in artifact["manifest"]["charts"] if chart["dataset"] == "validation_ratio")
    assert ratio_chart["encodings"]["y"]["field"] == "log10_ratio"
    assert "different horizons" in ratio_chart["subtitle"]
    assert any(line["value"] == 0 for line in ratio_chart["referenceLines"])
    assert [row["value"] for row in datasets["train_lr"]] == pytest.approx([10.0, 20.0])  # micro-units
    assert {row["metric"] for row in datasets["validation"]} >= {"pred_mse", "rollout_mse"}
    assert artifact["manifest"]["filters"] == []  # the reader ignores per-section selectors
    assert "learn" in artifact["manifest"]["blocks"][0]["body"]
    other = tmp_path / "runs" / "other"
    write_json(other / "manifest.json", {"config": {"seed": 2}})
    write_rows(other / "metrics.jsonl", [{"kind": "train", "step": 7, "loss": 5.0, "pred_loss": 4.0, "sigreg_loss": 1.0},
                                         {"kind": "complete", "step": 7, "total_steps": 7}])
    write_json(other / "status.json", {"kind": "complete", "step": 7, "total_steps": 7})
    results, notices = collect_run_results(tmp_path / "runs")
    focused = build_dashboard_artifact(results, notices, focus="learn")
    assert {row["run"] for row in focused["snapshot"]["datasets"]["train_pred_loss"]} == {"learn"}
    assert all("learn" in chart["title"] for chart in focused["manifest"]["charts"] if chart["dataset"].startswith("train_"))
    with pytest.raises(DashboardDataError, match="focus"):
        build_dashboard_artifact(results, notices, focus="missing")


PNG = bytes.fromhex("89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4890000000d4944415478da"
                    "63f8cfc0f00f0003010100c9fe92ef0000000049454e44ae426082")


def write_internals(root, name, step, panel="panels/attention.png", unchanged=True, family="pilot"):
    run = root / name
    write_json(run / "manifest.json", {"checkpoint": "x.pt", "checkpoint_sha256": "abc", "step": step,
                                       "windows": 4, "precision": "float32", "mode": "eval",
                                       "source_run_manifest": "runs/learn/manifest.json"})
    write_json(run / "internals.json", {
        "step": step, "family": family, "training_run": "learn" if family == "pilot" else None,
        "checkpoint_sha256": "abc", "checkpoint_unchanged": unchanged,
        "scalars": {"effective_rank": 10.5 + (step or 0), "rankme": 9.0, "gate_msa_mean": 0.0},
        "series": {"spectrum": [3.0, 1.0, 0.5],
                   "horizon": {"horizon": [1, 2], "prediction_mse": [0.1, 0.2], "copy_mse": [0.2, 0.4]},
                   "probe_r2": {"agent position": 0.9, "block position": 0.1}},
        "panels": [{"id": "encoder_attention", "title": "Encoder attention", "caption": "CLS to patches",
                    "path": panel}]})
    (run / "panels").mkdir(exist_ok=True)
    (run / "panels" / "attention.png").write_bytes(PNG)
    return run


def test_internals_ledger_indexes_scalars_series_and_embedded_panels(tmp_path):
    from viewer.dashboard import build_dashboard_artifact
    root = tmp_path / "runs" / "internals"
    write_internals(root, "pilot_0", 0)
    write_internals(root, "pilot_1000", 1000)
    write_internals(root, "released", None, family="released")
    learn = tmp_path / "runs" / "learn"
    write_json(learn / "manifest.json", {"config": {"seed": 1}})
    write_rows(learn / "metrics.jsonl", [{"kind": "train", "step": 1, "loss": 1.0}, {"kind": "complete", "step": 1}])
    write_json(learn / "status.json", {"kind": "complete", "step": 1})
    results, notices = collect_run_results(tmp_path / "runs")
    assert sorted(r.kind for r in results) == ["internals"] * 3 + ["training"]
    pilot = next(r for r in results if r.step == 1000 and r.kind == "internals")
    assert pilot.metrics["effective_rank"] == 1010.5 and pilot.status == "inspected"
    assert pilot.internals["series"]["spectrum"] == [3.0, 1.0, 0.5]
    artifact = build_dashboard_artifact(results, notices)
    datasets = artifact["snapshot"]["datasets"]
    assert {(row["step"], row["metric"]) for row in datasets["internals_rank"]} >= {(0, "effective_rank"), (1000, "effective_rank")}
    assert all(row["step"] is not None for row in datasets["internals_rank"])
    assert len(datasets["internals_spectrum"]) == 9 and {row["run"] for row in datasets["internals_spectrum"]} == {r.label for r in results if r.kind == "internals"}
    assert pilot.internals["series"]["horizon"] == {"horizon": [1, 2], "prediction_mse": [.1, .2], "copy_mse": [.2, .4]}
    ratios = {(row["run"], row["horizon"]): row["value"] for row in datasets["internals_horizon_autoregressive"]}
    assert ratios[(pilot.label, 2)] == pytest.approx(0.5) and len(ratios) == 6
    assert {row["run"] for row in datasets["internals_rank"]} == {r.label for r in results if r.kind == "internals" and r.step is not None}
    assert {(row["run"], row["target"]) for row in datasets["internals_probe"]} >= {(pilot.label, "agent position")}
    summary = {(row["run"], row["metric"]): row["value"] for row in datasets["internals_summary"]}
    assert summary[(pilot.label, "effective_rank")] == "1010.5"
    panels = [block for block in artifact["manifest"]["blocks"] if block["type"] == "html"]
    assert len(panels) == 1 and panels[0]["body"].count("data:image/png;base64,") == 3
    assert "CLS to patches" in panels[0]["body"] and "released" in panels[0]["body"]


def test_internals_ledger_rejects_missing_panels_and_marks_changed_checkpoints(tmp_path):
    root = tmp_path / "runs" / "internals"
    write_internals(root, "changed", 5, unchanged=False)
    results, _ = collect_run_results(tmp_path / "runs")
    assert results[0].status == "checkpoint_changed"
    write_internals(root, "broken", 6, panel="panels/missing.png")
    with pytest.raises(DashboardDataError, match="panel"):
        collect_run_results(tmp_path / "runs")


def test_training_time_internals_rows_get_their_own_panels(tmp_path):
    from viewer.dashboard import build_dashboard_artifact
    run = tmp_path / "runs" / "captured"
    write_json(run / "manifest.json", {"config": {"seed": 1, "introspect": True}})
    write_rows(run / "metrics.jsonl", [
        {"kind": "validation", "step": 0, "pred_mse": 0.5, "identity_mse": 1.0},
        {"kind": "internals", "step": 0, "elapsed_seconds": 1.0, "effective_rank": 3.0, "gate_msa_mean": 0.0, "examples": 8},
        {"kind": "train", "step": 5, "loss": 1.0},
        {"kind": "internals", "step": 5, "elapsed_seconds": 2.0, "effective_rank": 5.0, "gate_msa_mean": 0.1, "examples": 8},
        {"kind": "complete", "step": 5, "total_steps": 5},
    ])
    write_json(run / "status.json", {"kind": "complete", "step": 5, "total_steps": 5})
    results, notices = collect_run_results(tmp_path / "runs")
    assert len(results[0].internals["training_rows"]) == 2
    artifact = build_dashboard_artifact(results, notices)
    datasets = artifact["snapshot"]["datasets"]
    assert [(row["step"], row["value"]) for row in datasets["training_internals_rank"]] == [(0, 3.0), (5, 5.0)]
    assert [(row["step"], row["value"]) for row in datasets["training_internals_action_use"]] == [(0, 0.0), (5, 0.1)]
    assert any(chart["dataset"] == "training_internals_rank" and "captured during training" in chart["title"]
               for chart in artifact["manifest"]["charts"])
    assert "examples" not in {row["metric"] for key, rows in datasets.items() if key.startswith("training_internals_") for row in rows}


def test_embedded_panels_are_bounded_to_focus_endpoints_and_matching_reference(tmp_path):
    from viewer.dashboard import build_dashboard_artifact
    root = tmp_path / "runs" / "internals"
    for step in [0, 100, 200, 300]:
        write_internals(root, f"pilot_{step}", step)
    write_internals(root, "released", None, family="released")
    unrelated = write_internals(root, "other_reference", None, family="released")
    meta = json.loads((unrelated / "manifest.json").read_text())
    meta["source_run_manifest"] = "runs/other/manifest.json"
    write_json(unrelated / "manifest.json", meta)
    learn = tmp_path / "runs" / "learn"
    write_json(learn / "manifest.json", {"config": {"seed": 1}})
    write_rows(learn / "metrics.jsonl", [{"kind": "train", "step": 1, "loss": 1.0}, {"kind": "complete", "step": 1}])
    write_json(learn / "status.json", {"kind": "complete", "step": 1})
    results, notices = collect_run_results(tmp_path / "runs")
    artifact = build_dashboard_artifact(results, notices, focus="learn")
    bodies = "".join(b["body"] for b in artifact["manifest"]["blocks"] if b["type"] == "html")
    assert bodies.count("data:image/png;base64,") == 3
    assert "pilot_0" in bodies and "pilot_300" in bodies and "internals/released" in bodies
    assert "pilot_100" not in bodies and "pilot_200" not in bodies and "other_reference" not in bodies
    assert len({r["run"] for r in artifact["snapshot"]["datasets"]["internals_summary"]}) == 6
    notes = next(b["body"] for b in artifact["manifest"]["blocks"] if b["id"] == "notices")
    assert "earliest and latest" in notes and "All" in notes


def test_control_navigation_distinguishes_sources_and_keeps_initial_goal_denominators(tmp_path):
    from viewer.dashboard import build_dashboard_artifact
    cases = [dict(episode=0,start=5,success=True,initial_success=True),
             dict(episode=1,start=5,success=False,initial_success=False)]
    for name in ['pusht','tworoom']:
        run=tmp_path/'runs'/name
        write_json(run/'manifest.json',dict(dataset=dict(name=name,path=f'data/{name}.h5',revision='abc'),
            goal_offset=25,budget=50,cases=[dict(episode=i,start=5) for i in range(2)]))
        write_json(run/'summary.json',dict(successes=1,cases=2,initial_successes=1))
        write_rows(run/'cases.jsonl',cases)
    run=tmp_path/'runs'/'pusht'
    write_json(run/'action_baselines.json',dict(
        records=[dict(**case,kind='stationary') for case in cases],
        summary=dict(stationary=dict(successes=1,cases=2,initial_successes=1))))
    results,notices=collect_run_results(tmp_path/'runs')
    by_name={r.label:r for r in results}
    assert by_name['pusht'].context['case_set']!=by_name['tworoom'].context['case_set']
    baseline=by_name['pusht / stationary']
    assert baseline.context['case_set']==by_name['pusht'].context['case_set']
    assert baseline.context['dataset.name']=='pusht' and baseline.context['goal_offset']==25
    assert baseline.metrics['initial_successes']==1
    assert baseline.metrics['noninitial_cases']==1
    assert baseline.metrics['noninitial_successes']==0 and baseline.metrics['noninitial_success_rate']==0.
    artifact=build_dashboard_artifact(results,notices)
    rows={r['run']:r for r in artifact['snapshot']['datasets']['control_detail']}
    assert rows['pusht']['initial_successes']=='1' and rows['pusht']['noninitial_cases']=='1'
    assert rows['pusht']['success_rate']=='0.5' and rows['pusht']['noninitial_success_rate']=='0'
    assert any(c['id']=='control_noninitial' for c in artifact['manifest']['charts'])
    # An equal source window with a different target is a different case identity.
    manifest=__import__('json').loads((run/'manifest.json').read_text())
    manifest['goal_offset']=100;write_json(run/'manifest.json',manifest)
    changed,_=collect_run_results(tmp_path/'runs')
    assert next(r for r in changed if r.label=='pusht').context['case_set']!=by_name['pusht'].context['case_set']


def test_large_ledger_retains_every_exact_row_and_complete_chart_series(tmp_path):
    from dataclasses import replace
    from viewer.dashboard import MAX_DATASET_ROWS, build_dashboard_artifact
    root = tmp_path / 'runs'
    for i in range(12):
        path = write_internals(root, f'checkpoint_{i:02}', i)
        data = json.loads((path / 'internals.json').read_text())
        data['series']['spectrum'] = [1.0 / (j + 1) for j in range(192)]
        write_json(path / 'internals.json', data)
    results, notices = collect_run_results(root)
    results = [replace(r, context={f'field_{j}': f'{r.label}:{j}' for j in range(201)}) for r in results]
    artifact = build_dashboard_artifact(results, notices)
    datasets = artifact['snapshot']['datasets']
    assert all(len(rows) <= MAX_DATASET_ROWS for rows in datasets.values())
    spectra = [c for c in artifact['manifest']['charts'] if c['id'].startswith('internals_spectrum')]
    assert len(spectra) >= 2
    observed = {}
    for chart in spectra:
        assert 'part' in chart['title'].lower()
        rows = datasets[chart['dataset']]
        for run in {row['run'] for row in rows}:
            assert run not in observed  # A line must not be severed across charts.
            observed[run] = [row['component'] for row in rows if row['run'] == run]
    assert set(observed) == {r.label for r in results}
    assert all(components == list(range(1, 193)) for components in observed.values())
    context_tables = [t for t in artifact['manifest']['tables'] if t['id'].startswith('context')]
    values = [row['value'] for t in context_tables for row in datasets[t['dataset']]]
    assert sorted(values) == sorted(str(v) for r in results for v in r.context.values())


def test_forked_inspection_panels_match_recorded_population_not_run_directory(tmp_path):
    from viewer.dashboard import build_dashboard_artifact
    root=tmp_path/'runs'/'internals'
    population=dict(dataset={'name':'pusht','path':'data/pusht.h5','revision':'fixed','sha256':'data'},
        data_protocol={'split_protocol':'random_windows','val_indices_sha256':'split'},
        validation_window_indices=[1,5,9],probe_window_indices_sha256='probe',
        rollout_window_indices_sha256='rollout',history=3,image_size=224,
        validation_windows=3,probe_windows=4,rollout_windows=2,rollout_horizon=8,
        seeds={'torch':0,'window_sampling':0},precision='float32')
    for name,family in [('fork_final','pilot'),('matched_reference','released'),('wrong_windows','released'),('missing_identity','released')]:
        path=write_internals(root,name,2 if family=='pilot' else None,family=family)
        meta=json.loads((path/'manifest.json').read_text())
        meta.update(population)
        meta['source_run_manifest']='runs/parent/manifest.json' if name=='matched_reference' else 'runs/fork/manifest.json'
        if name=='wrong_windows':meta['validation_window_indices']=[1,5,10]
        if name=='missing_identity':del meta['probe_window_indices_sha256']
        write_json(path/'manifest.json',meta)
    learn=tmp_path/'runs'/'learn'
    write_json(learn/'manifest.json',{'config':{'seed':1}})
    write_rows(learn/'metrics.jsonl',[{'kind':'train','step':2,'loss':1.},{'kind':'complete','step':2}])
    write_json(learn/'status.json',{'kind':'complete','step':2})
    results,notices=collect_run_results(tmp_path/'runs')
    artifact=build_dashboard_artifact(results,notices,focus='learn')
    bodies=''.join(b['body'] for b in artifact['manifest']['blocks'] if b['type']=='html')
    assert 'matched_reference' in bodies
    assert 'wrong_windows' not in bodies and 'missing_identity' not in bodies
    assert bodies.count('data:image/png;base64,')==2
    assert len({row['run'] for row in artifact['snapshot']['datasets']['internals_summary']})==4


def test_training_gradient_audit_is_distinct_from_validation_and_indexes_evidence(tmp_path):
    from viewer.dashboard import build_dashboard_artifact
    run = tmp_path / 'runs' / 'gradient_probe'
    write_json(run / 'manifest.json', {'precision': 'bf16', 'population': 'Fixed TRAIN windows'})
    write_json(run / 'gradient_audit.json', {'step': 12, 'checkpoint_unchanged': True,
        'model_state_unchanged': True, 'sampling': {'processed_windows': 1536},
        'geometry': {'different_data': {'128': {'pairwise_cosine_mean': .4}}},
        'probes': [{'regime':'different_data', 'batch_size':32, 'replicate':i,
                    'modules':{'encoder':{'prediction_norm':v}}} for i,v in enumerate([2.,4.])]})
    (run / 'gradient_geometry.png').write_bytes(b'fixture image')
    results, notices = collect_run_results(tmp_path / 'runs')
    assert len(results) == 1
    assert results[0].kind == 'gradient_audit'
    assert results[0].metrics['geometry.different_data.128.pairwise_cosine_mean'] == .4
    assert results[0].metrics['probe_means.different_data.b32.encoder.prediction_norm'] == 3
    assert 'Training mode' in results[0].context['protocol']
    artifact = build_dashboard_artifact(results, notices)
    assert any(b['id'] == 'panel_training_gradient_geometry' for b in artifact['manifest']['blocks'])
    (run / 'gradient_geometry.png').unlink()
    with pytest.raises(DashboardDataError, match='panel'):
        collect_run_results(tmp_path / 'runs')


def test_portable_snapshot_contains_only_datasets_referenced_by_views(tmp_path):
    from viewer.dashboard import build_dashboard_artifact
    run = tmp_path / 'runs' / 'tiny'
    write_json(run / 'manifest.json', {'config': {'seed': 4}})
    write_rows(run / 'metrics.jsonl', [{'kind': 'train', 'step': 1, 'loss': .2}])
    results, notices = collect_run_results(tmp_path / 'runs')
    artifact = build_dashboard_artifact(results, notices)
    views = artifact['manifest']['charts'] + artifact['manifest']['tables'] + artifact['manifest']['cards']
    referenced = {view['dataset'] for view in views}
    assert set(artifact['snapshot']['datasets']) == referenced
    assert any(row.get('value') == .2 for name in referenced for row in artifact['snapshot']['datasets'][name])


def test_projection_comparison_retains_paired_populations_and_missing_arms(tmp_path):
    from scripts.paired_summary import summarize_pairs
    from viewer.dashboard import build_dashboard_artifact
    row = dict(dataset='toy', seed=3072, projections=1024, step=750, variant='saved',
               case_sha256='a'*64, calibration_rows_sha256=None, successes=2, cases=5,
               initial_successes=1, newly_solved=1, initial_model_sha256='b'*64)
    rows = [row, {**row, 'projections':4096, 'successes':3, 'newly_solved':2},
            {**row, 'seed':3073}]
    groups = summarize_pairs(rows)
    path = tmp_path/'runs'/'paired'/'projection_comparison.json'
    present={(r['seed'],r['projections'],r['variant']) for r in rows}
    missing=[dict(dataset='toy',step=750,seed=seed,projections=count,variant=variant)
             for seed in (3072,3073,3074) for count in (1024,4096) for variant in ('saved','calibrated')
             if (seed,count,variant) not in present]
    write_json(path, dict(version=1, rows=rows, groups=groups, expected_outcomes=12,
                          missing_outcomes=missing, sources=[]))
    results, notices = collect_run_results(tmp_path/'runs')
    comparison = next(r for r in results if r.kind=='projection_comparison')
    assert comparison.status=='incomplete'
    artifact = build_dashboard_artifact(results,notices)
    datasets = artifact['snapshot']['datasets']
    assert len(datasets['projection_control_detail'])==12
    assert sum(r['status']=='measured' for r in datasets['projection_control_detail'])==3
    curves=[c for c in artifact['manifest']['charts'] if c['id'].startswith('projection_control_')]
    assert curves and all(c['type']=='bar' for c in curves)  # Native reader rejects one-x line charts.
    assert any(t['dataset']=='projection_control_detail' for t in artifact['manifest']['tables'])
    uncertainty=datasets['projection_uncertainty'][0]
    assert uncertainty['complete_pairs']==1 and uncertainty['mean_pp']==pytest.approx(20.)
    assert uncertainty['sample_sd_pp'] is None and uncertainty['standard_error_pp'] is None
    value = json.loads(path.read_text()); value['groups'][0]['stats']['delta_successes']['mean']=100
    write_json(path,value)
    with pytest.raises(DashboardDataError,match='paired'):
        collect_run_results(tmp_path/'runs')
