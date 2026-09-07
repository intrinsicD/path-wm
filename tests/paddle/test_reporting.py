"""Paddle evidence remains distinct and reconciles with authoritative raw rows."""

import json

import pytest

from viewer.dashboard import build_dashboard_artifact
from viewer.ledger import DashboardDataError, collect_run_results


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def rows(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(v) + '\n' for v in values))


def stage(root):
    path = root / 'paddle' / 'smoke' / 'perception'
    write(path / 'paddle_manifest.json', {'stage': 'perception', 'horizon': 1,
          'config': {'smoke': True}, 'dataset_fingerprint': 'dataset-identity'})
    rows(path / 'training.jsonl', [{'step': i, 'loss': 1 / i} for i in range(1, 101)])
    rows(path / 'validation.jsonl', [{'step': 50, 'loss': .1, 'h_mae': [1, 2, 3]},
                                   {'step': 100, 'loss': .2, 'h_mae': [2, 3, 4]}])
    write(path / 'paddle_result.json', {'stage': 'perception', 'status': 'completed', 'smoke': True,
          'global_update': 100, 'selected_update': 50, 'metrics': {'loss': .1, 'h_mae': [1, 2, 3]}})
    return path


def test_paddle_curves_do_not_inherit_lewm_loss_or_stale_validation_labels(tmp_path):
    root = tmp_path / 'runs'
    stage(root)
    legacy = root / 'existing'
    rows(legacy / 'metrics.jsonl', [{'kind': 'train', 'step': 1, 'loss': 3}, {'kind': 'complete', 'step': 1}])
    write(legacy / 'manifest.json', {'config': {}})
    write(legacy / 'status.json', {'kind': 'complete', 'step': 1})
    results, notices = collect_run_results(root)
    paddle = next(r for r in results if r.kind == 'paddle_training')
    assert paddle.step == 100
    assert paddle.context['selected_update'] == 50
    assert paddle.metrics['selected_validation.loss'] == .1
    artifact = build_dashboard_artifact(results, notices)
    datasets = artifact['snapshot']['datasets']
    assert {r['run'] for r in datasets['train_loss']} == {'existing'}
    curve = datasets['paddle_training_perception']
    training = [r for r in curve if r['series'] == 'training objective']
    assert len(training) == 50
    assert training[0]['step'] == 1 and training[-1]['step'] == 100
    assert all('SIGReg' not in c['subtitle'] for c in artifact['manifest']['charts'] if c['id'].startswith('paddle_'))
    assert any('SMOKE' in b.get('body', '') for b in artifact['manifest']['blocks'])
    assert any('paddle_result.json' in source for source in paddle.source_paths)


def evaluation(root, declared_successes=1):
    path = root / 'paddle' / 'evaluation'
    cases = [{'case_id': 'a', 'population': 'ordinary', 'controller': 'learned',
              'first_hit_before_miss': True, 'decision_latency_ms': [1., 2.],
              'total_hits': 1, 'episode_length': 10}]
    write(path / 'control_records.json', cases)
    write(path / 'prediction_records.json', [])
    write(path / 'metrics.json', {'schema_version': 'paddle-evaluation-v1', 'status': 'completed',
          'smoke': True, 'ordinary': {'starts': 1, 'summary': {'learned': {'successes': declared_successes,
          'count': 1, 'success_rate': declared_successes}}}, 'paired': {'pairs': 0, 'summary': {}},
          'prediction': {'summary': {}}, 'targets': {'eligible_as_full_evidence': False}, 'visuals': []})
    return path


def test_paddle_control_summary_must_match_raw_case_successes(tmp_path):
    evaluation(tmp_path / 'runs', declared_successes=0)
    with pytest.raises(DashboardDataError, match='success'):
        collect_run_results(tmp_path / 'runs')


def test_paddle_prediction_summary_must_match_raw_matched_window_errors(tmp_path):
    path = evaluation(tmp_path / 'runs')
    value = json.loads((path / 'metrics.json').read_text())
    value['prediction']['summary'] = {'prediction': {'1': {'all': {'count': 1,
          'h_mae': [1., 2., 3.], 'r_mae': [0.] * 5, 'latent_error': .4}}}}
    write(path / 'metrics.json', value)
    write(path / 'prediction_records.json', [{'method': 'prediction', 'horizon': 1, 'collision': False,
          'h_abs_error': [9., 2., 3.], 'r_abs_error': [0.] * 5, 'latent_error': .4}])
    with pytest.raises(DashboardDataError, match='h_mae'):
        collect_run_results(tmp_path / 'runs')


def test_paddle_partial_run_is_visible_without_claiming_completion(tmp_path):
    path = stage(tmp_path / 'runs')
    (path / 'paddle_result.json').unlink()
    write(path / 'failure.json', {'error': 'interrupted'})
    results, notices = collect_run_results(tmp_path / 'runs')
    assert results[0].kind == 'paddle_training'
    assert results[0].status == 'failed'
    assert any('failed' in notice for notice in notices)


def test_existing_development_schema_stays_indexed_and_reconciled(tmp_path):
    path = tmp_path / 'runs' / 'dev' / 'earlier' / '0'
    metrics = {'transition_error_one_step': .2, 'transition_error_identity': .1}
    write(path / 'metrics.json', {'metrics': metrics, 'seed': 0, 'status': 'development', 'step': 10})
    write(path / 'run_summary.json', {'metrics': metrics, 'final_training': {'step': 10, 'total': .7}})
    rows(path / 'training.jsonl', [{'step': 1, 'total': 1.}, {'step': 10, 'total': .7}])
    results, _ = collect_run_results(tmp_path / 'runs')
    assert len(results) == 1 and results[0].kind == 'legacy_development'
    assert results[0].metrics['transition_error_one_step'] == .2
    write(path / 'run_summary.json', {'metrics': {**metrics, 'transition_error_one_step': .3}})
    with pytest.raises(DashboardDataError, match='metric copies'):
        collect_run_results(tmp_path / 'runs')


def test_paddle_prediction_cache_is_not_a_lewm_prediction_result(tmp_path):
    path = tmp_path / 'runs' / 'paddle' / 'evaluation' / 'diagnostics' / 'prediction.json'
    write(path, {'evaluation_fingerprint': 'identity', 'descriptor_fingerprint': 'window',
                 'payload_fingerprint': 'payload', 'payload': {'summary': {}}})
    results, _ = collect_run_results(tmp_path / 'runs')
    assert not any(r.kind == 'prediction' for r in results)
