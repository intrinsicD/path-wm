"""Diagnostic charts must reconcile recorded means and weighted image regions."""
import base64
import json

import pytest

from viewer.dashboard import build_dashboard_artifact
from viewer.ledger import DashboardDataError, collect_run_results


PNG = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII=')


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def diagnosis(root):
    directory = root / 'pusht_world_model' / 'collaboration' / 'perception_diagnosis'
    records, alignment, summary, coverage = [], [], {}, {}
    for split, first_group in [('train', 10), ('validation', 20)]:
        coverage[split] = {'episodes': 2, 'independent_groups': 2, 'frames': 100,
            'sampled_frames': 2, 'sampled_groups': 2,
            'sampled_per_group_counts': {str(first_group): 1, str(first_group + 1): 1}}
        for i in range(2):
            alignment.append({'split': split, 'source_episode': first_group + i,
                'frame_index': i + 3, 'source_row': first_group * 10 + i,
                'exact_source_pixels': True, 'exact_source_pose': True,
                'target_max_abs_difference': 1e-8})
        for update in (200, 1000):
            rows = []
            for i in range(2):
                scalars = [32., 96.][i]
                region = {'scalars': scalars, 'squared_error': scalars * (i * 2 + 1),
                          'white_squared_error': scalars * (i * 2 + 2)}
                row = {**alignment[-2 + i], 'checkpoint_update': update, 'group_id': first_group + i,
                    'position_abs_error': [i + j + 1. for j in range(4)],
                    'angle_error_deg': 10. + 20. * i, 'sin_cos_norm': .25 + .5 * i,
                    'pose_mse': .1 + .2 * i, 'image_mse': .01 + .02 * i,
                    'pusher_distance': 2. + i, 'block_distance': 3. + i,
                    'regions': {'block': region}}
                rows.append(row)
            records.extend(rows)
            summary.setdefault(str(update), {})[split] = {
                'frames': 2, 'groups': 2, 'position_mae': [1.5 + j for j in range(4)],
                'angle_error_deg': {'mean': 20.}, 'sin_cos_norm': {'mean': .5},
                'pusher_distance': {'mean': 2.5}, 'block_distance': {'mean': 3.5},
                'pose_mse': .2, 'image_mse': .02,
                'regions': {'block': {'scalars': 128., 'squared_error': 320.,
                    'mse': 2.5, 'white_image_mse': 3.5, 'pixel_fraction': 128. / (2 * 64 * 64 * 3)}}}
    report = {'schema_version': 'pusht-perception-diagnosis-v1', 'status': 'completed',
        'seed': 91731, 'device': 'cpu', 'threads': 2, 'sampling': 'Same fixed grouped frames at both checkpoints',
        'dataset_fingerprint': 'source-dataset', 'checkpoint_updates': [200, 1000],
        'protected_hashes_before': {'checkpoint.pt': 'a' * 64},
        'protected_hashes_after': {'checkpoint.pt': 'a' * 64}, 'script_sha256': 'b' * 64,
        'region_definition': 'Fractional RGB scalar weights at 64x64', 'coverage': coverage,
        'alignment': alignment, 'summary': summary, 'records': records, 'elapsed_seconds': 1.}
    write(directory / 'raw.json', report)
    for name in ('diagnostic_metrics.png', 'reconstruction_and_labels.png'):
        (directory / name).write_bytes(PNG)
    return directory, report


def test_diagnosis_is_source_backed_compact_and_keeps_both_inspected_panels(tmp_path):
    root = tmp_path / 'runs'
    directory, _ = diagnosis(root)
    write(root / 'unrelated' / 'raw.json', {'schema_version': 'different-diagnostic'})
    results, notices = collect_run_results(root)
    diagnoses = [r for r in results if r.kind == 'pusht_perception_diagnosis']
    assert len(diagnoses) == 1, 'The canonical reader must recognize the diagnosis schema'
    run = diagnoses[0]
    assert run.context['diagnostic_only'] is True
    assert run.metrics['checked_records'] == 8 and run.metrics['checked_alignment_frames'] == 4
    assert run.metrics['coverage.train.sampled_groups'] == 2
    assert run.metrics['summary.200.validation.angle_error_deg.mean'] == 20.
    assert run.metrics['summary.1000.train.regions.block.mse'] == 2.5
    assert {path.rsplit('/', 1)[-1] for path in run.source_paths} == {
        'raw.json', 'diagnostic_metrics.png', 'reconstruction_and_labels.png'}
    artifact = build_dashboard_artifact(results, notices)
    diagnosis_sets = {k: v for k, v in artifact['snapshot']['datasets'].items()
                      if k.startswith('pusht_diagnosis')}
    assert len(diagnosis_sets) <= 1
    blocks = [b for b in artifact['manifest']['blocks'] if b['id'].startswith('pusht_diagnosis')]
    assert sum(b.get('body', '').count('data:image/png;base64,') for b in blocks) == 2
    assert 'diagnostic' in ' '.join(b.get('body', '') for b in blocks).lower()


@pytest.mark.parametrize('tamper', ['angle_mean', 'position_mean', 'region_denominator',
                                  'region_white_ratio', 'pixel_fraction', 'protected_hash', 'sample_count'])
def test_diagnosis_rejects_tampered_means_denominators_and_provenance(tmp_path, tamper):
    root = tmp_path / 'runs'
    directory, report = diagnosis(root)
    summary = report['summary']['200']['validation']
    if tamper == 'angle_mean': summary['angle_error_deg']['mean'] += 1
    elif tamper == 'position_mean': summary['position_mae'][2] += 1
    elif tamper == 'region_denominator': summary['regions']['block']['scalars'] += 1
    elif tamper == 'region_white_ratio': summary['regions']['block']['white_image_mse'] += 1
    elif tamper == 'pixel_fraction': summary['regions']['block']['pixel_fraction'] *= 2
    elif tamper == 'protected_hash': report['protected_hashes_after']['checkpoint.pt'] = 'c' * 64
    elif tamper == 'sample_count': report['coverage']['validation']['sampled_frames'] += 1
    write(directory / 'raw.json', report)
    with pytest.raises(DashboardDataError, match='diagnosis|raw evidence|protected'):
        collect_run_results(root)


def history_stage(root, history):
    path = root / 'paddle' / ('history_start_v1' if history else 'baseline') / 'memory'
    write(path / 'paddle_manifest.json', {'stage': 'memory', 'horizon': 1,
        'config': {'history_starts': {'schema_version': 1}} if history else {}})
    rows = []
    for step, ordinary, suffix in [(0, 3., .25), (250, 1., .5)]:
        rows.append({'step': step, 'loss': .5 * (ordinary + suffix) if history else ordinary,
            'ordinary': {'loss': ordinary, 'squared_error_sum': ordinary * 6, 'supervised_scalar_count': 6},
            'suffix': {'loss': suffix, 'squared_error_sum': suffix * 16, 'supervised_scalar_count': 16}})
    (path / 'validation.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    (path / 'training.jsonl').write_text(json.dumps({'step': 250, 'loss': .2}) + '\n')
    return path, rows


def test_history_objective_preserves_equal_population_curves_and_baseline_caption(tmp_path):
    for history in (False, True):
        root = tmp_path / str(history) / 'runs'
        _, rows = history_stage(root, history)
        results, notices = collect_run_results(root)
        run = next(r for r in results if r.kind == 'paddle_training')
        artifact = build_dashboard_artifact(results, notices)
        chart = next(c for c in artifact['manifest']['charts'] if c['id'] == 'paddle_objective_validation_memory')
        if history:
            assert run.context['history_start_training'] is True
            values = artifact['snapshot']['datasets'][chart['dataset']]
            assert [(v['step'], v['loss'], v['ordinary_loss'], v['suffix_loss']) for v in values] == [
                (r['step'], r['loss'], r['ordinary']['loss'], r['suffix']['loss']) for r in rows]
            assert 'equal-weight ordinary/suffix' in chart['subtitle'].lower()
            assert 'diagnostic only' in chart['subtitle'].lower()
        else:
            assert 'Masked normalized state MSE' in chart['subtitle']
            assert 'equal-weight ordinary/suffix' not in chart['subtitle'].lower()


def test_history_reader_rejects_joint_score_that_pools_unequal_denominators(tmp_path):
    root = tmp_path / 'runs'
    path, rows = history_stage(root, True)
    rows[0]['loss'] = (18 + 4) / (6 + 16)
    (path / 'validation.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    with pytest.raises(DashboardDataError, match='joint|ordinary|suffix'):
        collect_run_results(root)


def history_assessment(root):
    """Small exact analogue of the saved independent CPU assessment schema."""
    scale = [64., 64., 6., 6., 64.]
    directory = root / 'paddle' / 'collaboration' / 'history_start_assessment'
    populations = {'ordinary': [[0, 0], [1, 0]], 'suffix': [[2, 1], [3, 1]],
                   'matched_observations': [[0, 2], [1, 2]], 'paired_count': 2}
    arms = {}
    def physical(factor, count):
        values = [x * factor for x in scale]
        return {'count': count, 'mae': values, 'p95': values.copy(), 'max': values.copy()}
    for name, factor, step in [('reference_10000', 1, 10000), ('mixed_selected_9750', 2, 9750),
                               ('mixed_final_10000', 3, 10000)]:
        records, summaries = {}, {}
        error = [x * factor for x in scale]
        for population in ('ordinary', 'suffix'):
            records[population] = [{'episode': ep, 'start': start, 'population': population,
                'target_state': [[0.] * 5 for _ in range(4)], 'predicted_state': [error.copy() for _ in range(4)]}
                for ep, start in populations[population]]
            summaries[population] = {'sequences': 2, 'observations': 8, 'supervised_scalar_count': 32,
                'normalized_squared_error_sum': 32. * factor ** 2, 'state_mse': float(factor ** 2),
                'all_valid': physical(factor, [8, 8, 4, 4, 8]), 'age2': physical(factor, [2] * 5),
                'later': physical(factor, [2] * 5), 'postwarm': physical(factor, [4] * 5)}
        records['paired'] = []
        for pair_seed in (7000, 7001):
            for direction in (-1, 1):
                target = [0., 0., float(direction), 0., 0.]
                predicted = [x + e for x, e in zip(target, error)]
                records['paired'].append({'pair_seed': pair_seed, 'direction': direction,
                    'target_state': [target.copy() for _ in range(3)],
                    'predicted_state': [predicted.copy() for _ in range(3)],
                    'reset_predicted_state': [x + 2 * e for x, e in zip(target, error)]})
        for population, multiplier in [('paired', 1), ('paired_reset', 2)]:
            summaries[population] = {**physical(factor * multiplier, [4] * 5),
                                     'pairs': 2, 'vx_sign_accuracy': .5}
        records['matched'] = [{'episode': ep, 'frame': frame, 'start': frame - 2,
            'target_state': [[0.] * 5 for _ in range(3)],
            'predicted_state': [[3 * e for e in error] for _ in range(3)],
            'full_predicted_state': error.copy()} for ep, frame in populations['matched_observations']]
        for population, multiplier in [('matched_full', 1), ('matched_three', 3)]:
            summaries[population] = physical(factor * multiplier, [2] * 5)
        arms[name] = {'checkpoint': f'runs/{name}.pt', 'checkpoint_sha256': str(factor) * 64,
                      'update': step, 'model_fingerprint': str(factor) * 64,
                      'records': records, 'summaries': summaries}
    comparisons = {}
    reference = arms['reference_10000']['summaries']
    for name in ('mixed_selected_9750', 'mixed_final_10000'):
        comparisons[name] = {}
        for population, current in arms[name]['summaries'].items():
            base = reference[population]
            metrics = [('state_mse', current['state_mse'], base['state_mse'])] if 'state_mse' in base else []
            if 'all_valid' in base:
                metrics += [(f'{age}_mae', current[age]['mae'], base[age]['mae'])
                            for age in ('all_valid', 'age2', 'later', 'postwarm')]
            else: metrics += [('mae', current['mae'], base['mae'])]
            comparisons[name][population] = {field: {
                'delta': [a - b for a, b in zip(value, old)] if isinstance(value, list) else value - old,
                'ratio': [a / b for a, b in zip(value, old)] if isinstance(value, list) else value / old}
                for field, value, old in metrics}
        delta = comparisons[name]['paired']['mae']['delta']
        comparisons[name]['paired']['mae_delta_pair_bootstrap_95'] = [delta.copy(), delta.copy()]
    report = {'schema': 'paddle-history-start-assessment-v1', 'status': 'completed', 'device': 'cpu',
        'coordinate_order': ['ball_x', 'ball_y', 'ball_vx', 'ball_vy', 'paddle_x'],
        'normalization': scale, 'populations': populations, 'arms': arms, 'comparisons': comparisons,
        'counter_audit': {'status': 'passed', 'logged_draws_checked': 8, 'logged_updates_checked': 1,
            'all_cumulative_counters_match': True, 'prospective_receipt_match': True, 'final_rng_states_match': True,
            'paired_never_selects': True, 'selection_uses': ['ordinary_state_mse', 'suffix_state_mse'],
            'counters': {'sequences': 8, 'full_sequences': 4, 'suffix_sequences': 4,
                'observations': 32, 'full_observations': 16, 'suffix_observations': 16,
                'supervised_scalar_count': 128, 'full_supervised_scalar_count': 64, 'suffix_supervised_scalar_count': 64}},
        'protected_hashes': {a['checkpoint']: a['checkpoint_sha256'] for a in arms.values()},
        'source_hashes_unchanged': True, 'model_tensors_unchanged': True, 'test_population_used': False,
        'bootstrap': {'seed': 38921, 'draws': 2000, 'unit': 'pair, both directions retained'}}
    write(directory / 'raw.json', report)
    (directory / 'comparison.png').write_bytes(PNG)
    return directory, report


def test_independent_history_assessment_preserves_original_followup_and_diagnostic_scope(tmp_path):
    root = tmp_path / 'runs'
    history_assessment(root)
    results, notices = collect_run_results(root)
    reports = [r for r in results if r.kind == 'paddle_history_assessment']
    assert len(reports) == 1, 'Independent history evaluation must enter the canonical report'
    run = reports[0]
    assert run.context['diagnostic_only'] is True and run.context['test_population_used'] is False
    assert run.metrics['arms.reference_10000.ordinary.state_mse'] == 1.
    assert run.metrics['arms.mixed_selected_9750.ordinary.state_mse'] == 4.
    assert run.metrics['arms.mixed_final_10000.ordinary.state_mse'] == 9.
    artifact = build_dashboard_artifact(results, notices)
    assert any('data:image/png;base64,' in b.get('body', '') for b in artifact['manifest']['blocks'])
    assert not any(k.startswith('paddle_history_assessment') for k in artifact['snapshot']['datasets'])


@pytest.mark.parametrize('tamper', ['masked_denominator', 'population', 'paired_ratio'])
def test_independent_history_assessment_rejects_wrong_denominators_or_unmatched_comparison(tmp_path, tamper):
    root = tmp_path / 'runs'
    directory, report = history_assessment(root)
    arm = report['arms']['mixed_selected_9750']
    if tamper == 'masked_denominator': arm['summaries']['ordinary']['supervised_scalar_count'] = 40
    elif tamper == 'population': arm['records']['suffix'][0]['episode'] = 99
    elif tamper == 'paired_ratio': report['comparisons']['mixed_selected_9750']['paired']['mae']['ratio'][2] = .1
    write(directory / 'raw.json', report)
    with pytest.raises(DashboardDataError, match='history|assessment|raw evidence'):
        collect_run_results(root)
