"""Read-only paddle ledger adapter and bounded canonical dashboard views.

Keep paddle objectives and physical units separate from the retained LeWM
experiment. Aggregate evaluations reconcile with raw windows/case outcomes before
they enter the instrument panel; source files remain authoritative and unchanged.
"""

from __future__ import annotations

import base64
import hashlib
import html
import json
import math
from pathlib import Path

from .ledger import DashboardDataError, RunResult, _check_finite, numeric, read_json, read_jsonl


POSITIONS = ('ball_x', 'ball_y', 'paddle_x')
STATES = ('ball_x', 'ball_y', 'ball_vx', 'ball_vy', 'paddle_x')


def _list(path):
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise DashboardDataError(f'{path}: cannot read paddle evidence: {error}') from error
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise DashboardDataError(f'{path}: expected a list of paddle records')
    _check_finite(value, str(path))
    return value


def _flat(value, prefix=''):
    result = {}
    for key, item in value.items():
        name = f'{prefix}.{key}' if prefix else key
        if isinstance(item, dict):
            result.update(_flat(item, name))
        elif isinstance(item, list):
            result.update(_flat(dict(enumerate(item)), name))
        elif isinstance(item, (int, float)) and not isinstance(item, bool):
            result[name] = item
    return result


def _equal(actual, recorded, name, path):
    if isinstance(actual, (list, tuple)):
        if not isinstance(recorded, (list, tuple)) or len(actual) != len(recorded):
            raise DashboardDataError(f'{path}: {name} shape disagrees with raw evidence')
        for a, b in zip(actual, recorded):
            _equal(a, b, name, path)
    elif recorded is None or not math.isclose(actual, recorded, rel_tol=1e-8, abs_tol=1e-10):
        raise DashboardDataError(f'{path}: {name} disagrees with raw evidence ({actual} versus {recorded})')


def _reconcile_history_assessment(report, path):
    """Recompute masked state errors and matched comparisons from CPU replay rows."""
    import numpy as np

    if (report['test_population_used'] or not report['source_hashes_unchanged']
            or not report['model_tensors_unchanged']):
        raise DashboardDataError(f'{path}: history assessment scope/integrity receipt failed')
    scale = np.asarray(report['normalization'])
    _equal(scale.tolist(), [64., 64., 6., 6., 64.], 'history assessment state normalization', path)
    reference = report['arms']['reference_10000']
    populations = report['populations']

    def physical(errors, mask, summary, label):
        for coordinate in range(5):
            values = errors[mask[:, coordinate], coordinate]
            _equal(len(values), summary['count'][coordinate], f'history {label} coordinate count', path)
            for field, function in [('mae', np.mean), ('p95', lambda x: np.quantile(x, .95)), ('max', np.max)]:
                _equal(float(function(values)), summary[field][coordinate], f'history {label} {field}', path)

    for name, arm in report['arms'].items():
        if report['protected_hashes'].get(arm['checkpoint']) != arm['checkpoint_sha256']:
            raise DashboardDataError(f'{path}: history checkpoint hash differs from protected source')
        for population in ('ordinary', 'suffix'):
            rows, summary = arm['records'][population], arm['summaries'][population]
            if [[r['episode'], r['start']] for r in rows] != populations[population]:
                raise DashboardDataError(f'{path}: history {population} source identities disagree')
            if [r['target_state'] for r in rows] != [r['target_state'] for r in reference['records'][population]]:
                raise DashboardDataError(f'{path}: history {population} reference targets differ')
            errors, masks, ages = [], [], []
            for row in rows:
                error = np.abs(np.asarray(row['predicted_state']) - row['target_state'])
                mask = np.ones(error.shape, dtype=bool); mask[:2, 2:4] = False
                errors.append(error); masks.append(mask); ages.extend(range(len(error)))
            errors, mask, age = np.concatenate(errors), np.concatenate(masks), np.asarray(ages)
            _equal(len(rows), summary['sequences'], 'history sequence count', path)
            _equal(len(errors), summary['observations'], 'history observation count', path)
            _equal(int(mask.sum()), summary['supervised_scalar_count'], 'history relative-mask denominator', path)
            squared = float(((errors / scale) ** 2)[mask].sum())
            _equal(squared, summary['normalized_squared_error_sum'], 'history normalized squared error', path)
            _equal(squared / mask.sum(), summary['state_mse'], 'history normalized state MSE', path)
            for cut, keep in [('all_valid', np.ones(len(age), bool)), ('age2', age == 2),
                              ('later', age > 2), ('postwarm', age >= 2)]:
                physical(errors, mask & keep[:, None], summary[cut], f'{population}/{cut}')
        pairs = arm['records']['paired']
        pair_ids = [(r['pair_seed'], r['direction']) for r in pairs]
        if pair_ids != [(r['pair_seed'], r['direction']) for r in reference['records']['paired']]:
            raise DashboardDataError(f'{path}: history paired reference identities differ')
        seeds = {r['pair_seed'] for r in pairs}
        if len(pair_ids) != len(set(pair_ids)) or len(pairs) != 2 * len(seeds):
            raise DashboardDataError(f'{path}: history assessment must preserve two directions per pair')
        _equal(len(seeds), populations['paired_count'], 'history diagnostic pair count', path)
        matched = arm['records']['matched']
        if [[r['episode'], r['frame']] for r in matched] != populations['matched_observations']:
            raise DashboardDataError(f'{path}: history matched source identities disagree')
        full = {r['episode']: r for r in arm['records']['ordinary']}
        for row in matched:
            _equal(row['full_predicted_state'], full[row['episode']]['predicted_state'][row['frame']],
                   'history matched full-history prediction', path)
        for population, rows, field, sequence in [('paired', pairs, 'predicted_state', True),
                ('paired_reset', pairs, 'reset_predicted_state', False),
                ('matched_full', matched, 'full_predicted_state', False),
                ('matched_three', matched, 'predicted_state', True)]:
            truth = np.asarray([r['target_state'][-1] for r in rows])
            base_rows = reference['records']['paired' if population.startswith('paired') else 'matched']
            if not np.array_equal(truth, [r['target_state'][-1] for r in base_rows]):
                raise DashboardDataError(f'{path}: history {population} reference targets differ')
            predicted = np.asarray([r[field][-1] if sequence else r[field] for r in rows])
            physical(np.abs(predicted - truth), np.ones(truth.shape, bool), arm['summaries'][population], population)
            if population.startswith('paired'):
                _equal(len(seeds), arm['summaries'][population]['pairs'], 'history pair count', path)
                _equal(float(np.mean(np.sign(predicted[:, 2]) == np.sign(truth[:, 2]))),
                       arm['summaries'][population]['vx_sign_accuracy'], 'history vx sign accuracy', path)
    for name, populations_comparison in report['comparisons'].items():
        for population, comparisons in populations_comparison.items():
            current, base = report['arms'][name]['summaries'][population], reference['summaries'][population]
            for field, comparison in comparisons.items():
                if field == 'mae_delta_pair_bootstrap_95':
                    def pair_errors(arm):
                        rows = arm['records']['paired']
                        return np.asarray([np.abs(np.asarray(r['predicted_state'][-1]) - r['target_state'][-1])
                                           for r in rows]).reshape(-1, 2, 5).mean(1)
                    delta = pair_errors(report['arms'][name]) - pair_errors(reference)
                    draws = np.random.default_rng(report['bootstrap']['seed']).integers(
                        0, len(delta), size=(report['bootstrap']['draws'], len(delta)))
                    _equal(np.quantile(delta[draws].mean(1), [.025, .975], axis=0).tolist(), comparison,
                           'history paired bootstrap interval', path)
                    continue
                a = current[field[:-4]]['mae'] if field.endswith('_mae') else current[field]
                b = base[field[:-4]]['mae'] if field.endswith('_mae') else base[field]
                _equal((np.asarray(a) - b).tolist(), comparison['delta'], 'history comparison delta', path)
                _equal((np.asarray(a) / b).tolist(), comparison['ratio'], 'history comparison ratio', path)
    audit = report['counter_audit']; counters = audit['counters']
    if audit['status'] != 'passed' or not all(audit[k] for k in ('all_cumulative_counters_match',
            'prospective_receipt_match', 'final_rng_states_match', 'paired_never_selects')):
        raise DashboardDataError(f'{path}: history counter audit failed')
    for field in ('sequences', 'observations', 'supervised_scalar_count'):
        _equal(counters['full_' + field] + counters['suffix_' + field], counters[field], 'history mixed counters', path)
    _equal(counters['sequences'], audit['logged_draws_checked'], 'history logged draw count', path)
    _equal(5 * counters['observations'] - 4 * counters['sequences'], counters['supervised_scalar_count'],
           'history training mask denominator', path)


def _reconcile_evaluation(report, directory):
    controls_path, predictions_path = directory / 'control_records.json', directory / 'prediction_records.json'
    controls, predictions = _list(controls_path), _list(predictions_path)
    if len({r['case_id'] for r in controls}) != len(controls):
        raise DashboardDataError(f'{controls_path}: duplicate case identities')
    for population in ('ordinary', 'paired'):
        for policy, summary in report[population]['summary'].items():
            rows = [r for r in controls if r['population'] == population and r['controller'] == policy]
            count = len(rows)
            successes = sum(r['first_hit_before_miss'] for r in rows)
            _equal(count, summary['count'], f'{population}.{policy}.count', controls_path)
            _equal(successes, summary['successes'], f'{population}.{policy}.successes', controls_path)
            if count:
                _equal(successes / count, summary['success_rate'], f'{population}.{policy}.success_rate', controls_path)
            if 'failure_case_ids' in summary:
                failures = [r['case_id'] for r in rows if not r['first_hit_before_miss']]
                if sorted(failures) != sorted(summary['failure_case_ids']):
                    raise DashboardDataError(f'{controls_path}: {population}.{policy} failure identities disagree with raw evidence')
            if 'first_action_correct' in summary and summary['first_action_correct'] is not None:
                _equal(sum(r['first_action'] == r['correct_action'] for r in rows), summary['first_action_correct'],
                       'first_action_correct', controls_path)
    for method, horizons in report['prediction']['summary'].items():
        for horizon, groups in horizons.items():
            matched = [r for r in predictions if r['method'] == method and r['horizon'] == int(horizon)]
            for group, summary in groups.items():
                selected = [r for r in matched if group == 'all'
                            or group == 'collision' and r['collision']
                            or group == 'no_collision' and not r['collision']
                            or group not in ('all', 'collision', 'no_collision') and group in r.get('collision_types', [])]
                _equal(len(selected), summary['count'], f'{method}.h{horizon}.{group}.count', predictions_path)
                if not selected:
                    continue
                for field, raw in (('h_mae', 'h_abs_error'), ('r_mae', 'r_abs_error')):
                    averages = [sum(r[raw][i] for r in selected) / len(selected) for i in range(len(selected[0][raw]))]
                    _equal(averages, summary[field], f'{method}.h{horizon}.{group}.{field}', predictions_path)
                _equal(sum(r['latent_error'] for r in selected) / len(selected), summary['latent_error'],
                       f'{method}.h{horizon}.{group}.latent_error', predictions_path)
    sources = [controls_path, predictions_path]
    if report.get('reconstruction'):
        path = directory / 'reconstruction_records.json'
        rows = _list(path)
        sources.append(path)
        summary = report['reconstruction']
        _equal(sum(row['count'] for row in rows), summary['count'], 'reconstruction.count', path)
        for region in ('global', 'ball_region', 'paddle_region'):
            scalars = sum(row[f'{region}_scalars'] for row in rows)
            squared = sum(row[f'{region}_squared_error'] for row in rows)
            if scalars:
                _equal(squared / scalars, summary[f'{region}_mse'], f'{region}_mse', path)
    if report.get('actual_frame_readout') and (report['actual_frame_readout'].get('source_records')
                                              or (directory / 'actual_frame_readout_records.json').exists()):
        path = directory / 'actual_frame_readout_records.json'
        rows = _list(path)
        sources.append(path)
        summary = report['actual_frame_readout']
        _equal(len(rows), summary['count'], 'all-frame H count', path)
        if rows:
            mean = [sum(row['h_abs_error'][i] for row in rows) / len(rows) for i in range(3)]
            _equal(mean, summary['h_mae'], 'all-frame H MAE', path)
    return sources


def collect_paddle_results(runs_root):
    results, notices = [], []

    def add(directory, kind, status, metrics, context, sources, step=None, internals=None):
        results.append(RunResult(directory.relative_to(runs_root).as_posix(), kind, status, step,
                       metrics, context, tuple(p.relative_to(runs_root.parent).as_posix() for p in sources),
                       max(p.stat().st_mtime for p in sources), internals=internals or {}))

    for path in sorted(runs_root.rglob('paddle_manifest.json')):
        directory = path.parent
        manifest = read_json(path)
        result_path = directory / 'paddle_result.json'
        sources = [path]
        training, validation = (), ()
        for name in ('training', 'validation'):
            ledger = directory / f'{name}.jsonl'
            if ledger.exists():
                sources.append(ledger)
                values = read_jsonl(ledger)
                if name == 'training': training = values
                else: validation = values
        if result_path.exists():
            result = read_json(result_path)
            sources.append(result_path)
            status = result['status']
        else:
            status_path = directory / 'status.json'
            result = read_json(status_path) if status_path.exists() else {}
            if status_path.exists(): sources.append(status_path)
            status = 'incomplete'
            failure_path = directory / 'failure.json'
            if failure_path.exists():
                sources.append(failure_path)
                result['failure'] = read_json(failure_path)
                status = 'failed'
        selected = result.get('selected_update')
        step = result.get('global_update', training[-1]['step'] if training else None)
        if selected is not None:
            selected_rows = [r for r in validation if r['step'] == selected]
            if not selected_rows:
                raise DashboardDataError(f'{result_path}: selected validation step missing from raw ledger')
            for key, value in _flat(result.get('metrics', {})).items():
                _equal(value, _flat(selected_rows[-1]).get(key), f'selected validation {key}', result_path)
        if status == 'completed' and training and step != training[-1]['step']:
            raise DashboardDataError(f'{result_path}: completed update disagrees with training ledger')
        smoke = bool(result.get('smoke', manifest.get('config', {}).get('smoke', False)))
        history_start = manifest['stage'] == 'memory' and 'history_starts' in (manifest.get('config') or {})
        if history_start:
            for row in validation:
                for population in ('ordinary', 'suffix'):
                    values = row[population]
                    count = values['supervised_scalar_count']
                    if count <= 0:
                        raise DashboardDataError(f'{directory}: {population} validation has no supervised scalars')
                    _equal(values['squared_error_sum'] / count, values['loss'],
                           f'{population} normalized state MSE', directory / 'validation.jsonl')
                _equal(.5 * (row['ordinary']['loss'] + row['suffix']['loss']), row['loss'],
                       'joint ordinary/suffix validation MSE', directory / 'validation.jsonl')
        context = {'architecture': 'Paddle E/U/P; separate observation features and persistent memory',
                   'stage': manifest['stage'], 'horizon': manifest.get('horizon'), 'smoke': smoke,
                   'selected_update': selected, 'dataset_fingerprint': manifest.get('dataset_fingerprint'),
                   'config': manifest.get('config'), 'dependencies': manifest.get('dependencies'),
                   'versions': manifest.get('versions'), 'quality_gate': result.get('quality_gate'),
                   'baseline_gate': json.dumps(result.get('quality_gate', {})),
                   'checkpoint': result.get('checkpoint'), 'failure': result.get('failure')}
        if history_start:
            context.update(history_start_training=True,
                           validation_objective='Equal-weight ordinary/suffix normalized state MSE; paired results diagnostic only')
        if manifest.get('continuation'):
            # Keep lineage reviewable within a bounded context cell. The full
            # parent config and ancestry remain in the raw manifest source.
            context['continuation'] = {key: manifest['continuation'][key] for key in (
                'parent_checkpoint', 'parent_sha256', 'parent_global_update',
                'parent_budget_updates', 'child_budget_updates', 'added_budget_updates',
                'remaining_updates_from_parent_checkpoint', 'counter_semantics',
            ) if key in manifest['continuation']}
        metrics = {**numeric(result), **_flat(result.get('metrics', {}), 'selected_validation')}
        if training: metrics.update(_flat(training[-1], 'latest_training'))
        if validation: metrics.update(_flat(validation[-1], 'latest_validation'))
        add(directory, 'paddle_training', status, metrics, context, sources, step,
            {'paddle_training': training, 'paddle_validation': validation})
        if smoke:
            notices.append(f'{directory.relative_to(runs_root)}: SMOKE execution evidence only; engineering targets are not established')
        if status != 'completed':
            notices.append(f'{directory.relative_to(runs_root)}: {status}; preserve selected/latest validation steps separately')

    for path in sorted(runs_root.rglob('metrics.json')):
        report = read_json(path)
        schema = report.get('schema_version')
        if schema not in ('paddle-evaluation-v1', 'paddle-demo-v1'):
            continue
        sources = [path]
        if schema == 'paddle-demo-v1':
            add(path.parent, 'paddle_demo', report['status'], numeric(report.get('case', {})),
                {'protocol': report.get('note'), 'seed': report.get('seed')}, sources)
            continue
        if report['status'] != 'completed' and not all((path.parent / name).exists()
                for name in ('control_records.json', 'prediction_records.json')):
            add(path.parent, 'paddle_reporting_failure', report['status'], {},
                {'raw_status': report.get('raw_status'), 'error': report.get('report_error'),
                 'protocol': 'Raw report retained; derived case/window records are missing. Control charts remain unavailable until reporting resumes.'}, sources)
            notices.append(f'{path.parent.relative_to(runs_root)}: {report["status"]}; missing derived record exports prevent independent aggregate reconciliation')
            continue
        sources += _reconcile_evaluation(report, path.parent)
        probe_path = path.parent / 'probe_records.json'
        if probe_path.exists(): sources.append(probe_path)
        panels = []
        for visual in report.get('visuals', []):
            if 'png' not in visual:
                continue
            from .ledger import resolve_evidence_path
            panel = resolve_evidence_path(visual['png'], runs_root)
            if not panel.is_file():
                raise DashboardDataError(f'{path}: qualitative panel missing: {panel}')
            sources.append(panel)
            panels.append({**visual, 'png': str(panel)})
        context = {'smoke': report.get('smoke', False), 'targets': report.get('targets'),
                   'baseline_gate': json.dumps(report.get('targets', {})), 'definitions': report.get('definitions'),
                   'hardware': report.get('hardware'), 'checkpoints': report.get('checkpoints'),
                   'dataset_fingerprint': report.get('dataset_fingerprint'),
                   'protocol': 'Paired ordinary initial states across controllers; distinct identical-frame paired-history population; test-only evaluation',
                   'probe_training_frames': report.get('velocity_probe', {}).get('fit', {}).get('count'),
                   'probe_fit_split': report.get('velocity_probe', {}).get('fit', {}).get('split')}
        add(path.parent, 'paddle_evaluation', report['status'],
            {'ordinary_starts': report['ordinary']['starts'], 'paired_pairs': report['paired']['pairs'],
             **_flat(report.get('reconstruction', {}), 'reconstruction'),
             **_flat(report.get('actual_frame_readout', {}), 'h_allframes'),
             **_flat({k: v for k, v in report.get('velocity_probe', {}).items() if k != 'fit'}, 'velocity_probe')},
            context, sources, internals={'paddle_report': report, 'paddle_panels': panels})
        if report['status'] != 'completed':
            notices.append(f'{path.parent.relative_to(runs_root)}: {report["status"]}; raw metrics may be complete while reporting remains incomplete')
        if report.get('smoke'):
            notices.append(f'{path.parent.relative_to(runs_root)}: SMOKE counts and failures are execution diagnostics, not full-population success evidence')
        failed = [k for k, value in report.get('targets', {}).items() if value is False]
        if failed:
            notices.append(f'{path.parent.relative_to(runs_root)}: unmet targets / evidence requirements: {", ".join(failed)}')
    for path in sorted(runs_root.rglob('ordinary.partial.json')):
        if not (path.parent / 'metrics.json').exists():
            notices.append(f'{path.parent.relative_to(runs_root)}: incomplete paddle evaluation; partial case evidence retained')
    for path in sorted(runs_root.rglob('history_checks.json')):
        report = read_json(path)
        if not str(report.get('schema_version', '')).startswith('paddle-history-verification'):
            continue
        cases = report['cases']
        if bool(report['passed']) != all(case['passed'] for case in cases):
            raise DashboardDataError(f'{path}: paired-history pass summary disagrees with cases')
        if len(cases) != 2 * (report['validation_pairs'] + report['test_pairs']):
            raise DashboardDataError(f'{path}: paired-history member count disagrees with declared pairs')
        add(path.parent, 'paddle_history', 'verified' if report['passed'] else 'failed',
            {**numeric(report), 'passed_members': sum(case['passed'] for case in cases), 'members': len(cases)},
            {'environment_fingerprint': report.get('environment_fingerprint'), 'schema_version': report['schema_version'],
             'protocol': 'Exact simulator enumeration of paired identical-current-frame histories; distinct validation/test seeds'}, [path])
    for path in sorted(runs_root.rglob('raw.json')):
        report = read_json(path)
        if report.get('schema') != 'paddle-history-start-assessment-v1':
            continue
        _reconcile_history_assessment(report, path)
        panel = path.parent / 'comparison.png'
        if not panel.is_file():
            raise DashboardDataError(f'{path}: history assessment comparison image missing')
        metrics, context = {}, {'diagnostic_only': True, 'test_population_used': False,
            'protocol': 'Independent CPU replay on fixed validation populations; one training seed; paired results never select',
            'counter_audit': report['counter_audit'], 'bootstrap': report['bootstrap']}
        for name, arm in report['arms'].items():
            context[f'checkpoint.{name}'] = {k: v for k, v in arm.items() if k not in ('records', 'summaries')}
            for population, summary in arm['summaries'].items():
                context[f'{name}.{population}'] = summary
                metrics.update(_flat({k: v for k, v in summary.items() if k in
                    ('state_mse', 'supervised_scalar_count', 'observations', 'sequences', 'mae', 'vx_sign_accuracy')},
                    f'arms.{name}.{population}'))
            if name in report['comparisons']:
                for population, comparison in report['comparisons'][name].items():
                    context[f'comparison.{name}.{population}'] = comparison
        add(path.parent, 'paddle_history_assessment', report['status'], metrics, context, [path, panel],
            internals={'arms': {name: {'update': arm['update'], 'summaries': arm['summaries']}
                               for name, arm in report['arms'].items()}, 'panel': str(panel)})
    return results, notices


def _sample(rows, limit=50):
    if len(rows) <= limit:
        return list(rows)
    return [rows[i] for i in sorted({round(j * (len(rows) - 1) / (limit - 1)) for j in range(limit)})]


def add_paddle_views(runs, datasets, charts, tables, cards, chart, table, source_id):
    """Append native charts/exact aggregate tables; return introductory/image blocks."""
    training = [r for r in runs if r.kind == 'paddle_training']
    evaluations = [r for r in runs if r.kind == 'paddle_evaluation']
    assessments = [r for r in runs if r.kind == 'paddle_history_assessment']
    if not training and not evaluations and not assessments:
        return []
    number = lambda key: {'field': key, 'type': 'quantitative'}
    category = lambda key: {'field': key, 'type': 'nominal'}
    blocks = [{'id': 'paddle_intro', 'type': 'markdown', 'sourceId': source_id,
               'body': '## Paddle world model\n\nE/U/P learning, physical readouts and paired control results. '
                       'SMOKE runs establish execution only. Actual versus imagined R updates are distinct; '
                       'training completion and engineering target achievement are reported separately. '
                       'The retained LeWM experiment appears in its own charts and complete inventory.'}]
    datasets['paddle_coverage'] = [{'completed_stages': sum(r.status == 'completed' for r in training),
                                   'evaluations': len(evaluations)}]
    cards.append({'id': 'paddle_completed_stages', 'dataset': 'paddle_coverage', 'sourceId': source_id,
                  'description': 'Completed paddle optimizer stages including smoke; failed quality gates remain separate.',
                  'metrics': [{'label': 'Completed paddle stages', 'field': 'completed_stages', 'format': 'number'}]})
    latest_stages = {}
    for run in training:
        key = run.context['stage']
        if key == 'predictor': key += f"_{run.context['horizon']}"
        latest_stages[key] = run
    for key, run in latest_stages.items():
        training_values, validation_values = [], []
        for source, series in (('paddle_training', 'training objective'), ('paddle_validation', 'validation objective')):
            values = training_values if source == 'paddle_training' else validation_values
            for row in _sample(run.internals[source]):
                if 'loss' in row:
                    values.append({'step': row['step'], 'value': row['loss'], 'series': series})
                if source == 'paddle_validation' and 'copy_loss' in row:
                    values.append({'step': row['step'], 'value': row['copy_loss'], 'series': 'matched copy objective'})
        objective = {'perception': 'RGB MSE plus normalized position MSE',
                     'memory': 'Masked normalized state MSE; initial velocity entries excluded'}.get(run.context['stage'],
                     'Frozen-latent variance-normalized MSE; fine/coarse equal weight')
        if run.context.get('continuation'):
            objective += '; update numbers are cumulative from the parent checkpoint, while additional_updates counts this continuation only'
        history_start = run.context.get('history_start_training', False)
        if history_start:
            objective = 'Mixed full/suffix batch masked normalized state MSE; first two relative velocity entries excluded'
            validation_values = [{'step': row['step'], 'loss': row['loss'],
                                  'ordinary_loss': row['ordinary']['loss'], 'suffix_loss': row['suffix']['loss'],
                                  'ordinary_scalars': row['ordinary']['supervised_scalar_count'],
                                  'suffix_scalars': row['suffix']['supervised_scalar_count']}
                                 for row in _sample(run.internals['paddle_validation'])]
        # The portable reader uses encounter-order categories and breaks lines
        # at missing series values. Training and validation use different step
        # grids, so plot them separately without inventing intermediate values.
        for name, label, values in ((f'paddle_training_{key}', 'training objective', training_values),
                                    (f'paddle_objective_validation_{key}', 'validation objective', validation_values)):
            if not values:
                continue
            datasets[name] = sorted(values, key=lambda row: row['step'])
            kind = 'line' if len({row['step'] for row in values}) > 1 else 'bar'
            if history_start and values is validation_values:
                charts.append(chart(name, f'Paddle memory: joint and population validation objectives · {run.label}',
                    'Equal-weight ordinary/suffix normalized state MSE: 0.5 × ordinary + 0.5 × suffix. '
                    'Paired results are diagnostic only and never select the checkpoint. '
                    'Each population uses its own supervised-scalar denominator; first two relative velocity entries are masked. '
                    'At most 50 recorded updates; connecting curves are visual guides.',
                    name, kind, number('step'),
                    {'fields': ['loss', 'ordinary_loss', 'suffix_loss'], 'type': 'quantitative', 'label': 'Normalized state MSE'}))
                charts[-1]['palette'] = {'kind': 'categorical', 'name': 'PATH-WM blue-orange'}
                charts[-1]['legend'] = {'position': 'bottom', 'sort': 'spec'}
                charts[-1]['surface']['interactiveLegend'] = True
                if kind == 'line': charts[-1]['settings']['showPoints'] = 'always'
                continue
            charts.append(chart(name, f'Paddle {key}: {label} · {run.label}',
                objective + '. At most 50 recorded updates per curve; raw ledgers retain every update. '
                'Line x positions are ordered sampled updates with equal spacing; labels retain the exact update numbers.',
                name, kind, number('step') if kind == 'line' else category('step'), number('value'), color=category('series')))
            if kind == 'line' and values is validation_values:
                charts[-1]['settings']['showPoints'] = 'always'
        for readout, coordinates, indices, units in (
            ('h_mae', POSITIONS, range(3), 'world units/pixels'),
            ('r_mae', STATES, (2, 3), 'world units per decision interval'),
            ('r_position_mae', STATES, (0, 1, 4), 'world units/pixels'),
        ):
            source_key = 'r_mae' if readout == 'r_position_mae' else readout
            values = []
            for row in _sample(run.internals['paddle_validation']):
                errors = row.get(source_key)
                if not errors:
                    continue
                # Predictor validation has [horizon,coordinate] errors. Display
                # the final trained horizon and retain all coordinates/horizons
                # in the selected/latest exact-value records.
                if isinstance(errors[0], list): errors = errors[-1]
                values.extend({'step': row['step'], 'coordinate': coordinates[i], 'value': errors[i]} for i in indices)
            if values:
                name = f'paddle_validation_{readout}_{key}'
                datasets[name] = values
                charts.append(chart(name, f'Paddle {key}: validation {readout}',
                    f'Each displayed coordinate is MAE in {units}. Predictor charts show the last trained horizon; other charts use real observations.',
                    name, 'line', number('step'), number('value'), color=category('coordinate')))
        if run.context['stage'] == 'memory':
            for coordinate, index, units in (('ball_vx', 2, 'world units per decision interval'),
                                              ('ball_vy', 3, 'world units per decision interval'),
                                              ('ball_y', 1, 'world units/pixels')):
                values = []
                for row in _sample(run.internals['paddle_validation']):
                    if 'paired_validation_r_mae' not in row:
                        continue
                    for field, label in (('r_mae', 'ordinary history R'),
                                         ('paired_validation_r_mae', 'paired history R'),
                                         ('paired_validation_reset_r_mae', 'paired reset R')):
                        if row.get(field):
                            values.append({'step': row['step'], 'value': row[field][index], 'readout': label})
                if values:
                    name = f'paddle_memory_paired_{coordinate}'
                    datasets[name] = values
                    charts.append(chart(name, f'Paddle memory: {coordinate} validation MAE · {run.label}',
                        f'MAE in {units}. Ordinary episodes and paired three-frame histories are distinct validation populations. '
                        'Paired reset uses only the final image. At most 50 recorded steps; the exact table retains every step and coordinate.',
                        name, 'line', number('step'), number('value'), color=category('readout')))
    datasets['paddle_paired_validation_detail'] = []
    for run in training:
        if run.context['stage'] != 'memory':
            continue
        config = run.context.get('config') or {}
        pair_count = config.get('training', {}).get('memory', {}).get('validation_pairs')
        for row in run.internals['paddle_validation']:
            if 'paired_validation_r_mae' in row:
                datasets['paddle_paired_validation_detail'].append({
                    'run': run.label, 'step': row['step'],
                    'ordinary_r_mae': json.dumps(row.get('r_mae')),
                    'paired_r_mae': json.dumps(row['paired_validation_r_mae']),
                    'paired_reset_r_mae': json.dumps(row.get('paired_validation_reset_r_mae')),
                    'ordinary_observations': row.get('ordinary', {}).get('observations', row.get('observations')),
                    'ordinary_velocity_observations': (row['ordinary']['post_warmup']['coordinate_count'][2]
                        if 'ordinary' in row else row.get('post_warmup_observations')),
                    'configured_pairs': pair_count,
                })
    datasets['paddle_prediction_detail'], datasets['paddle_control_detail'] = [], []
    datasets['paddle_failure_cases'] = []
    for run in evaluations:
        report = run.internals['paddle_report']
        for population in ('ordinary', 'paired'):
            for controller, summary in report[population]['summary'].items():
                failures = summary.get('failure_case_ids', [])
                datasets['paddle_control_detail'].append({'run': run.label, 'population': population,
                     'controller': controller, **{k: json.dumps(v) if isinstance(v, (list, dict)) else v
                                                   for k, v in summary.items() if k != 'failure_case_ids'},
                     'failure_case_count': len(failures),
                     'smoke': report.get('smoke', False)})
                record_key = 'R' + hashlib.sha256(run.label.encode()).hexdigest()[:10]
                datasets['paddle_failure_cases'].extend({'record_key': record_key, 'case_id': case_id,
                      'population': population, 'controller': controller} for case_id in failures)
        for method, horizons in report['prediction']['summary'].items():
            for horizon, groups in horizons.items():
                for group, summary in groups.items():
                    datasets['paddle_prediction_detail'].append({'run': run.label, 'method': method,
                        'horizon': int(horizon), 'population': group,
                        **{k: json.dumps(v) if isinstance(v, list) else v for k, v in summary.items()}})
    if evaluations:
        run = evaluations[-1]
        report = run.internals['paddle_report']
        for population in ('ordinary', 'paired'):
            name = f'paddle_control_{population}'
            datasets[name] = [r for r in datasets['paddle_control_detail'] if r['run'] == run.label and r['population'] == population]
            if datasets[name]:
                charts.append(chart(name, f'Paddle {population}: first interception · {run.label}',
                    'Successes / recorded starts after the same two stay actions. Privileged uses exact simulation. Smoke sample sizes do not establish the 90% target.',
                    name, 'bar', category('controller'), number('success_rate'),
                    tooltip=[number('successes'), number('count')],
                    reference_lines=[{'axis': 'y', 'value': .9, 'label': 'Engineering target'}]))
        for field, names, title, target in (('h_mae', POSITIONS, 'Actual H position MAE', 1.),
                                           ('r_mae', STATES, 'Actual R velocity MAE', .5)):
            actual = report['prediction']['summary'].get('actual', {}).get('0', {}).get('all')
            if field == 'h_mae' and report.get('actual_frame_readout'):
                actual = report['actual_frame_readout']
            if not actual or not actual['count']:
                continue
            ix = range(3) if field == 'h_mae' else (2, 3)
            name = f'paddle_actual_{field}'
            datasets[name] = [{'coordinate': names[i], 'value': actual[field][i], 'count': actual['count']} for i in ix]
            charts.append(chart(name, f'Paddle: {title}',
                ('H uses all test frames when the all-frame readout is recorded; R excludes the first two frames. '
                 'H units are world units/pixels; R velocity units are world units per decision interval.'),
                name, 'bar', category('coordinate'), number('value'), tooltip=[number('count')],
                reference_lines=[{'axis': 'y', 'value': target, 'label': 'Engineering target'}]))
        name = 'paddle_matched_horizons'
        values = []
        for method, horizons in report['prediction']['summary'].items():
            if method == 'actual': continue
            for horizon, groups in horizons.items():
                row = groups['all']
                if row['count']:
                    values.append({'horizon': int(horizon), 'latent': row['latent_error'],
                                   **dict(zip(POSITIONS, row['h_mae'])),
                                   'method': method, 'count': row['count']})
        if values:
            datasets[name] = sorted(values, key=lambda r: (r['horizon'], r['method']))
            for coordinate in ('latent', *POSITIONS):
                charts.append(chart(f'paddle_horizon_{coordinate}', f'Paddle matched rollout: {coordinate} error',
                    ('Variance-normalized latent MSE.' if coordinate == 'latent' else 'Coordinate MAE in world units/pixels.')
                    + ' Prediction, copy and fresh-memory reset share exact windows and actions; collision subsets stay in the exact table.',
                    name, 'line', number('horizon'), number(coordinate), color=category('method'), tooltip=[number('count')]))
        probe = report.get('velocity_probe', {})
        if probe.get('frame_probe_mae') and probe.get('memory_mae'):
            name = 'paddle_paired_velocity_probe'
            datasets[name] = [{'coordinate': coordinate, 'value': probe[field][i], 'readout': label}
                              for field, label in (('frame_probe_mae', 'current frame linear probe'), ('memory_mae', 'history memory R'))
                              for i, coordinate in enumerate(('vx', 'vy'))]
            charts.append(chart(name, 'Paddle identical-current-frame pairs: velocity MAE',
                'Physical interval units; current-frame diagnostic fitted on training only. Opposite histories share the same last raw image.',
                name, 'bar', category('coordinate'), number('value'), color=category('readout'), group_mode='grouped'))
        images = []
        for panel in run.internals['paddle_panels'][:2]:
            encoded = base64.b64encode(Path(panel['png']).read_bytes()).decode('ascii')
            images.append(f'<figure><img style="max-width:100%;height:auto" alt="Paddle actual, reconstructed and imagined frames" src="data:image/png;base64,{encoded}">'
                          f'<figcaption>{html.escape(panel["case_id"])}; first interception: {panel["first_hit_before_miss"]}. '
                          'R after real and imagined observations is labeled separately.</figcaption></figure>')
        if images:
            blocks.append({'id': 'paddle_qualitative', 'type': 'html', 'layout': 'full',
                           'body': '<h3>Paddle actual / reconstruction / imagination</h3>' + ''.join(images)})
    for name, title, columns, sort, subtitle in (
        ('paddle_paired_validation_detail', 'Paddle memory: every paired-history validation checkpoint',
         [('run','Run'),('step','Update'),('ordinary_r_mae','Ordinary R MAE [x,y,vx,vy,paddle]'),
          ('paired_r_mae','Paired R MAE [x,y,vx,vy,paddle]'),('paired_reset_r_mae','Paired reset R MAE [x,y,vx,vy,paddle]'),
          ('ordinary_observations','Ordinary frames'),('ordinary_velocity_observations','Ordinary velocity frames'),
          ('configured_pairs','Configured pairs')],
         'run', 'Every recorded validation step and all five physical-coordinate errors are retained. '
         'Ordinary velocity excludes warm-up frames; paired history and reset each measure the final frame of both pair members. '
         'These populations are separate diagnostics, not pooled estimates. Exact rows come from each memory validation.jsonl.'),
        ('paddle_control_detail', 'Paddle control: exact paired-population results',
         [('run','Run'),('population','Population'),('controller','Controller'),('successes','First interceptions'),('count','Cases'),
          ('success_rate','Success fraction'),('first_action_correct','Correct first actions'),('first_action_count','First-action cases'),
          ('total_hits','Total hits'),('mean_episode_length','Mean intervals'),('latency_median_ms','Decision median ms'),
          ('latency_p95_ms','Decision p95 ms'),('planning_failures','Planning failures'),('evaluation_capped','Capped cases'),('smoke','Smoke')],
         'run', 'Counts reconcile with raw case records. Latency is synchronized decision time excluding rendering and real observation encoding; full scope/device are in protocol context.'),
        ('paddle_prediction_detail', 'Paddle readouts: exact horizons, coordinates and collision populations',
         [('run','Run'),('method','Method'),('horizon','Horizon'),('population','Reflection population'),('count','Rows'),
          ('latent_error','Normalized latent MSE'),('h_mae','H MAE [x,y,paddle]'),('r_mae','R MAE [x,y,vx,vy,paddle]'),
          ('h_p95','H coordinate p95'),('h_max','H coordinate max'),('r_p95','R coordinate p95'),('r_max','R coordinate max'),
          ('terminal_count','Terminal targets'),('terminal_false_negative_count','Underestimated miss boundary')],
         'run', 'Actual rows are observed states after warm-up. Other methods use matched windows; collision means any reflection in the predicted prefix. Positions and velocities have different physical units.'),
        ('paddle_failure_cases', 'Paddle first-interception failures: every case identity',
         [('record_key','Evaluation record'),('case_id','Case identity'),('population','Population'),('controller','Controller')],
         'record_key', 'One row per failed first interception. Record keys resolve to full evaluation identities and raw control trajectories in the inventory; all cases are retained across numbered table parts.'),
    ):
        if datasets[name]: tables.append(table(name, title, columns, sort, subtitle))
    if assessments:
        run = assessments[-1]
        body = ('<h3>Independent history-start assessment</h3><p>The original observer, selected mixed-history '
                'checkpoint and final checkpoint are replayed on the same ordinary, suffix and paired validation '
                'populations. This one-seed assessment is diagnostic only; no test population is used.</p>'
                '<div style="overflow-x:auto"><table><thead><tr><th>Observer</th><th>Update</th>'
                '<th>Ordinary state MSE</th><th>Suffix state MSE</th><th>Paired vx MAE</th></tr></thead><tbody>')
        for name, arm in run.internals['arms'].items():
            summary = arm['summaries']
            body += (f'<tr><td>{html.escape(name.replace("_", " "))}</td><td>{arm["update"]}</td>'
                     f'<td>{summary["ordinary"]["state_mse"]:.6g}</td><td>{summary["suffix"]["state_mse"]:.6g}</td>'
                     f'<td>{summary["paired"]["mae"][2]:.6g}</td></tr>')
        body += ('</tbody></table></div><p>Each state MSE uses its own masked scalar denominator. '
                 'The first two relative velocity labels are excluded. Physical velocity errors use pixels per '
                 'decision interval. Paired bootstrap intervals preserve both directions of each pair and describe '
                 'this fixed diagnostic population, not uncertainty across training seeds.</p>')
        encoded = base64.b64encode(Path(run.internals['panel']).read_bytes()).decode()
        body += (f'<img style="max-width:100%;height:auto" alt="Matched history-start observer comparison" '
                 f'src="data:image/png;base64,{encoded}"><p>Source identities, physical means, quantiles, maxima, '
                 'normalized errors, comparison ratios and pair-bootstrap intervals reconcile with replay records. '
                 'The exact context retains both selected and final results; paired outcomes never select a checkpoint.</p>')
        blocks.append({'id': 'paddle_history_assessment_panel', 'type': 'html', 'layout': 'full',
                       'sourceId': source_id, 'body': body})
    return blocks
