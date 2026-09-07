"""Source-backed views for the separate CCHI E/U/P task; never relabel LeWM."""
from __future__ import annotations

import base64
from collections import Counter
import hashlib
import html
import json
from pathlib import Path

from .ledger import DashboardDataError, RunResult, numeric, read_json, read_jsonl
from .paddle import _equal, _flat, _list, _sample


POSITIONS = ('pusher_x', 'pusher_y', 'block_x', 'block_y')
MOTION = (*POSITIONS, 'block_angle')


def _reconcile_diagnosis(report, path, independent_splits=None):
    """Audit only recorded populations; weighted ROI scalars need not be integers."""
    before = report['protected_hashes_before']
    if not before or before != report['protected_hashes_after']:
        raise DashboardDataError(f'{path}: diagnosis protected before/after hashes disagree')
    records, alignment = report['records'], report['alignment']
    identity = lambda row: (row['split'], row['source_episode'], row['frame_index'], row['source_row'])
    observed = {identity(row) for row in alignment}
    if len(observed) != len(alignment) or not observed:
        raise DashboardDataError(f'{path}: diagnosis alignment identities are empty or repeated')
    keys = {(r['checkpoint_update'], identity(r)) for r in records}
    expected = {(u, key) for u in report['checkpoint_updates'] for key in observed}
    if len(keys) != len(records) or keys != expected:
        raise DashboardDataError(f'{path}: diagnosis checkpoints do not share exactly the aligned frames')
    if set(report['summary']) != {str(u) for u in report['checkpoint_updates']}:
        raise DashboardDataError(f'{path}: diagnosis checkpoint summary population disagrees')
    if any(r['split'] not in report['coverage'] for r in records):
        raise DashboardDataError(f'{path}: diagnosis contains an undeclared split')

    def means(rows, summary, prefix):
        _equal(len(rows), summary['frames'], f'{prefix} frames', path)
        _equal(len({r['group_id'] for r in rows}), summary['groups'], f'{prefix} groups', path)
        if not rows:
            raise DashboardDataError(f'{path}: diagnosis empty summary population')
        _equal([sum(r['position_abs_error'][i] for r in rows) / len(rows) for i in range(4)],
               summary['position_mae'], f'{prefix} position MAE', path)
        for raw, field in [('angle_error_deg', 'angle_mae_deg'), ('sin_cos_norm', 'sin_cos_norm_mean')]:
            recorded = summary[field] if field in summary else summary[raw]['mean']
            _equal(sum(r[raw] for r in rows) / len(rows), recorded, f'{prefix} {raw} mean', path)

    compact, group_sets = {}, {}
    for update, populations in report['summary'].items():
        if set(populations) != set(report['coverage']):
            raise DashboardDataError(f'{path}: diagnosis split summary population disagrees')
        compact[update] = {}
        for split, summary in populations.items():
            rows = [r for r in records if str(r['checkpoint_update']) == update and r['split'] == split]
            means(rows, summary, f'diagnosis {update}/{split}')
            counts = Counter(str(r['group_id']) for r in rows)
            coverage = report['coverage'][split]
            _equal(len(rows), coverage['sampled_frames'], 'diagnosis sampled frames', path)
            _equal(len(counts), coverage['sampled_groups'], 'diagnosis sampled groups', path)
            if dict(counts) != coverage['sampled_per_group_counts']:
                raise DashboardDataError(f'{path}: diagnosis sampled per-group counts disagree')
            group_sets[split] = set(counts)
            for field in ('pusher_distance', 'block_distance', 'pose_mse', 'image_mse'):
                recorded = summary[field]['mean'] if isinstance(summary[field], dict) else summary[field]
                _equal(sum(r[field] for r in rows) / len(rows), recorded, f'diagnosis {field} mean', path)
            for region, values in summary['regions'].items():
                totals = {k: sum(r['regions'][region][k] for r in rows)
                          for k in ('scalars', 'squared_error', 'white_squared_error')}
                scalars = totals['scalars']
                if scalars <= 0 or any(r['regions'][region]['scalars'] < 0 for r in rows):
                    raise DashboardDataError(f'{path}: diagnosis nonpositive region denominator')
                for field in ('scalars', 'squared_error'):
                    _equal(totals[field], values[field], f'diagnosis {region} {field}', path)
                _equal(totals['squared_error'] / scalars, values['mse'], f'diagnosis {region} MSE', path)
                _equal(totals['white_squared_error'] / scalars, values['white_image_mse'],
                       f'diagnosis {region} white MSE', path)
                _equal(scalars / (len(rows) * 64 * 64 * 3), values['pixel_fraction'],
                       f'diagnosis {region} pixel fraction', path)
            # The fractional foreground/background masks form an exact partition.
            for row in rows:
                if {'dynamic_foreground', 'background'} <= row['regions'].keys():
                    parts = [row['regions'][k] for k in ('dynamic_foreground', 'background')]
                    _equal(sum(p['scalars'] for p in parts), 64 * 64 * 3, 'diagnosis RGB denominator', path)
                    _equal(sum(p['squared_error'] for p in parts) / (64 * 64 * 3), row['image_mse'],
                           'diagnosis foreground/background MSE', path)
            for field in ('truth_angle_bin', 'wrap_bin', 'norm_bin', 'contact_gap_bin', 'goal_overlap_bin', 'group_id'):
                for key, value in summary.get(field, {}).items():
                    means([r for r in rows if str(r[field]) == key], value, f'diagnosis {field}/{key}')
            compact[update][split] = {k: summary[k] for k in
                ('frames', 'groups', 'position_mae', 'pose_mse', 'image_mse', 'regions')}
            compact[update][split].update({k: {'mean': summary[k]['mean']} for k in
                ('angle_error_deg', 'sin_cos_norm', 'pusher_distance', 'block_distance')})
    splits = list(group_sets) if independent_splits is None else independent_splits
    if any(group_sets[a] & group_sets[b] for i, a in enumerate(splits) for b in splits[i + 1:]):
        raise DashboardDataError(f'{path}: diagnosis sampled groups cross splits')
    coverage = {split: {k: value[k] for k in ('episodes', 'independent_groups', 'frames', 'sampled_frames', 'sampled_groups')}
                for split, value in report['coverage'].items()}
    metrics = {'checked_records': len(records), 'checked_alignment_frames': len(alignment),
               'protected_files_unchanged': len(before),
               'exact_source_pixel_frames': sum(r['exact_source_pixels'] for r in alignment),
               'exact_source_pose_frames': sum(r['exact_source_pose'] for r in alignment),
               'target_max_abs_difference': max(r['target_max_abs_difference'] for r in alignment),
               **_flat(coverage, 'coverage'), **_flat(compact, 'summary')}
    return compact, coverage, metrics


def _reconcile_coverage(report, path, root):
    """Share error arithmetic while keeping the saved synthetic grid a separate domain."""
    import numpy as np

    source, grid = report['source_population'], report['grid_population']
    parent_path, grid_path = root / source['parent_raw'], root / grid['source']
    for artifact, expected in ((parent_path, source['parent_sha256']), (grid_path, grid['sha256'])):
        if not artifact.is_file() or hashlib.sha256(artifact.read_bytes()).hexdigest() != expected:
            raise DashboardDataError(f'{path}: coverage input fingerprint differs from declared source')
    if report['test_population_used'] or not report['source_hashes_unchanged'] or not report['model_tensors_unchanged']:
        raise DashboardDataError(f'{path}: coverage scope/integrity receipt failed')
    parent = read_json(parent_path)
    if report['dataset_fingerprint'] != parent['dataset_fingerprint']:
        raise DashboardDataError(f'{path}: coverage dataset differs from original diagnosis')
    fields = ('split', 'source_episode', 'group_id', 'frame_index', 'source_row')
    identity = lambda row: tuple(row[k] for k in fields)
    parent_rows = [r for r in parent['records'] if r['checkpoint_update'] == parent['checkpoint_updates'][0]]
    expected_source = {identity(r) for r in parent_rows}
    if {identity(r) for r in source['identities']} != expected_source or len(source['identities']) != len(expected_source):
        raise DashboardDataError(f'{path}: coverage fixed source identities differ from parent diagnosis')
    _equal(len(expected_source), source['frames'], 'coverage source frame count', path)
    for split in ('train', 'validation'):
        _equal(sum(r['split'] == split for r in parent_rows), source[f'{split}_frames'], f'coverage {split} count', path)
    with np.load(grid_path, allow_pickle=False) as inputs:
        poses = inputs['poses']
        if inputs['frames'].shape != (grid['frames'], 64, 64, 3) or poses.shape != (grid['frames'], 5):
            raise DashboardDataError(f'{path}: coverage saved grid shape differs')
    _equal(grid['independent_position_settings'] * grid['angles_per_position'], grid['frames'], 'coverage grid shape', path)
    coverage = {k: dict(v) for k, v in parent['coverage'].items()}
    coverage['synthetic_grid'] = {'episodes': 0, 'independent_groups': grid['independent_position_settings'],
        'frames': grid['frames'], 'sampled_frames': grid['frames'], 'sampled_groups': grid['independent_position_settings'],
        'sampled_per_group_counts': {str(i): grid['angles_per_position'] for i in range(grid['independent_position_settings'])}}
    normalized, alignment = [], []
    for arm, model in report['models'].items():
        if report['protected_hashes_before'].get(model['checkpoint']) != model['sha256']:
            raise DashboardDataError(f'{path}: coverage model hash differs from protected checkpoint')
        rows = [r for r in report['records'] if r['arm'] == arm]
        source_rows = [r for r in rows if r['split'] != 'synthetic_grid']
        if {identity(r) for r in source_rows} != expected_source or len(source_rows) != len(expected_source):
            raise DashboardDataError(f'{path}: coverage arm does not use exactly the fixed source identities')
        synthetic = [r for r in rows if r['split'] == 'synthetic_grid']
        if sorted(r['frame_index'] for r in synthetic) != list(range(grid['frames'])):
            raise DashboardDataError(f'{path}: coverage arm does not use exactly the saved grid')
        for row in synthetic:
            i = row['frame_index']
            if (row['source_episode'] != -1 - i // grid['angles_per_position'] or row['source_row'] != -1 - i
                    or row['group_id'] != i // grid['angles_per_position']
                    or not np.array_equal(row['truth_pose_world'], poses[i])):
                raise DashboardDataError(f'{path}: coverage grid identity/pose differs from saved input')
        if any(r['checkpoint_update'] != model['update'] for r in rows):
            raise DashboardDataError(f'{path}: coverage checkpoint update differs from model provenance')
        normalized.extend({**r, 'checkpoint_update': arm} for r in rows)
        if not alignment:
            alignment = [{**r, 'exact_source_pixels': False, 'exact_source_pose': False,
                          'target_max_abs_difference': 0.} for r in rows]
        if arm.startswith('source_'):
            previous = {identity(r): r for r in parent['records'] if r['checkpoint_update'] == model['update']}
            for row in source_rows:
                for field in ('h_pose6', 'truth_pose_world', 'image_mse', 'position_abs_error', 'angle_error_deg'):
                    if field in previous[identity(row)]:
                        _equal(row[field], previous[identity(row)][field], f'coverage preserved source {field}', path)
    if len(normalized) != len(report['records']):
        raise DashboardDataError(f'{path}: coverage contains an undeclared model arm')
    # Arm names distinguish checkpoints with equal update numbers. Grid group ids
    # describe synthetic positions and are not CCHI configuration-group ids.
    compact, _, _ = _reconcile_diagnosis({**report, 'checkpoint_updates': list(report['models']),
        'coverage': coverage, 'records': normalized, 'alignment': alignment}, path,
        independent_splits=['train', 'validation'])
    for comparison, populations in report['comparisons'].items():
        current, base = ('coverage_final1000', 'source_final1000') if comparison == 'matched_final1000' else (
                         'coverage_selected500', 'source_selected200')
        for split, values in populations.items():
            a, b = compact[current][split], compact[base][split]
            for field in ('position_mae', 'pose_mse', 'image_mse'):
                _equal((np.asarray(a[field]) - b[field]).tolist(), values[field + '_delta'], 'coverage comparison ' + field, path)
            _equal(a['angle_error_deg']['mean'] - b['angle_error_deg']['mean'], values['angle_mae_deg_delta'],
                   'coverage angle comparison', path)
            _equal(a['regions']['dynamic_foreground']['mse'] - b['regions']['dynamic_foreground']['mse'],
                   values['foreground_mse_delta'], 'coverage foreground comparison', path)
    return compact, [parent_path, grid_path]


def _reconcile(report, directory):
    source = directory/'control_records.json'; controls = _list(source)
    if len({(r['case_id'],r['controller']) for r in controls}) != len(controls):
        raise DashboardDataError(f'{source}: duplicate controller/case identity')
    for controller, summary in report['control'].items():
        rows = [r for r in controls if r['controller'] == controller]
        _equal(len(rows),summary['count'],'controller count',source)
        successes = sum(r['success'] for r in rows)
        _equal(successes,summary['successes'],'controller successes',source)
        if rows: _equal(successes/len(rows),summary['success_rate'],'final success fraction',source)
        if 'any_time_successes' in summary:
            _equal(sum(r['any_time_success'] for r in rows),summary['any_time_successes'],'any-time successes',source)
        if 'failure_case_ids' in summary:
            if sorted(summary['failure_case_ids']) != sorted(r['case_id'] for r in rows if not r['success']):
                raise DashboardDataError(f'{source}: failure identities disagree')
    prediction_source = directory/'prediction_records.json'; predictions = _list(prediction_source)
    for method, horizons in report['prediction'].items():
        for horizon, summary in horizons.items():
            rows = [r for r in predictions if r['method'] == method and r['horizon'] == int(horizon)]
            _equal(len(rows),summary['count'],'prediction count',prediction_source)
            if not rows: continue
            for raw, field in [('h_position_abs_error','h_position_mae'),('r_position_abs_error','r_position_mae'),
                               ('h_angle_abs_error_deg','h_angle_mae_deg'),('r_angle_abs_error_deg','r_angle_mae_deg')]:
                values = [r[raw] for r in rows]
                average = [sum(v[i] for v in values)/len(values) for i in range(len(values[0]))] if isinstance(values[0],list) else sum(values)/len(values)
                _equal(average,summary[field],field,prediction_source)
            motions = [r['r_motion_abs_error'] for r in rows if r.get('r_motion_abs_error') is not None]
            _equal(len(motions),summary['motion_count'],'post-warm-up motion count',prediction_source)
            if motions:
                _equal([sum(v[i] for v in motions)/len(motions) for i in range(5)],summary['r_motion_mae'],'physical motion MAE',prediction_source)
            for field in ('latent_error','image_mse'):
                values = [r[field] for r in rows if r.get(field) is not None]
                if values: _equal(sum(values)/len(values),summary[field],field,prediction_source)
    return [source,prediction_source]


def collect_pusht_results(runs_root):
    results, notices = [], []
    def add(path,kind,status,metrics,context,sources,step=None,internals=None):
        results.append(RunResult(path.relative_to(runs_root).as_posix(),kind,status,step,metrics,context,
            tuple(p.relative_to(runs_root.parent).as_posix() for p in sources),
            max(p.stat().st_mtime for p in sources),internals=internals or {}))
    for path in sorted(runs_root.rglob('pusht_manifest.json')):
        manifest = read_json(path); directory = path.parent; sources = [path]
        rows = {}
        for name in ('training','validation'):
            ledger = directory/f'{name}.jsonl'
            rows[name] = read_jsonl(ledger) if ledger.exists() else ()
            if ledger.exists(): sources.append(ledger)
        result_path = directory/'pusht_result.json'
        result = read_json(result_path) if result_path.exists() else {}
        if result_path.exists(): sources.append(result_path)
        for name in ('status.json','failure.json'):
            if (directory/name).exists(): sources.append(directory/name)
        status = result.get('status','failed' if (directory/'failure.json').exists() else 'incomplete')
        selected = result.get('selected_update')
        step = result.get('global_update',rows['training'][-1]['step'] if rows['training'] else None)
        if selected is not None:
            matched = [r for r in rows['validation'] if r['step'] == selected]
            if not matched: raise DashboardDataError(f'{result_path}: selected validation checkpoint missing')
            for key,value in _flat(result.get('metrics',{})).items():
                _equal(value,_flat(matched[-1]).get(key),f'selected validation {key}',result_path)
        if status == 'completed' and rows['training'] and step != rows['training'][-1]['step']:
            raise DashboardDataError(f'{result_path}: completed update differs from raw training')
        metrics = {**numeric(result),**_flat(result.get('metrics',{}),'selected_validation')}
        for name in ('training','validation'):
            if rows[name]: metrics.update(_flat(rows[name][-1],f'latest_{name}'))
        context = {k:manifest.get(k) for k in ('stage','horizon','config','dependencies','dataset_fingerprint','versions','normalization')}
        context.update(task='CCHI PushT E/U/P; primitive 10 Hz absolute actions',selected_update=selected,
            checkpoint=result.get('checkpoint'),quality_gate=result.get('quality_gate'),smoke=result.get('smoke',manifest.get('config',{}).get('smoke',False)))
        if manifest.get('training_population'):
            context['training_population'] = manifest['training_population']
            context['population_presentations'] = result.get('population_presentations',
                rows['training'][-1].get('population_presentations', {}) if rows['training'] else {})
        add(directory,'pusht_training',status,metrics,context,sources,step,rows)
        if context['smoke']: notices.append(f'{directory.relative_to(runs_root)}: PushT SMOKE is execution evidence only')
        if status != 'completed': notices.append(f'{directory.relative_to(runs_root)}: PushT stage {status}; quality gates remain separate')
    for path in sorted(runs_root.rglob('metrics.json')):
        report = read_json(path)
        if report.get('schema_version') != 'pusht-eup-evaluation-v1': continue
        sources = [path]; directory = path.parent
        raw_ready = all((directory/name).exists() for name in ('control_records.json','prediction_records.json'))
        if raw_ready: sources += _reconcile(report,directory)
        elif report['status'] == 'completed': raise DashboardDataError(f'{path}: completed PushT raw records missing')
        for name in ('protocol.json','case_manifest.json','case_eligibility_records.json'):
            if (directory/name).exists(): sources.append(directory/name)
        panels=[]
        for visual in report.get('visuals',[]):
            from .ledger import resolve_evidence_path
            panel = resolve_evidence_path(visual['png'], runs_root)
            if not panel.exists(): raise DashboardDataError(f'{path}: PushT image missing {panel}')
            sources.append(panel); panels.append({**visual,'png':str(panel)})
        # Protocol sections retain their exact values without one oversized cell.
        context={f'protocol.{key}':value for key,value in report.get('protocol',{}).items()}
        context.update(case_selection=report.get('case_selection'),
            smoke=report.get('smoke'),raw_status=report.get('raw_status'),error=report.get('error'),
            goal_task='Block XY <20 world units AND circular angle <pi/9; not 95% coverage',
            oracle_budget='Replay is 5-interval reachability only, separate from fixed-budget controllers')
        add(directory,'pusht_evaluation' if raw_ready else 'pusht_reporting_failure',report['status'],
            _flat({k:report.get(k,{}) for k in ('prediction','control')}),context,sources,
            internals={'report':report,'panels':panels})
        if report['status'] != 'completed': notices.append(f'{directory.relative_to(runs_root)}: PushT evaluation/report {report["status"]}: {report.get("error","")}')
    for path in sorted(runs_root.rglob('raw.json')):
        report = read_json(path)
        if report.get('schema_version') == 'pusht-perception-coverage-comparison-v1':
            summary, extra_sources = _reconcile_coverage(report, path, runs_root.parent)
            panel = path.parent / 'examples.png'
            if not panel.is_file():
                raise DashboardDataError(f'{path}: coverage comparison image missing')
            context = {'diagnostic_only': True, 'test_population_used': False,
                'source_frames': report['source_population']['frames'],
                'source_train_frames': report['source_population']['train_frames'],
                'source_validation_frames': report['source_population']['validation_frames'],
                'synthetic_grid_frames': report['grid_population']['frames'],
                'grid_scope': report['grid_population']['scope'], 'dataset_fingerprint': report['dataset_fingerprint']}
            metrics = {'checked_records': len(report['records'])}
            for arm, populations in summary.items():
                context[f'model.{arm}'] = report['models'][arm]
                for split, values in populations.items():
                    context[f'{arm}.{split}'] = values
                    metrics.update(_flat({k: v for k, v in values.items() if k in
                        ('frames', 'groups', 'position_mae', 'angle_error_deg', 'pose_mse', 'image_mse')}, f'summary.{arm}.{split}'))
            for comparison, populations in report['comparisons'].items():
                for split, values in populations.items(): context[f'comparison.{comparison}.{split}'] = values
            add(path.parent, 'pusht_perception_coverage', report['status'], metrics, context, [path, panel, *extra_sources],
                internals={'summary': summary, 'panel': str(panel)})
            continue
        if report.get('schema_version') != 'pusht-perception-diagnosis-v1':
            continue
        summary, coverage, metrics = _reconcile_diagnosis(report, path)
        panels = [path.parent / name for name in ('diagnostic_metrics.png', 'reconstruction_and_labels.png')]
        if not all(p.is_file() for p in panels):
            raise DashboardDataError(f'{path}: diagnosis inspected image panel missing')
        context = {k: report.get(k) for k in ('schema_version', 'seed', 'device', 'threads', 'sampling',
                   'dataset_fingerprint', 'checkpoint_updates', 'script_sha256', 'region_definition')}
        context.update(diagnostic_only=True, protected_hashes_verified=True,
                       interpretation='Fixed grouped frame diagnosis; no optimizer updates or control-quality claim')
        add(path.parent, 'pusht_perception_diagnosis', report['status'], metrics, context, [path, *panels],
            internals={'summary': summary, 'coverage': coverage, 'panels': [str(p) for p in panels]})
    return results, notices


def add_pusht_views(runs,datasets,charts,tables,cards,chart,table,source_id):
    training=[r for r in runs if r.kind=='pusht_training']; evaluations=[r for r in runs if r.kind=='pusht_evaluation']
    diagnoses=[r for r in runs if r.kind=='pusht_perception_diagnosis']
    coverage_reports=[r for r in runs if r.kind=='pusht_perception_coverage']
    if not training and not evaluations and not diagnoses and not coverage_reports: return []
    number=lambda field:{'field':field,'type':'quantitative'}
    category=lambda field:{'field':field,'type':'nominal'}
    blocks=[{'id':'pusht_intro','type':'markdown','sourceId':source_id,
        'body':'## CCHI PushT E/U/P\n\nSeparate new-model experiment: absolute XY actions at 10 Hz. '
        'Training, prediction, and block-goal control are distinct evidence. Motion labels are backward displacements, not instantaneous velocities. '
        'Final controller success uses the complete fixed budget; any-time success is secondary. The 5-action replay oracle only checks reachability. '
        'SMOKE results and the retained LeWM PushT experiments do not establish this model’s control quality.'}]
    latest={}
    for run in training:
        key=run.context['stage']
        if key=='predictor':key+=f'_{run.context["horizon"]}'
        latest[key]=run
    for key,run in latest.items():
        name=f'pusht_training_{key}'
        population_note = ''
        population = run.context.get('training_population')
        if population:
            counts = run.context.get('population_presentations', {})
            unit = population['presentation_unit']
            population_note = (f' Population: {population["source_frames"]:,} source frames and '
                f'{population["supplement_frames"]:,} supplement frames. '
                f'{population.get("sampling", "Source data only")}. '
                f'Recorded {unit} presentations: source {counts.get("source", 0):,}, '
                f'supplement {counts.get("supplement", 0):,}; repeated presentations are not independent samples.')
        values=sorted([dict(step=row['step'],value=row['loss'])
                       for row in _sample(run.internals['training']) if 'loss' in row],key=lambda r:r['step'])
        if values:
            datasets[name]=values
            charts.append(chart(name,f'PushT {key}: training objective · {run.label}',
                'Task-specific recorded objective; no LeWM SIGReg meaning. At most 50 raw updates; ordered categorical step spacing. Training and validation grids are separate.' + population_note,
                name,'line' if len(values)>1 else 'bar',number('step'),number('value')))
        # Each validation checkpoint is one wide row, reused by all metric charts.
        # This preserves its exact step grid and keeps the portable dataset budget bounded.
        name=f'pusht_validation_objective_{key}';values=[]
        for row in _sample(run.internals['validation']):
            value={'step':row['step']}
            for field in ('loss','copy_loss'):
                if field in row:value[field]=row[field]
            for field,prefix,coordinates in [('position_mae','position',POSITIONS),('motion_mae','motion',MOTION)]:
                vector=row.get(field)
                if vector is None:continue
                if isinstance(vector[0],list):vector=vector[-1]
                value.update({f'{prefix}_{coordinate}':vector[i] for i,coordinate in enumerate(coordinates)})
            if 'motion_block_angle' in value:value['motion_block_angle_mrad']=value['motion_block_angle']*1000
            angle=row.get('angle_mae_deg')
            if angle is not None:value['angle_mae_deg']=angle[-1] if isinstance(angle,list) else angle
            values.append(value)
        if not values:continue
        datasets[name]=sorted(values,key=lambda r:r['step'])
        available=set().union(*(row.keys() for row in values))
        kind='line' if len(values)>1 else 'bar'
        groups=[('objective','validation objective',['loss','copy_loss'],'recorded objective'),
                ('position_mae','validation position MAE',[f'position_{c}' for c in POSITIONS],'world units'),
                ('angle_mae_deg','validation block-angle MAE',['angle_mae_deg'],'degrees'),
                ('motion_mae','validation displacement MAE',[f'motion_{c}' for c in POSITIONS],'world units per 0.1 second'),
                ('angular_motion_mae','validation angular-displacement MAE',['motion_block_angle_mrad'],'milliradians per 0.1 second (raw radians ×1000)')]
        for suffix,label,fields,units in groups:
            fields=[f for f in fields if f in available]
            if not fields:continue
            identifier=name if suffix=='objective' else f'pusht_validation_{suffix}_{key}'
            y={'fields':fields,'type':'quantitative','label':units} if len(fields)>1 else number(fields[0])
            charts.append(chart(identifier,f'PushT {key}: {label} · {run.label}',
                f'Exact validation checkpoints in {units}; predictor physical metrics show the final trained horizon. Ordered categorical step spacing; dots are recorded values and connecting curves are visual guides.',
                name,kind,number('step'),y))
            if len(fields)>1:
                charts[-1]['palette']={'kind':'categorical','name':'PATH-WM blue-orange'}
                charts[-1]['legend']={'position':'bottom','sort':'spec'}
                charts[-1]['surface']['interactiveLegend']=True
            if kind=='line':charts[-1]['settings']['showPoints']='always'
    datasets['pusht_control_detail']=[];datasets['pusht_prediction_detail']=[]
    for run in evaluations:
        report=run.internals['report']
        for controller,summary in report['control'].items():
            datasets['pusht_control_detail'].append(dict(run=run.label,controller=controller,
                budget_scope='5-action oracle reachability' if controller=='replay' else 'fixed controller budget',
                **{k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in summary.items() if k!='failure_case_ids'}))
        for method,horizons in report['prediction'].items():
            for horizon,summary in horizons.items():
                datasets['pusht_prediction_detail'].append(dict(run=run.label,method=method,horizon=int(horizon),
                    **{k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in summary.items()}))
    if evaluations:
        run=evaluations[-1];report=run.internals['report']
        control=[row for row in datasets['pusht_control_detail'] if row['run']==run.label and row['controller']!='replay']
        if control:
            name='pusht_final_control';datasets[name]=control
            charts.append(chart(name,f'PushT final block-goal success · {run.label}',
                'Final success after the same fixed primitive-action budget. Excludes the separately reported 5-action replay oracle; no 95% coverage claim.',
                name,'bar',category('controller'),number('success_rate'),tooltip=[number('successes'),number('count')]))
        name='pusht_matched_horizons'
        values=[]
        for method,horizons in report['prediction'].items():
            if method=='actual':continue
            for h,row in horizons.items():
                value=dict(horizon=int(h),method=method,count=row['count'])
                for field in ('latent_error','h_angle_mae_deg','r_angle_mae_deg','image_mse'):
                    if row.get(field) is not None:value[field]=row[field]
                for prefix in ('h','r'):
                    for i,coordinate in enumerate(POSITIONS):
                        if row.get(f'{prefix}_position_mae') is not None:
                            value[f'{prefix}_{coordinate}']=row[f'{prefix}_position_mae'][i]
                values.append(value)
        if values:
            datasets[name]=sorted(values,key=lambda row:(row['horizon'],row['method']))
            fields=[('latent_error','variance-normalized latent MSE'),('h_angle_mae_deg','H block-angle MAE (degrees)')]
            fields += [(f'h_{c}',f'H {c} MAE (world units)') for c in POSITIONS]
            for field,label in fields:
                if not any(field in row for row in values):continue
                charts.append(chart(f'pusht_horizon_{field}',f'PushT matched rollout: {label}',
                    'Prediction, copy and reset share exact windows/actions. One horizon step is 0.1 seconds.',name,'line',
                    number('horizon'),number(field),color=category('method'),tooltip=[number('count')]))
        for visual in run.internals['panels'][:1]:
            encoded=base64.b64encode(Path(visual['png']).read_bytes()).decode()
            blocks.append({'id':'pusht_qualitative','type':'html','layout':'full','body':
                '<h3>PushT actual / reconstruction / imagination</h3>'
                f'<img style="max-width:100%;height:auto" alt="Matched PushT observations and imagined frames" src="data:image/png;base64,{encoded}">'
                f'<p>{html.escape(visual.get("action_alignment","Matched recorded actions"))}. Separate real/imagined R appears in the raw GIF.</p>'})
    for name,title,columns,subtitle in [
        ('pusht_control_detail','PushT exact controller and oracle results',
         [('run','Run'),('controller','Controller'),('budget_scope','Budget meaning'),('count','Cases'),('successes','Final successes'),
          ('success_rate','Final fraction'),('any_time_successes','Any-time successes'),('mean_episode_length','Executed intervals'),
          ('planning_failures','Planning failures'),('latency_median_ms','Decision median ms'),('latency_p95_ms','Decision p95 ms')],
         'Replay uses five privileged actions to verify reachable goal construction. All other controllers share the declared fixed budget. Raw case identities remain in control_records.json.'),
        ('pusht_prediction_detail','PushT exact pose and observable-motion errors',
         [('run','Run'),('method','Method'),('horizon','Horizon'),('count','Pose frames'),('motion_count','Motion frames'),
          ('h_position_mae','H position MAE [pusherXY,blockXY]'),('h_angle_mae_deg','H angle MAE degrees'),
          ('r_position_mae','R position MAE'),('r_angle_mae_deg','R angle MAE degrees'),
          ('r_motion_mae','R displacement MAE [dx4,dangle]'),('latent_error','Normalized latent MSE'),('image_mse','RGB MSE')],
         'Actual H/R pose uses all recorded held-out frames; motion excludes the first two. dx has world units and dangle radians per 0.1 second, not simulator velocity.')]:
        if datasets[name]:tables.append(table(name,title,columns,'run',subtitle))
    if diagnoses:
        run = diagnoses[-1]
        name = 'pusht_diagnosis_summary'
        values = []
        for update, splits in run.internals['summary'].items():
            for split, summary in splits.items():
                value = {'population': f'{update} / {split}', 'checkpoint_update': int(update), 'split': split,
                         'frames': summary['frames'], 'groups': summary['groups'],
                         'angle_mae_deg': summary['angle_error_deg']['mean'],
                         'sin_cos_norm': summary['sin_cos_norm']['mean'], 'image_mse': summary['image_mse'],
                         'pose_mse': summary['pose_mse']}
                value.update({f'position_{c}': summary['position_mae'][i] for i, c in enumerate(POSITIONS)})
                for region, metrics in summary['regions'].items():
                    value.update({f'{region}_{key}': item for key, item in metrics.items()})
                values.append(value)
        datasets[name] = values
        coverage = run.internals['coverage']
        scope = '; '.join(f'{split}: {c["sampled_frames"]} sampled frames across {c["sampled_groups"]} groups'
                          for split, c in coverage.items())
        endpoints = sorted(run.internals['summary'], key=int)
        comparisons = []
        for split in ('train', 'validation'):
            if all(split in run.internals['summary'][u] for u in (endpoints[0], endpoints[-1])):
                first, last = (run.internals['summary'][u][split] for u in (endpoints[0], endpoints[-1]))
                comparisons.append(f'{split} angle MAE {first["angle_error_deg"]["mean"]:.2f}° → '
                                   f'{last["angle_error_deg"]["mean"]:.2f}°')
        body = (f'<h3>PushT perception diagnosis</h3><p>From update {endpoints[0]} to {endpoints[-1]}: '
                f'{html.escape("; ".join(comparisons))}. '
                'Fixed frame comparisons separate pose generalization from reconstruction. '
                'These checkpoint comparisons are diagnostic only; they do not establish control quality.</p>'
                f'<p>{html.escape(scope)}. Both checkpoints use exactly the same source frames within each split. '
                'Angle error is circular MAE in degrees; position error uses world units. Region RGB MSE divides '
                'summed squared error by summed fractional RGB scalar weights, rather than averaging per-frame ratios.</p>')
        for panel, caption in zip(run.internals['panels'], (
            'Angle, position and image-region diagnostics. Full-source angle histograms and initial poses are descriptive coverage context.',
            'Actual source RGB, geometry labels and reconstructions from the two saved checkpoints. Better reconstruction alone does not establish pose generalization.')):
            encoded = base64.b64encode(Path(panel).read_bytes()).decode()
            body += (f'<p>{html.escape(caption)}</p><img style="max-width:100%;height:auto" '
                     f'alt="{html.escape(caption)}" src="data:image/png;base64,{encoded}">')
        body += (f'<p>Reconciled {run.metrics["checked_records"]:,} prediction records and '
                 f'{run.metrics["checked_alignment_frames"]:,} alignment frames, including angle/position means, '
                 'sampled counts and regional error denominators. '
                 f'{run.metrics["protected_files_unchanged"]} protected before/after file hashes agree. '
                 'These associations do not identify a causal failure mechanism; follow-up training requires its own declared comparison.</p>')
        blocks.append({'id': 'pusht_diagnosis_panels', 'type': 'html', 'layout': 'full', 'sourceId': source_id, 'body': body})
        for field, title in [('angle_mae_deg', 'Circular block-angle MAE (degrees)'),
                             ('image_mse', 'Whole-image RGB MSE')]:
            charts.append(chart(f'pusht_diagnosis_{field}', f'PushT diagnosis: {title}',
                'Paired saved checkpoints on fixed group-balanced train/validation frames; descriptive diagnosis only.',
                name, 'bar', category('population'), number(field), tooltip=[number('frames'), number('groups')]))
        regions = [f'{region}_mse' for region in ('pusher', 'block', 'background')
                   if all(f'{region}_mse' in row for row in values)]
        if regions:
            charts.append(chart('pusht_diagnosis_regions', 'PushT diagnosis: regional RGB reconstruction MSE',
                'Each region uses summed fractional RGB scalar weights. Background is not pooled with foreground; '
                'the all-white reference, region numerators and denominators remain in exact values.',
                name, 'bar', category('population'), {'fields': regions, 'type': 'quantitative', 'label': 'Regional RGB MSE'}))
            charts[-1]['palette'] = {'kind': 'categorical', 'name': 'PATH-WM blue-orange'}
            charts[-1]['legend'] = {'position': 'bottom', 'sort': 'spec'}
            charts[-1]['surface']['interactiveLegend'] = True
    if coverage_reports:
        run = coverage_reports[-1]
        body = ('<h3>PushT perception coverage comparison</h3>'
            f'<p>Four saved model arms share {run.context["source_train_frames"]} fixed train frames and '
            f'{run.context["source_validation_frames"]} fixed validation frames. '
            f'The {run.context["synthetic_grid_frames"]} saved synthetic stress frames are a separate out-of-source '
            'diagnostic population. Improvement on that grid does not establish CCHI generalization or useful control.</p>'
            '<div style="overflow-x:auto"><table><thead><tr><th>Model</th><th>Population</th>'
            '<th>Frames</th><th>Angle MAE (degrees)</th><th>Pose-vector MSE</th><th>RGB MSE</th></tr></thead><tbody>')
        for arm, populations in run.internals['summary'].items():
            for split, values in populations.items():
                body += (f'<tr><td>{html.escape(arm.replace("_", " "))}</td><td>{html.escape(split.replace("_", " "))}</td>'
                    f'<td>{values["frames"]}</td><td>{values["angle_error_deg"]["mean"]:.5g}</td>'
                    f'<td>{values["pose_mse"]:.6g}</td><td>{values["image_mse"]:.6g}</td></tr>')
        body += ('</tbody></table></div><p>Selected checkpoints and equal-budget final checkpoints answer different '
                 'comparisons. Pose-vector MSE includes the predicted sine/cosine magnitude; circular angle error '
                 'and the four world-position MAEs remain separate in the exact context. Regional errors use '
                 'summed fractional RGB scalar weights. The original diagnosis and its two images remain intact.</p>')
        encoded = base64.b64encode(Path(run.internals['panel']).read_bytes()).decode()
        body += (f'<img style="max-width:100%;height:auto" alt="Fixed source and separate synthetic stress examples" '
                 f'src="data:image/png;base64,{encoded}"><p>This source-reconciled comparison is diagnostic only. '
                 'All original source identities, grid inputs, model hashes and protected before/after hashes are checked.</p>')
        blocks.append({'id': 'pusht_coverage_comparison_panel', 'type': 'html', 'layout': 'full',
                       'sourceId': source_id, 'body': body})
    return blocks
