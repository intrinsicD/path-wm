"""Comparable capability tables from hash-verified overnight analysis records."""
from pathlib import Path
import json
import numpy as np
from .ledger import DashboardDataError

PROJECT = Path(__file__).resolve().parents[1]


def source_file(run, basename):
    names = [name for name in run.source_paths if Path(name).name == basename]
    if len(names) != 1: raise DashboardDataError(f'{run.label}: expected one {basename}')
    for path in (PROJECT / names[0], PROJECT / 'runs' / names[0]):
        if path.is_file(): return path
    raise DashboardDataError(f'{run.label}: missing {basename}')


def verify_pose_record(path, expected):
    raw = dict(np.load(path)); prediction, target = raw['predictions'], raw['targets']
    xy = np.abs(prediction[:, :4] - target[:, :4]) * 512
    angle = np.arctan2(prediction[:, 4], prediction[:, 5]) - np.arctan2(target[:, 4], target[:, 5])
    angle = np.abs(np.arctan2(np.sin(angle), np.cos(angle))) * 180 / np.pi
    q = max(xy.mean(0).max() / 8, angle.mean() / 10)
    if not np.allclose(xy.mean(0), expected['position_mae'], atol=1e-11) or not np.isclose(q, expected['q'], atol=1e-12):
        raise DashboardDataError(f'{path}: raw pose predictions disagree with summary')


def add_perception_views(runs, datasets, charts, tables, chart, table, source_id):
    candidates = [r for r in runs if r.kind == 'curriculum_analysis' and 'development' not in Path(r.label).parts]
    rows = []; semantics = []; fresh = []
    for run in candidates:
        purpose = run.context.get('purpose')
        if purpose in ('Frozen representation and independent typed readouts', 'Mixed-supervision encoder continuation'):
            path = source_file(run, 'evaluation.json'); evaluation = json.loads(path.read_text())
            manifest = json.loads(source_file(run, 'curriculum_manifest.json').read_text())
            result = json.loads(source_file(run, 'curriculum_result.json').read_text())
            p, c = evaluation['selected']['test']['pusht'], evaluation['selected']['test']['coco']
            verify_pose_record(path.parent / 'selected_pusht_test_errors.npz', p)
            raw = dict(np.load(path.parent / 'selected_coco_test_errors.npz'))
            if not np.isclose(raw['iou'][raw['has_valid'].astype(bool)].mean(), c['iou'], atol=1e-8):
                raise DashboardDataError(f'{path}: mask IoU mismatch')
            config = manifest['config']; family = 'Frozen package' if 'encoder' in config else 'Encoder continuation'
            arm = config.get('encoder', config.get('kind')); seed = config['seed']
            rows.append(dict(study=family, arm=arm, seed=seed, label=f'{arm} · {seed}',
                status=result['status'], updates=result['step'], selected_update=result['selected_step'],
                validation_q=result['selected']['q'], test_q=p['q'], angle_mae_deg=p['angle_mae_deg'],
                pusher_x_mae=p['position_mae'][0], pusher_y_mae=p['position_mae'][1],
                block_x_mae=p['position_mae'][2], block_y_mae=p['position_mae'][3],
                coco_mse=c['image_mse'], pusht_mse=p['image_mse'], mask_iou=c['iou'],
                mask_baseline=c['full_iou'], test_pose_frames=p['frames'], test_coco_frames=c['frames'],
                fitting_seconds=result['elapsed_seconds'], parameters=sum(manifest['parameters'].values()),
                source=str(path.relative_to(PROJECT))))
        elif purpose == 'Generic category-accessibility readout':
            from world_model.curriculum.perception_semantics import average_precision
            path = source_file(run, 'result.json'); r = json.loads(path.read_text())
            manifest = json.loads(source_file(run, 'semantic_manifest.json').read_text()); config = manifest['config']
            raw = dict(np.load(path.parent / 'selected_test_scores.npz'))
            aps = [average_precision(raw['targets'][:, i], raw['scores'][:, i], raw['known'][:, i]) for i in range(raw['targets'].shape[1])]
            calculated = float(np.mean([v for v in aps if v is not None])); test = r['evaluation']['selected']['test']
            if not np.isclose(calculated, test['macro_ap'], atol=1e-12):
                raise DashboardDataError(f'{path}: semantic AP disagrees with scores/unknown mask')
            semantics.append(dict(arm=config['encoder'], seed=config['seed'], label=f'{config["encoder"]} · {config["seed"]}',
                status=r['status'], updates=r['step'], selected_update=r['selected']['step'],
                validation_ap=r['selected']['macro_ap'], test_ap=test['macro_ap'], baseline_ap=test['baseline_macro_ap'],
                covered_classes=test['covered_classes'], retained_classes=test['classes'], frames=test['frames'],
                device=r['device'], fitting_seconds=r['seconds'], source=str(path.relative_to(PROJECT))))
        elif purpose == 'Fresh frozen perception evaluation':
            path = source_file(run, 'evaluation.json'); r = json.loads(path.read_text()); m = r['metrics']['fresh']
            verify_pose_record(path.parent / 'fresh_errors.npz', m)
            fresh.append(dict(arm=r['encoder'], seed=r['seed'], label=f'{r["encoder"]} · {r["seed"]}',
                q=m['q'], case_q_p95=m['case_q_p95'], per_case_pass=m['per_case_tolerance_pass'],
                angle_mae_deg=m['angle_mae_deg'], frames=m['frames'],
                original_calibration_q=r['metrics']['calibration_original']['q'],
                rendered_calibration_q=r['metrics']['calibration_rendered']['q'], source=str(path.relative_to(PROJECT))))
    number = lambda field: {'field': field, 'type': 'quantitative'}
    category = lambda field: {'field': field, 'type': 'nominal'}
    for family, key in (('Frozen package', 'package'), ('Encoder continuation', 'continuation')):
        data = sorted([r for r in rows if r['study'] == family], key=lambda r: (r['seed'], r['arm']))
        if not data: continue
        name = f'perception_{key}'; datasets[name] = data
        for metric, title, unit in [('test_q', 'Pose readiness on reused held-outs', 'q; lower is better; target≤1'),
                                     ('mask_iou', 'COCO foreground accessibility', 'mean per-image IoU; higher is better'),
                                     ('coco_mse', 'COCO reconstruction', 'RGB MSE in[0,1]; lower is better')]:
            charts.append(chart(f'{name}_{metric}', f'{title} · {family}',
                f'{unit}. Individual paired seeds; pose-selected snapshots. 2506 PushT or512 COCO test frames. '
                'Only recorded completed/evaluated fits appear; no confidence-interval claim.',
                name, 'bar', category('seed'), number(metric), color=category('arm'), group_mode='grouped',
                reference_lines=[{'axis':'y','value':1,'label':'readiness target','lineStyle':'dashed'}] if metric=='test_q' else None))
        tables.append(table(name, f'{family}: exact capabilities',
            [('arm','Arm'),('seed','Seed'),('status','Status'),('selected_update','Selected update'),
             ('validation_q','Validation q'),('test_q','Test q'),('pusher_x_mae','Pusher x MAE'),
             ('pusher_y_mae','Pusher y MAE'),('block_x_mae','Block x MAE'),('block_y_mae','Block y MAE'),
             ('angle_mae_deg','Angle MAE°'),('mask_iou','Mask IoU'),('coco_mse','COCO MSE'),
             ('pusht_mse','PushT MSE'),('fitting_seconds','Fit seconds'),('source','Raw evaluation')],
            'seed', 'Position in world units. All heads at one pose-selected snapshot; endpoints remain in raw evaluations.'))
    if semantics:
        name = 'perception_semantics'; datasets[name] = semantics
        charts.append(chart(name, 'Category presence on COCO',
            'Macro average precision over supported test classes; higher is better. Three paired head seeds, frozen encoders. '
            '78 training-supported classes; crowd unknowns excluded. Presence does not establish localization.',
            name, 'bar', category('seed'), number('test_ap'), color=category('arm'), group_mode='grouped',
            reference_lines=[{'axis':'y','value':semantics[0]['baseline_ap'],'label':'constant scores','lineStyle':'dashed'}]))
        tables.append(table(name, 'Exact category accessibility',
            [('arm','Encoder'),('seed','Seed'),('selected_update','Selected update'),('validation_ap','Validation AP'),
             ('test_ap','Test AP'),('baseline_ap','Constant-score AP'),('covered_classes','Covered classes'),
             ('frames','Test images'),('device','Fit device'),('fitting_seconds','Fit seconds'),('source','Raw scores')],
            'seed', 'The simple mean-pooled classifier tests accessibility under this head; a different readout could recover other information.'))
    if fresh:
        name = 'perception_fresh'; datasets[name] = fresh
        charts.append(chart(name, 'Per-case tolerance pass on fresh simulator observations',
            'Fraction satisfying all four8-unit coordinate tolerances and10-degree angle tolerance. 512 random reset cases, no adaptation. '
            'Includes the declared near-match cases. This is perception, not control success.',
            name, 'bar', category('seed'), number('per_case_pass'), color=category('arm'), group_mode='grouped'))
        tables.append(table(name, 'Fresh stress cohort and renderer calibration',
            [('arm','Encoder'),('seed','Seed'),('q','Mean q'),('per_case_pass','Per-case pass fraction'),
             ('case_q_p95','95th percentile case q'),('angle_mae_deg','Angle MAE°'),
             ('original_calibration_q','Source calibration q'),('rendered_calibration_q','Rendered calibration q'),('source','Raw evaluation')],
            'seed', 'Primary512-case stress cohort and separate40-pair source/render calibration. Mean q≤1 does not imply every case passes.'))
    return bool(rows or semantics or fresh)
