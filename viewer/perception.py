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
    if 'group_position_mae' in expected:
        groups=raw['groups']; keys=np.unique(groups)
        gx=np.array([xy[groups==g].mean(0) for g in keys]).mean(0)
        ga=np.array([angle[groups==g].mean() for g in keys]).mean()
        if not np.allclose(gx,expected['group_position_mae'],atol=1e-11) or not np.isclose(ga,expected['group_angle_mae_deg'],atol=1e-11):
            raise DashboardDataError(f'{path}: group-balanced pose metrics disagree')


def add_perception_views(runs, datasets, charts, tables, chart, table, source_id):
    candidates = [r for r in runs if r.kind == 'curriculum_analysis' and 'development' not in Path(r.label).parts]
    rows = []; semantics = []; fresh = []; decoders = []; reliance = []; localization = []; interventions = []; directional = []; attention = []
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
                group_balanced_q=max(max(p['group_position_mae'])/8,p['group_angle_mae_deg']/10),groups=p['groups'],
                angle_norm_mean=p['angle_norm_mean'],near_zero_angle_count=p['near_zero_angle_norm_count'],
                coco_mse=c['image_mse'], pusht_mse=p['image_mse'], mask_iou=c['iou'],
                mask_baseline=c['full_iou'], test_pose_frames=p['frames'], test_coco_frames=c['frames'],
                fitting_seconds=result['elapsed_seconds'], parameters=sum(manifest['parameters'].values()),
                source=str(path.relative_to(PROJECT))))
        elif purpose in ('Read-only cross-scale attention intervention','Read-only directional attention intervention'):
            path=source_file(run,'evaluation.json'); r=json.loads(path.read_text())
            if r['development']: continue
            for seed,conditions in r['evaluations'].items():
                for condition,m in conditions.items():
                    verify_pose_record(path.parent/f'{condition}_{seed}_errors.npz',m)
                    destination=directional if r.get('directional') else interventions
                    destination.append(dict(seed=int(seed),condition=condition,q=m['q'],
                        per_case_pass=m['per_case_tolerance_pass'],image_mse=m['image_mse'],source=str(path.relative_to(PROJECT))))
            for direction,m in ({} if r.get('directional') else r['attention']).items():
                for head,h in enumerate(m['per_head_mean_entropy']):
                    attention.append(dict(direction=direction,head=head+1,entropy=h,
                        mean_maximum_probability=m['per_head_mean_maximum_probability'][head],
                        keys=m['keys'],relative_uniform_difference=m['relative_uniform_difference'],source=str(path.relative_to(PROJECT))))
        elif purpose == 'Frozen decoder input reliance':
            path=source_file(run,'evaluation.json'); e=json.loads(path.read_text())
            if e['development']: continue
            for row in e['evaluations']:
                raw=dict(np.load(path.parent/f'{row["arm"]}_{row["seed"]}_{row["condition"]}_errors.npz'))
                if not np.isclose(raw['image_mse'].mean(),row['image_mse'],atol=1e-12) or not np.isclose(raw['iou'][raw['has_valid'].astype(bool)].mean(),row['iou'],atol=1e-12):
                    raise DashboardDataError(f'{path}: decoder reliance summaries disagree')
                reliance.append(dict(arm=row['arm'],seed=row['seed'],condition=row['condition'],frames=row['frames'],
                    image_mse=row['image_mse'],mask_iou=row['iou'],image_mse_ratio=row['image_mse_ratio'],iou_delta=row['iou_delta'],
                    status=e['status'],source=str(path.relative_to(PROJECT))))
        elif purpose == 'Location-distribution supervision':
            path=source_file(run,'evaluation.json'); e=json.loads(path.read_text())
            manifest=json.loads(source_file(run,'curriculum_manifest.json').read_text())
            result=json.loads(source_file(run,'curriculum_result.json').read_text()); config=manifest['config']
            m=e['selected']['test']; f=e['fresh']
            verify_pose_record(path.parent/'selected_test_errors.npz',m)
            verify_pose_record(path.parent/'fresh_errors.npz',f)
            raw=dict(np.load(path.parent/'fresh_errors.npz'))
            if not np.isclose((raw['case_q']<=1).mean(),f['per_case_tolerance_pass'],atol=1e-12):
                raise DashboardDataError(f'{path}: localization pass fraction mismatch')
            localization.append(dict(arm=config['encoder'],seed=config['seed'],status=result['status'],
                updates=result['step'],selected_update=result['selected_step'],test_q=m['q'],fresh_q=f['q'],
                fresh_pass=f['per_case_tolerance_pass'],case_q_p95=f['case_q_p95'],angle_mae_deg=f['angle_mae_deg'],
                boundary_frames=f['boundary_slices']['within_48_units']['frames'],
                boundary_pass=f['boundary_slices']['within_48_units']['per_case_pass'],
                interior_pass=f['boundary_slices']['interior']['per_case_pass'],
                location_entropy=f['location_entropy_mean'],target_support_mass=f['target_support_mass_mean'],
                fitting_seconds=result['elapsed_seconds'],source=str(path.relative_to(PROJECT))))
        elif purpose == 'Task-conditioned dense decoding':
            path=source_file(run,'evaluation.json'); e=json.loads(path.read_text())
            manifest=json.loads(source_file(run,'curriculum_manifest.json').read_text())
            result=json.loads(source_file(run,'curriculum_result.json').read_text()); config=manifest['config']
            for domain in ('coco','pusht'):
                raw=dict(np.load(path.parent/f'{domain}_test_errors.npz'))
                if not np.isclose(raw['image_mse'].mean(),e['test'][domain]['image_mse'],atol=1e-12):
                    raise DashboardDataError(f'{path}: decoder RGB mismatch')
                if domain=='coco' and not np.isclose(raw['iou'][raw['has_valid'].astype(bool)].mean(),e['test']['coco']['iou'],atol=1e-12):
                    raise DashboardDataError(f'{path}: decoder IoU mismatch')
            c,p=e['test']['coco'],e['test']['pusht']
            decoders.append(dict(arm=config['kind'],seed=config['seed'],status=result['status'],updates=result['step'],
                mask_iou=c['iou'],mask_dice=c['dice'],coco_mse=c['image_mse'],pusht_mse=p['image_mse'],
                parameters=manifest['parameters']['decoder'],fitting_seconds=result['elapsed_seconds'],
                rgb_ms_per_image=e['timing']['rgb_milliseconds_per_image'],mask_ms_per_image=e['timing']['mask_milliseconds_per_image'],
                both_ms_per_image=e['timing']['both_milliseconds_per_image'],source=str(path.relative_to(PROJECT))))
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
    if decoders:
        name='perception_decoders'; datasets[name]=decoders
        for field,title,unit in [('mask_iou','COCO foreground decoding','mean per-image IoU; higher is better'),
                                ('coco_mse','COCO reconstruction','RGB MSE in[0,1]; lower is better'),
                                ('pusht_mse','PushT reconstruction','RGB MSE in[0,1]; lower is better')]:
            charts.append(chart(name+'_'+field,title+' · local inputs and task conditioning',
                unit+'. Fixed update endpoints, three paired seeds. Frozen final ViT features, shared dense trunk. '
                'Raw input has fewer parameters and different preprocessing; no future prediction claim.',
                name,'bar',category('seed'),number(field),color=category('arm'),group_mode='grouped'))
        tables.append(table(name,'Exact fixed-endpoint decoder comparison',
            [('arm','Arm'),('seed','Seed'),('status','Status'),('updates','Updates'),('mask_iou','Mask IoU'),
             ('mask_dice','Mask Dice'),('coco_mse','COCO MSE'),('pusht_mse','PushT MSE'),('parameters','Parameters'),
             ('fitting_seconds','Fit seconds'),('both_ms_per_image','Both outputs ms/image'),('source','Raw evaluation')],
            'seed','Decoder timing uses prepared inputs, batch32. Unconditioned trunks are reused for both outputs; conditioned outputs use two passes.'))
    if reliance:
        name='perception_decoder_reliance';datasets[name]=reliance
        tables.append(table(name,'Frozen decoder reliance on supplied inputs',
            [('arm','Decoder'),('seed','Seed'),('condition','Input intervention'),('frames','Cases'),('mask_iou','Mask IoU'),
             ('iou_delta','IoU change'),('image_mse','RGB MSE'),('image_mse_ratio','RGB MSE / normal'),('source','Raw evaluation')],
            'arm','CPU baselines checked against GPU endpoints. Donor local/context features come from the same fixed permutation, targets stay fixed. Evaluation perturbations are not retrained ablations.'))
    if localization:
        name='perception_localization'; datasets[name]=localization
        charts.append(chart(name,'Fresh geometry after location-distribution supervision',
            'Fraction passing all coordinate/angle tolerances on512 fresh cases. Same frozen P1 encoders and pose-head topology; '
            'only the objective changes. Adaptive follow-up; read against the original P1 rows and paired draw checks.',
            name,'bar',category('seed'),number('fresh_pass'),color=category('arm'),group_mode='grouped'))
        tables.append(table(name,'Exact geometry objective follow-up',
            [('arm','Encoder'),('seed','Seed'),('selected_update','Selected update'),('test_q','Test q'),('fresh_q','Fresh q'),
             ('fresh_pass','Fresh case pass'),('case_q_p95','95th percentile q'),('angle_mae_deg','Angle MAE°'),
             ('boundary_frames','Boundary cases'),('boundary_pass','Boundary pass'),('interior_pass','Interior pass'),
             ('location_entropy','Map entropy'),('target_support_mass','Target support mass'),('source','Raw evaluation')],
            'seed','Bilinear target supervision fixes the map convention; entropy and support mass check its effect. Probabilities are not calibrated uncertainty.'))
    if interventions:
        name='perception_attention_interventions'; datasets[name]=interventions
        charts.append(chart(name,'Reliance on cross-scale attention after training',
            'Fresh512-case perception q; lower is better, target≤1. Both directions changed simultaneously. '
            'Uniform keeps value/output projections; zero removes branch outputs. These are evaluation-time perturbations, not retrained controls.',
            name,'bar',category('seed'),number('q'),color=category('condition'),group_mode='grouped',
            reference_lines=[{'axis':'y','value':1,'label':'readiness target','lineStyle':'dashed'}]))
        tables.append(table(name,'Exact attention intervention outcomes',
            [('seed','Readout seed'),('condition','Condition'),('q','Mean q'),('per_case_pass','Per-case pass fraction'),
             ('image_mse','RGB MSE'),('source','Raw predictions')],'seed','Same source encoder and frozen heads; no optimization during interventions.'))
        name='perception_attention_entropy'; datasets[name]=attention
        charts.append(chart(name,'Normalized cross-scale entropy, separately by head',
            'Mean over all512 fresh images and query positions of H(p)/log(number of keys). Key-axis probabilities are computed independently per head before averaging. '
            'Values near1 do not establish that attention is unused.',name,'bar',category('head'),number('entropy'),
            color=category('direction'),group_mode='grouped'))
        tables.append(table(name,'Exact attention statistics',
            [('direction','Direction'),('head','Head'),('keys','Keys'),('entropy','Normalized entropy'),
             ('mean_maximum_probability','Mean largest key probability'),('relative_uniform_difference','Uniform-output difference / output norm'),
             ('source','Raw diagnostic')],'direction','Entropy is not a causal usefulness measure; matched intervention outcomes answer a separate reliance question.'))
    if directional:
        name='perception_attention_directional'; datasets[name]=directional
        charts.append(chart(name,'Direction-specific attention intervention',
            'Fresh512-case q; lower is better. Fine changes only fine-from-coarse; coarse changes only coarse-from-fine. '
            'Frozen readouts, three paired seeds; perturbation sensitivity does not identify retrained architecture quality.',
            name,'bar',category('seed'),number('q'),color=category('condition'),group_mode='grouped'))
        tables.append(table(name,'Exact direction-specific outcomes',
            [('seed','Seed'),('condition','Changed receiving direction'),('q','Mean q'),
             ('per_case_pass','Per-case pass fraction'),('image_mse','RGB MSE'),('source','Raw predictions')],
            'seed','Uniform weights keep projected values; zero removes the receiving branch output.'))
    return bool(rows or semantics or fresh or decoders or reliance or localization or interventions or directional)
