"""Reconcile declared overnight fits and produce auditable descriptive comparisons."""
from collections import defaultdict
from datetime import datetime,timezone
import json
from pathlib import Path
import numpy as np
from viewer.ledger import collect_run_results
from viewer.dashboard import build_dashboard_artifact
from .perception_cache import ROOT
from .data import file_hash
from world_model.pusht.checkpoints import json_atomic

FAMILIES={'p1':['cnn','vit'],'extensions':['joint','conv','transformer'],
          'decoders':['late','early','raw','conditioned'],'independent':['split'],'localization':['cnn','vit']}
SEEDS=(9107,9108,9109)


def read(path): return json.loads(Path(path).read_text())
def lines(path): return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def reconcile():
    runs,notices=collect_run_results(ROOT.resolve())
    artifact=build_dashboard_artifact(runs,notices)
    datasets={k:v for k,v in artifact['snapshot']['datasets'].items() if k.startswith('perception_')}
    checks={}; pending=[]; streams={}; completed=0; updates=0; presentations=0; seconds=0.; diagnostics=[]; trends=[]
    for family,arms in FAMILIES.items():
        for seed in SEEDS:
            for arm in arms:
                out=ROOT/family/f'seed_{seed}'/arm; key=str(out.relative_to(ROOT))
                if not (out/'curriculum_analysis.json').exists(): pending.append(key); continue
                result=read(out/'curriculum_result.json'); manifest=read(out/'curriculum_manifest.json')
                cfg=manifest['config']; training=lines(out/'training.jsonl'); validation=lines(out/'validation.jsonl')
                if result['step']!=len(training) or result['examples']!=len(training)*cfg['batch_size']:
                    raise AssertionError('exposure count differs: '+key)
                expected=validation[-1] if family in ('decoders','independent') else min(validation,key=lambda r:r['q'])
                if result['selected_step']!=expected['step'] or result['selected']!=expected:
                    raise AssertionError('wrong checkpoint selector: '+key)
                draws=[r['sample_indices_sha256'] for r in training]
                if seed in streams:
                    if draws!=streams[seed][:len(draws)]: raise AssertionError('unpaired image stream: '+key)
                elif len(draws)==4000: streams[seed]=draws
                checks[key]=dict(status=result['status'],updates=result['step'],selected_step=result['selected_step'],
                    selector='fixed endpoint' if family in ('decoders','independent') else 'minimum validation q; earliest tie',
                    result_sha256=file_hash(out/'curriculum_result.json'),paired_stream_verified=seed in streams)
                if result['status']=='completed' and result['step']==cfg['updates']:
                    completed+=1; updates+=result['step']; presentations+=result['examples']; seconds+=result['elapsed_seconds']
                else: pending.append(key+': '+result['status'])
                earlier=[r for r in validation if 3000<r['step']<=3500]
                later=[r for r in validation if 3500<r['step']<=4000]
                if len(earlier)==len(later)==5:
                    trend=dict(family=family,arm=arm,seed=seed,source=str(out/'validation.jsonl'))
                    for field in ('q','coco_image_mse','mask_iou'):
                        if field not in later[0]: continue
                        a=float(np.mean([r[field] for r in earlier])); b=float(np.mean([r[field] for r in later]))
                        trend.update({field+'_earlier':a,field+'_later':b,field+'_ratio':b/a,field+'_delta':b-a})
                    trends.append(trend)
                if family=='extensions':
                    audited=[r for r in training if r['step']>=1000 and 'encoder_domain_gradient_cosine' in r]
                    diagnostics.append(dict(seed=seed,arm=arm,observations=len(audited),
                        generic_manipulation_gradient_norm_ratio_median=float(np.median([r['encoder_generic_grad_norm']/max(r['encoder_manipulation_grad_norm'],1e-20) for r in audited])),
                        domain_gradient_cosine_median=float(np.median([r['encoder_domain_gradient_cosine'] for r in audited])),
                        negative_gradient_cosine_fraction=float(np.mean([r['encoder_domain_gradient_cosine']<0 for r in audited])),
                        encoder_clip_fraction=float(np.mean([r['grad_norm_encoder']>1 for r in training])),
                        last_gates=audited[-1]['gates'],scope='Audited updates1000 onward; norm ratio is generic RGB+mask / manipulation RGB+pose, before clipping'))
    semantic_checks={}; semantic_completed=0; semantic_updates=0; semantic_examples=0; semantic_seconds=0.
    semantic_roots=[ROOT/'semantics/fits'/f'seed_{s}'/e for s in SEEDS for e in ('cnn','vit')]
    semantic_roots += [ROOT/'extension_audits'/f'seed_{s}'/e/'semantics' for s in SEEDS for e in ('joint','conv','transformer')]
    semantic_streams={}
    for out in semantic_roots:
        key=str(out.relative_to(ROOT))
        if not (out/'curriculum_analysis.json').exists(): pending.append(key); continue
        result=read(out/'result.json'); manifest=read(out/'semantic_manifest.json'); seed=manifest['config']['seed']
        training=lines(out/'training.jsonl'); validation=lines(out/'validation.jsonl')
        expected=max(validation,key=lambda r:r['macro_ap'])
        if result['selected']!=expected or len(training)!=result['step']: raise AssertionError('semantic selector/updates differ: '+key)
        if result['examples']!=result['step']*64: raise AssertionError('semantic exposures differ')
        draws=[r['sample_indices_sha256'] for r in training]
        if seed in semantic_streams and draws!=semantic_streams[seed]: raise AssertionError('semantic draws unpaired')
        semantic_streams[seed]=draws
        semantic_checks[key]=dict(status=result['status'],updates=result['step'],selected_step=result['selected']['step'],result_sha256=file_hash(out/'result.json'))
        if result['status']=='completed' and result['step']==2000:
            semantic_completed+=1; semantic_updates+=result['step']; semantic_examples+=result['examples']; semantic_seconds+=result['seconds']
        else: pending.append(key+': '+result['status'])
    fresh_roots=[ROOT/'fresh/evaluations'/f'seed_{s}'/e for s in SEEDS for e in ('cnn','vit')]
    fresh_roots += [ROOT/'extension_audits'/f'seed_{s}'/e/'fresh' for s in SEEDS for e in ('joint','conv','transformer')]
    for out in fresh_roots:
        if not (out/'curriculum_analysis.json').exists(): pending.append(str(out.relative_to(ROOT)))
    reliance=ROOT/'decoder_reliance'
    if not (reliance/'curriculum_analysis.json').exists(): pending.append('decoder_reliance')
    else:
        record=read(reliance/'evaluation.json')
        if record['status']!='completed' or len(record['evaluations'])!=42: pending.append('decoder_reliance: incomplete conditions')
    if not (ROOT/'figures/decoders/curriculum_analysis.json').exists(): pending.append('figures/decoders')
    if not (ROOT/'figures/split/curriculum_analysis.json').exists(): pending.append('figures/split')
    if not (ROOT/'figures/localization/curriculum_analysis.json').exists(): pending.append('figures/localization')
    if not (ROOT/'runtime/curriculum_analysis.json').exists(): pending.append('runtime')
    else:
        runtime=read(ROOT/'runtime/evaluation.json')
        if runtime['status']!='completed' or len(runtime['rows'])!=40: pending.append('runtime: incomplete timing conditions')
    groups={}
    metrics={
        'perception_package':['test_q','mask_iou','coco_mse','pusht_mse','angle_mae_deg','fitting_seconds'],
        'perception_continuation':['test_q','mask_iou','coco_mse','pusht_mse','angle_mae_deg','fitting_seconds'],
        'perception_semantics':['test_ap'],
        'perception_fresh':['q','per_case_pass','case_q_p95','angle_mae_deg'],
        'perception_decoders':['mask_iou','coco_mse','pusht_mse','both_ms_per_image','fitting_seconds'],
        'perception_independent':['mask_iou','coco_mse','pusht_mse','both_ms_per_image','fitting_seconds'],
        'perception_localization':['test_q','fresh_q','fresh_pass','case_q_p95','angle_mae_deg',
            'boundary_pass','interior_pass','location_entropy','target_support_mass','fitting_seconds']}
    for dataset,fields in metrics.items():
        by_arm=defaultdict(list)
        for row in datasets.get(dataset,[]):
            if row.get('status','completed')=='completed': by_arm[row['arm']].append(row)
        groups[dataset]={}
        for arm,rows in by_arm.items():
            groups[dataset][arm]=dict(seeds=[r['seed'] for r in rows],runs=len(rows),
                metrics={field:dict(mean=float(np.mean([r[field] for r in rows])),
                    minimum=float(min(r[field] for r in rows)),maximum=float(max(r[field] for r in rows))) for field in fields})
    sources=sorted({name for run in runs for name in run.source_paths})
    report=dict(generated=datetime.now(timezone.utc).isoformat(),status='partial' if pending else 'complete',
        checks=checks,semantic_checks=semantic_checks,pending=pending,datasets=datasets,descriptive_groups=groups,
        encoder_training_diagnostics=diagnostics,validation_tail_trends=trends,
        counts=dict(completed_vision_fits=completed,expected_vision_fits=36,vision_updates=updates,vision_presentations=presentations,
            vision_fit_seconds_sum=seconds,completed_semantic_fits=semantic_completed,expected_semantic_fits=15,
            semantic_updates=semantic_updates,semantic_presentations=semantic_examples,semantic_fit_seconds_sum=semantic_seconds),
        sources=sources,transformation_sha256=file_hash(__file__),
        interpretation='Descriptive means/ranges over fixed-source head/module seeds. No confidence interval or architecture/pretraining isolation. Fit-time sums include validation; concurrent jobs do not sum to wall time.')
    out=ROOT/'report'; out.mkdir(exist_ok=True)
    json_atomic(out/('reconciled.json' if not pending else 'reconciled_partial.json'),report)
    return report,artifact


if __name__=='__main__':
    report,_=reconcile()
    print(json.dumps(dict(status=report['status'],counts=report['counts'],pending=len(report['pending'])),indent=2))
