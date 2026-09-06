"""Reconcile the frozen paired screen from native case ledgers, without pooling.

The derived file is restartable and explicitly incomplete until all expected
outcomes exist. Native viewer reconciliation verifies boolean case counts.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import yaml
from scripts.paired_summary import summarize_pairs
from scripts.projection_experiment import ROOT, DEFAULT_BASE, CASE_FILES, CASE_HASHES, read, sha
from viewer.ledger import collect_run_results
from world_model.train import write_json


def collect(base=DEFAULT_BASE):
    base=Path(base).resolve(); rows=[]; missing=[]; sources=set(); training=[]; pair_populations={}
    controls,_=collect_run_results(base/'evaluations')
    controls={r.label:r for r in controls if r.kind=='control'}
    for config in read(base/'planned_configs.json')['training']:
        path=ROOT/config;cfg=yaml.safe_load(path.read_text());run=ROOT/cfg['run_dir'];name=run.name
        sources.add(path)
        meta=read(run/'manifest.json') if (run/'manifest.json').exists() else None
        dataset=yaml.safe_load((ROOT/cfg['dataset']).read_text())['name']
        if meta is not None:
            identity=hashlib.sha256(json.dumps({k:meta[k] for k in
                ('data_protocol','validation_window_indices','action_stats','train_episodes','val_episodes','model','initial_model_sha256')},sort_keys=True).encode()).hexdigest()
            pair=(dataset,cfg['seed'])
            if pair in pair_populations and pair_populations[pair]!=identity:
                raise ValueError('Paired training populations, initialization or model configuration differ')
            pair_populations[pair]=identity
        status=read(run/'status.json') if (run/'status.json').exists() else {'kind':'not_started','step':0}
        training.append(dict(run=name,dataset=dataset,seed=cfg['seed'],projections=cfg['sigreg_projections'],status=status['kind'],step=status.get('step'),planned_steps=1500))
        for step in (750,1500):
            for variant,directory in [('saved','saved'),('calibrated','calibrated_control')]:
                key=f'{name}/step{step}/{directory}'; control=controls.get(key)
                identity=dict(dataset=dataset,seed=cfg['seed'],projections=cfg['sigreg_projections'],step=step,variant=variant)
                if control is None:
                    missing.append({**identity,'training_status':status['kind'],'training_step':status.get('step')});continue
                if meta is None:raise ValueError('Control exists without training provenance')
                out=base/'evaluations'/key;manifest=read(out/'manifest.json')
                expected=read(ROOT/CASE_FILES[dataset])
                if (manifest['case_manifest_sha256'] != CASE_HASHES[dataset]
                    or sha(ROOT/CASE_FILES[dataset]) != CASE_HASHES[dataset]
                    or manifest['step'] != step or control.status != 'evaluated'
                    or any(manifest.get(k)!=v for k,v in expected.items())):
                    raise ValueError('Control differs from the frozen case protocol or checkpoint step')
                calibration_hash=None
                if variant=='calibrated':
                    receipt=base/'evaluations'/name/f'step{step}'/'calibration.json';cal=read(receipt)
                    if not cal['checkpoint_unchanged'] or cal['checkpoint_sha256']!=manifest['parent_checkpoint_sha256']:
                        raise ValueError('Invalid calibration parent provenance')
                    if len(cal['calibration_source_rows'])!=512 or set(cal['calibration_source_rows']) & set(cal['validation_source_rows']):
                        raise ValueError('Calibration sample differs or overlaps validation windows')
                    calibration_hash=hashlib.sha256(json.dumps(cal['calibration_source_rows']).encode()).hexdigest();sources.add(receipt)
                metrics=control.metrics
                row={**identity,'case_sha256':manifest['case_manifest_sha256'],'calibration_rows_sha256':calibration_hash,
                     'successes':metrics['successes'],'cases':metrics['cases'],'initial_successes':metrics['initial_successes'],
                     'newly_solved':metrics['noninitial_successes'],'initial_model_sha256':meta['initial_model_sha256'],
                     'run':name,'checkpoint_sha256':manifest['checkpoint_sha256'],'source':str((out/'cases.jsonl').relative_to(ROOT))}
                rows.append(row)
                sources.update([run/'manifest.json',out/'manifest.json',out/'summary.json',out/'cases.jsonl',ROOT/CASE_FILES[dataset]])
    rows.sort(key=lambda r:(r['dataset'],r['variant'],r['seed'],r['projections'],r['step']))
    groups=summarize_pairs(rows)
    result=dict(version=1,generated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),rows=rows,groups=groups,
                expected_outcomes=len(training)*4,missing_outcomes=missing,training=training,
                limitation='Three paired seeds, fixed source cases, short cosine schedule. Descriptive uncertainty; no pass threshold or generalization claim.',
                sources=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for p in sorted(sources)])
    write_json(base/'projection_comparison.json',result)
    print(json.dumps(dict(control_outcomes=len(rows),expected_outcomes=len(training)*4,complete_pairs=sum(g['seeds_complete'] for g in groups))))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',type=Path,default=DEFAULT_BASE)
    collect(p.parse_args().base)
