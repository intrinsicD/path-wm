"""Serial paired projection-count experiment; durable stages and immutable inputs.

Every training/evaluation command uses run.py. Re-running this coordinator skips
verified stages and refreshes reporting for already complete raw outputs. A failed
or partial control directory is preserved and requires explicit investigation.
"""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch
import yaml
from world_model.train import write_json

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = ROOT / 'runs/projection_training_2026-09-06'
CASE_FILES = {
    'pusht': 'runs/reproduction/pusht_source_scale_preparation/control_cases.json',
    'tworoom': 'runs/overnight_2026-09-06/tworoom_preparation/control_cases.json',
}
CASE_HASHES = {
    'pusht': '3fa2a20eb393e21c609f5f118dd3f7f43fd2683e88a628842510f5db4abbb4a8',
    'tworoom': '330f1d7995decf4c6b8fed75fe0e9800cc246c7af6bbf244bfe6179ebc497835',
}


def sha(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()


def read(path): return json.loads(Path(path).read_text())


def record(base, kind, **fields):
    row = dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(), kind=kind, **fields)
    with (base/'stages.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
    write_json(base/'progress.json', row)
    print(json.dumps(row), flush=True)


def run_stage(base, name, arguments, complete, timeout):
    log = base/'logs'/f'{name}.log'
    receipts = [json.loads(s) for s in (base/'stages.jsonl').read_text().splitlines()] if (base/'stages.jsonl').exists() else []
    if complete() and any(r.get('name')==name and r['kind']=='end' and r.get('returncode')==0 for r in receipts):
        return
    record(base, 'start', name=name, command=arguments, recovered_raw=complete())
    # If scientific output is complete, repair/verify reporting without recomputing it.
    command = [sys.executable, '-m', 'viewer.dashboard'] if complete() else [sys.executable, 'run.py', '-m', *arguments]
    tick = time.monotonic()
    with log.open('a') as stream:
        result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                timeout=timeout)
    valid = complete()
    receipt = read(ROOT/'runs/experiment_dashboard.receipt.json')
    verified = receipt.get('stages', {}).get('verification') == 'passed'
    code = result.returncode or (0 if valid and verified else 1)
    record(base, 'end', name=name, returncode=code, elapsed_seconds=time.monotonic()-tick,
           raw_complete=valid, dashboard_verified=verified, log=str(log.relative_to(ROOT)))
    if code: raise RuntimeError(f'{name} failed; preserved log {log}')


def training_complete(run):
    return (run/'checkpoint.pt').exists() and (run/'status.json').exists() and read(run/'status.json')['kind'] in ('complete','time_limit','step_limit','stop_requested')


def verify_training(run, previous):
    manifest = read(run/'manifest.json'); saved = torch.load(run/'checkpoint.pt', weights_only=True, map_location='cpu')
    status = read(run/'status.json')
    if saved['step'] != saved['validation_step'] or saved['step'] != status['step']:
        raise ValueError('Saved checkpoint, validation and final status disagree')
    if saved['regularizer']['_extra_state']['generator_state'] is None:
        raise ValueError('Missing active private sketch stream')
    cfg = manifest['config']; key = (manifest['dataset']['name'], cfg['seed'])
    if key in previous:
        other = previous[key]
        if manifest['initial_model_sha256'] != other['initial_model_sha256']:
            raise ValueError('Paired initialization differs')
        if saved['step'] == other['step']:
            if not torch.equal(saved['rng'], other['rng']) or not all(torch.equal(a,b) for a,b in zip(saved['cuda_rng'],other['cuda_rng'])):
                raise ValueError('Projection arms changed the global random stream')
    previous[key] = dict(initial_model_sha256=manifest['initial_model_sha256'], step=saved['step'], rng=saved['rng'], cuda_rng=saved['cuda_rng'])
    return dict(step=saved['step'], validation_step=saved['validation_step'], checkpoint_sha256=sha(run/'checkpoint.pt'), initial_model_sha256=manifest['initial_model_sha256'])


def execute(base):
    base=Path(base).resolve(); (base/'logs').mkdir(parents=True, exist_ok=True)
    with (base/'coordinator.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write_json(base/'coordinator_pid.json', dict(pid=os.getpid()))
        for dataset,path in CASE_FILES.items():
            if sha(ROOT/path) != CASE_HASHES[dataset]: raise ValueError('Frozen cases changed')
        planned=read(base/'planned_configs.json')['training']
        configs=[(path,yaml.safe_load((ROOT/path).read_text())) for path in planned]
        # Preserve counterbalanced within-pair order; finish a pair per dataset first.
        configs.sort(key=lambda item: (item[1]['seed'], 0 if 'pusht_' in Path(item[0]).stem else 1))
        previous={}
        record(base, 'experiment_start', planned_runs=len(configs), planned_control_evaluations=len(configs)*4)
        for path,cfg in configs:
            run=ROOT/cfg['run_dir']; name=run.name
            args=['world_model.train',path]
            if (run/'checkpoint.pt').exists() and not training_complete(run): args.append('--resume')
            run_stage(base,name+'_train',args,lambda:training_complete(run),timeout=3300)
            evidence=verify_training(run,previous);record(base,'training_verified',name=name,**evidence)
            dataset=read(run/'manifest.json')['dataset']['name']
            for step in (750,1500):
                checkpoint=run/f'checkpoint_{step:06d}.pt'
                if not checkpoint.exists():
                    record(base,'missing_checkpoint',name=name,step=step,actual_step=evidence['step']);continue
                evalbase=base/'evaluations'/name/f'step{step}'
                clone=evalbase/'calibrated'; receipt=evalbase/'calibration.json'
                run_stage(base,f'{name}_{step}_calibrate',[
                    'scripts.check_training_modes',str(run),'--checkpoint',str(checkpoint),'--output',str(receipt),
                    '--save-calibrated',str(clone),'--calibration-only'],
                    lambda:receipt.exists() and (clone/'checkpoint.pt').exists(),timeout=420)
                for variant,weights in [('saved',checkpoint),('calibrated',clone/'checkpoint.pt')]:
                    output=evalbase/variant if variant=='saved' else evalbase/'calibrated_control'
                    run_stage(base,f'{name}_{step}_{variant}',[
                        'scripts.evaluate_prepared_control',CASE_FILES[dataset],str(weights),str(output),'--device','cuda'],
                        lambda:(output/'summary.json').exists() and read(output/'summary.json').get('checkpoint_unchanged') is True,timeout=900)
            # An independent collector can be rerun after every completed arm.
            if (ROOT/'scripts/collect_projection_experiment.py').exists():
                from scripts.collect_projection_experiment import collect
                collect(base)
        record(base,'complete',planned_runs=len(configs))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',type=Path,default=DEFAULT_BASE)
    args=p.parse_args()
    try: execute(args.base)
    except BlockingIOError:
        raise SystemExit('Experiment coordinator already running; active progress was not changed')
    except Exception as error:
        record(args.base,'failed',error=f'{type(error).__name__}: {error}');raise
