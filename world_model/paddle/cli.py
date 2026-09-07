"""Reproducible commands for the fixed paddle world-model baseline."""
from __future__ import annotations

import argparse
import json
import hashlib
from pathlib import Path
import shutil
import subprocess
import sys

import torch
import yaml

from .checkpoints import json_atomic, versions


def load_config(path):
    config = yaml.safe_load(Path(path).read_text())
    if not isinstance(config,dict) or config.get('schema_version') != 1:
        raise ValueError('Expected version 1 paddle configuration')
    for split in ('train','validation','test'):
        if not isinstance(config['dataset'][split],int) or config['dataset'][split] <= 0:
            raise ValueError(f'dataset.{split} must be a positive episode count')
    for stage in ('perception','memory','predictor_1','predictor_5'):
        for key in ('updates','batch_size','validation_examples'):
            if config['training'][stage][key] <= 0:
                raise ValueError(f'{stage}.{key} must be positive')
    if config['training']['validate_every'] <= 0:
        raise ValueError('validate_every must be positive')
    return config


def doctor():
    info = versions(); info['cuda_available'] = torch.cuda.is_available()
    info['disk_free_bytes'] = shutil.disk_usage('.').free
    if shutil.which('nvidia-smi'):
        result = subprocess.run(['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv,noheader'],
                                capture_output=True,text=True)
        info['nvidia_smi'] = result.stdout.strip() or result.stderr.strip()
    if info['cuda_available']:
        value = torch.randn(16,16,device='cuda',requires_grad=True)
        value.square().mean().backward(); torch.cuda.synchronize()
        info['cuda_forward_backward'] = bool(torch.isfinite(value.grad).all())
    return info


def refresh_dashboard():
    from viewer.dashboard import write_experiment_dashboard
    artifact, html, receipt = write_experiment_dashboard()
    if receipt.get('stages',{}).get('verification') != 'passed':
        raise RuntimeError(f'Raw experiment saved, but dashboard visual verification is pending: {html}')
    print(f'Verified dashboard: {html}',flush=True)


def _reported(action):
    try:
        result = action()
    except BaseException:
        try:
            refresh_dashboard()
        except Exception as reporting_error:
            print(f'Dashboard also failed: {reporting_error}; original experiment failure is preserved.',file=sys.stderr)
        raise
    refresh_dashboard()
    return result


def run_all(config,data,run):
    from .data import generate_dataset, verify_dataset, test_history_cases, EpisodeDataset
    from .training import train_perception, train_memory, train_predictor
    from .evaluation import evaluate
    from .checkpoints import export_bundle, read_checkpoint
    run = Path(run); run.mkdir(parents=True,exist_ok=True)
    generate_dataset(config,data); verify_dataset(data)
    dataset_identity = EpisodeDataset(data,'train').fingerprint
    test_history_cases(run/'history_checks')
    a,b,c,d = (run/s for s in ('perception','memory','predictor_1','predictor_5'))
    stages = [
        (a,lambda:train_perception(config,data,a,resume=(a/'last.pt').exists())),
        (b,lambda:train_memory(config,data,a/'best.pt',b,resume=(b/'last.pt').exists())),
        (c,lambda:train_predictor(config,data,a/'best.pt',b/'best.pt',1,c,resume=(c/'last.pt').exists())),
        (d,lambda:train_predictor(config,data,a/'best.pt',b/'best.pt',5,d,initialize_from=c/'best.pt',resume=(d/'last.pt').exists())),
    ]
    for directory, action in stages:
        result_path = directory/'paddle_result.json'
        if result_path.exists():
            result = json.loads(result_path.read_text())
            if result['status'] == 'completed':
                checkpoint = read_checkpoint(directory/'best.pt',dataset_fingerprint=dataset_identity)
                if checkpoint['config'] != config:
                    raise ValueError(f'Cannot reuse completed run with different config: {directory}')
                print(f'Reusing completed {directory}',flush=True)
                continue
        _reported(action)
    evaluation_path = run/'evaluation'/'metrics.json'
    if evaluation_path.exists() and json.loads(evaluation_path.read_text()).get('status') == 'completed':
        result = json.loads(evaluation_path.read_text())
        if result['config'] != config or result['dataset_fingerprint'] != dataset_identity:
            raise ValueError('Completed evaluation has different config or data; use a new run directory')
        for name, checkpoint_path in (('perception',a/'best.pt'),('memory',b/'best.pt'),('predictor',d/'best.pt')):
            if hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() != result['checkpoints'][name]['sha256']:
                raise ValueError('Completed evaluation belongs to different checkpoint bytes; use a new evaluation directory')
    else:
        result = _reported(lambda:evaluate(config,data,a/'best.pt',b/'best.pt',d/'best.pt',run/'evaluation'))
    export_bundle(a/'best.pt',b/'best.pt',d/'best.pt',run/'inference.pt')
    refresh_dashboard()
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command',required=True)
    commands.add_parser('doctor')
    for name in ('generate','verify-data','test-history-cases','train-perception','train-memory',
                 'train-predictor','evaluate','demo','run-all','export-bundle','fork-predictor'):
        p = commands.add_parser(name)
        if name in ('generate','train-perception','train-memory','train-predictor','evaluate','run-all','fork-predictor'):
            p.add_argument('--config',required=True,type=Path)
        if name in ('verify-data','train-perception','train-memory','train-predictor','evaluate','run-all'):
            p.add_argument('--data',required=True,type=Path)
        if name in ('train-perception','train-memory','train-predictor','run-all','fork-predictor'):
            p.add_argument('--run',required=True,type=Path)
        if name.startswith('train-'):
            p.add_argument('--resume',action='store_true')
        if name == 'fork-predictor':
            p.add_argument('--checkpoint',required=True,type=Path)
        if name in ('train-memory','train-predictor','evaluate','demo','export-bundle'):
            p.add_argument('--perception',required=True,type=Path)
        if name in ('train-predictor','evaluate','demo','export-bundle'):
            p.add_argument('--memory',required=True,type=Path)
        if name in ('evaluate','demo','export-bundle'):
            p.add_argument('--predictor',required=True,type=Path)
        if name == 'train-predictor':
            p.add_argument('--horizon',required=True,type=int,choices=(1,5))
            p.add_argument('--initialize-from',type=Path)
        if name in ('generate','test-history-cases','evaluate','demo','export-bundle'):
            p.add_argument('--output',required=True,type=Path)
    args = vars(parser.parse_args(argv)); command = args.pop('command')
    if 'config' in args: args['config'] = load_config(args['config'])
    if command == 'doctor': function = doctor
    elif command == 'fork-predictor':
        from .continuation import fork_predictor
        function = fork_predictor
    elif command == 'run-all': function = run_all
    elif command in ('generate','verify-data','test-history-cases'):
        from .data import generate_dataset, verify_dataset, test_history_cases
        function = {'generate':generate_dataset,'verify-data':verify_dataset,'test-history-cases':test_history_cases}[command]
        if command == 'verify-data': args = {'path':args['data']}
    elif command.startswith('train-'):
        from .training import train_perception,train_memory,train_predictor
        function = {'train-perception':train_perception,'train-memory':train_memory,'train-predictor':train_predictor}[command]
    elif command == 'export-bundle':
        from .checkpoints import export_bundle
        function = export_bundle
    else:
        from .evaluation import evaluate,demo
        function = {'evaluate':evaluate,'demo':demo}[command]
    try:
        result = function(**args)
        print(json.dumps(result,indent=2,default=str,allow_nan=False))
        return 0
    except (ValueError,RuntimeError,FileNotFoundError,FileExistsError) as error:
        parser.exit(1,f'{command}: {error}\n')
