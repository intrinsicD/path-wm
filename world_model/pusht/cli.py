"""Commands for the separate CCHI PushT E/U/P experiment."""
import argparse
import json
from pathlib import Path

import yaml

from . import checkpoints, data, training


def load_config(path):
    config=yaml.safe_load(Path(path).read_text())
    if config.get('schema_version')!=1 or config.get('task')!='pusht-eup':
        raise ValueError('Expected version1 pusht-eup configuration')
    for stage in ('perception','memory','predictor_1','predictor_5'):
        for name in ('updates','batch_size','validation_examples'):
            value=config['training'][stage][name]
            if isinstance(value,bool) or not isinstance(value,int) or value<=0:
                raise ValueError(f'{stage}.{name} must be a positive integer')
    if config['training']['validate_every']<=0: raise ValueError('validate_every must be positive')
    return config


def run_all(config,data,run):
    from world_model.paddle.cli import _reported
    from .evaluation import evaluate
    run=Path(run)
    a,b,c,d=[run/name for name in ('perception','memory','predictor_1','predictor_5')]
    stages=[(a,lambda:training.train_perception(config,data,a,resume=(a/'last.pt').exists())),
        (b,lambda:training.train_memory(config,data,a/'best.pt',b,resume=(b/'last.pt').exists())),
        (c,lambda:training.train_predictor(config,data,a/'best.pt',b/'best.pt',1,c,resume=(c/'last.pt').exists())),
        (d,lambda:training.train_predictor(config,data,a/'best.pt',b/'best.pt',5,d,
             initialize_from=c/'best.pt',resume=(d/'last.pt').exists()))]
    for _,action in stages: _reported(action)
    checkpoints.export_bundle(a/'best.pt',b/'best.pt',d/'best.pt',run/'inference.pt')
    return _reported(lambda:evaluate(config,data,a/'best.pt',b/'best.pt',d/'best.pt',run/'evaluation'))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    for name in ('prepare','verify-data','train-perception','train-memory','train-predictor',
                 'evaluate','export-bundle','run-all'):
        p=commands.add_parser(name)
        if name in ('prepare',): p.add_argument('--source',required=True,type=Path)
        if name=='prepare': p.add_argument('--seed',type=int,default=3107)
        if name not in ('prepare','verify-data','export-bundle'): p.add_argument('--config',required=True,type=Path)
        if name not in ('prepare','export-bundle'): p.add_argument('--data',required=True,type=Path)
        if name.startswith('train-') or name=='run-all': p.add_argument('--run',required=True,type=Path)
        if name.startswith('train-'): p.add_argument('--resume',action='store_true')
        if name in ('train-memory','train-predictor','evaluate','export-bundle'): p.add_argument('--perception',required=True,type=Path)
        if name in ('train-predictor','evaluate','export-bundle'): p.add_argument('--memory',required=True,type=Path)
        if name in ('evaluate','export-bundle'): p.add_argument('--predictor',required=True,type=Path)
        if name in ('prepare','evaluate','export-bundle'): p.add_argument('--output',required=True,type=Path)
        if name=='train-predictor':
            p.add_argument('--horizon',required=True,type=int,choices=(1,5))
            p.add_argument('--initialize-from',type=Path)
    args=vars(parser.parse_args(argv)); command=args.pop('command')
    if 'config' in args: args['config']=load_config(args['config'])
    if command=='prepare': function=data.prepare
    elif command=='verify-data': function=data.verify_dataset; args={'output':args['data']}
    elif command.startswith('train-'): function=getattr(training,command.replace('-','_'))
    elif command=='run-all': function=run_all
    elif command=='export-bundle': function=checkpoints.export_bundle
    else:
        from .evaluation import evaluate
        function=evaluate
    try: print(json.dumps(function(**args),indent=2,default=str,allow_nan=False))
    except (ValueError,RuntimeError,FileNotFoundError,FileExistsError) as error:
        parser.exit(1,f'{command}: {error}\n')


if __name__=='__main__': main()
