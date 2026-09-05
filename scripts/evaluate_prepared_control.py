"""Evaluate saved or released weights on a predeclared matched control case file."""
import argparse
import json
from pathlib import Path
import subprocess
import h5py
import numpy as np
import torch
from world_model.model import build_model
from world_model.eval_pusht import evaluate_case
from world_model.train import write_json
from scripts.diagnose_pusht_control import sha256


@torch.inference_mode()
def evaluate(case_file,checkpoint,output,released=False):
    protocol=json.loads(Path(case_file).read_text())
    if any(protocol.get(k)!=v for k,v in {'goal_offset':25,'horizon':5,'action_block':5,'receding_horizon':5}.items()):
        raise ValueError('Prepared evaluator requires the frozen five-block/25-action protocol')
    digest=sha256(checkpoint)
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if released:
        reference=json.loads(Path('runs/diagnostics/reference_full_source/manifest.json').read_text())
        if digest!=reference['checkpoint_sha256'] or protocol['dataset']['revision']!=reference['dataset']['revision']:
            raise ValueError('Released checkpoint/source does not match reference evidence')
        model=build_model();model.load_state_dict(saved,strict=True)
        stats=reference['normalization']['action'];step=None
        normalization='Released full-source sklearn population statistics'
    else:
        training=json.loads((Path(checkpoint).parent/'manifest.json').read_text())
        if training['dataset']!=protocol['dataset'] or training['fingerprint']!=saved['fingerprint']:
            raise ValueError('Checkpoint training source/fingerprint mismatch')
        model=build_model(saved['model_config']);model.load_state_dict(saved['model'],strict=True)
        stats=saved['action_stats'];step=saved['step']
        normalization='Saved training action statistics; see training data_protocol'
    cases=protocol['cases'];seeds=protocol['solver_and_reset_seeds']
    if len(cases)!=len(seeds):raise ValueError('Case/reset seed count differs')
    with h5py.File(protocol['dataset']['path'],'r') as data:
        for case in cases:
            episode,start=case['episode'],case['start']
            if not 0<=episode<len(data['ep_len']) or not 0<=start<int(data['ep_len'][episode])-25:
                raise ValueError('Control goal crosses episode boundary')
            if case['row']!=int(data['ep_offset'][episode])+start:raise ValueError('Case row mapping mismatch')
    torch.set_num_threads(4);torch.manual_seed(42);model=model.cuda().eval()
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    settings={**protocol,'checkpoint':str(checkpoint),'checkpoint_sha256':digest,'step':step,
              'action_stats':stats,'normalization':normalization,'released':released,
              'case_manifest_sha256':sha256(case_file),'batchnorm':'saved buffers, unchanged',
              'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()}
    write_json(out/'manifest.json',settings);records=[]
    with h5py.File(protocol['dataset']['path'],'r') as data:
        for i,case in enumerate(cases):
            row=case['row']
            result=evaluate_case(model,data['pixels'][row],data['pixels'][row+25],data['state'][row],data['state'][row+25],stats,
                seed=seeds[i],samples=protocol['samples'],iterations=protocol['iterations'],elites=protocol['elites'],
                budget=protocol['budget'],frameskip=protocol['action_block'])
            np.save(out/f'actions_{i}.npy',result.pop('actions'));result.update(case);records.append(result)
            with (out/'cases.jsonl').open('a') as stream:stream.write(json.dumps(result)+'\n')
            print(json.dumps(dict(case=i+1,**{k:v for k,v in result.items() if k!='plans'})),flush=True)
    if sha256(checkpoint)!=digest:raise RuntimeError('Checkpoint changed')
    summary=dict(successes=sum(r['success'] for r in records),cases=len(records),
                 initial_successes=sum(r['initial_success'] for r in records),
                 mean_steps=float(np.mean([r['steps'] for r in records])),checkpoint_unchanged=True)
    write_json(out/'summary.json',summary);print(json.dumps(summary),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('cases');parser.add_argument('checkpoint');parser.add_argument('output')
    parser.add_argument('--released',action='store_true')
    a=parser.parse_args();evaluate(a.cases,a.checkpoint,a.output,a.released)
