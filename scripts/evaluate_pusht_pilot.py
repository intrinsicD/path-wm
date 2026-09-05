"""Evaluate an untouched pilot/released checkpoint on the pre-frozen shared cases."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from world_model.data import TrajectoryDataset
from world_model.model import build_model
from world_model.evaluation import evaluate_prediction
from world_model.eval_pusht import evaluate_case
from world_model.train import write_json


def sha256(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()


@torch.inference_mode()
def evaluate(run, checkpoint, output, released=False):
    meta=json.loads((Path(run)/'manifest.json').read_text())
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    digest=sha256(checkpoint)
    torch.set_num_threads(4);torch.manual_seed(42)
    saved=torch.load(checkpoint,weights_only=True,map_location='cuda')
    if released:
        reference_path=Path('runs/diagnostics/reference_full_source/manifest.json')
        reference=json.loads(reference_path.read_text())
        if digest!=reference['checkpoint_sha256']: raise ValueError('Released checkpoint digest mismatch')
        if meta['dataset']['revision']!=reference['dataset']['revision']: raise ValueError('Source revision mismatch')
        model=build_model().cuda().eval();model.load_state_dict(saved,strict=True)
        stats=reference['normalization']['action'];step=None
        normalization='Released evaluation: full-source sklearn population scaler'
    else:
        if saved['fingerprint']!=meta['fingerprint']: raise ValueError('Checkpoint belongs to a different run')
        model=build_model(saved['model_config']).cuda().eval();model.load_state_dict(saved['model'],strict=True)
        stats=saved['action_stats'];step=saved['step']
        normalization='Pilot: saved training-only unbiased action statistics'
    split=meta['episode_split'];cases=split['cases']
    if len(cases)!=20 or len({c['episode'] for c in cases})!=20:
        raise ValueError('Expected 20 distinct, frozen held-out episodes')
    if not {c['episode'] for c in cases}<=set(meta['val_episodes']): raise ValueError('Non-held-out control case')
    settings=dict(checkpoint=str(checkpoint),checkpoint_sha256=digest,step=step,
        released=released,dataset=meta['dataset'],cases=cases,action_stats=stats,
        normalization=normalization,sampling_seed=split['sampling_seed'],
        solver_and_reset_seeds=split['solver_and_reset_seeds'],
        population='20 held-out initial-configuration groups for pilot; source training population for released weights',
        batchnorm='saved buffers, unchanged',precision='float32',goal_offset=25,budget=50,
        samples=300,iterations=30,elites=30,horizon=5,action_block=5,receding_horizon=5,
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        environment_commit='6f1e499e9cc0c898d326112f485c1062c3d20f24',
        protocol='Local paired development evaluation, not an exact historical paper benchmark',
        validation_window_indices=meta['validation_window_indices'])
    write_json(out/'manifest.json',settings)
    dataset=TrajectoryDataset(meta['dataset']['path'],meta['val_episodes'],frameskip=5,num_steps=4)
    loader=DataLoader(dataset,batch_size=32,sampler=meta['validation_window_indices'],num_workers=0)
    prediction=evaluate_prediction(model,loader,stats,torch.device('cuda'),batches=16,precision='float32')
    write_json(out/'prediction.json',prediction)
    print(json.dumps(dict(stage='prediction',step=step,released=released,**prediction)),flush=True)
    results=[]
    with h5py.File(meta['dataset']['path'],'r') as f:
        for i,case in enumerate(cases):
            row=case['row']
            result=evaluate_case(model,f['pixels'][row],f['pixels'][row+25],
                f['state'][row],f['state'][row+25],stats,seed=split['solver_and_reset_seeds'][i])
            np.save(out/f'actions_{i}.npy',result.pop('actions'))
            result.update(case);results.append(result)
            with (out/'cases.jsonl').open('a') as log:log.write(json.dumps(result)+'\n')
            print(json.dumps(dict(stage='control',case=i+1,total=20,**{k:v for k,v in result.items() if k!='plans'})),flush=True)
    if sha256(checkpoint)!=digest: raise RuntimeError('Evaluation changed the checkpoint')
    summary=dict(successes=sum(r['success'] for r in results),cases=len(results),
        initial_successes=sum(r['initial_success'] for r in results),
        mean_steps=float(np.mean([r['steps'] for r in results])),
        control_seconds=sum(r['seconds'] for r in results),checkpoint_unchanged=True)
    write_json(out/'summary.json',summary);print(json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run');p.add_argument('checkpoint');p.add_argument('output')
    p.add_argument('--released',action='store_true')
    a=p.parse_args();evaluate(a.run,a.checkpoint,a.output,a.released)
