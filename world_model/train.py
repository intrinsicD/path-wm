"""Local reproducible baseline training on a selected trajectory dataset."""
import argparse
import hashlib
import json
import math
import random
import subprocess
import time
from pathlib import Path
import h5py
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from world_model.data import TrajectoryDataset, split_episodes, action_statistics, preprocess_pixels, normalize_actions
from world_model.model import build_model
from world_model.objective import SIGReg
from world_model.training import backward_batch
from world_model.evaluation import evaluate_prediction


def write_json(path,value):
    path=Path(path); tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def train(config_path, resume=False):
    cfg=yaml.safe_load(Path(config_path).read_text())
    ds_cfg=yaml.safe_load(Path(cfg['dataset']).read_text())
    if ds_cfg['kind']!='action_trajectory': raise ValueError('Dataset has no action-conditioned protocol')
    run=Path(cfg['run_dir']);run.mkdir(parents=True,exist_ok=True)
    if (run/'checkpoint.pt').exists() and not resume:
        raise FileExistsError(f'{run} already has a checkpoint; use --resume or a new run_dir')
    seed=cfg['seed'];torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    torch.set_num_threads(4)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type=='cpu' and cfg['precision']=='bf16': raise ValueError('bf16 recipe expects CUDA')
    with h5py.File(ds_cfg['path'],'r') as f: count=len(f['ep_len'])
    tr,va=split_episodes(count,seed,cfg['train_fraction'])
    if cfg.get('train_episodes'): tr=tr[:cfg['train_episodes']]
    stats=action_statistics(ds_cfg['path'],tr)
    args=dict(path=ds_cfg['path'],frameskip=ds_cfg['frameskip'],num_steps=ds_cfg['history']+1,cache_bytes=cfg.get('cache_bytes',0))
    train_ds=TrajectoryDataset(episodes=tr,**args); val_ds=TrajectoryDataset(episodes=va,**args)
    if train_ds.action_dim!=ds_cfg['action_dim']: raise ValueError('Dataset action dimension mismatch')
    model_cfg={**cfg.get('model',{}),'action_dim':ds_cfg['action_dim'],
               'frameskip':ds_cfg['frameskip'],'history':ds_cfg['history']}
    model=build_model(model_cfg).to(device)
    reg=SIGReg(knots=cfg['sigreg_knots'],num_proj=cfg['sigreg_projections']).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    batches_per_epoch=len(train_ds)//cfg['batch_size']
    if batches_per_epoch<1: raise ValueError('Training dataset smaller than batch size')
    total_steps=batches_per_epoch*cfg['epochs']
    if cfg.get('max_steps'): total_steps=min(total_steps,cfg['max_steps'])
    warmup=max(1,int(total_steps*cfg['warmup_fraction']))
    val_order=torch.randperm(len(val_ds),generator=torch.Generator().manual_seed(seed+100000)).tolist()
    val_order=val_order[:cfg['eval_batches']*min(cfg['batch_size'],32)]
    val_loader=DataLoader(val_ds,batch_size=min(cfg['batch_size'],32),sampler=val_order,
        num_workers=cfg['workers'],pin_memory=device.type=='cuda')
    signature=dict(config=cfg,dataset=ds_cfg,model=model_cfg,train_episodes=tr,val_episodes=va,
                   action_stats=stats,total_steps=total_steps,train_windows=len(train_ds),
                   val_windows=len(val_ds),validation_window_indices=val_order,initialization='random',seed=seed)
    fingerprint=hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()
    step=0;elapsed=0
    if resume:
        saved=torch.load(run/'checkpoint.pt',map_location=device,weights_only=True)
        if saved['fingerprint']!=fingerprint: raise ValueError('Resume configuration or dataset changed')
        model.load_state_dict(saved['model'],strict=True);optimizer.load_state_dict(saved['optimizer'])
        step=saved['step'];elapsed=saved['elapsed_seconds']
        torch.set_rng_state(saved['rng'].cpu())
        if device.type=='cuda':torch.cuda.set_rng_state_all([s.cpu() for s in saved['cuda_rng']])
    else:
        signature.update(code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            code_dirty=bool(subprocess.check_output(['git','status','--porcelain'],text=True).strip()),
            torch_version=torch.__version__,device=str(device),fingerprint=fingerprint)
        write_json(run/'manifest.json',signature)
    start=time.monotonic()
    def record(kind,metrics):
        row=dict(kind=kind,step=step,elapsed_seconds=elapsed+time.monotonic()-start,**metrics)
        with (run/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        write_json(run/'status.json',row);print(json.dumps(row),flush=True)
    def save():
        checkpoint=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),step=step,
            elapsed_seconds=elapsed+time.monotonic()-start,fingerprint=fingerprint,
            model_config=model_cfg,action_stats=stats,rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else [])
        torch.save(checkpoint,run/'checkpoint.pt.tmp');(run/'checkpoint.pt.tmp').replace(run/'checkpoint.pt')
    if not resume:
        record('validation',evaluate_prediction(model,val_loader,stats,device,
            model_cfg.get('image_size',224),cfg['eval_batches'],cfg['precision']))
        save()
    while step<total_steps:
        epoch=step//batches_per_epoch
        skip=step%batches_per_epoch
        # Regenerate exact epoch permutation on resume; no mutable sampler state.
        order=torch.randperm(len(train_ds),generator=torch.Generator().manual_seed(seed+epoch)).tolist()
        order=order[skip*cfg['batch_size']:batches_per_epoch*cfg['batch_size']]
        loader=DataLoader(train_ds,batch_size=cfg['batch_size'],sampler=order,
            num_workers=cfg['workers'],pin_memory=device.type=='cuda',drop_last=True,
            generator=torch.Generator().manual_seed(seed+epoch))
        model.train()
        for batch in loader:
            if step>=total_steps: break
            if cfg.get('max_seconds') and time.monotonic()-start>=cfg['max_seconds']:
                record('time_limit',dict(total_steps=total_steps))
                save()
                return
            tick=time.monotonic()
            lr_scale=(step+1)/warmup if step<warmup else .5*(1+math.cos(math.pi*(step-warmup)/max(1,total_steps-warmup)))
            for group in optimizer.param_groups:group['lr']=cfg['lr']*lr_scale
            pixels=preprocess_pixels(batch['pixels'].to(device,non_blocking=True),model_cfg.get('image_size',224))
            actions=normalize_actions(batch['action'].to(device,non_blocking=True),stats)
            optimizer.zero_grad(set_to_none=True)
            terms=backward_batch(model,pixels,actions,reg,cfg['sigreg_weight'],cfg['encoder_chunk'],cfg['precision'])
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['grad_clip'],error_if_nonfinite=True)
            if not all(torch.isfinite(x) for x in terms.values()): raise FloatingPointError('Non-finite loss')
            optimizer.step();step+=1
            if step==1 or step%cfg['log_every']==0:
                record('train',{**{k:float(v) for k,v in terms.items()},'grad_norm':float(norm),
                    'lr':optimizer.param_groups[0]['lr'],'step_seconds':time.monotonic()-tick,
                    'peak_gpu_bytes':torch.cuda.max_memory_allocated() if device.type=='cuda' else 0})
            if step%cfg['eval_every']==0 or step==total_steps:
                record('validation',evaluate_prediction(model,val_loader,stats,device,
                    model_cfg.get('image_size',224),cfg['eval_batches'],cfg['precision']))
                save()
    record('complete',dict(total_steps=total_steps,baseline_gate='pending_closed_loop_evaluation'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('--resume',action='store_true')
    a=p.parse_args();train(a.config,a.resume)
