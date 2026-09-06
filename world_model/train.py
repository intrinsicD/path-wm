"""Local reproducible baseline training on a selected trajectory dataset."""
import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import time
from pathlib import Path
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset
from world_model.data import preprocess_pixels, normalize_actions
from world_model.model import build_model
from world_model.protocol import prepare_training_data
from world_model.objective import SIGReg
from world_model.training import backward_batch
from world_model.evaluation import evaluate_prediction
from world_model.introspection import scalar_summary


def write_json(path,value):
    path=Path(path); tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


# These affect execution/reporting, not data, model, objective or LR schedule.
# Unknown keys remain frozen. Legacy fingerprints retain their original contract.
OPERATIONAL_KEYS = ('max_seconds', 'workers', 'cache_bytes', 'log_every',
                    'eval_every', 'checkpoint_steps', 'introspect')


def configuration_fingerprint(signature, version=2):
    frozen = dict(signature)
    if version == 2:
        frozen['config'] = {k: v for k, v in signature['config'].items()
                            if k not in OPERATIONAL_KEYS}
        frozen['fingerprint_version'] = version
    elif version != 1:
        raise ValueError(f'Unsupported fingerprint version: {version}')
    return hashlib.sha256(json.dumps(frozen, sort_keys=True).encode()).hexdigest()


def train(config_path, resume=False):
    cfg=yaml.safe_load(Path(config_path).read_text())
    ds_cfg=yaml.safe_load(Path(cfg['dataset']).read_text())
    if ds_cfg['kind']!='action_trajectory': raise ValueError('Dataset has no action-conditioned protocol')
    run=Path(cfg['run_dir']);run.mkdir(parents=True,exist_ok=True)
    if (run/'checkpoint.pt').exists() and not resume:
        raise FileExistsError(f'{run} already has a checkpoint; use --resume or a new run_dir')
    saved = torch.load(run/'checkpoint.pt', map_location='cpu', weights_only=True) if resume else None
    fingerprint_version = saved.get('fingerprint_version', 1) if resume else 2
    seed=cfg['seed'];torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    torch.set_num_threads(4)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type=='cpu' and cfg['precision']=='bf16': raise ValueError('bf16 recipe expects CUDA')
    train_ds,val_ds,tr,va,split_stats,split_receipt=prepare_training_data(cfg,ds_cfg)
    stats=split_stats
    base=train_ds.dataset if isinstance(train_ds,Subset) else train_ds
    if base.action_dim!=ds_cfg['action_dim']: raise ValueError('Dataset action dimension mismatch')
    model_cfg={**cfg.get('model',{}),'action_dim':ds_cfg['action_dim'],
               'frameskip':ds_cfg['frameskip'],'history':ds_cfg['history']}
    model=build_model(model_cfg).to(device)
    if cfg.get('encoder_gradient_checkpointing'):
        if cfg.get('encoder_chunk'):
            raise ValueError('Full-batch activation checkpointing cannot use encoder slicing')
        model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
    reg=SIGReg(knots=cfg['sigreg_knots'],num_proj=cfg['sigreg_projections']).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    batches_per_epoch=len(train_ds)//cfg['batch_size']
    if batches_per_epoch<1: raise ValueError('Training dataset smaller than batch size')
    total_steps=batches_per_epoch*cfg['epochs']
    if cfg.get('max_steps'): total_steps=min(total_steps,cfg['max_steps'])
    if cfg.get('expected_train_windows',len(train_ds))!=len(train_ds) or cfg.get('expected_total_steps',total_steps)!=total_steps:
        raise ValueError('Training source/window/update budget differs from the prepared protocol')
    warmup=max(1,int(total_steps*cfg['warmup_fraction']))
    val_order=torch.randperm(len(val_ds),generator=torch.Generator().manual_seed(seed+100000)).tolist()
    val_order=val_order[:cfg['eval_batches']*min(cfg['batch_size'],32)]
    val_loader=DataLoader(val_ds,batch_size=min(cfg['batch_size'],32),sampler=val_order,
        num_workers=cfg['workers'],pin_memory=device.type=='cuda',
        # Isolate DataLoader's base-seed draws from optimization randomness.
        # Preserve the legacy stream when recovering a version-one checkpoint.
        generator=torch.Generator().manual_seed(seed+200000) if fingerprint_version==2 else None)
    signature=dict(config=cfg,dataset=ds_cfg,model=model_cfg,train_episodes=tr,val_episodes=va,
                   action_stats=stats,total_steps=total_steps,train_windows=len(train_ds),
                   val_windows=len(val_ds),validation_window_indices=val_order,initialization='random',seed=seed)
    if split_receipt is not None:
        if cfg.get('split_protocol')=='random_windows':
            signature.update(data_protocol=split_receipt,population=split_receipt['population'])
        else: signature['episode_split']=split_receipt
    fingerprint=configuration_fingerprint(signature,fingerprint_version)
    step=0;elapsed=0;validation_step=None
    if resume:
        if saved['fingerprint']!=fingerprint: raise ValueError('Resume configuration or dataset changed')
        model.load_state_dict(saved['model'],strict=True);optimizer.load_state_dict(saved['optimizer'])
        step=saved['step'];elapsed=saved['elapsed_seconds']
        validation_step=saved.get('validation_step')
        original=json.loads((run/'manifest.json').read_text())
        previous=saved.get('operational_config', original['config'])
        overrides={k:dict(previous=previous.get(k),current=cfg.get(k))
                   for k in OPERATIONAL_KEYS if previous.get(k)!=cfg.get(k)}
        receipt=dict(step=step,elapsed_seconds=elapsed,fingerprint=fingerprint,
            fingerprint_version=fingerprint_version,operational_overrides=overrides,
            resumed_at=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
            code_dirty=bool(subprocess.check_output(['git','status','--porcelain'],text=True).strip()),
            code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
        with (run/'resumes.jsonl').open('a') as stream:stream.write(json.dumps(receipt)+'\n')
        torch.set_rng_state(saved['rng'].cpu())
        if device.type=='cuda':torch.cuda.set_rng_state_all([s.cpu() for s in saved['cuda_rng']])
    else:
        signature.update(code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            code_dirty=bool(subprocess.check_output(['git','status','--porcelain'],text=True).strip()),
            torch_version=torch.__version__,device=str(device),fingerprint=fingerprint,
            fingerprint_version=fingerprint_version,operational_keys=list(OPERATIONAL_KEYS),
            time_budget='Cumulative training-loop seconds, including validation/checkpointing; setup excluded. Final validation may extend the ceiling.')
        write_json(run/'manifest.json',signature)
    start=time.monotonic()
    interval=dict(updates=0,compute=0.,wait=0.)
    def take_timing():
        n=interval['updates']
        result=dict(timing_updates=n,mean_step_seconds=interval['compute']/n if n else None,
                    mean_data_wait_seconds=interval['wait']/n if n else None)
        interval.update(updates=0,compute=0.,wait=0.)
        return result
    def record(kind,metrics):
        row=dict(kind=kind,step=step,elapsed_seconds=elapsed+time.monotonic()-start,**metrics)
        with (run/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        write_json(run/'status.json',row);print(json.dumps(row),flush=True)
    def save(final=False):
        checkpoint=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),step=step,
            elapsed_seconds=elapsed+time.monotonic()-start,fingerprint=fingerprint,
            model_config=model_cfg,action_stats=stats,rng=torch.get_rng_state(),
            fingerprint_version=fingerprint_version,validation_step=validation_step,
            operational_config={k:cfg.get(k) for k in OPERATIONAL_KEYS},
            cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else [])
        torch.save(checkpoint,run/'checkpoint.pt.tmp');(run/'checkpoint.pt.tmp').replace(run/'checkpoint.pt')
        if final or step in cfg.get('checkpoint_steps',[]):
            snapshot=run/f'checkpoint_{step:06d}.pt'
            if not snapshot.exists(): os.link(run/'checkpoint.pt',snapshot)
    def introspect():
        # Opt-in read-only internals on the first validation batch; eval mode keeps buffers unchanged.
        if not cfg.get('introspect'): return
        batch=next(iter(val_loader))
        pixels=preprocess_pixels(batch['pixels'].to(device),model_cfg.get('image_size',224))
        record('internals',scalar_summary(model,pixels,normalize_actions(batch['action'].to(device),stats)))
    def validate():
        nonlocal validation_step
        record('validation',evaluate_prediction(model,val_loader,stats,device,
            model_cfg.get('image_size',224),cfg['eval_batches'],cfg.get('eval_precision',cfg['precision'])))
        introspect()
        validation_step=step
    if not resume:
        validate()
        save()
    else:
        record('resumed',dict(resumed_from_step=step))
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
        wait_start=time.monotonic()
        for batch in loader:
            data_wait_seconds=time.monotonic()-wait_start
            if step>=total_steps: break
            requested=(run/'STOP').exists()
            expired=cfg.get('max_seconds') and elapsed+time.monotonic()-start>=cfg['max_seconds']
            if requested or expired:
                if validation_step!=step:validate()
                save(final=True)
                record('stop_requested' if requested else 'time_limit',
                    dict(total_steps=total_steps,validation_step=validation_step,**take_timing(),
                         baseline_gate='pending_closed_loop_evaluation'))
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
            step_seconds=time.monotonic()-tick
            interval['updates']+=1;interval['compute']+=step_seconds;interval['wait']+=data_wait_seconds
            if step==1 or step%cfg['log_every']==0 or step==total_steps:
                record('train',{**{k:float(v) for k,v in terms.items()},'grad_norm':float(norm),
                    'lr':optimizer.param_groups[0]['lr'],'step_seconds':step_seconds,**take_timing(),
                    'data_wait_seconds':data_wait_seconds,
                    'peak_gpu_bytes':torch.cuda.max_memory_allocated() if device.type=='cuda' else 0})
            if step%cfg['eval_every']==0 or step==total_steps or step in cfg.get('checkpoint_steps',[]):
                validate()
                save(final=step==total_steps)
            wait_start=time.monotonic()
    record('complete',dict(total_steps=total_steps,validation_step=validation_step,**take_timing(),baseline_gate='pending_closed_loop_evaluation'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('--resume',action='store_true')
    a=p.parse_args();train(a.config,a.resume)
