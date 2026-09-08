"""Matched E/D warmup and E/D/H adaptation with auditable atomic recovery."""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import time
import fcntl

import numpy as np
import torch
from world_model.paddle.models import Encoder, Decoder
from world_model.pusht.models import PoseReadout
from world_model.paddle.training import optimizer_for
from world_model.pusht.checkpoints import (
    SCHEMA_VERSION,TENSOR_SCHEMA,ACTION_SCHEMA_VERSION,atomic_checkpoint,atomic_checkpoint_copy,
    fingerprint_modules,json_atomic,rng_state,restore_rng,versions,read_checkpoint)
from .data import digest

def initial_models(seed,warmup=None):
    # CPU construction is isolated from global training RNG and does not seed CUDA.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        models={'E':Encoder(),'D':Decoder(),'H':PoseReadout()}
    if warmup is not None:
        for name in ('E','D'):models[name].load_state_dict(warmup[name])
    return models

def phase_sampler(seed,phase):
    if phase not in ('warmup','supervised'):raise ValueError('unknown sampler phase')
    return np.random.default_rng(seed+(10001 if phase=='warmup' else 20003))

def objective(models,x,targets=None):
    encoded=models['E'](x);decoded=models['D'](encoded)
    pixel=(decoded-x).square().mean();loss=pixel
    metrics={'image_mse':float(pixel.detach())}
    if targets is not None:
        if targets.shape!=(len(x),6):raise ValueError('pose target shape')
        pose=(models['H'](encoded)-targets).square().mean()
        loss=loss+pose;metrics['pose_mse']=float(pose.detach())
    return loss,{'loss':float(loss.detach()),**metrics}

def backward_batch(models,x,targets,microbatch):
    if microbatch<1 or not len(x):raise ValueError('nonempty batch and positive microbatch required')
    for model in models.values():
        for p in model.parameters():p.grad=None
    metrics={}
    for start in range(0,len(x),microbatch):
        stop=min(start+microbatch,len(x));weight=(stop-start)/len(x)
        loss,row=objective(models,x[start:stop],None if targets is None else targets[start:stop])
        if not torch.isfinite(loss):raise RuntimeError('nonfinite perception objective')
        (loss*weight).backward()
        for k,v in row.items():metrics[k]=metrics.get(k,0.)+weight*v
    return metrics

def selection_key(metrics,step):
    values=[*metrics['position_mae'],metrics['angle_mae_deg'],metrics['image_mse']]
    if len(metrics['position_mae'])!=4 or not np.isfinite(values).all():raise ValueError('finite physical metrics required')
    q=max([v/8 for v in metrics['position_mae']]+[metrics['angle_mae_deg']/10])
    return q,metrics['image_mse'],step

@torch.no_grad()
def evaluate(models,dataset,indices,batch=128,device='cpu',labelled=True,return_records=False):
    indices=np.asarray(indices,dtype=np.int64)
    if not len(indices):raise ValueError('empty evaluation population')
    modes={k:m.training for k,m in models.items()}
    for m in models.values():m.eval()
    image_error=[];predictions=[];targets=[]
    try:
        for start in range(0,len(indices),batch):
            ids=indices[start:start+batch];x,y=dataset.batch(ids,device,labelled)
            latent=models['E'](x);decoded=models['D'](latent)
            image_error.extend((decoded-x).square().mean((1,2,3)).cpu().double().tolist())
            if y is not None:
                predictions.append(models['H'](latent).cpu().double().numpy())
                targets.append(y.cpu().double().numpy())
    finally:
        for k,m in models.items():m.train(modes[k])
    metrics={'frames':len(indices),'image_mse':float(np.mean(image_error))}
    raw={'indices':indices.tolist(),'image_mse':image_error}
    if predictions:
        pred=np.concatenate(predictions);target=np.concatenate(targets)
        err=np.abs(pred[:,:4]-target[:,:4])*512
        angle=np.arctan2(pred[:,4],pred[:,5])-np.arctan2(target[:,4],target[:,5])
        angle=np.abs(np.arctan2(np.sin(angle),np.cos(angle)))*180/np.pi
        norm=np.linalg.norm(pred[:,4:6],axis=1)
        metrics.update(position_mae=err.mean(0).tolist(),position_p95=np.percentile(err,95,axis=0).tolist(),
                       position_max=err.max(0).tolist(),angle_mae_deg=float(angle.mean()),
                       angle_p95_deg=float(np.percentile(angle,95)),angle_norm_mean=float(norm.mean()),
                       near_zero_angle_norm_count=int((norm<1e-6).sum()),
                       pose_mse=float(np.square(pred-target).mean()))
        metrics['q']=selection_key(metrics,0)[0]
        metrics['perception_numeric_gate']=metrics['q']<=1
        raw.update(predictions=pred.tolist(),targets=target.tolist(),position_abs_error=err.tolist(),
                   angle_abs_error_deg=angle.tolist(),angle_norm=norm.tolist())
    metrics['loss']=metrics['image_mse']+metrics.get('pose_mse',0.)
    return (metrics,raw) if return_records else metrics

def source_hash():
    return digest({str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                   for folder in ['world_model/curriculum','world_model/paddle','world_model/pusht']
                   for p in sorted(Path(folder).glob('*.py'))})

def train_phase(config,train,validation,validation_indices,run,warmup=None,resume=False,stop_after=None):
    """One phase; generic targets are structurally absent, task outputs stay loadable."""
    run=Path(run);run.mkdir(parents=True,exist_ok=True)
    lock=(run/'.trainer.lock').open('w')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError('another curriculum trainer owns this run')
    try:
        return _train(config,train,validation,validation_indices,run,warmup,resume,stop_after)
    finally:
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()

def _train(config,train,validation,validation_indices,run,warmup,resume,stop_after):
    config=copy.deepcopy(config);labelled=config['phase']=='supervised'
    decoder_only=config.get('decoder_only',False)
    if decoder_only and (config['phase']!='warmup' or warmup is None):
        raise ValueError('decoder refit requires an image-only phase and a parent checkpoint')
    device=config.get('device','cpu');torch.set_num_threads(config.get('cpu_threads',4))
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    parent=read_checkpoint(warmup) if warmup else None
    models=initial_models(config['seed'],None if parent is None else parent['models'])
    if decoder_only:
        initialization=config.get('decoder_initialization','parent')
        if initialization not in ('fresh','parent'):raise ValueError('unknown decoder initialization')
        if initialization=='fresh':models['D'].load_state_dict(initial_models(config['seed'])['D'].state_dict())
        if 'H' in parent['models']:models['H'].load_state_dict(parent['models']['H'])
    keys=('E','D','H') if labelled or decoder_only else ('E','D')
    for name,model in models.items():
        trainable=name=='D' if decoder_only else name in keys
        model.to(device).train(trainable).requires_grad_(trainable)
    active={k:models[k] for k in (('D',) if decoder_only else keys)}
    checkpoint_models={k:models[k] for k in keys}
    frozen={k:models[k] for k in ('E','H')} if decoder_only else {}
    frozen_fingerprint=fingerprint_modules(frozen) if decoder_only else None
    init=fingerprint_modules(checkpoint_models);ed_init=fingerprint_modules({k:models[k] for k in ('E','D')})
    head_init=fingerprint_modules({'H':models['H']})
    optimizer=optimizer_for(active,config);rng=phase_sampler(config['seed'],config['phase'])
    ids=np.asarray(validation_indices,dtype=np.int64)
    dependencies={} if parent is None else {'warmup':parent['model_fingerprint']}
    identity={'config':config,'dataset':train.fingerprint,'validation_dataset':validation.fingerprint,
              'validation_indices':ids.tolist(),'dependencies':dependencies,
              'training_rows_sha256':digest(train.rows.tolist()),
              'validation_rows_sha256':digest(validation.rows.tolist())}
    code_id=source_hash()
    start=0;elapsed=0.;best_key=None;best_checkpoint=None;examples=0
    if resume:
        saved=read_checkpoint(run/'last.pt')
        if saved['curriculum_identity']!=identity:raise ValueError('resume configuration/population/dependency mismatch')
        for k in keys:models[k].load_state_dict(saved['models'][k])
        optimizer.load_state_dict(saved['optimizer']);restore_rng(saved['rng'],rng)
        start=saved['global_update'];elapsed=saved['elapsed_seconds'];examples=saved['examples_processed']
        best_key=tuple(saved['best_key']);best_checkpoint=saved['best_checkpoint']
        init=saved['initial_fingerprint'];ed_init=saved['initial_ed_fingerprint'];head_init=saved['initial_h_fingerprint']
        for name in ('training.jsonl','validation.jsonl'):
            p=run/name
            if p.exists():
                rows=[json.loads(l) for l in p.read_text().splitlines() if l.strip()]
                p.write_text(''.join(json.dumps(r)+'\n' for r in rows if r['step']<=start))
        atomic_checkpoint_copy(run/best_checkpoint,run/'best.pt')
        if saved.get('training_complete'):
            return json.loads((run/'curriculum_result.json').read_text())
    elif (run/'last.pt').exists():
        raise ValueError('existing run requires explicit resume')
    json_atomic(run/'curriculum_manifest.json',dict(schema='curriculum-v1',**identity,
        initial_fingerprint=init,initial_ed_fingerprint=ed_init,initial_h_fingerprint=head_init,
        versions=versions(),code_fingerprint=code_id,selector='minimax physical error q; RGB MSE then earliest' if labelled else 'reconstruction MSE'))
    begin=time.monotonic()
    if device=='cuda':torch.cuda.reset_peak_memory_stats()
    def save(step,metrics,complete=False,reason=None):
        nonlocal best_key,best_checkpoint
        if decoder_only and fingerprint_modules(frozen)!=frozen_fingerprint:
            raise RuntimeError('decoder refit changed frozen encoder/readout state')
        key=selection_key(metrics,step) if labelled else (metrics['image_mse'],step)
        improved=best_key is None or key<best_key
        if improved:best_key=key;best_checkpoint=f'checkpoints/best_{step:08d}.pt'
        states={k:m.state_dict() for k,m in checkpoint_models.items()}
        value=dict(schema_version=SCHEMA_VERSION,tensor_schema=TENSOR_SCHEMA,action_schema=ACTION_SCHEMA_VERSION,
                   normalization=parent.get('normalization') if decoder_only else train.normalization,
                   stage='decoder_refit' if decoder_only else ('perception' if labelled else 'image_pretraining'),
                   horizon=1,global_update=step,models=states,model_fingerprint=fingerprint_modules(checkpoint_models),
                   dataset_fingerprint=train.fingerprint,dependencies=dependencies,config=config,
                   versions=versions(),code_fingerprint=code_id,optimizer=optimizer.state_dict(),
                   rng=rng_state(rng),curriculum_identity=identity,metrics=metrics,best_key=list(best_key),
                   best_checkpoint=best_checkpoint,examples_processed=examples,
                   elapsed_seconds=elapsed+time.monotonic()-begin,initial_fingerprint=init,
                   initial_ed_fingerprint=ed_init,initial_h_fingerprint=head_init,
                   training_complete=complete,stop_reason=reason)
        if decoder_only:value['frozen_fingerprint']=frozen_fingerprint
        if improved:atomic_checkpoint(run/best_checkpoint,value)
        atomic_checkpoint(run/'last.pt',value)
        if improved:atomic_checkpoint_copy(run/best_checkpoint,run/'best.pt')
        if step in (0,config['updates']//2,config['updates']):
            atomic_checkpoint(run/f'update_{step:08d}.pt',value)
        json_atomic(run/'status.json',{'status':'completed' if complete else 'running','step':step,'metrics':metrics})
        return value
    def validate(step):
        metrics=evaluate(models,validation,ids,config.get('validation_batch',128),device,labelled)
        with (run/'validation.jsonl').open('a') as f:f.write(json.dumps({'step':step,**metrics})+'\n')
        print(f"{config.get('arm','diagnostic')} {config['phase']} {step}: image={metrics['image_mse']:.6g} q={metrics.get('q')} angle={metrics.get('angle_mae_deg')}",flush=True)
        return metrics
    if start==0 and not resume:
        metrics=validate(0);save(0,metrics)
    last_valid=start;step=start;stop_reason='update_budget'
    for step in range(start+1,config['updates']+1):
        if elapsed+time.monotonic()-begin>=config['max_seconds']:
            step-=1;stop_reason='wall_budget';break
        indices=rng.integers(len(train),size=config['batch_size'])
        x,y=train.batch(indices,device,labelled)
        batch_begin=time.monotonic()
        metrics=backward_batch(models,x,y,config['microbatch'])
        norm=torch.nn.utils.clip_grad_norm_([p for m in active.values() for p in m.parameters()],config['grad_clip'])
        if not torch.isfinite(norm):raise RuntimeError('nonfinite perception gradients')
        optimizer.step();examples+=len(indices)
        row={'step':step,**metrics,'grad_norm':float(norm),'examples':examples,
             'sample_indices_sha256':hashlib.sha256(indices.tobytes()).hexdigest(),
             'update_seconds':time.monotonic()-batch_begin}
        with (run/'training.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        if step%config['validate_every']==0 or step==config['updates'] or step==stop_after:
            metrics=validate(step);save(step,metrics);last_valid=step
        if stop_after is not None and step>=stop_after:
            return {'status':'interrupted_for_test','step':step}
    metrics=validate(step) if last_valid!=step else metrics
    complete=step==config['updates'];saved=save(step,metrics,complete,stop_reason)
    selected=read_checkpoint(run/best_checkpoint)
    result=dict(schema='curriculum-v1',status='completed' if complete else 'stopped',config=config,
                step=step,selected_step=selected['global_update'],selected_checkpoint=best_checkpoint,
                selected=selected['metrics'],final=metrics,examples=examples,elapsed_seconds=saved['elapsed_seconds'],
                peak_cuda_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else None,
                dataset_fingerprint=train.fingerprint,initial_ed_fingerprint=ed_init,initial_h_fingerprint=head_init,
                model_fingerprint=saved['model_fingerprint'],dependencies=dependencies,stop_reason=stop_reason)
    json_atomic(run/'curriculum_result.json',result)
    json_atomic(run/'status.json',{'status':result['status'],'step':step,'metrics':metrics})
    return result
