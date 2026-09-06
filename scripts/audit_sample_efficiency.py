"""Frozen training-mode gradient and sampling audit. Always launch via run.py."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from scripts.gradient_audit_math import gradient_geometry, module_geometry, adam_delta, preserved_state
from scripts.inspect_checkpoint import inspection_datasets, load, sha256
from world_model.data import preprocess_pixels, normalize_actions
from world_model.introspection import state_digest
from world_model.objective import SIGReg
from world_model.training import autocast_context
from world_model.train import write_json

GROUPS = ('encoder', 'projector', 'predictor', 'action_encoder', 'pred_proj')


def sampling_audit(train, manifest, step):
    base = train.dataset
    cfg = manifest['config']; batch = cfg['batch_size']
    order = torch.randperm(len(train), generator=torch.Generator().manual_seed(cfg['seed'])).numpy()
    # Current checkpoints are in epoch zero. Do not silently generalize this count.
    if step > len(train)//batch: raise ValueError('Audit currently supports first-epoch checkpoints')
    raw = np.asarray(train.indices)[order[:step*batch]]
    ep = np.searchsorted(base.cumulative, raw, side='right')
    local = raw - np.r_[0, base.cumulative[:-1]][ep]
    starts = base.offsets[ep] + local
    frame_rows = (starts[:,None] + np.arange(base.num_steps)*base.frameskip).ravel()
    transition_rows = (starts[:,None] + np.arange(base.num_steps-1)*base.frameskip).ravel()
    unique_frames = len(np.unique(frame_rows)); unique_transitions = len(np.unique(transition_rows))
    first = slice(0, min(len(ep), 100*batch))
    episodes_per_batch = [len(np.unique(v)) for v in ep[first].reshape(-1,batch)]
    frames_per_batch = [len(np.unique(v)) for v in frame_rows[:len(ep[first])*base.num_steps].reshape(-1,batch*base.num_steps)]
    result = dict(source_episodes=len(base.lengths), source_frames=int(base.lengths.sum()),
                  train_windows=len(train), valid_windows=len(base), processed_windows=len(raw),
                  processed_unique_episodes=len(np.unique(ep)), encoded_frame_slots=len(frame_rows),
                  unique_encoded_frame_rows=unique_frames, frame_reuse=len(frame_rows)/unique_frames,
                  supervised_transition_slots=len(transition_rows), unique_transition_rows=unique_transitions,
                  transition_reuse=len(transition_rows)/unique_transitions,
                  mean_unique_episodes_per_batch=float(np.mean(episodes_per_batch)),
                  mean_unique_frames_per_batch=float(np.mean(frames_per_batch)), audited_batches=len(episodes_per_batch),
                  epoch_drop_last_windows=len(train)%batch,
                  definition='Frame identity is source row; transition identity is source start row with fixed frameskip. Context lengths differ across repeated transitions. Counts are not effective sample size.')
    with h5py.File(base.path) as f:
        if manifest['dataset']['name']=='pusht':
            initial=f['state'][base.offsets, :5]
            _, groups = np.unique(initial, axis=0, return_inverse=True)
            result.update(initial_configuration_groups=int(groups.max()+1),
                          initial_configuration_definition='Exact first-frame state[:5]: agent xy, block xy and angle; grouping proxy, not full trajectory equivalence.',
                          mean_unique_initial_groups_per_batch=float(np.mean([len(np.unique(groups[v])) for v in ep[first].reshape(-1,batch)])))
    return result


def logged_training(manifest_path):
    manifest=json.loads(manifest_path.read_text()); paths=[manifest_path.parent/'metrics.jsonl']
    if manifest.get('parent'): paths.insert(0,Path(manifest['parent']['manifest']).parent/'metrics.jsonl')
    result=[]
    for path in paths:
        rows=[json.loads(x) for x in path.read_text().splitlines()]
        rows=[x for x in rows if x.get('kind')=='train']
        result.append(dict(source=str(path), logged_updates=len(rows),
            clip_fraction=sum(x['grad_norm']>manifest['config']['grad_clip'] for x in rows)/len(rows),
            quantiles={k:np.quantile([x[k] for x in rows if k in x],[0,.25,.5,.75,1]).tolist()
                       for k in ('grad_norm','loss','pred_loss','sigreg_loss','lr','mean_step_seconds','mean_data_wait_seconds')
                       if any(k in x for x in rows)}))
    return result


def flatten_grad(loss, params, retain_graph=True):
    grads=torch.autograd.grad(loss,params,retain_graph=retain_graph,allow_unused=True)
    return torch.cat([(torch.zeros_like(p) if g is None else g).detach().float().cpu().flatten()
                      for p,g in zip(params,grads)])


def group_slices(model):
    result={};start=0
    for name,p in model.named_parameters():
        group=name.split('.')[0]
        if group not in result:result[group]=[start,start]
        start+=p.numel();result[group][1]=start
    # Top-level modules must be contiguous for these slices to be valid.
    assert sum(b-a for a,b in result.values())==start
    return {k:slice(a,b) for k,(a,b) in result.items()}


def probe(model, pixels, actions, cfg, seed, branches=False, reg_seed=None, fp32_sigreg=False):
    params=list(model.parameters()); slices=group_slices(model)
    with preserved_state(model):
        model.train(); torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        reg=SIGReg(cfg['sigreg_knots'],cfg['sigreg_projections']).to(pixels.device)
        with autocast_context(pixels.device,cfg['precision']):
            z=model.encode(pixels);prediction=model.predict(z[:,:-1],actions[:,:-1])
            pred=(prediction-z[:,1:]).square().mean()
            with torch.random.fork_rng(devices=[pixels.device.index] if pixels.is_cuda else []):
                if reg_seed is not None: torch.manual_seed(reg_seed)
                with autocast_context(pixels.device, 'float32') if not fp32_sigreg else torch.autocast(device_type=pixels.device.type, enabled=False):
                    reg_z=z.float() if fp32_sigreg else z
                    sig=cfg['sigreg_weight']*reg(reg_z.transpose(0,1))
            loss=pred+sig
            input_loss=(prediction-z[:,1:].detach()).square().mean()
            target_loss=(prediction.detach()-z[:,1:]).square().mean()
        gp=flatten_grad(pred,params);gr=flatten_grad(sig,params)
        gt=flatten_grad(loss,params)
        row=dict(seed=seed,batch_size=len(pixels),prediction_loss=float(pred.detach()),weighted_sigreg_loss=float(sig.detach()),
                 loss=float(loss.detach()),preclip_norm=float(gt.norm()),
                 decomposition_relative_error=float((gt-gp-gr).norm()/gt.norm()),
                 modules={k:module_geometry(gp[s],gr[s]) for k,s in slices.items()})
        if branches:
            gi=flatten_grad(input_loss,params); gz=flatten_grad(target_loss,params)
            row['branches']={}
            for k in ('encoder','projector'):
                s=slices[k]; geom=module_geometry(gi[s],gz[s])
                row['branches'][k]=dict(input_norm=geom['prediction_norm'],target_norm=geom['weighted_sigreg_norm'],
                    cosine=geom['cosine'],cancellation_ratio=geom['total_norm']/(geom['prediction_norm']+geom['weighted_sigreg_norm']),
                    decomposition_relative_error=float((gp[s]-gi[s]-gz[s]).norm()/gp[s].norm()))
        return row,gt


def optimizer_audit(model, gradient, saved, cfg, total_steps):
    groups=saved['optimizer']['param_groups']
    if len(groups)!=1:raise ValueError('Expected one saved Adam group')
    group=dict(groups[0]);step=saved['step'];warm=max(1,int(total_steps*cfg['warmup_fraction']))
    scale=(step+1)/warm if step<warm else .5*(1+math.cos(math.pi*(step-warm)/(total_steps-warm)))
    group['lr']=cfg['lr']*scale
    clip=min(1.,cfg['grad_clip']/(float(gradient.norm())+1e-6));pos=0;parts=[]
    for p,identity in zip(model.parameters(),group['params']):
        g=gradient[pos:pos+p.numel()].reshape(p.shape)*clip;pos+=p.numel()
        parts.append(adam_delta(p.detach().cpu(),g,saved['optimizer']['state'][identity],group).flatten())
    delta=torch.cat(parts);parameters=torch.cat([p.detach().cpu().flatten() for p in model.parameters()])
    return dict(next_lr=group['lr'],clip_multiplier=clip,
                modules={k:dict(parameter_norm=float(parameters[s].norm()),delta_norm=float(delta[s].norm()),
                    relative_delta=float(delta[s].norm()/parameters[s].norm())) for k,s in group_slices(model).items()},
                definition='Analytical next AdamW delta using saved moments, current probe gradient, clipping and next schedule LR. No optimizer update executed.')


def plot(result,path):
    rows=result['probes'];fig,axes=plt.subplots(1,3,figsize=(14,4),dpi=130)
    for group in GROUPS:
        vals=[np.mean([r['modules'][group]['weighted_sigreg_norm']/(r['modules'][group]['prediction_norm']+1e-30)
                       for r in rows if r['regime']=='different_data' and r['batch_size']==b]) for b in (32,64,128)]
        if group in ('encoder','projector'):axes[0].plot([32,64,128],vals,'o-',label=group)
    axes[0].set(title='Weighted SIGReg / prediction gradient norm',xlabel='Batch windows',ylabel='Norm ratio');axes[0].legend()
    for key,label in [('different_data','Different data, fixed seed'),('same_data','Same data, varied seed')]:
        vals=result['geometry'][key]
        axes[1].plot([int(b) for b in vals],[v['pairwise_cosine_mean'] for v in vals.values()],'o-',label=label)
    axes[1].set(title='Total-gradient consistency (4 probes)',xlabel='Batch windows',ylabel='Mean pairwise cosine',ylim=(-.1,1.05));axes[1].legend(fontsize=8)
    branch=next(r['branches'] for r in rows if 'branches' in r)
    axes[2].bar(list(branch),[v['cosine'] for v in branch.values()],color='#cc6d42')
    axes[2].set(title='Prediction input / target branch cosine',ylabel='Cosine (one batch of 128)',ylim=(-1,1));axes[2].axhline(0,color='gray',lw=.7)
    fig.suptitle(f"{result['dataset']} · step {result['step']} · train mode / bf16 · no updates",fontsize=12)
    fig.tight_layout();fig.savefig(path);plt.close(fig)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=False);tick=time.monotonic()
    torch.set_num_threads(4);device=torch.device('cuda')
    manifest_path=args.checkpoint.parent/'manifest.json';manifest=json.loads(manifest_path.read_text());digest=sha256(args.checkpoint)
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    model,stats,step,cfg=load(args.checkpoint,manifest,False);model.to(device)
    model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    before=state_digest(model);train,_,_=inspection_datasets(manifest)
    indices=torch.randperm(len(train),generator=torch.Generator().manual_seed(cfg['seed']+600000))[:512].tolist()
    raw=[int(train.indices[i]) for i in indices]
    write_json(out/'manifest.json',dict(dataset=manifest['dataset'],checkpoint=str(args.checkpoint),checkpoint_sha256=digest,
        source_run_manifest=str(manifest_path),step=step,precision=cfg['precision'],population='512 frozen TRAIN windows; not held-out evaluation',
        protocol='docs/sample-efficiency-plan-2026-09-06.md',sampling_seed=cfg['seed']+600000,train_indices=indices,base_window_indices=raw,
        base_indices_sha256=hashlib.sha256(np.asarray(raw,dtype='<i8').tobytes()).hexdigest(),config=cfg,
        batchnorm='Training batch statistics; buffers restored after each probe',optimizer_updates=0,
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()))
    result=dict(dataset=manifest['dataset']['name'],step=step,checkpoint_sha256=digest,
                sampling=sampling_audit(train,manifest,step),logs=logged_training(manifest_path),probes=[],geometry={})
    print(json.dumps(dict(stage='sampling',**result['sampling'])),flush=True)
    batches=list(DataLoader(train,batch_size=128,sampler=indices,num_workers=4))
    vectors={b:[] for b in (32,64,128)};same=[]
    for i,batch in enumerate(batches):
        pixels=preprocess_pixels(batch['pixels'].to(device),model.encoder.config.image_size)
        actions=normalize_actions(batch['action'].to(device),stats)
        for b in (32,64,128):
            row,g=probe(model,pixels[:b],actions[:b],cfg,530001,branches=i==0 and b==128)
            row.update(regime='different_data',replicate=i);result['probes'].append(row);vectors[b].append(g)
            if i==0 and b==128:
                same.append(g);result['adam']=optimizer_audit(model,g,saved,cfg,manifest['total_steps'])
            with (out/'probes.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            print(json.dumps(dict(stage='gradient',replicate=i,batch_size=b,norm=row['preclip_norm'])),flush=True)
        if i==0:
            for j in range(1,4):
                row,g=probe(model,pixels,actions,cfg,530001+j)
                row.update(regime='same_data',replicate=j);result['probes'].append(row);same.append(g)
                with (out/'probes.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        del pixels,actions
    result['geometry']['different_data']={str(b):gradient_geometry(gs) for b,gs in vectors.items()}
    result['geometry']['same_data']={'128':gradient_geometry(same)}
    result.update(checkpoint_unchanged=sha256(args.checkpoint)==digest,model_state_unchanged=state_digest(model)==before,
                  elapsed_seconds=time.monotonic()-tick,peak_gpu_bytes=torch.cuda.max_memory_allocated(),
                  limitations='4 batches at one checkpoint per dataset; descriptive, no confidence bounds or critical batch estimate. Nested batches change BN and SIGReg. Same-data variance is not an additive decomposition of total variance.')
    assert result['checkpoint_unchanged'] and result['model_state_unchanged']
    write_json(out/'gradient_audit.json',result);plot(result,out/'gradient_geometry.png')
    print(json.dumps(dict(stage='complete',output=str(out),seconds=result['elapsed_seconds'],geometry=result['geometry'])),flush=True)


if __name__=='__main__':main()
