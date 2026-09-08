"""Matched frozen-input dense decoding, with fixed endpoints and raw evidence."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .perception_cache import ROOT, cache_root, check_storage
from .perception_decoder import DenseDecoder, raw_patches
from .perception_heads import feature_grids
from .perception_training import (load_data, append_json, checkpoint_state, restore_checkpoint,
    recover_ledgers, TRAIN_CUTOFF)
from .encoder_reference import pinned_backbone
from .dino_reference import preprocess
from .encoder_masks import masked_bce, mask_statistics
from .data import file_hash
from .pose_accessibility import setup, analyze
from world_model.pusht.checkpoints import json_atomic, atomic_checkpoint, fingerprint_modules, versions


@torch.no_grad()
def batch(data, ids, patch, kind, device):
    rgb,_=data['ds'].batch(ids,device)
    tokens=torch.from_numpy(np.asarray(data['cache'][ids]).copy()).to(device).float()
    fine,coarse=feature_grids(tokens,'vit')
    local=raw_patches(rgb) if kind=='raw' else (fine if kind=='late' else patch(preprocess(rgb)))
    return (local,fine,coarse),rgb


def step_update(model, optimizer, patch, data, draws, device, audit):
    optimizer.zero_grad(set_to_none=True); model.train(); stats={}; pixels=0.
    shared=[p for name,p in model.named_parameters() if not name.startswith('outputs.')]
    for domain in ('coco','pusht'):
        d=data[domain,'train']; ids=draws[domain]
        features,rgb=batch(d,ids,patch,model.kind,device)
        pixel=F.mse_loss(model(features,'rgb'),rgb)
        (.5*pixel).backward(); pixels+=.5*float(pixel.detach())
        stats[domain+'_image_mse']=float(pixel.detach())
        if domain=='coco':
            a=[torch.zeros_like(p) if p.grad is None else p.grad.detach().clone() for p in shared] if audit else None
            labels=d['labels']
            mask=torch.from_numpy(labels['masks'][ids].copy()).float().to(device)[:,None]
            valid=torch.from_numpy(labels['valid'][ids].copy()).float().to(device)[:,None]
            loss=masked_bce(model(features,'mask'),mask,valid); loss.backward()
            stats['mask_bce']=float(loss.detach())
            if audit:
                b=[(torch.zeros_like(p) if p.grad is None else p.grad.detach())-g for p,g in zip(shared,a)]
                a2=torch.stack([g.square().sum() for g in a]).sum(); b2=torch.stack([g.square().sum() for g in b]).sum()
                dot=torch.stack([(x*y).sum() for x,y in zip(a,b)]).sum()
                stats.update(rgb_shared_grad_norm=float(a2.sqrt()),mask_shared_grad_norm=float(b2.sqrt()),
                    rgb_mask_gradient_cosine=float(dot/(a2*b2).sqrt().clamp_min(1e-20)))
    norm=nn.utils.clip_grad_norm_(model.parameters(),1.)
    stats.update(image_mse=pixels,loss=pixels+stats['mask_bce'],grad_norm=float(norm))
    if not torch.isfinite(norm) or not np.isfinite(stats['loss']): raise RuntimeError('nonfinite dense decoder update')
    optimizer.step()
    if audit:
        stats['film_weight_norms']=[float(m.weight.detach().norm()) for m in model.films]
        stats['film_bias_norms']=[float(m.bias.detach().norm()) for m in model.films]
    return stats


@torch.no_grad()
def evaluate_domain(model,patch,data,domain,split,device,panels=False,limit=None):
    model.eval(); d=data[domain,split]; labels=d['labels']; chunks={}; panel={}
    ids_all=np.arange(len(d['ds']))[:limit]
    def append(key,values): chunks.setdefault(key,[]).append(np.asarray(values))
    for start in range(0,len(ids_all),64):
        ids=ids_all[start:start+64]; features,rgb=batch(d,ids,patch,model.kind,device)
        image,logits=model.both(features) if domain=='coco' else (model(features,'rgb'),None)
        append('image_mse',(image-rgb).square().mean((1,2,3)).cpu().numpy())
        baseline=torch.from_numpy(d['train_mean_rgb']).to(device)[None]
        append('train_mean_image_mse',(baseline-rgb).square().mean((1,2,3)).cpu().numpy())
        if domain=='coco':
            mask=torch.from_numpy(labels['masks'][ids].copy()).to(device).float()[:,None]
            valid=torch.from_numpy(labels['valid'][ids].copy()).to(device).float()[:,None]
            probability=logits.sigmoid()
            append('mask_bce',((F.binary_cross_entropy_with_logits(logits,mask,reduction='none')*valid).sum((1,2,3))/valid.sum((1,2,3)).clamp_min(1)).cpu().numpy())
            for key,value in mask_statistics(probability,mask,valid).items(): append(key,value)
            constants={'empty':torch.zeros_like(probability),'full':torch.ones_like(probability),
                'train_mean':torch.from_numpy(d['train_mean_mask']).to(device)[None,None].expand_as(probability)}
            for name,constant in constants.items():
                for key,value in mask_statistics(constant,mask,valid).items():
                    if key in ('iou','dice'): append(name+'_'+key,value)
            if panels and start==0:
                panel.update(mask=mask[:6].cpu().numpy(),valid=valid[:6].cpu().numpy(),probability=probability[:6].cpu().numpy())
        if panels and start==0:
            panel.update(rgb=rgb[:6].cpu().numpy(),reconstruction=image[:6].cpu().numpy(),indices=ids[:6])
    raw={k:np.concatenate(v) for k,v in chunks.items()}
    metric=dict(frames=len(ids_all),image_mse=float(raw['image_mse'].mean()),train_mean_image_mse=float(raw['train_mean_image_mse'].mean()))
    if domain=='coco':
        keep=raw['has_valid'].astype(bool); metric['mask_frames']=int(keep.sum())
        for k in ('mask_bce','iou','dice','empty_iou','full_iou','empty_dice','full_dice','train_mean_iou','train_mean_dice'):
            metric[k]=float(raw[k][keep].mean())
    raw.update(indices=ids_all,source_rows=labels['rows'][ids_all],groups=labels['groups'][ids_all])
    return metric,raw,panel


@torch.no_grad()
def benchmark(model,patch,data,device):
    count=min(32,len(data['coco','test']['ds']))
    features,_=batch(data['coco','test'],np.arange(count),patch,model.kind,device)
    result={}
    for task in ('rgb','mask','both'):
        operation=(lambda:model.both(features)) if task=='both' else (lambda:model(features,task))
        for _ in range(5): operation()
        if device=='cuda': torch.cuda.synchronize()
        start=time.perf_counter()
        for _ in range(20): operation()
        if device=='cuda': torch.cuda.synchronize()
        result[task+'_milliseconds_per_image']=(time.perf_counter()-start)*1000/(20*count)
    return dict(**result,batch_size=count,repeats=20,scope='decoder only; prepared feature inputs; both reuses unconditioned trunk, conditioned uses two passes')


def train(kind,seed,development=False,device='cuda',resume=False):
    setup(); check_storage(100*1024**2)
    out=ROOT/('development/decoders' if development else 'decoders')/f'seed_{seed}'/kind; out.mkdir(parents=True,exist_ok=True)
    if (out/'curriculum_result.json').exists():
        if resume and not (out/'curriculum_analysis.json').exists(): return evaluate_saved(out,kind,seed,development,device)
        raise FileExistsError('preserve completed/stopped dense decoder')
    if (out/'curriculum_manifest.json').exists() and not resume: raise FileExistsError('explicit decoder resume required')
    source=pinned_backbone(device); patch=source.backbone.patch_embed; del source
    before=fingerprint_modules({'patch':patch}); model=DenseDecoder(kind,seed).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    config=dict(seed=seed,kind=kind,arm=kind,device=device,phase='supervised',updates=50 if development else 4000,
        batch_size=64,microbatch_size=32,validate_every=10 if development else 100,max_seconds=1200,
        learning_rate=3e-4,weight_decay=1e-4,grad_clip=1.,development=development,
        protocol='overnight D2: local evidence and task FiLM in a shared dense decoder',
        selector='fixed update endpoint; validation is monitoring only',precision='FP32, TF32 off; FP16 final features and FP32 local inputs')
    data,cache_manifest=load_data('vit',development)
    manifest=dict(config=config,dataset=file_hash(cache_root(development)/'vit/manifest.json'),
        source_hashes={str(p):file_hash(p) for p in (Path(__file__),Path('world_model/curriculum/perception_decoder.py'))},
        protocol_sha256=file_hash('docs/perception-decoder-protocol-2026-09-08.md'),
        initial=fingerprint_modules({'decoder':model}),frozen_patch=before,parameters={'decoder':sum(p.numel() for p in model.parameters())},versions=versions())
    rng=np.random.default_rng(seed+230003); models={'decoder':model}; optimizers={'decoder':optimizer}
    begin=time.monotonic(); previous=0.; step=0
    if resume:
        if json.loads((out/'curriculum_manifest.json').read_text())!=manifest: raise ValueError('decoder resume identity changed')
        state=torch.load(out/'last.pt',map_location='cpu',weights_only=False)
        restore_checkpoint(state,models,optimizers,rng); previous=state['elapsed_seconds']; step=state['step']; recover_ledgers(out,step)
    else:
        json_atomic(out/'curriculum_manifest.json',manifest); (out/'training.jsonl').touch()
    if device=='cuda': torch.cuda.reset_peak_memory_stats()
    def elapsed(): return previous+time.monotonic()-begin
    def validate(update):
        c,_,_=evaluate_domain(model,patch,data,'coco','validation',device)
        p,_,_=evaluate_domain(model,patch,data,'pusht','validation',device)
        row=dict(step=update,elapsed_seconds=elapsed(),coco_image_mse=c['image_mse'],pusht_image_mse=p['image_mse'],
            image_mse=.5*(c['image_mse']+p['image_mse']),mask_iou=c['iou'],mask_dice=c['dice'],mask_bce=c['mask_bce'],
            loss=.5*(c['image_mse']+p['image_mse'])+c['mask_bce'])
        append_json(out/'validation.jsonl',row)
        atomic_checkpoint(out/'last.pt',checkpoint_state(models,optimizers,rng,step=update,selected=row,
            elapsed_seconds=elapsed(),config=config,manifest_sha256=file_hash(out/'curriculum_manifest.json')))
        print('decoder',kind,seed,update,'IoU',round(c['iou'],4),'COCO MSE',round(c['image_mse'],5),'seconds',round(elapsed(),1),flush=True)
        return row
    final=json.loads((out/'validation.jsonl').read_text().splitlines()[-1]) if resume else validate(0)
    status='completed'
    for update in range(step+1,config['updates']+1):
        if elapsed()>=config['max_seconds'] or datetime.now(timezone.utc)>=TRAIN_CUTOFF:
            status='stopped_wall_budget'; break
        draws={d:rng.integers(len(data[d,'train']['ds']),size=32) for d in ('coco','pusht')}
        stats=step_update(model,optimizer,patch,data,draws,device,update<=100 or update%100==0)
        step=update
        from .data import digest
        append_json(out/'training.jsonl',dict(step=step,**stats,examples=step*64,
            sample_indices_sha256=digest({k:v.tolist() for k,v in draws.items()}),elapsed_seconds=elapsed()))
        if step%config['validate_every']==0 or step==config['updates']: final=validate(step)
    if final['step']!=step: final=validate(step)
    if before!=fingerprint_modules({'patch':patch}): raise RuntimeError('decoder training changed frozen patch source')
    result=dict(status=status,step=step,selected_step=step,selected=final,final=final,examples=step*64,
        elapsed_seconds=elapsed(),peak_cuda_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else None)
    json_atomic(out/'curriculum_result.json',result)
    return evaluate_saved(out,kind,seed,development,device,model,patch,data)


def evaluate_saved(out,kind,seed,development,device,model=None,patch=None,data=None):
    if model is None:
        model=DenseDecoder(kind,seed).to(device)
        model.load_state_dict(torch.load(out/'last.pt',map_location='cpu',weights_only=False)['heads']['decoder'])
    if patch is None:
        source=pinned_backbone(device); patch=source.backbone.patch_embed; del source
    if data is None: data,_=load_data('vit',development)
    model.eval(); before=fingerprint_modules({'decoder':model,'patch':patch}); start=time.monotonic()
    result=json.loads((out/'curriculum_result.json').read_text()); evaluation={}; sources=[]
    for split in ('train','validation','test'):
        evaluation[split]={}
        for domain in ('coco','pusht'):
            metric,raw,panels=evaluate_domain(model,patch,data,domain,split,device,panels=split=='test',limit=512 if split=='train' else None)
            evaluation[split][domain]=metric
            path=out/f'{domain}_{split}_errors.npz'; np.savez_compressed(path,**raw); sources.append(path)
            if panels:
                path=out/f'{domain}_panels.npz'; np.savez_compressed(path,**panels); sources.append(path)
    evaluation.update(checkpoint_sha256=file_hash(out/'last.pt'),step=result['step'],
        timing=benchmark(model,patch,data,device),seconds=time.monotonic()-start,
        scope='Fixed endpoint shared dense decoder; no pose, encoder adaptation, future prediction or control')
    if before!=fingerprint_modules({'decoder':model,'patch':patch}): raise RuntimeError('dense evaluation mutated state')
    json_atomic(out/'evaluation.json',evaluation); sources+=[out/n for n in ('evaluation.json','curriculum_manifest.json','curriculum_result.json','training.jsonl','validation.jsonl')]
    c,p=evaluation['test']['coco'],evaluation['test']['pusht']
    analyze(ROOT,out,'Task-conditioned dense decoding',dict(test_mask_iou=c['iou'],test_coco_mse=c['image_mse'],test_pusht_mse=p['image_mse']),
        f'## Dense decoder {kind} · seed{seed}\n\n{"Development only. " if development else "Adaptive fixed-endpoint comparison. "}'
        f'{result["status"]}, {result["step"]}updates. COCO foreground IoU={c["iou"]:.4f}; RGB MSE={c["image_mse"]:.6f}. '
        'Final semantic features stay frozen. Local-source and task-signal changes are explicit. Raw-patch input has fewer parameters and different preprocessing support. '
        'Present-frame detail skips do not establish future reconstruction or predictive-state quality.',sources)
    return result


def main():
    p=argparse.ArgumentParser(); p.add_argument('--kind',choices=['late','early','raw','conditioned'],required=True)
    p.add_argument('--seed',type=int,default=9107); p.add_argument('--development',action='store_true')
    p.add_argument('--device',default='cuda'); p.add_argument('--resume',action='store_true'); a=p.parse_args()
    train(a.kind,a.seed,a.development,a.device,a.resume)


if __name__=='__main__': main()
