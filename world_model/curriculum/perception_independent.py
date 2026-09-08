"""Split dense decoder trunks with the declared joint optimizer/clipping rule."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from .perception_decoder import DenseDecoder
from .perception_decoder_training import batch, step_update, evaluate_domain, benchmark
from .perception_cache import ROOT, cache_root, check_storage
from .perception_training import (load_data, append_json, checkpoint_state, restore_checkpoint,
    recover_ledgers, TRAIN_CUTOFF)
from .encoder_reference import pinned_backbone
from .data import file_hash, digest
from .pose_accessibility import setup, analyze
from world_model.pusht.checkpoints import json_atomic, atomic_checkpoint, fingerprint_modules, versions


class IndependentDecoder(nn.Module):
    def __init__(self, seed):
        super().__init__()
        self.kind = 'early'
        self.heads = nn.ModuleDict({task: DenseDecoder('early', seed) for task in ('rgb', 'mask')})
        for task, head in self.heads.items():
            del head.outputs['mask' if task == 'rgb' else 'rgb']

    def forward(self, features, task):
        return self.heads[task](features, task)

    def both(self, features):
        return self(features, 'rgb'), self(features, 'mask')


@torch.no_grad()
def initial_agreement(model, patch, data, seed, device):
    reference = DenseDecoder('early', seed).to(device).eval()
    model.eval(); differences = {}
    for domain in ('coco', 'pusht'):
        features, _ = batch(data[domain, 'train'], np.arange(4), patch, 'early', device)
        for task in ('rgb', 'mask'):
            differences[domain+'_'+task] = float((reference(features, task)-model(features, task)).abs().max())
    result = dict(device=device, maximum=max(differences.values()), differences=differences,
        tolerance=2e-5, scope='FP32 real inputs before the first optimizer update; mask comparison uses logits')
    if result['maximum'] > result['tolerance']: raise AssertionError('split initialization changes initial function')
    return result


def train(seed, development=False, device='cuda', resume=False):
    setup(); check_storage(200*1024**2)
    out=ROOT/('development/independent' if development else 'independent')/f'seed_{seed}'/'split'
    out.mkdir(parents=True, exist_ok=True)
    if (out/'curriculum_result.json').exists():
        if resume and not (out/'curriculum_analysis.json').exists(): return evaluate_saved(out,seed,development,device)
        raise FileExistsError('preserve completed/stopped split decoder')
    if (out/'curriculum_manifest.json').exists() and not resume: raise FileExistsError('explicit split resume required')
    source=pinned_backbone(device); patch=source.backbone.patch_embed; del source
    before=fingerprint_modules({'patch':patch}); model=IndependentDecoder(seed).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    config=dict(seed=seed,kind='split',arm='split',device=device,phase='supervised',updates=50 if development else 4000,
        batch_size=64,microbatch_size=32,validate_every=10 if development else 100,max_seconds=1200,
        learning_rate=3e-4,weight_decay=1e-4,grad_clip=1.,development=development,
        protocol='overnight D3: split trunks, joint clipping; frozen local and contextual inputs',
        selector='fixed update endpoint; validation is monitoring only',precision='FP32, TF32 off; FP16 final features and FP32 local inputs')
    data,_=load_data('vit',development)
    manifest=dict(config=config,dataset=file_hash(cache_root(development)/'vit/manifest.json'),
        source_hashes={str(p):file_hash(p) for p in (Path(__file__),Path('world_model/curriculum/perception_decoder.py'),Path('world_model/curriculum/perception_decoder_training.py'))},
        protocol_sha256=file_hash('docs/perception-independent-decoder-protocol-2026-09-09.md'),
        initial=fingerprint_modules({'decoder':model}),frozen_patch=before,
        parameters={'decoder':sum(p.numel() for p in model.parameters())},versions=versions())
    rng=np.random.default_rng(seed+230003); models={'decoder':model}; optimizers={'decoder':optimizer}
    begin=time.monotonic(); previous=0.; step=0
    if resume:
        if json.loads((out/'curriculum_manifest.json').read_text())!=manifest: raise ValueError('split resume identity changed')
        state=torch.load(out/'last.pt',map_location='cpu',weights_only=False)
        restore_checkpoint(state,models,optimizers,rng); previous=state['elapsed_seconds']; step=state['step']; recover_ledgers(out,step)
    else:
        json_atomic(out/'initial_agreement.json',initial_agreement(model,patch,data,seed,device))
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
        print('split decoder',seed,update,'IoU',round(c['iou'],4),'COCO MSE',round(c['image_mse'],5),'seconds',round(elapsed(),1),flush=True)
        return row
    final=json.loads((out/'validation.jsonl').read_text().splitlines()[-1]) if resume else validate(0)
    status='completed'
    for update in range(step+1,config['updates']+1):
        if elapsed()>=config['max_seconds'] or datetime.now(timezone.utc)>=TRAIN_CUTOFF:
            status='stopped_wall_budget'; break
        draws={d:rng.integers(len(data[d,'train']['ds']),size=32) for d in ('coco','pusht')}
        stats=step_update(model,optimizer,patch,data,draws,device,audit=False); step=update
        append_json(out/'training.jsonl',dict(step=step,**stats,examples=step*64,
            sample_indices_sha256=digest({k:v.tolist() for k,v in draws.items()}),elapsed_seconds=elapsed()))
        if step%config['validate_every']==0 or step==config['updates']: final=validate(step)
    if final['step']!=step: final=validate(step)
    if before!=fingerprint_modules({'patch':patch}): raise RuntimeError('split training changed frozen patch source')
    result=dict(status=status,step=step,selected_step=step,selected=final,final=final,examples=step*64,
        elapsed_seconds=elapsed(),peak_cuda_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else None)
    json_atomic(out/'curriculum_result.json',result)
    return evaluate_saved(out,seed,development,device,model,patch,data)


def evaluate_saved(out,seed,development,device,model=None,patch=None,data=None):
    if model is None:
        model=IndependentDecoder(seed).to(device)
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
    timing=benchmark(model,patch,data,device)
    timing['scope']='decoder only, prepared feature inputs; one branch per output and two passes for both'
    training=[json.loads(line) for line in (out/'training.jsonl').read_text().splitlines()]
    evaluation.update(checkpoint_sha256=file_hash(out/'last.pt'),step=result['step'],timing=timing,
        global_clip_fraction=float(np.mean([r['grad_norm']>1 for r in training])),seconds=time.monotonic()-start,
        scope='Split parameter ownership and cross-domain transfer; joint gradient clipping still couples branch scales. No encoder, prediction or control change.')
    if before!=fingerprint_modules({'decoder':model,'patch':patch}): raise RuntimeError('split evaluation mutated state')
    json_atomic(out/'evaluation.json',evaluation)
    sources+=[out/n for n in ('evaluation.json','initial_agreement.json','curriculum_manifest.json','curriculum_result.json','training.jsonl','validation.jsonl')]
    c,p=evaluation['test']['coco'],evaluation['test']['pusht']
    analyze(ROOT,out,'Split dense decoder trunks',dict(test_mask_iou=c['iou'],test_coco_mse=c['image_mse'],test_pusht_mse=p['image_mse']),
        f'## Split trunks, joint clipping · seed{seed}\n\n{"Development only. " if development else "Adaptive fixed-endpoint comparison. "}'
        f'{result["status"]}, {result["step"]}updates. COCO foreground IoU={c["iou"]:.4f}; RGB MSE={c["image_mse"]:.6f}. '
        'Both branches receive identical early/final/pooled frozen ViT inputs. Sharing and cross-domain gradient transfer change; '
        'the common clipping coefficient still couples branch scales. Parameter count is higher. No future-pixel or control claim.',sources)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--seed',type=int,default=9107)
    parser.add_argument('--development',action='store_true'); parser.add_argument('--device',default='cuda')
    parser.add_argument('--resume',action='store_true'); args=parser.parse_args()
    train(args.seed,args.development,args.device,args.resume)
