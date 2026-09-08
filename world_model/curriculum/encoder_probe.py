"""Matched fresh RGB/mask readouts on frozen task and generic representations."""
import json,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .encoder_masks import ROOT,mask_statistics,masked_bce
from .encoder_reference import pinned_backbone
from .encoder_variants import matched_models
from .encoder_factorial import ARMS
from .dino_reference import DinoAdapter
from .data import file_hash,digest
from .pose_accessibility import setup,analyze,PARENT
from .training import initial_models
from world_model.paddle.models import Decoder
from world_model.paddle.types import ObservationLatent
from world_model.paddle.training import optimizer_for
from world_model.pusht.checkpoints import read_checkpoint,fingerprint_modules,json_atomic,atomic_checkpoint,versions


class MaskDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(128,64,3,padding=1),nn.GELU(),nn.Upsample(scale_factor=2,mode='nearest'),
            nn.Conv2d(64,32,3,padding=1),nn.GELU(),nn.Upsample(scale_factor=2,mode='nearest'),
            nn.Conv2d(32,16,3,padding=1),nn.GELU(),nn.Conv2d(16,1,1))
    def forward(self,z):
        fine=z.fine.transpose(1,2).reshape(-1,64,16,16);coarse=z.coarse.transpose(1,2).reshape(-1,64,8,8)
        return self.net(torch.cat((fine,F.interpolate(coarse,size=(16,16),mode='nearest')),1))


def source_models(arm,seed,development,device):
    if arm=='dino':
        checkpoint=ROOT/('development/reference/dino/best.pt' if development else 'reference/dino/best.pt')
        state=torch.load(checkpoint,map_location='cpu',weights_only=False);adapter=DinoAdapter()
        adapter.load_state_dict(state['models']['A']);models={'E':pinned_backbone(device),'A':adapter.to(device)}
    else:
        checkpoint=PARENT if arm=='custom' else Path('runs/curriculum_2026-09-07/seed_4107/B/warmup/best.pt') if arm=='warmup' else ROOT/('development/factorial' if development else 'factorial')/f'seed_{seed}'/arm/'best.pt'
        state=read_checkpoint(checkpoint)
        model=initial_models(6107)['E'] if arm in ('custom','warmup') else matched_models(seed,*ARMS[arm])['E']
        model.load_state_dict(state['models']['E']);models={'E':model.to(device)}
    for model in models.values():model.eval().requires_grad_(False)
    return models,checkpoint


def source_forward(models,rgb):
    z=models['E'](rgb)
    return models['A'](z) if 'A' in models else z


def masks(development):
    root=ROOT/('development/coco_masks' if development else 'coco_masks');manifest=json.loads((root/'manifest.json').read_text())
    data={}
    for split in ('train','validation','test'):
        p=root/f'{split}.npz'
        if file_hash(p)!=manifest['splits'][split]['sha256']:raise ValueError('mask population changed')
        data[split]=dict(np.load(p))
    return root,manifest,data


def rgb_batch(frames,rows,device):
    return torch.from_numpy(np.asarray(frames[rows]).copy()).permute(0,3,1,2).float().to(device)/255


def prepare_features(out,arm,seed,development,device,labels,frames):
    if (out/'features.json').exists():raise FileExistsError('preserve completed probe features')
    models,checkpoint=source_models(arm,seed,development,device);before=fingerprint_modules(models);splits={}
    for split,lab in labels.items():
        path=out/f'{split}_features.npy';cache=np.lib.format.open_memmap(path,mode='w+',dtype='float16',shape=(len(lab['rows']),320,64))
        num=den=0.
        for start in range(0,len(cache),32):
            rgb=rgb_batch(frames,lab['rows'][start:start+32],device)
            with torch.no_grad():z=source_forward(models,rgb).tokens()
            cache[start:start+len(z)]=z.cpu().half().numpy()
            num+=float((z-z.half().float()).square().sum());den+=float(z.square().sum())
        cache.flush();del cache
        if num/max(den,1e-30)>1e-6:raise RuntimeError('probe cache quantization gate')
        splits[split]=dict(sha256=file_hash(path),relative_mse=num/max(den,1e-30),rows_sha256=digest(lab['rows'].tolist()))
    if before!=fingerprint_modules(models):raise RuntimeError('probe changed frozen representation')
    receipt=dict(source_checkpoint=str(checkpoint),source_checkpoint_sha256=file_hash(checkpoint),frozen_fingerprint=before,splits=splits,
                 precision='FP32 inference, TF32 disabled, FP16 cache then FP32 head training')
    json_atomic(out/'features.json',receipt)
    return receipt


def batch(out,split,labels,frames,ids,device):
    cache=np.load(out/f'{split}_features.npy',mmap_mode='r')
    z=ObservationLatent.from_tokens(torch.from_numpy(np.asarray(cache[ids]).copy()).float().to(device))
    lab=labels[split];rgb=rgb_batch(frames,lab['rows'][ids],device)
    target=torch.from_numpy(lab['masks'][ids].copy()).float().unsqueeze(1).to(device)
    valid=torch.from_numpy(lab['valid'][ids].copy()).float().unsqueeze(1).to(device)
    return z,rgb,target,valid


@torch.no_grad()
def evaluate(models,out,split,labels,frames,device,return_panels=False):
    for m in models.values():m.eval()
    records={k:[] for k in ('image_mse','mask_bce','iou','dice','has_valid','foreground_fraction')};panels=[]
    for start in range(0,len(labels[split]['rows']),64):
        ids=np.arange(start,min(start+64,len(labels[split]['rows'])))
        z,rgb,target,valid=batch(out,split,labels,frames,ids,device);recon=models['D'](z);logits=models['M'](z)
        per_bce=(F.binary_cross_entropy_with_logits(logits,target,reduction='none')*valid).sum((1,2,3))/valid.sum((1,2,3)).clamp_min(1)
        records['image_mse'].extend((recon-rgb).square().mean((1,2,3)).cpu().tolist());records['mask_bce'].extend(per_bce.cpu().tolist())
        for k,v in mask_statistics(logits.sigmoid(),target,valid).items():records[k].extend(v.tolist())
        if return_panels and start==0:
            panels=[t[:6].cpu().numpy() for t in (rgb,recon,target,logits.sigmoid(),valid)]
    raw={k:np.asarray(v) for k,v in records.items()};keep=raw['has_valid'].astype(bool)
    metric=dict(frames=len(keep),mask_frames=int(keep.sum()),image_mse=float(raw['image_mse'].mean()),
                mask_bce=float(raw['mask_bce'][keep].mean()),iou=float(raw['iou'][keep].mean()),dice=float(raw['dice'][keep].mean()))
    metric['loss']=metric['image_mse']+metric['mask_bce']
    strata={}
    for name,lo,hi in [('empty',-.1,0),('small',0,.1),('medium',.1,.5),('large',.5,1.)]:
        which=keep&(raw['foreground_fraction']>lo)&(raw['foreground_fraction']<=hi)
        strata[name]=dict(frames=int(which.sum()),iou=float(raw['iou'][which].mean()) if which.any() else None)
    return metric,raw,strata,panels


def train(arm,seed=7107,development=False,device='cuda'):
    setup();key=arm if arm in ('custom','warmup','dino') else f'{seed}_{arm}'
    out=ROOT/('development/probes' if development else 'probes')/key;out.mkdir(parents=True,exist_ok=True)
    if (out/'last.pt').exists():raise FileExistsError('preserve probe run')
    mask_root,mask_manifest,labels=masks(development);frames=np.load('data/curriculum/coco_v1/frames.npy',mmap_mode='r')
    features_receipt=prepare_features(out,arm,seed,development,device,labels,frames)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(8157);models={'D':Decoder().to(device),'M':MaskDecoder().to(device)}
    config=dict(seed=8157,source_seed=seed,arm=key,phase='warmup',updates=3 if development else 2000,batch_size=64,
                learning_rate=.0003,weight_decay=.0001,grad_clip=1.,validate_every=1 if development else 100,
                max_seconds=900,development=development,protocol='A3/B audit: frozen E, fresh independent RGB/mask decoders')
    opt=optimizer_for(models,config);rng=np.random.default_rng(8158);begin=time.monotonic();best=None;selected=None;curve=[]
    json_atomic(out/'curriculum_manifest.json',dict(config=config,dataset=file_hash(mask_root/'manifest.json'),
        features=features_receipt,initial=fingerprint_modules(models),source_code_sha256=file_hash(__file__),versions=versions()))
    def validate(step):
        nonlocal best,selected
        metric,*_=evaluate(models,out,'validation',labels,frames,device)
        row=dict(step=step,elapsed_seconds=time.monotonic()-begin,**metric);curve.append(row)
        with (out/'validation.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        state=dict(models={k:{n:v.detach().cpu().clone() for n,v in m.state_dict().items()} for k,m in models.items()},config=config,step=step,metrics=row,
                   features_sha256=file_hash(out/'features.json'))
        atomic_checkpoint(out/'last.pt',state)
        if best is None or (metric['loss'],step)<best:
            best=(metric['loss'],step);selected=row;atomic_checkpoint(out/'best.pt',state)
        print('probe',key,step,'RGB',round(metric['image_mse'],5),'IoU',round(metric['iou'],4),flush=True)
    validate(0)
    for step in range(1,config['updates']+1):
        if time.monotonic()-begin>config['max_seconds']:raise TimeoutError('probe wall cap')
        ids=rng.integers(len(labels['train']['rows']),size=64);z,rgb,target,valid=batch(out,'train',labels,frames,ids,device)
        for m in models.values():m.train()
        opt.zero_grad(set_to_none=True);pixel=(models['D'](z)-rgb).square().mean();bce=masked_bce(models['M'](z),target,valid);loss=pixel+bce
        loss.backward();grad=nn.utils.clip_grad_norm_([p for m in models.values() for p in m.parameters()],1.)
        if not torch.isfinite(loss) or not torch.isfinite(grad):raise RuntimeError('nonfinite probe')
        opt.step()
        with (out/'training.jsonl').open('a') as f:f.write(json.dumps(dict(step=step,loss=float(loss.detach()),image_mse=float(pixel.detach()),mask_bce=float(bce.detach()),
            examples=step*64,sample_indices_sha256=digest(ids.tolist()),elapsed_seconds=time.monotonic()-begin))+'\n')
        if step%config['validate_every']==0 or step==config['updates']:validate(step)
    result=dict(status='completed',step=step,selected_step=selected['step'],selected=selected,final=curve[-1],examples=step*64,elapsed_seconds=time.monotonic()-begin)
    json_atomic(out/'curriculum_result.json',result)
    saved=torch.load(out/'best.pt',map_location='cpu',weights_only=False)
    for k,m in models.items():m.load_state_dict(saved['models'][k])
    metrics={};strata={};sources=[out/'curriculum_manifest.json',out/'curriculum_result.json',out/'training.jsonl',out/'validation.jsonl',out/'features.json']
    for split in ('train','validation','test'):
        metric,raw,s,p=evaluate(models,out,split,labels,frames,device,return_panels=split=='test');metrics[split]=metric;strata[split]=s
        path=out/f'{split}_errors.npz';np.savez_compressed(path,**raw,rows=labels[split]['rows'],groups=labels[split]['groups']);sources.append(path)
        if p:np.savez_compressed(out/'panels.npz',rgb=p[0],reconstruction=p[1],mask=p[2],probability=p[3],valid=p[4],rows=labels[split]['rows'][:6])
    # Baselines use training masks only; no validation threshold optimization.
    train=labels['train'];counts=train['valid'].sum(0);mean=(train['masks']*train['valid']).sum(0)/np.maximum(counts,1)
    baselines={}
    for name,prob in [('empty',np.zeros((64,64))),('full',np.ones((64,64))),('training_mean',mean)]:
        lab=labels['test'];target=torch.from_numpy(lab['masks']).float()[:,None];valid=torch.from_numpy(lab['valid']).float()[:,None]
        s=mask_statistics(torch.from_numpy(prob).float()[None,None].expand_as(target),target,valid);keep=s['has_valid']
        baselines[name]={k:float(s[k][keep].mean()) for k in ('iou','dice')}
    json_atomic(out/'evaluation.json',dict(metrics=metrics,strata=strata,baselines=baselines,
        empty_convention='IoU/Dice=1 if both masks empty on valid pixels; no-valid images excluded from mask aggregates',
        selected_checkpoint_sha256=file_hash(out/'best.pt'),scope='annotated foreground coverage on reused COCO groups; DINO pretraining overlap not excluded'))
    sources.append(out/'evaluation.json')
    analyze(ROOT,out,'Frozen representation RGB/foreground audit',{'test_image_mse':metrics['test']['image_mse'],'test_iou':metrics['test']['iou']},
        f"## {key} · RGB and foreground audit\n\nFresh readouts on frozen representation; same COCO views/draws. Test RGB MSE={metrics['test']['image_mse']:.6f}; annotated foreground IoU={metrics['test']['iou']:.4f}. Crowd ignored; empty/full/mean-mask controls retained. This does not demonstrate separate object instances or general-agent capability.",sources)
    return result
