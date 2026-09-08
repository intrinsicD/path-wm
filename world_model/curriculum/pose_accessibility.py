"""Bounded frozen-E PushT readout comparison with exact cached features."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .bottleneck_models import NonlinearPoseHead, SpatialPoseHead
from .data import task_frames, file_hash, digest
from .training import initial_models
from .decoder_recovery import paired_group_bootstrap
from world_model.paddle.types import ObservationLatent
from world_model.paddle.training import optimizer_for
from world_model.pusht.models import PoseReadout
from world_model.pusht.checkpoints import read_checkpoint, fingerprint_modules, json_atomic, atomic_checkpoint, versions

ROOT=Path('runs/bottlenecks_2026-09-08/pose')
PARENT=Path('runs/curriculum_2026-09-07/seed_4107/A/supervised/best.pt')
SEED=6107
HEADS={'linear':PoseReadout,'nonlinear':NonlinearPoseHead,'spatial':SpatialPoseHead}


def setup():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False


def analyze(root,out,purpose,metrics,narrative,sources,panels=()):
    json_atomic(out/'curriculum_analysis.json',dict(status='completed',purpose=purpose,metrics=metrics,
        narrative=narrative,sources={p.relative_to(root).as_posix():file_hash(p) for p in sources},panels=list(panels)))


def prepare(root,development=False,device='cuda'):
    out=root/'features';out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.json').exists():raise FileExistsError('preserve completed feature extraction')
    parent=read_checkpoint(PARENT);models=initial_models(SEED,parent['models'])
    models['H'].load_state_dict(parent['models']['H'])
    encoder=models['E'].to(device).eval().requires_grad_(False)
    before=fingerprint_modules({'E':encoder});begin=time.monotonic();splits={}
    for split in ('train','validation','test'):
        ds=task_frames('data/pusht_world_model/cchi_v1',split)
        ids=np.arange(min(len(ds),48 if split=='train' else 16)) if development else np.arange(len(ds))
        cache=np.lib.format.open_memmap(out/f'{split}.npy',mode='w+',dtype='float32',shape=(len(ids),320,64))
        for start in range(0,len(ids),128):
            rows=ids[start:start+128];x,_=ds.batch(rows,device)
            with torch.no_grad():cache[start:start+len(rows)]=encoder(x).tokens().cpu().numpy()
        cache.flush();del cache
        np.savez_compressed(out/f'{split}_labels.npz',targets=ds.targets[ids],indices=ids,
            groups=np.array([ds.metadata[i]['group'] for i in ids]))
        splits[split]=dict(frames=len(ids),dataset=ds.fingerprint,rows_sha256=digest(ds.rows[ids].tolist()),
            features_sha256=file_hash(out/f'{split}.npy'),labels_sha256=file_hash(out/f'{split}_labels.npz'))
        print('cached',split,len(ids),flush=True)
    if fingerprint_modules({'E':encoder})!=before:raise RuntimeError('feature extraction changed encoder')
    manifest=dict(parent=str(PARENT),parent_sha256=file_hash(PARENT),encoder_fingerprint=before,
        frozen_encoder_exact=True,precision='FP32; TF32 disabled',versions=versions(),seed=SEED,splits=splits,
        development=development,seconds=time.monotonic()-begin,
        code={str(p):file_hash(p) for p in [Path(__file__),Path('world_model/curriculum/bottleneck_models.py')]})
    json_atomic(out/'manifest.json',manifest)
    analyze(root,out,'Frozen PushT features prepared',{'training_frames':splits['train']['frames']},
        '## Frozen PushT feature cache\n\n'+('Development subset only. ' if development else '')+
        'Selected task-only A encoder; exact FP32 features, original grouped splits. Encoder state unchanged. No pose-quality result yet.',[out/'manifest.json'])


def features(root,split):
    m=json.loads((root/'features/manifest.json').read_text())
    p=root/'features'/f'{split}.npy';label=root/'features'/f'{split}_labels.npz'
    if file_hash(p)!=m['splits'][split]['features_sha256'] or file_hash(label)!=m['splits'][split]['labels_sha256']:
        raise ValueError('feature cache changed')
    return np.load(p,mmap_mode='r'),dict(np.load(label))


def batch(features,ids,device):
    return ObservationLatent.from_tokens(torch.from_numpy(np.asarray(features[ids]).copy()).to(device))


def pose_metrics(pred,target):
    delta=np.abs(pred[:,:4]-target[:,:4])*512
    angles=np.arctan2(pred[:,4],pred[:,5])-np.arctan2(target[:,4],target[:,5])
    angle=np.abs(np.arctan2(np.sin(angles),np.cos(angles)))*180/np.pi
    norms=np.linalg.norm(pred[:,4:6],axis=1)
    metric=dict(position_mae=delta.mean(0).tolist(),angle_mae_deg=float(angle.mean()),
        pose_mse=float(np.square(pred-target).mean()),angle_norm_mean=float(norms.mean()),
        near_zero_angle_norm_count=int((norms<1e-6).sum()),frames=len(pred))
    metric['q']=max([v/8 for v in metric['position_mae']]+[metric['angle_mae_deg']/10])
    return metric,dict(predictions=pred,targets=target,position_abs_error=delta,angle_abs_error_deg=angle,angle_norm=norms)


@torch.no_grad()
def evaluate_head(head,x,y,ids,device):
    head.eval();pred=[]
    for start in range(0,len(ids),128):pred.append(head(batch(x,ids[start:start+128],device)).cpu().double().numpy())
    return pose_metrics(np.concatenate(pred),y[ids].astype(np.float64))


def train(root,arm,development=False,device='cuda'):
    out=root/arm;out.mkdir(parents=True,exist_ok=True)
    if (out/'last.pt').exists():raise FileExistsError('preserve completed/interrupted head run')
    manifest=json.loads((root/'features/manifest.json').read_text())
    if file_hash(PARENT)!=manifest['parent_sha256']:raise ValueError('parent changed')
    x,train_labels=features(root,'train');v,val_labels=features(root,'validation')
    ids=np.random.default_rng(SEED).choice(len(v),min(2048,len(v)),replace=False)
    torch.manual_seed(SEED);head=HEADS[arm]().to(device)
    config=dict(seed=SEED,arm=arm,updates=3 if development else 2000,batch_size=128,
        learning_rate=.0003,weight_decay=.0001,grad_clip=1.,max_seconds=1800,
        validate_every=1 if development else 100,landmark_weight=1. if arm=='spatial' else 0.,development=development)
    opt=optimizer_for({'H':head},config);rng=np.random.default_rng(SEED+1)
    initial=fingerprint_modules({'H':head});begin=time.monotonic();best=None;selected=None;curve=[]
    json_atomic(out/'manifest.json',dict(config=config,parent_sha256=manifest['parent_sha256'],
        encoder_fingerprint=manifest['encoder_fingerprint'],validation_indices=ids.tolist(),initial_head=initial,
        parameters=sum(p.numel() for p in head.parameters()),source_sha256=file_hash(__file__)))
    def validate(step):
        nonlocal best,selected
        metric,_=evaluate_head(head,v,val_labels['targets'],ids,device)
        row=dict(step=step,**metric);curve.append(row)
        with (out/'validation.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        key=(metric['q'],metric['pose_mse'],step)
        saved=dict(schema='frozen-pose-head-v1',arm=arm,step=step,head={k:t.detach().cpu().clone() for k,t in head.state_dict().items()},
            parent_sha256=manifest['parent_sha256'],encoder_fingerprint=manifest['encoder_fingerprint'],config=config,metrics=metric)
        atomic_checkpoint(out/'last.pt',saved)
        if best is None or key<best:
            best=key;selected=step;atomic_checkpoint(out/'best.pt',saved)
        print(arm,step,'q',round(metric['q'],4),'angle',round(metric['angle_mae_deg'],3),flush=True)
        return metric
    validate(0);step=0
    for step in range(1,config['updates']+1):
        if time.monotonic()-begin>config['max_seconds']:raise TimeoutError('head wall budget reached; last checkpoint retained')
        draw=rng.integers(len(x),size=128);z=batch(x,draw,device)
        y=torch.from_numpy(train_labels['targets'][draw].copy()).to(device)
        head.train();opt.zero_grad(set_to_none=True)
        if arm=='spatial':
            pred,points,_=head.details(z)
            loss=(pred-y).square().mean()+(points-head.target_points(y)).square().mean()
        else:loss=(head(z)-y).square().mean()
        if not torch.isfinite(loss):raise RuntimeError('nonfinite pose loss')
        loss.backward();grad=nn.utils.clip_grad_norm_(head.parameters(),1.)
        if not torch.isfinite(grad):raise RuntimeError('nonfinite pose gradient')
        opt.step()
        with (out/'training.jsonl').open('a') as f:f.write(json.dumps(dict(step=step,loss=float(loss.detach()),
            grad_norm=float(grad),sample_indices_sha256=digest(draw.tolist()),examples=step*128))+'\n')
        if step%config['validate_every']==0 or step==config['updates']:validate(step)
    if file_hash(PARENT)!=manifest['parent_sha256']:raise ValueError('parent changed during head fitting')
    result=dict(status='completed',step=step,selected_step=selected,selected=curve[[r['step'] for r in curve].index(selected)],
        final=curve[-1],seconds=time.monotonic()-begin,examples=128*step,frozen_encoder_exact=True,
        peak_cuda_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else None)
    json_atomic(out/'result.json',result)
    fig,ax=plt.subplots(figsize=(7,3.5),layout='constrained');ax.plot([r['step'] for r in curve],[r['q'] for r in curve],color='#245a90')
    ax.axhline(1,color='#777777',ls='--');ax.set(xlabel='Head optimizer updates',ylabel='Worst normalized pose MAE (q)',title=f'{arm} head · fixed validation · seed {SEED}');ax.set_ylim(bottom=0)
    image=out/'validation.png';fig.savefig(image,dpi=120);plt.close(fig)
    analyze(root,out,f'Frozen PushT {arm} head',{'selected_q':result['selected']['q'],'selected_angle_mae_deg':result['selected']['angle_mae_deg']},
        f'## Frozen PushT {arm} head\n\n'+('Development subset only. ' if development else '')+
        f"{step} updates; selected update {selected}; q={result['selected']['q']:.4f}. Encoder frozen. q≤1 remains the readiness target; no downstream training follows.",
        [out/'manifest.json',out/'result.json',out/'training.jsonl',out/'validation.jsonl',image],
        [dict(file=str(image),title=f'{arm} validation',caption='Same fixed validation population; lower is better.',embed=True)])


def final_evaluation(root,device='cuda'):
    out=root/'evaluation';out.mkdir(parents=True,exist_ok=True)
    if (out/'metrics.json').exists():raise FileExistsError('preserve completed head evaluation')
    fm=json.loads((root/'features/manifest.json').read_text());parent=read_checkpoint(PARENT)
    if file_hash(PARENT)!=fm['parent_sha256']:raise ValueError('parent mutated')
    reference=PoseReadout().to(device);reference.load_state_dict(parent['models']['H'])
    heads={'original':reference};steps={};streams=[];sources=[]
    for arm,cls in HEADS.items():
        streams.append([json.loads(s)['sample_indices_sha256'] for s in (root/arm/'training.jsonl').read_text().splitlines()])
        for which in ('best','last'):
            saved=torch.load(root/arm/f'{which}.pt',map_location='cpu',weights_only=False)
            if saved['parent_sha256']!=fm['parent_sha256']:raise ValueError('head source mismatch')
            head=cls().to(device);head.load_state_dict(saved['head']);heads[f'{arm}_{which}']=head;steps[f'{arm}_{which}']=saved['step']
    if any(s!=streams[0] for s in streams[1:]):raise ValueError('unmatched training streams')
    metrics={name:{} for name in heads};raw={};data={};groups={};split_ids={}
    for split in ('train','validation','test'):
        x,labels=features(root,split);data[split]=(x,labels)
        ids=(np.random.default_rng(SEED).choice(len(x),min(2048,len(x)),replace=False) if split!='test' else np.arange(len(x)))
        groups[split]=labels['groups'][ids];split_ids[split]=ids
        for name,head in heads.items():
            m,r=evaluate_head(head,x,labels['targets'],ids,device);metrics[name][split]=m;raw[(name,split)]=r
            path=out/f'{name}_{split}.npz';np.savez_compressed(path,**r,indices=labels['indices'][ids],groups=groups[split]);sources.append(path)
            print(name,split,'q',round(m['q'],4),'angle',round(m['angle_mae_deg'],3),flush=True)
    comparisons={}
    for i,arm in enumerate(HEADS):
        name=f'{arm}_best';a=raw[(name,'test')];b=raw[('original','test')]
        ci,draws=paired_group_bootstrap(a['angle_abs_error_deg'],b['angle_abs_error_deg'],groups['test'],SEED+100+i)
        ci['unit']='degrees; whole configuration-group resampling, frame-weighted mean'
        ci['delta_angle_mae_deg']=ci.pop('delta_mse');comparisons[arm]=ci
        p=out/f'{arm}_angle_bootstrap.npy';np.save(p,draws);sources.append(p)
    group_report={}
    for name in heads:
        record=raw[(name,'test')];group_report[name]={}
        for group in np.unique(groups['test']):
            mask=groups['test']==group
            group_report[name][str(int(group))]=dict(frames=int(mask.sum()),
                angle_mae_deg=float(record['angle_abs_error_deg'][mask].mean()),
                position_mae=record['position_abs_error'][mask].mean(0).tolist())
    json_atomic(out/'configuration_metrics.json',group_report);sources.append(out/'configuration_metrics.json')
    gates={arm:dict(relative_validation_improvement=metrics[f'{arm}_best']['validation']['q']<=.9*metrics['original']['validation']['q'],
        numeric_readiness=metrics[f'{arm}_best']['validation']['q']<=1) for arm in HEADS}
    report=dict(status='completed',metrics=metrics,selected_steps=steps,comparisons=comparisons,gates=gates,
        same_training_streams=True,frozen_encoder_exact=True,parent_sha256=fm['parent_sha256'],
        scope='development subset' if fm['development'] else 'one seed; reused internal configuration-disjoint splits; no independent confirmation')
    json_atomic(out/'metrics.json',report);sources.append(out/'metrics.json')
    selected=['original']+[a+'_best' for a in HEADS];colors=['#777777','#245a90','#245a90','#ac651f'];styles=[':',':','--','-']
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for name,color,style in zip(selected,colors,styles):
        axes[0].plot(['train sample','validation','test'],[metrics[name][s]['angle_mae_deg'] for s in ('train','validation','test')],label=name,color=color,ls=style,marker='o')
        axes[1].plot(range(4),metrics[name]['test']['position_mae'],label=name,color=color,ls=style,marker='o')
    axes[0].set(ylabel='Orientation MAE (degrees)',title='Orientation by population');axes[1].set(ylabel='Position MAE (world units)',xticks=range(4),xticklabels=['pusher x','pusher y','object x','object y'],title='Held-out position')
    for ax in axes:ax.set_ylim(bottom=0);ax.grid(axis='y',alpha=.2)
    axes[0].legend(fontsize=8);fig.suptitle('Frozen-encoder pose heads · seed 6107 · selected by validation q')
    plot=out/'pose_errors.png';fig.savefig(plot,dpi=120);fig.savefig(plot.with_suffix('.svg'));plt.close(fig);sources.append(plot)
    x,labels=data['test'];ids=np.random.default_rng(SEED+7).choice(len(x),min(6,len(x)),replace=False)
    ds=task_frames('data/pusht_world_model/cchi_v1','test');rgb,_=ds.batch(labels['indices'][ids])
    z=batch(x,ids,device)
    with torch.no_grad():pose,points,heatmaps=heads['spatial_best'].details(z)
    np.savez_compressed(out/'spatial_panels.npz',indices=labels['indices'][ids],points=points.cpu().numpy(),heatmaps=heatmaps.cpu().numpy(),targets=labels['targets'][ids]);sources.append(out/'spatial_panels.npz')
    fig,axes=plt.subplots(4,len(ids),figsize=(11,7),layout='constrained')
    for c in range(len(ids)):
        axes[0,c].imshow(rgb[c].permute(1,2,0));truth=SpatialPoseHead.target_points(torch.from_numpy(labels['targets'][ids]))[c].numpy()*64
        pred=points[c].cpu().numpy()*64
        axes[0,c].scatter(truth[:,0],truth[:,1],marker='+',c='#245a90',s=45,label='truth')
        axes[0,c].scatter(pred[:,0],pred[:,1],marker='x',c='#ac651f',s=30,label='prediction')
        axes[0,c].set_title(f'Test row {labels["indices"][ids[c]]}',fontsize=8)
        for j in range(3):axes[j+1,c].imshow(heatmaps[c,j].cpu(),cmap='Greys',vmin=0,vmax=float(heatmaps.max()))
    for ax in axes.flat:ax.set_xticks([]);ax.set_yticks([])
    for row,title in enumerate(['Image + points','Pusher heatmap','Object center','Orientation point']):axes[row,0].set_ylabel(title,fontsize=8)
    axes[0,0].legend(fontsize=6,loc='lower left');fig.suptitle('Spatial head · six fixed test frames\nBlue + = true landmark; orange × = predicted; shared heatmap scale',fontsize=11)
    panel=out/'spatial_heatmaps.png';fig.savefig(panel,dpi=120);fig.savefig(panel.with_suffix('.svg'));plt.close(fig);sources.append(panel)
    curvefig,ax=plt.subplots(figsize=(8,4),layout='constrained')
    for arm,color,style in zip(HEADS,colors[1:],styles[1:]):
        rows=[json.loads(s) for s in (root/arm/'validation.jsonl').read_text().splitlines()]
        ax.plot([r['step'] for r in rows],[r['q'] for r in rows],color=color,ls=style,label=arm)
    ax.axhline(metrics['original']['validation']['q'],color='#777777',ls=':',label='original head')
    ax.axhline(1,color='#555555',ls='--',label='readiness');ax.set(xlabel='Head updates',ylabel='Validation q · log scale',yscale='log',title='Matched frozen-feature head fitting');ax.legend(fontsize=8)
    curve=out/'learning_curves.png';curvefig.savefig(curve,dpi=120);curvefig.savefig(curve.with_suffix('.svg'));plt.close(curvefig);sources.append(curve)
    text='## Frozen PushT pose accessibility\n\n'+report['scope']+'. All encoders unchanged; fresh-head sample streams identical.\n\n| Head | Validation q | Test angle MAE (degrees) | Test q |\n|---|---:|---:|---:|\n'
    for n in selected:text+=f"| {n} | {metrics[n]['validation']['q']:.4f} | {metrics[n]['test']['angle_mae_deg']:.4f} | {metrics[n]['test']['q']:.4f} |\n"
    text+='\nReadiness is q≤1. Head capacity and spatial supervision differ; this does not isolate an architecture-only effect or prove information loss. No PushT dynamics or control was trained. Further experiments are deferred for user reassessment.'
    analyze(root,out,'Frozen PushT pose-head comparison', {n+'_test_angle_mae_deg':metrics[n]['test']['angle_mae_deg'] for n in selected},text,sources,
        [dict(file=str(p),title=p.stem,caption='Fixed protocol; one seed; validation selection, separate held-out evaluation.',embed=True) for p in [plot,panel,curve]])


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','train','evaluate']);p.add_argument('--arm',choices=list(HEADS));p.add_argument('--development',action='store_true');p.add_argument('--device',default='cuda');a=p.parse_args()
    setup();root=ROOT/('development' if a.development else 'experiment')
    if a.command=='prepare':prepare(root,a.development,a.device)
    elif a.command=='train':
        if not a.arm:p.error('--arm required')
        train(root,a.arm,a.development,a.device)
    else:final_evaluation(root,a.device)

if __name__=='__main__':main()
