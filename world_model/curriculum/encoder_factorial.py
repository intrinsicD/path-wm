"""Frozen, paired depth-by-exchange study; legacy models and runs stay intact."""
import json
from pathlib import Path
import numpy as np
from .encoder_variants import matched_models
from .training import train_phase,evaluate
from .data import task_frames,file_hash,digest
from .pose_accessibility import analyze
from world_model.pusht.checkpoints import read_checkpoint,json_atomic

ROOT=Path('runs/encoder_study_2026-09-08')
ARMS={'reference':(0,True),'deeper':(2,True),'no_exchange':(0,False),'deeper_no_exchange':(2,False)}


def train(arm,seed=7107,development=False,device='cuda',resume=False):
    if arm not in ARMS:raise ValueError('unknown factorial arm')
    if seed not in (7107,7108,7109) and not development:raise ValueError('undeclared formal seed')
    depth,exchange=ARMS[arm]
    out=ROOT/('development/factorial' if development else 'factorial')/f'seed_{seed}'/arm
    train=task_frames('data/pusht_world_model/cchi_v1','train');val=task_frames('data/pusht_world_model/cchi_v1','validation')
    ids=np.random.default_rng(seed).choice(len(val),16 if development else 2048,replace=False)
    config=dict(seed=seed,arm=arm,phase='supervised',encoder_variant=dict(depth=depth,exchange=exchange),
                updates=3 if development else 4000,batch_size=128,microbatch=128,learning_rate=.0003,
                weight_decay=.0001,grad_clip=1.,validate_every=1 if development else 100,max_seconds=1200,
                validation_batch=128,device=device,cpu_threads=4,development=development,
                protocol='B: paired residual-depth package by cross-scale exchange; original RGB+pose objective')
    result=train_phase(config,train,val,ids,out,resume=resume)
    if result['status']!='completed':raise RuntimeError('factorial run stopped before declared cap')
    selected=read_checkpoint(out/'best.pt');models=matched_models(seed,depth,exchange)
    for name,m in models.items():m.load_state_dict(selected['models'][name]);m.to(device).eval().requires_grad_(False)
    metrics={};sources=[out/'curriculum_manifest.json',out/'curriculum_result.json',out/'training.jsonl',out/'validation.jsonl']
    for split in ('train','validation','test'):
        ds=task_frames('data/pusht_world_model/cchi_v1',split)
        rows=np.random.default_rng(seed).choice(len(ds),16 if development else min(2048,len(ds)),replace=False) if split!='test' or development else np.arange(len(ds))
        m,raw=evaluate(models,ds,rows,128,device,True,True);metrics[split]=m
        p=out/f'{split}_errors.npz';np.savez_compressed(p,**{k:np.asarray(v) for k,v in raw.items()},
            groups=np.array([ds.metadata[i]['group'] for i in rows]));sources.append(p)
    json_atomic(out/'evaluation.json',dict(metrics=metrics,selected_checkpoint_sha256=file_hash(out/'best.pt'),
        gate=metrics['validation']['perception_numeric_gate'],scope='exploratory reused test groups; perception, not planning'))
    sources.append(out/'evaluation.json')
    analyze(ROOT,out,'Depth-by-exchange perception comparison',{'selected_q':result['selected']['q'],'test_q':metrics['test']['q']},
        f"## {arm} · seed {seed}\n\n{'Development only. ' if development else 'Paired four-arm perception recipe. '}Selected validation q={result['selected']['q']:.4f}; test q={metrics['test']['q']:.4f}. q≤1 numeric gate is {'passed' if result['selected']['q']<=1 else 'failed'}. This comparison does not establish cross-scale attention's benefit for imagination or planning. All downstream weights would need compatible retraining.",sources)
    return result
