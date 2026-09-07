"""Execute one bounded curriculum stage; run through run.py for dashboard QA."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from .data import FrameSet,prepare_coco,task_frames,coco_frames
from .training import train_phase

ROOT=Path('runs/curriculum_2026-09-07')
BASE=dict(seed=4107,device='cuda',cpu_threads=4,batch_size=128,microbatch=128,
          validation_batch=128,validate_every=100,learning_rate=3e-4,weight_decay=1e-4,grad_clip=1.,
          max_seconds=3600)
def validation_ids():
    return json.loads(Path('runs/pusht_world_model/baseline/perception/pusht_manifest.json').read_text())['validation_indices']
def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare-coco','profile','diagnostic','phase'])
    p.add_argument('--arm',choices=['A','B','C'],default='A');p.add_argument('--phase',choices=['warmup','supervised'],default='supervised')
    p.add_argument('--seed',type=int,default=4107);p.add_argument('--microbatch',type=int,default=128)
    p.add_argument('--updates',type=int);p.add_argument('--resume',action='store_true')
    a=p.parse_args()
    if a.command=='prepare-coco':
        prepare_coco('/home/alex/Documents/datasets/train2014','data/curriculum/coco_v1');return
    train=task_frames('data/pusht_world_model/cchi_v1','train')
    val=task_frames('data/pusht_world_model/cchi_v1','validation');ids=validation_ids()
    config={**BASE,'seed':a.seed,'microbatch':a.microbatch,'phase':a.phase,'arm':a.arm}
    warmup=None
    if a.command=='profile':
        config.update(arm='profile',phase='supervised',updates=100,max_seconds=180)
        run=ROOT/f'profile_micro{a.microbatch}'
    elif a.command=='diagnostic':
        # 64 distinct training episodes, one private fixed frame per episode.
        rng=np.random.default_rng(4107);episodes={}
        for i,m in enumerate(train.metadata):episodes.setdefault(m['source_episode'],[]).append(i)
        chosen=[int(rng.choice(episodes[int(e)])) for e in rng.choice(sorted(episodes),64,replace=False)]
        train=FrameSet(train.frames,train.rows[chosen],train.targets[chosen],
                       [train.metadata[i] for i in chosen],train.fingerprint,train.normalization)
        val=train;ids=list(range(64))
        config.update(arm='diagnostic',phase='supervised',updates=500,max_seconds=600)
        run=ROOT/'diagnostic'
    else:
        run=ROOT/f'seed_{a.seed}'/a.arm/a.phase
        if a.phase=='warmup':
            if a.arm=='A':raise ValueError('A has no warmup')
            config['updates']=2000
            if a.arm=='B':
                train=coco_frames('data/curriculum/coco_v1','train');val=coco_frames('data/curriculum/coco_v1','validation')
                ids=np.random.default_rng(4107).choice(len(val),min(2048,len(val)),replace=False).tolist()
        else:
            config['updates']=4000 if a.arm=='A' else 2000
            if a.arm!='A':
                warmup=ROOT/f'seed_{a.seed}'/a.arm/'warmup/last.pt'
                previous=json.loads((warmup.parent/'curriculum_result.json').read_text())
                if previous['status']!='completed':raise ValueError('warmup did not complete')
                config['max_seconds']=max(0.,3600-previous['elapsed_seconds'])
    if a.updates is not None:config['updates']=a.updates
    print(json.dumps(train_phase(config,train,val,ids,run,warmup=warmup,resume=a.resume)),flush=True)
if __name__=='__main__':main()
