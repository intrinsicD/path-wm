"""Read-only frame populations and conservative grouped COCO preparation."""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import hashlib
import io
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from world_model.pusht.checkpoints import json_atomic

TRANSFORM = 'RGB; LANCZOS shorter-side64; integer-floor size; centered64 crop; uint8-v1'

def file_hash(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def group_hashes(hashes, exact, radius=4):
    """All distance<=4 dHash pairs via five disjoint bands; transitive union.

    A <=4-bit difference must leave one of five bands identical. This candidate
    index is exhaustive for that criterion, not a universal near-duplicate test.
    Conservatively group ambiguous hash matches as well; inspect them separately.
    """
    if radius != 4 or len(hashes)!=len(exact):
        raise ValueError('expected paired hashes and radius four')
    parent=list(range(len(hashes))); buckets=defaultdict(list); seen={}; edges=[]
    def root(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]];i=parent[i]
        return i
    def join(a,b,kind,distance):
        parent[root(b)]=root(a);edges.append([a,b,kind,distance])
    widths=(13,13,13,13,12)
    for i,h in enumerate(map(int,hashes)):
        if exact[i] in seen:join(seen[exact[i]],i,'exact_canonical',0)
        seen[exact[i]]=i
        keys=[];offset=0;candidates=set()
        for band,width in enumerate(widths):
            key=(band,(h>>offset)&((1<<width)-1));keys.append(key)
            candidates.update(buckets[key]);offset+=width
        for j in sorted(candidates):
            distance=(h^int(hashes[j])).bit_count()
            if distance<=radius:join(j,i,'dhash_conservative',distance)
        for key in keys:buckets[key].append(i)
    roots=[root(i) for i in range(len(parent))]
    mapping={r:i for i,r in enumerate(sorted(set(roots)))}
    return [mapping[r] for r in roots],edges

def split_groups(groups,seed):
    unique=np.unique(groups)
    if len(unique)<3:raise ValueError('at least three distinct groups required')
    order=np.random.default_rng(seed).permutation(unique)
    validation=max(1,round(len(order)*.05));test=max(1,round(len(order)*.05))
    train=len(order)-validation-test
    labels={int(g):name for name,part in [
        ('train',order[:train]),('validation',order[train:train+validation]),
        ('test',order[train+validation:])] for g in part}
    return {name:[i for i,g in enumerate(groups) if labels[int(g)]==name]
            for name in ('train','validation','test')}

def canonical_image(image):
    image=image.convert('RGB');width,height=image.size
    size=(max(64,int(width*64/min(width,height))),max(64,int(height*64/min(width,height))))
    image=image.resize(size,Image.Resampling.LANCZOS)
    x,y=(size[0]-64)//2,(size[1]-64)//2
    return np.asarray(image.crop((x,y,x+64,y+64)),dtype=np.uint8).copy()

def _decode(path):
    raw=path.read_bytes()
    with Image.open(io.BytesIO(raw)) as image:
        shape=list(image.size);frame=canonical_image(image)
        tiny=np.asarray(Image.fromarray(frame).convert('L').resize((9,8),Image.Resampling.LANCZOS))
    bits=(tiny[:,1:]>tiny[:,:-1]).reshape(-1)
    dhash=sum(int(b)<<i for i,b in enumerate(bits))
    return frame,dict(file=path.name,bytes=len(raw),source_sha256=hashlib.sha256(raw).hexdigest(),
                      canonical_sha256=hashlib.sha256(frame.tobytes()).hexdigest(),
                      source_size=shape,dhash=str(dhash))

def prepare_coco(source,output,seed=4107,workers=4):
    source,output=Path(source),Path(output)
    manifest_path=output/'manifest.json'
    if manifest_path.exists():
        m=json.loads(manifest_path.read_text())
        if m['transform']!=TRANSFORM or m['seed']!=seed or m['source']!=str(source.resolve()):
            raise ValueError('incompatible COCO preparation')
        if file_hash(output/'frames.npy')!=m['frames_sha256']:raise ValueError('COCO frame cache hash mismatch')
        return m
    output.mkdir(parents=True,exist_ok=True)
    paths=sorted(source.glob('COCO_train2014_*.jpg'))
    if not paths:raise ValueError('no COCO images')
    partial=output/'frames.partial.npy'
    frames=np.lib.format.open_memmap(partial,mode='w+',dtype='uint8',shape=(len(paths),64,64,3))
    begin=time.monotonic();records=[]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i,(frame,record) in enumerate(pool.map(_decode,paths)):
            frames[i]=frame;records.append(record)
            if (i+1)%5000==0:print(f'COCO decoded {i+1}/{len(paths)} in {time.monotonic()-begin:.1f}s',flush=True)
    frames.flush();del frames
    groups,edges=group_hashes([int(r['dhash']) for r in records],[r['canonical_sha256'] for r in records])
    splits=split_groups(groups,seed)
    for r,g in zip(records,groups):r['group']=g
    partial.replace(output/'frames.npy')
    m=dict(schema='curriculum-coco-v1',source=str(source.resolve()),transform=TRANSFORM,seed=seed,
           count=len(records),records=records,splits=splits,duplicate_edges=edges,
           duplicate_rule='canonical RGB64 equality or exhaustive dHash64 radius<=4 connected components',
           limitation='conservative hash grouping; transformations beyond this criterion can evade detection',
           frames_sha256=file_hash(output/'frames.npy'),seconds=time.monotonic()-begin)
    m['fingerprint']=digest(m);json_atomic(manifest_path,m)
    print({k:len(v) for k,v in splits.items()},'groups',len(set(groups)),'edges',len(edges),flush=True)
    return m

class FrameSet:
    def __init__(self,frames,rows,targets=None,metadata=None,fingerprint=None,normalization=None):
        self.frames=frames;self.rows=np.asarray(rows,dtype=np.int64)
        self.targets=None if targets is None else np.asarray(targets,dtype=np.float32)
        self.metadata=metadata
        self.fingerprint=fingerprint or digest(self.rows.tolist())
        self.normalization=normalization
        if self.targets is not None and self.targets.shape!=(len(self.rows),6):
            raise ValueError('pose targets must align with exactly the retained frame population')
        if self.rows.size and (self.rows.min()<0 or self.rows.max()>=len(frames)):
            raise ValueError('frame rows out of bounds')
    def __len__(self):return len(self.rows)
    def batch(self,indices,device='cpu',labelled=True):
        indices=np.asarray(indices,dtype=np.int64)
        if indices.size and (indices.min()<0 or indices.max()>=len(self)):raise ValueError('population index out of bounds')
        x=torch.from_numpy(np.asarray(self.frames[self.rows[indices]]).copy()).to(device)
        x=x.permute(0,3,1,2).float().div_(255)
        y=None if self.targets is None or not labelled else torch.from_numpy(self.targets[indices].copy()).to(device)
        return x,y

def task_frames(directory,split):
    from world_model.pusht.data import EpisodeDataset
    ds=EpisodeDataset(directory,split);rows=[];targets=[];metadata=[]
    for ep in ds:
        n=len(ep['frames']);rows.extend(ep['source_rows'].tolist());targets.extend(ep['pose_targets'])
        metadata.extend(dict(source_episode=int(ep['source_episode']),frame=t,source_row=int(ep['source_rows'][t]),
                             group=int(ep['group_id'])) for t in range(n))
    return FrameSet(ds.frames,rows,targets,metadata,ds.fingerprint,ds.manifest['normalization'])

def coco_frames(directory,split):
    directory=Path(directory);m=json.loads((directory/'manifest.json').read_text())
    frames=np.load(directory/'frames.npy',mmap_mode='r')
    indices=m['splits'][split]
    return FrameSet(frames,indices,metadata=[dict(source_file=m['records'][i]['file'],group=m['records'][i]['group'])
                                           for i in indices],fingerprint=m['fingerprint'])
