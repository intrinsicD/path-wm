"""Bounded COCO foreground audit with explicit annotation and transform semantics."""
import sys,json,time
from pathlib import Path
from collections import defaultdict
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from .data import file_hash,digest,coco_frames
from .pose_accessibility import analyze
from world_model.pusht.checkpoints import json_atomic

ROOT=Path('runs/encoder_study_2026-09-08')
ANNOTATIONS=Path('/home/alex/Documents/datasets/annotations_trainval2014/annotations/instances_train2014.json')


def mask_api():
    try:
        from pycocotools import mask
    except ModuleNotFoundError:
        sys.path.insert(0,str(Path('.runtime/coco').resolve()))
        from pycocotools import mask
    return mask


def transform_mask(mask):
    height,width=mask.shape
    size=(max(64,int(width*64/min(width,height))),max(64,int(height*64/min(width,height))))
    resized=Image.fromarray(mask.astype('uint8')).resize(size,Image.Resampling.NEAREST)
    x,y=(size[0]-64)//2,(size[1]-64)//2
    return np.asarray(resized.crop((x,y,x+64,y+64)),dtype='uint8').copy()


def decode_union(annotations,height,width):
    api=mask_api();foreground=np.zeros((height,width),dtype=bool);crowd=foreground.copy()
    for ann in annotations:
        segmentation=ann['segmentation']
        # Official COCO annToRLE conventions: polygon, uncompressed RLE, or RLE.
        if isinstance(segmentation,list):
            if not segmentation:continue
            rle=api.merge(api.frPyObjects(segmentation,height,width))
        elif isinstance(segmentation['counts'],list):rle=api.frPyObjects(segmentation,height,width)
        else:rle=segmentation
        decoded=api.decode(rle).astype(bool)
        if decoded.ndim==3:decoded=decoded.any(-1)
        if ann.get('iscrowd',0):crowd|=decoded
        else:foreground|=decoded
    # Any crowd overlap is ignored, including overlap with a non-crowd instance.
    return foreground.astype('uint8'),(~crowd).astype('uint8')


def masked_bce(logits,target,valid):
    per=F.binary_cross_entropy_with_logits(logits,target,reduction='none')*valid
    denom=valid.sum((1,2,3));keep=denom>0
    if not keep.any():return logits.sum()*0
    return (per.sum((1,2,3))[keep]/denom[keep]).mean()


def mask_statistics(probability,target,valid):
    p=(probability>=.5)&(valid>0);t=(target>0)&(valid>0)
    intersect=(p&t).sum((1,2,3));union=(p|t).sum((1,2,3));total=p.sum((1,2,3))+t.sum((1,2,3))
    has_valid=valid.sum((1,2,3))>0
    iou=torch.where(union>0,intersect/union,torch.ones_like(union,dtype=torch.float32))
    dice=torch.where(total>0,2*intersect/total,torch.ones_like(total,dtype=torch.float32))
    return {k:v.detach().cpu().numpy() for k,v in dict(iou=iou,dice=dice,has_valid=has_valid,
        foreground_fraction=t.sum((1,2,3))/valid.sum((1,2,3)).clamp_min(1)).items()}


def prepare(development=False):
    out=ROOT/('development/coco_masks' if development else 'coco_masks');out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.json').exists():raise FileExistsError('preserve COCO audit subset')
    source=json.loads(ANNOTATIONS.read_text());by_id=defaultdict(list);images={r['id']:r for r in source['images']}
    for ann in source['annotations']:by_id[ann['image_id']].append(ann)
    manifest=json.loads(Path('data/curriculum/coco_v1/manifest.json').read_text());rng=np.random.default_rng(8107);splits={};begin=time.monotonic()
    for split,size in [('train',4096),('validation',512),('test',512)]:
        size=min(size,16 if development else size)
        rows=np.sort(rng.choice(manifest['splits'][split],size,replace=False))
        masks=[];valid=[];ids=[];groups=[];annotations=[]
        for row in rows:
            record=manifest['records'][row];image_id=int(Path(record['file']).stem.split('_')[-1]);info=images[image_id]
            if [info['width'],info['height']]!=record['source_size']:raise ValueError('COCO source geometry mismatch')
            m,v=decode_union(by_id[image_id],info['height'],info['width'])
            masks.append(transform_mask(m));valid.append(transform_mask(v));ids.append(image_id);groups.append(record['group'])
            annotations.append([a['id'] for a in by_id[image_id]])
        path=out/f'{split}.npz';np.savez_compressed(path,masks=np.array(masks),valid=np.array(valid),rows=rows,image_ids=ids,groups=groups)
        splits[split]=dict(frames=len(rows),sha256=file_hash(path),rows_sha256=digest(rows.tolist()),annotation_ids_sha256=digest(annotations),
                           zero_valid_frames=int((np.array(valid).sum((1,2))==0).sum()))
        print('COCO masks',split,len(rows),flush=True)
    groups={s:set(np.load(out/f'{s}.npz')['groups'].tolist()) for s in splits}
    if any(groups[a]&groups[b] for a,b in [('train','validation'),('train','test'),('validation','test')]):raise RuntimeError('mask split leakage')
    receipt=dict(status='completed',development=development,splits=splits,seconds=time.monotonic()-begin,
                  source=str(ANNOTATIONS),source_sha256=file_hash(ANNOTATIONS),coco_manifest_sha256=file_hash('data/curriculum/coco_v1/manifest.json'),
                  source_code_sha256=file_hash(__file__),decoder='pycocotools2.0.11; official annToRLE conventions',seed=8107,
                  semantics='union of non-crowd annotated instances; all crowd overlap ignored; binary nearest labels, original RGB crop geometry',
                  limitation='unannotated objects are background; semantic foreground coverage, not instance separation/extent')
    json_atomic(out/'manifest.json',receipt)
    analyze(ROOT,out,'COCO foreground audit preparation',{'train_frames':splits['train']['frames']},
            '## COCO foreground audit population\n\nFixed grouped train/validation/test subsets. Non-crowd instance union; crowd pixels ignored. Labels use nearest resampling with the existing RGB crop geometry. This does not label every possible object or separate instances.',[out/'manifest.json'])
