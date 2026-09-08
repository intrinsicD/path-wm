"""Read-only donor-feature and task-context interventions on fixed D2 decoders."""
import argparse
from contextlib import contextmanager
from datetime import datetime,timezone
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F
from .perception_cache import ROOT
from .perception_decoder import DenseDecoder
from .perception_decoder_training import batch
from .perception_training import load_data
from .encoder_reference import pinned_backbone
from .encoder_masks import mask_statistics
from .pose_accessibility import setup,analyze
from .data import file_hash
from world_model.pusht.checkpoints import json_atomic,fingerprint_modules


def routed_features(original,donor,condition):
    if condition=='donor_local': return donor[0],original[1],original[2]
    if condition=='donor_context': return original[0],donor[1],donor[2]
    if condition in ('normal','neutral','flipped'): return original
    raise ValueError('unknown decoder intervention')


@contextmanager
def task_context(model,condition):
    handles=[]
    try:
        if condition in ('neutral','flipped'):
            def change(module,args):
                return (torch.zeros_like(args[0]) if condition=='neutral' else -args[0],)
            handles=[film.register_forward_pre_hook(change) for film in model.films]
        yield
    finally:
        for handle in handles: handle.remove()


def derangement(count):
    rng=np.random.default_rng(20260911)
    while True:
        indices=rng.permutation(count)
        if (indices!=np.arange(count)).all(): return indices


@torch.no_grad()
def evaluate(model,patch,data,donors,condition):
    d=data['coco','test']; chunks={}; panel={}; n=len(donors)
    def add(k,v): chunks.setdefault(k,[]).append(np.asarray(v))
    with task_context(model,condition):
        for start in range(0,n,16):
            ids=np.arange(start,min(start+16,n)); original,rgb=batch(d,ids,patch,model.kind,'cpu')
            donor,_=batch(d,donors[ids],patch,model.kind,'cpu') if condition.startswith('donor') else (original,rgb)
            features=routed_features(original,donor,condition); image,logits=model.both(features); probability=logits.sigmoid()
            mask=torch.from_numpy(d['labels']['masks'][ids].copy()).float()[:,None]
            valid=torch.from_numpy(d['labels']['valid'][ids].copy()).float()[:,None]
            add('image_mse',(image-rgb).square().mean((1,2,3)).numpy())
            add('mask_bce',((F.binary_cross_entropy_with_logits(logits,mask,reduction='none')*valid).sum((1,2,3))/valid.sum((1,2,3)).clamp_min(1)).numpy())
            for key,value in mask_statistics(probability,mask,valid).items():add(key,value)
            if start==0: panel=dict(rgb=rgb[:6].numpy(),reconstruction=image[:6].numpy(),mask=mask[:6].numpy(),valid=valid[:6].numpy(),probability=probability[:6].numpy())
    raw={k:np.concatenate(v) for k,v in chunks.items()}; raw.update(indices=np.arange(n),donor_indices=donors)
    keep=raw['has_valid'].astype(bool)
    metric=dict(frames=n,image_mse=float(raw['image_mse'].mean()),
        **{key:float(raw[key][keep].mean()) for key in ('mask_bce','iou','dice')})
    return metric,raw,panel


def run(development=False):
    setup(); root=ROOT/('development/decoder_reliance' if development else 'decoder_reliance')
    if (root/'evaluation.json').exists():
        if not (root/'curriculum_analysis.json').exists(): return publish(root)
        raise FileExistsError('preserve completed reliance evaluation')
    root.mkdir(parents=True,exist_ok=True); n=16 if development else 512; donors=derangement(n)
    combinations=[('conditioned',9107)] if development else [(kind,seed) for seed in (9107,9108,9109) for kind in ('late','early','raw','conditioned')]
    source=pinned_backbone('cpu'); patch=source.backbone.patch_embed; del source
    data,_=load_data('vit',False); begin=time.monotonic(); rows=[]; sources=[]; baseline_checks=[]
    np.save(root/'donor_indices.npy',donors); sources.append(root/'donor_indices.npy')
    manifest=dict(development=development,device='cpu',precision='FP32, TF32 off',rows=n,donor_seed=20260911,
        source_hashes={str(p):file_hash(p) for p in (Path(__file__),Path('world_model/curriculum/perception_decoder.py'),Path('world_model/curriculum/perception_decoder_training.py'))},
        cache_sha256=file_hash(ROOT/'cache/vit/manifest.json'),protocol_sha256=file_hash('docs/perception-decoder-reliance-protocol-2026-09-08.md'))
    json_atomic(root/'manifest.json',manifest); sources.append(root/'manifest.json'); status='completed'
    for kind,seed in combinations:
        out=ROOT/'decoders'/f'seed_{seed}'/kind
        if not (out/'curriculum_analysis.json').exists(): raise RuntimeError('decoder endpoint not ready')
        model=DenseDecoder(kind,seed).eval().requires_grad_(False)
        checkpoint=out/'last.pt'; model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['heads']['decoder'])
        before=fingerprint_modules({'decoder':model,'patch':patch}); baseline=None
        for condition in ['normal','donor_local','donor_context']+(['neutral','flipped'] if kind=='conditioned' else []):
            if time.monotonic()-begin>=1200: status='stopped_wall_budget'; break
            metric,raw,panel=evaluate(model,patch,data,donors,condition)
            if condition=='normal':
                baseline=metric; reference=dict(np.load(out/'coco_test_errors.npz')); ref_panel=dict(np.load(out/'coco_panels.npz'))
                expected_iou=float(reference['iou'][:n][reference['has_valid'][:n].astype(bool)].mean())
                check=dict(arm=kind,seed=seed,frames=n,image_mse_difference=abs(metric['image_mse']-float(reference['image_mse'][:n].mean())),
                    iou_difference=abs(metric['iou']-expected_iou),maximum_panel_rgb_difference=float(np.abs(panel['reconstruction']-ref_panel['reconstruction']).max()),
                    maximum_panel_probability_difference=float(np.abs(panel['probability']-ref_panel['probability']).max()),tolerance=1e-5)
                baseline_checks.append(check); json_atomic(root/'baseline_checks.json',baseline_checks)
                if check['image_mse_difference']>1e-5 or check['iou_difference']>1e-5: raise RuntimeError('CPU/GPU decoder baseline discrepancy')
            row=dict(arm=kind,seed=seed,condition=condition,**metric,
                image_mse_ratio=metric['image_mse']/baseline['image_mse'],iou_delta=metric['iou']-baseline['iou'],
                checkpoint_sha256=file_hash(checkpoint))
            rows.append(row)
            prefix=f'{kind}_{seed}_{condition}'
            for suffix,value in [('errors',raw),('panels',panel)]:
                path=root/f'{prefix}_{suffix}.npz'; np.savez_compressed(path,**value); sources.append(path)
            json_atomic(root/'progress.json',dict(status='running',evaluations=rows,seconds=time.monotonic()-begin))
            print(kind,seed,condition,'IoU',round(metric['iou'],4),'RGB ratio',round(row['image_mse_ratio'],3),flush=True)
        if before!=fingerprint_modules({'decoder':model,'patch':patch}): raise RuntimeError('reliance diagnostic mutated frozen state')
        if status!='completed': break
    json_atomic(root/'evaluation.json',dict(status=status,development=development,evaluations=rows,baseline_checks=baseline_checks,
        expected_conditions=5 if development else 42,seconds=time.monotonic()-begin,sources=[str(p) for p in sources]))
    return publish(root)


def publish(root):
    result=json.loads((root/'evaluation.json').read_text()); sources=[Path(p) for p in result['sources']]+[root/'evaluation.json',root/'baseline_checks.json']
    analyze(ROOT,root,'Frozen decoder input reliance',dict(conditions=len(result['evaluations']),expected_conditions=result['expected_conditions'],seconds=result['seconds']),
        '## Frozen decoder input reliance\n\n'+('Development only. ' if result['development'] else 'Read-only fixed-endpoint diagnostic. ')+
        f'{result["status"]}: {len(result["evaluations"])}/{result["expected_conditions"]}conditions. '
        'Targets remain fixed while donor local/context features or task FiLM inputs change. '
        'CPU baselines are checked against saved GPU metrics. Degradation measures fitted dependence, not a retrained architecture comparison.',sources)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--development',action='store_true');args=parser.parse_args();run(args.development)
