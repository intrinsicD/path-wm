"""Read-only observation-to-output costs for the declared frozen packages."""
import argparse
import gc
import json
from pathlib import Path
import time
import numpy as np
import torch
from .perception_cache import ROOT,CNN_SHA
from .perception_training import load_data
from .perception_heads import make_heads,feature_grids
from .perception_decoder import DenseDecoder
from .perception_independent import IndependentDecoder
from .encoder_probe import source_models
from .encoder_reference import pinned_backbone
from .dino_reference import preprocess
from .pose_accessibility import setup,analyze
from .data import file_hash
from world_model.pusht.checkpoints import json_atomic,fingerprint_modules,versions


def load_package(name,device):
    encoder='cnn' if name=='cnn_p1' else 'vit'; sources=[]
    if encoder=='cnn':
        models,source=source_models('deeper',7107,False,device)
        if file_hash(source)!=CNN_SHA: raise ValueError('runtime CNN reference changed')
        backbone=models['E']; del models; sources.append(source)
    else: backbone=pinned_backbone(device)
    path=ROOT/f'p1/seed_9107/{encoder}/best.pt'; sources.append(path)
    state=torch.load(path,map_location='cpu',weights_only=False)
    heads=make_heads(64 if encoder=='cnn' else 384,9107,device)
    for key,head in heads.items(): head.load_state_dict(state['heads'][key])
    if not name.endswith('_p1'):
        heads={'pose':heads['pose']}
        arm=name.removeprefix('vit_')
        path=ROOT/('independent/seed_9107/split' if arm=='split' else f'decoders/seed_9107/{arm}')/'last.pt'
        dense=IndependentDecoder(9107) if arm=='split' else DenseDecoder(arm,9107)
        dense.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['heads']['decoder'])
        heads['dense']=dense.to(device); sources.append(path)
    modules={'encoder':backbone,**heads}
    for model in modules.values(): model.eval().requires_grad_(False)
    return encoder,backbone,heads,sources


def operation(encoder,backbone,heads,rgb,task):
    z=backbone(rgb); tokens=z.tokens() if encoder=='cnn' else z
    fine,coarse=feature_grids(tokens.half().float(),encoder)
    outputs={}
    if task in ('pose','all'): outputs['pose']=heads['pose'](fine,coarse)
    if task=='pose': return outputs
    if 'dense' in heads:
        local=backbone.backbone.patch_embed(preprocess(rgb)); features=(local,fine,coarse)
        if task=='all': outputs['rgb'],outputs['mask']=heads['dense'].both(features)
        else: outputs[task]=heads['dense'](features,task)
    else:
        for output in (('rgb','mask') if task=='all' else (task,)):
            outputs[output]=heads[output](fine,coarse)
    return outputs


@torch.no_grad()
def run(development=False,device='cuda'):
    setup(); root=ROOT/('development/runtime' if development else 'runtime'); root.mkdir(parents=True,exist_ok=True)
    if (root/'evaluation.json').exists(): raise FileExistsError('preserve runtime evidence')
    packages=['vit_early'] if development else ['cnn_p1','vit_p1','vit_early','vit_conditioned','vit_split']
    data,_=load_data('vit',False); rgb,_=data['coco','test']['ds'].batch(np.arange(32),device)
    source_files=[Path('world_model/curriculum')/n for n in ('perception_runtime.py','perception_independent.py',
        'perception_decoder.py','perception_heads.py','perception_training.py','perception_cache.py','encoder_reference.py','dino_reference.py')]
    protocol=Path('docs/perception-runtime-protocol-2026-09-09.md')
    identity=dict(development=development,device=device,packages=packages,seed=9107,
        batches=[1] if development else [1,32],warmup=1 if development else 5,repeats=2 if development else 20,
        max_seconds=600,source_sha256={str(p):file_hash(p) for p in source_files},
        protocol_sha256=file_hash(protocol),cache_identity=file_hash(ROOT/'cache/vit/manifest.json'),versions=versions())
    json_atomic(root/'manifest.json',identity); (root/'measurements.jsonl').touch()
    rows=[]; evidence=[]; sources=[root/'manifest.json',root/'measurements.jsonl']; checks=[]; start=time.monotonic(); status='completed'
    for name in packages:
        gc.collect()
        if device=='cuda': torch.cuda.empty_cache()
        encoder,backbone,heads,paths=load_package(name,device)
        modules={'encoder':backbone,**heads}; before=fingerprint_modules(modules)
        metadata=dict(package=name,checkpoints={str(p):file_hash(p) for p in paths},
            parameters={key:sum(p.numel() for p in m.parameters()) for key,m in modules.items()},frozen_fingerprint=before)
        evidence.append(metadata)
        for count in identity['batches']:
            x=rgb[:count]; combined=operation(encoder,backbone,heads,x,'all'); differences={}
            for task in ('rgb','mask','pose'):
                single=operation(encoder,backbone,heads,x,task)[task]
                differences[task]=float((single-combined[task]).abs().max())
            check=dict(package=name,batch=count,differences=differences,tolerance=2e-5)
            if max(differences.values())>check['tolerance']: raise AssertionError('combined inference changes an output')
            checks.append(check); del combined,single
            for task in ('rgb','mask','pose','all'):
                if time.monotonic()-start>identity['max_seconds']:
                    status='stopped_wall_budget'; break
                for _ in range(identity['warmup']): operation(encoder,backbone,heads,x,task)
                if device=='cuda':
                    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); baseline=torch.cuda.memory_allocated()
                else: baseline=None
                samples=[]
                for repetition in range(identity['repeats']):
                    if device=='cuda': torch.cuda.synchronize()
                    begin=time.perf_counter(); operation(encoder,backbone,heads,x,task)
                    if device=='cuda': torch.cuda.synchronize()
                    milliseconds=(time.perf_counter()-begin)*1000; samples.append(milliseconds)
                    with (root/'measurements.jsonl').open('a') as stream:
                        stream.write(json.dumps(dict(package=name,batch=count,output=task,repetition=repetition,milliseconds=milliseconds))+'\n')
                peak=torch.cuda.max_memory_allocated() if device=='cuda' else None
                row=dict(package=name,batch=count,output=task,repeats=len(samples),milliseconds=samples,
                    median_batch_ms=float(np.median(samples)),p95_batch_ms=float(np.percentile(samples,95)),
                    median_ms_per_image=float(np.median(samples))/count,parameters=sum(metadata['parameters'].values()),
                    resident_cuda_bytes=baseline,peak_cuda_bytes=peak,incremental_peak_cuda_bytes=peak-baseline if device=='cuda' else None)
                rows.append(row); print(name,count,task,'median batch ms',round(row['median_batch_ms'],3),flush=True)
            if status!='completed': break
        if before!=fingerprint_modules(modules): raise RuntimeError('runtime audit changed a frozen package')
        del modules,heads,backbone
        if status!='completed': break
    evaluation=dict(status=status,development=development,device=device,rows=rows,checks=checks,packages=evidence,seconds=time.monotonic()-start,
        scope='GPU/CPU-resident RGB64 to outputs, including encoder preprocessing. Excludes capture, file IO, host transfer and planning. All outputs include original P1 pose; no new quality evaluation.')
    json_atomic(root/'evaluation.json',evaluation); sources.append(root/'evaluation.json')
    analyze(ROOT,root,'Observation-to-output runtime',dict(conditions=len(rows),packages=len(evidence),seconds=evaluation['seconds']),
        '## Observation-to-output compute\n\n'+('Development only. ' if development else 'Read-only audit, seed9107 checkpoints. ')+
        f'{status}. Encoder, its preprocessing and requested decoding are inside each invocation. '
        'The input is already resident on the measured device; capture/IO/host transfer/planning are excluded. '
        'All-output requests include the same original P1 pose head within each encoder family. Raw timings and frozen-state/output agreement checks remain available.',sources)
    return evaluation


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--development',action='store_true');parser.add_argument('--device',default='cuda')
    args=parser.parse_args();run(args.development,args.device)
