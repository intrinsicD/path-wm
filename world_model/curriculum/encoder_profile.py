"""Development-only measurements to fix a feasible matched formal budget."""
import json,time
from pathlib import Path
import numpy as np
import torch
from .encoder_variants import matched_models
from .dino_reference import FrozenDino,DinoAdapter
from .data import task_frames,file_hash
from .pose_accessibility import setup,analyze
from .training import backward_batch
from world_model.paddle.training import optimizer_for
from world_model.pusht.checkpoints import json_atomic,fingerprint_modules,versions


def profile(device='cuda'):
    setup();root=Path('runs/encoder_study_2026-09-08');out=root/'development/profile';out.mkdir(parents=True,exist_ok=True)
    if (out/'profile.json').exists():raise FileExistsError('preserve completed profile')
    ds=task_frames('data/pusht_world_model/cchi_v1','train');x,y=ds.batch(np.arange(128),device)
    rows=[]
    for depth,exchange in ((0,True),(2,True),(0,False),(2,False)):
        models=matched_models(7106,depth,exchange)
        for m in models.values():m.to(device)
        opt=optimizer_for(models,dict(learning_rate=.0003,weight_decay=.0001))
        times=[]
        if device=='cuda':torch.cuda.reset_peak_memory_stats()
        for i in range(25):
            if device=='cuda':torch.cuda.synchronize()
            start=time.monotonic();loss=backward_batch(models,x,y,128);opt.step()
            if device=='cuda':torch.cuda.synchronize()
            times.append(time.monotonic()-start)
        encoder=models['E'].eval();latency=[]
        for i in range(30):
            if device=='cuda':torch.cuda.synchronize()
            start=time.monotonic()
            with torch.no_grad():encoder(x[:1])
            if device=='cuda':torch.cuda.synchronize()
            latency.append(time.monotonic()-start)
        row=dict(depth=depth,exchange=exchange,updates=25,examples=3200,final_training=loss,
                 median_update_seconds=float(np.median(times[5:])),update_seconds=times,
                 active_encoder_parameters=sum(p.numel() for p in encoder.parameters() if p.requires_grad),
                 total_encoder_parameters=sum(p.numel() for p in encoder.parameters()),
                 peak_cuda_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else None,
                 frame_latency_median_ms=1000*float(np.median(latency[5:])))
        rows.append(row);print(row,flush=True)
        del models,encoder,opt
    if device=='cuda':torch.cuda.empty_cache()
    upstream=json.loads((root/'upstream/manifest.json').read_text())
    dino=FrozenDino(upstream['source'],upstream['weights']).to(device);before=fingerprint_modules({'backbone':dino})
    times=[]
    if device=='cuda':torch.cuda.reset_peak_memory_stats()
    for i in range(10):
        if device=='cuda':torch.cuda.synchronize()
        start=time.monotonic();patch=dino(x[:32])
        if device=='cuda':torch.cuda.synchronize()
        times.append(time.monotonic()-start)
    difference=(patch-patch.half().float()).abs();relative=float(difference.square().mean()/patch.square().mean())
    adapter=DinoAdapter().to(device)
    with torch.no_grad():a,b=adapter(patch).tokens(),adapter(patch.half().float()).tokens()
    if fingerprint_modules({'backbone':dino})!=before:raise RuntimeError('DINO state mutated')
    if not torch.isfinite(patch).all() or relative>1e-6:raise RuntimeError('FP16 cache gate failed')
    dino_row=dict(batch_size=32,median_seconds=float(np.median(times[2:])),shape=list(patch.shape),
                  peak_cuda_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else None,
                  cache_relative_mse=relative,cache_max_abs=float(difference.max()),
                  projected_cache_mse=float((a-b).square().mean()),frozen_fingerprint=before,
                  patch_norm_percentiles=torch.quantile(patch.norm(dim=-1).flatten(),torch.tensor([0.,.5,.95,.99,1.],device=device)).tolist())
    result=dict(status='completed',purpose='development profiling; not formal evidence',custom=rows,dino=dino_row,
                precision='FP32 compute, TF32 disabled; proposed DINO FP16 storage',versions=versions(),source_sha256=file_hash(__file__))
    json_atomic(out/'profile.json',result);print('DINO',dino_row,flush=True)
    analyze(root,out,'Encoder development profile',{'max_custom_update_seconds':max(r['median_update_seconds'] for r in rows),
            'dino_batch_seconds':dino_row['median_seconds']},
            '## Encoder development profile\n\n25 development updates per custom variant; 32-frame frozen DINO profile. These timing and storage checks fix the formal budget and are not model-quality comparisons.',[out/'profile.json'])
