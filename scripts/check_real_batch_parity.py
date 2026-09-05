"""Compare native JEPA and local backward on discarded real-data checkpoint clones.

Reference loss follows le-wm@8edfeb336732b5f3ce7b8b210d0ba370a09e2cac:
train.py:lejepa_forward. No saved checkpoint or training run is modified.
"""
import gc
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from third_party.lewm.jepa import JEPA
from third_party.lewm.module import SIGReg
from world_model.model import build_model
from world_model.data import TrajectoryDataset, preprocess_pixels, normalize_actions
from world_model.training import backward_batch, autocast_context
from scripts.diagnose_pusht_control import sha256
from world_model.train import write_json


def check(mode='basic'):
    root=Path('runs/diagnostics/pusht_control_diagnosis')
    path=Path('runs/diagnostics/pusht_broader_pilot/checkpoint_001000.pt')
    digest=sha256(path)
    out=root/({'basic':'real_batch_parity.json','expanded':'real_batch_parity_expanded.json','checkpointed':'real_batch_parity_checkpointed.json'}[mode])
    if out.exists(): raise FileExistsError(out)
    meta=json.loads((path.parent/'manifest.json').read_text())
    saved=torch.load(path,weights_only=True,map_location='cpu')
    ds=TrajectoryDataset(meta['dataset']['path'],meta['train_episodes'])
    indices=torch.randperm(len(ds),generator=torch.Generator().manual_seed(9201))[:4].tolist()
    batch=next(iter(DataLoader(ds,batch_size=4,sampler=indices,num_workers=0)))
    pixels=preprocess_pixels(batch['pixels'].cuda())
    actions=normalize_actions(batch['action'].cuda(),saved['action_stats'])
    torch.set_num_threads(4)
    results=[]
    started=time.monotonic()
    for precision in ('float32','bf16'):
        records={}
        for variant in {'basic':('native','local_recomputed'),'expanded':('native','local_full','local_recomputed','local_full_recomputed'),'checkpointed':('native','local_checkpointed')}[mode]:
            model=build_model(saved['model_config'])
            model.load_state_dict(saved['model'],strict=True)
            if variant=='native':
                model=JEPA(*(getattr(model,k) for k in ('encoder','predictor','action_encoder','projector','pred_proj')))
            if variant=='local_checkpointed':
                model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
            model=model.cuda().train()
            reg=SIGReg(knots=17,num_proj=1024).cuda()
            torch.manual_seed(9202)
            if variant=='native':
                with autocast_context(torch.device('cuda'),precision):
                    encoded=model.encode(dict(pixels=pixels,action=actions))
                    prediction=model.predict(encoded['emb'][:,:3],encoded['act_emb'][:,:3])
                    pred=(prediction-encoded['emb'][:,1:]).square().mean()
                    sigreg=reg(encoded['emb'].transpose(0,1))
                    loss=pred+.09*sigreg
                loss.backward()
                terms=dict(loss=loss.detach(),pred_loss=pred.detach(),sigreg_loss=sigreg.detach())
            else:
                terms=backward_batch(model,pixels,actions,reg,encoder_chunk={'local_full':0,'local_checkpointed':0,'local_recomputed':8,'local_full_recomputed':16}[variant],precision=precision)
            records[variant]=dict(terms={k:float(v) for k,v in terms.items()},
                grads={k:p.grad.detach().cpu().clone() for k,p in model.named_parameters() if p.grad is not None},
                buffers={k:v.detach().cpu().clone() for k,v in model.named_buffers()})
            del model,reg,terms
            if variant=='native': del encoded,prediction,pred,sigreg,loss
            gc.collect();torch.cuda.empty_cache()
        for variant in [v for v in records if v!='native']:
            a,b=records['native'],records[variant]
            if a['grads'].keys()!=b['grads'].keys(): raise RuntimeError('Gradient parameter sets differ')
            delta=sum(float((v-b['grads'][k]).double().square().sum()) for k,v in a['grads'].items())
            norm=sum(float(v.double().square().sum()) for v in a['grads'].values())
            maxabs=max(float((v-b['grads'][k]).abs().max()) for k,v in a['grads'].items())
            buffer_error=max(float((v.float()-b['buffers'][k].float()).abs().max()) for k,v in a['buffers'].items())
            results.append(dict(precision=precision,variant=variant,native=a['terms'],local=b['terms'],
                                gradient_relative_l2=(delta/norm)**.5,gradient_max_abs=maxabs,
                                buffer_max_abs=buffer_error,gradient_parameters=len(a['grads'])))
            print(json.dumps(results[-1]),flush=True)
    if sha256(path)!=digest: raise RuntimeError('Checkpoint changed')
    write_json(out,dict(checkpoint_sha256=digest,checkpoint_unchanged=True,
        batch_size=4,window_indices=indices,encoder_chunk=8,random_seed=9202,
        source='Pinned native JEPA + train.py:lejepa_forward versus local backward_batch',
        limitation='One real batch on a discarded clone; not batch-128 long-run optimizer trajectory parity.',
        results=results,seconds=time.monotonic()-started))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['basic','expanded','checkpointed'],default='basic')
    check(parser.parse_args().mode)
