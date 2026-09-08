"""Objective-only point-localization follow-up on the frozen P1 representations."""
import argparse
import copy
from datetime import datetime,timezone
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from .perception_cache import ROOT,CNN_SHA,cache_root,check_storage
from .perception_heads import make_heads,feature_grids
from .perception_training import (load_data,checkpoint_state,restore_checkpoint,recover_ledgers,
    append_json,selected_better,TRAIN_CUTOFF)
from .perception_fresh import cohorts
from .encoder_probe import source_models
from .encoder_reference import pinned_backbone
from .pose_accessibility import setup,pose_metrics,analyze
from .data import file_hash,digest
from world_model.pusht.checkpoints import json_atomic,atomic_checkpoint,fingerprint_modules,versions


def state_equivalence(current,reference):
    if current.keys()!=reference.keys(): raise ValueError('pose state keys differ')
    redundant=('locations.bias','orientation_attention.bias')
    differences={name:float((value.detach().cpu()-reference[name].detach().cpu()).abs().max()) for name,value in current.items()}
    return dict(maximum_all_parameters=max(differences.values()),
        maximum_identifiable_parameters=max(v for n,v in differences.items() if n not in redundant),
        softmax_bias_differences={n:differences[n] for n in redundant})


def barycentric_targets(xy,side=16):
    if xy.ndim!=3 or xy.shape[-1]!=2 or not torch.isfinite(xy).all() or ((xy<0)|(xy>1)).any():
        raise ValueError('finite point coordinates in[0,1] required')
    scaled=xy*(side-1); low=scaled.floor().long().clamp(max=side-2); fraction=scaled-low
    x,y=low.unbind(-1); dx,dy=fraction.unbind(-1)
    indices=torch.stack((y*side+x,y*side+x+1,(y+1)*side+x,(y+1)*side+x+1),-1)
    weights=torch.stack(((1-dx)*(1-dy),dx*(1-dy),(1-dx)*dy,dx*dy),-1)
    result=xy.new_zeros((*xy.shape[:-1],side*side)).scatter_add(-1,indices,weights)
    return result.reshape(*xy.shape[:-1],side,side)


def location_kl(logits,xy):
    target=barycentric_targets(xy,logits.shape[-1]).detach().flatten(2)
    log_probability=logits.flatten(2).log_softmax(-1)
    return (torch.special.xlogy(target,target)-target*log_probability).sum(-1).mean()


def map_diagnostics(probability,xy):
    target=barycentric_targets(xy,probability.shape[-1]).flatten(2)
    p=probability.flatten(2); log_p=p.clamp_min(torch.finfo(p.dtype).tiny).log()
    return dict(location_entropy=(-(p*log_p).sum(-1)/np.log(p.shape[-1])).cpu().numpy(),
        target_support_mass=(p*(target>0)).sum(-1).cpu().numpy(),
        location_kl=(torch.special.xlogy(target,target)-target*log_p).sum(-1).cpu().numpy())


def attach_map_metrics(metric,raw,diagnostics):
    for name in diagnostics[0]:
        values=np.concatenate([d[name] for d in diagnostics]); raw[name]=values
        metric[name+'_mean']=float(values.mean())
        metric[name+'_by_object']=values.mean(0).tolist()


def pose_and_logits(head,fine,coarse):
    captured=[]
    handle=head.locations.register_forward_hook(lambda module,args,output:captured.append(output))
    try: pose=head(fine,coarse)
    finally: handle.remove()
    return pose,captured[0]


def pose_batch(data,ids,encoder,device):
    tokens=torch.from_numpy(np.asarray(data['cache'][ids]).copy()).float().to(device)
    fine,coarse=feature_grids(tokens,encoder)
    target=torch.from_numpy(data['labels']['targets'][ids].copy()).float().to(device)
    return fine,coarse,target


def extend_metrics(prediction,target):
    metric,raw=pose_metrics(prediction,target)
    case_q=np.maximum(raw['position_abs_error'].max(1)/8,raw['angle_abs_error_deg']/10)
    edge=np.minimum(target[:,:2],1-target[:,:2]).min(1)*512<48
    metric.update(per_case_tolerance_pass=float((case_q<=1).mean()),case_q_p95=float(np.percentile(case_q,95)))
    metric['boundary_slices']={}
    for name,keep in [('within_48_units',edge),('interior',~edge)]:
        metric['boundary_slices'][name]=dict(frames=int(keep.sum()),
            per_case_pass=float((case_q[keep]<=1).mean()) if keep.any() else None,
            angle_mae_deg=float(raw['angle_abs_error_deg'][keep].mean()) if keep.any() else None)
    return metric,dict(**raw,case_q=case_q,near_boundary=edge)


@torch.no_grad()
def evaluate_head(head,data,encoder,device,limit=None,maps=False):
    head.eval(); ids_all=np.arange(len(data['ds']))[:limit]; predictions=[]; locations=[]; pools=[]; diagnostics=[]
    for start in range(0,len(ids_all),64):
        ids=ids_all[start:start+64]; fine,coarse,target=pose_batch(data,ids,encoder,device)
        prediction,location,pool=head(fine,coarse,return_maps=True)
        predictions.append(prediction.cpu().double().numpy())
        diagnostics.append(map_diagnostics(location,target[:,:4].reshape(-1,2,2)))
        if maps: locations.append(location.cpu().numpy()); pools.append(pool.cpu().numpy())
    metric,raw=extend_metrics(np.concatenate(predictions),data['labels']['targets'][ids_all].astype('float64'))
    raw.update(indices=ids_all,source_rows=data['labels']['rows'][ids_all],groups=data['labels']['groups'][ids_all])
    attach_map_metrics(metric,raw,diagnostics)
    if maps: raw.update(locations=np.concatenate(locations),orientation_pool=np.concatenate(pools))
    return metric,raw


@torch.no_grad()
def evaluate_fresh(head,encoder,device):
    if encoder=='cnn':
        models,source=source_models('deeper',7107,False,device)
        if file_hash(source)!=CNN_SHA: raise ValueError('CNN source changed')
        model=models['E']; del models
    else: model=pinned_backbone(device)
    model.eval().requires_grad_(False); head.eval()
    before=fingerprint_modules({'encoder':model,'pose':head}); frames,target=cohorts()['fresh']
    predictions=[]; locations=[]; pools=[]; diagnostics=[]
    for start in range(0,len(frames),32):
        rgb=torch.from_numpy(frames[start:start+32].copy()).permute(0,3,1,2).float().to(device)/255
        z=model(rgb); tokens=z.tokens() if encoder=='cnn' else z
        fine,coarse=feature_grids(tokens.half().float(),encoder)
        prediction,location,pool=head(fine,coarse,return_maps=True)
        predictions.append(prediction.cpu().double().numpy()); locations.append(location.cpu().numpy()); pools.append(pool.cpu().numpy())
        xy=torch.from_numpy(target[start:start+32,:4].copy()).float().to(device).reshape(-1,2,2)
        diagnostics.append(map_diagnostics(location,xy))
    metric,raw=extend_metrics(np.concatenate(predictions),target.astype('float64'))
    raw.update(locations=np.concatenate(locations),orientation_pool=np.concatenate(pools),indices=np.arange(len(frames)))
    attach_map_metrics(metric,raw,diagnostics)
    if before!=fingerprint_modules({'encoder':model,'pose':head}): raise RuntimeError('fresh localization evaluation mutated frozen state')
    return metric,raw


def train(encoder,seed,development=False,control=False,device='cuda',resume=False):
    if control and not development: raise ValueError('zero-weight control is a declared development equivalence check only')
    setup(); check_storage(80*1024**2)
    out=ROOT/('development/localization_v2' if development else 'localization')/f'seed_{seed}'/(encoder+('_control' if control else ''))
    out.mkdir(parents=True,exist_ok=True)
    if (out/'curriculum_result.json').exists():
        if resume and not (out/'curriculum_analysis.json').exists(): return evaluate_saved(out,encoder,seed,development,device)
        raise FileExistsError('preserve localization result')
    if (out/'curriculum_manifest.json').exists() and not resume: raise FileExistsError('explicit localization resume required')
    head=make_heads(64 if encoder=='cnn' else 384,seed,device)['pose']
    optimizer=torch.optim.AdamW(head.parameters(),lr=3e-4,weight_decay=1e-4)
    data,_=load_data(encoder,development); d=data['pusht','train']
    config=dict(encoder=encoder,seed=seed,arm=encoder+('_control' if control else '_localized'),phase='supervised',
        updates=50 if development else 4000,batch_size=32,validate_every=10 if development else 100,max_seconds=900,
        discarded_coco_draws_per_update=32,location_kl_weight=0. if control else .001,
        learning_rate=3e-4,weight_decay=1e-4,grad_clip=1.,development=development,device=device,
        protocol='overnight localization: objective-only geometry decoder comparison',
        selector='minimum validation q; earliest exact tie',precision='FP32, TF32 off; audited FP16 frozen features')
    manifest=dict(config=config,dataset=file_hash(cache_root(development)/encoder/'manifest.json'),
        initial=fingerprint_modules({'pose':head}),parameters={'pose':sum(p.numel() for p in head.parameters())},
        source_sha256=file_hash(__file__),protocol_sha256=file_hash('docs/perception-localization-protocol-2026-09-08.md'),versions=versions())
    rng=np.random.default_rng(seed+230003); begin=time.monotonic(); previous=0.; selected=None; selected_models=None; step=0
    if resume:
        if read(out/'curriculum_manifest.json')!=manifest: raise ValueError('localization resume identity changed')
        state=torch.load(out/'last.pt',map_location='cpu',weights_only=False)
        restore_checkpoint(state,{'pose':head},{'pose':optimizer},rng)
        previous=state['elapsed_seconds']; step=state['step']; selected=state['selected']; selected_models=state['selected_models']; recover_ledgers(out,step)
        atomic_checkpoint(out/'best.pt',dict(state,heads=selected_models,step=selected['step']))
    else:
        json_atomic(out/'curriculum_manifest.json',manifest); (out/'training.jsonl').touch()
    if device=='cuda': torch.cuda.reset_peak_memory_stats()
    def elapsed(): return previous+time.monotonic()-begin
    def validate(update):
        nonlocal selected,selected_models
        metric,_=evaluate_head(head,data['pusht','validation'],encoder,device)
        # Keep nested exploratory slices in evaluations, not the numerical training ledger.
        metric.pop('boundary_slices')
        row=dict(step=update,elapsed_seconds=elapsed(),**metric,loss=metric['pose_mse'])
        improved=selected_better(row,selected)
        if improved:
            selected=row; selected_models={'pose':{n:p.detach().cpu().clone() for n,p in head.state_dict().items()}}
        state=checkpoint_state({'pose':head},{'pose':optimizer},rng,step=update,selected=selected,selected_models=selected_models,
            elapsed_seconds=elapsed(),config=config,manifest_sha256=file_hash(out/'curriculum_manifest.json'))
        append_json(out/'validation.jsonl',row); atomic_checkpoint(out/'last.pt',state)
        if improved: atomic_checkpoint(out/'best.pt',state)
        print('localization',encoder,seed,config['location_kl_weight'],update,'q',round(row['q'],4),'seconds',round(elapsed(),1),flush=True)
        return row
    final=json.loads((out/'validation.jsonl').read_text().splitlines()[-1]) if resume else validate(0); status='completed'
    for update in range(step+1,config['updates']+1):
        if elapsed()>=config['max_seconds'] or datetime.now(timezone.utc)>=TRAIN_CUTOFF:
            status='stopped_wall_budget'; break
        draws={domain:rng.integers(len(data[domain,'train']['ds']),size=32) for domain in ('coco','pusht')}
        fine,coarse,target=pose_batch(d,draws['pusht'],encoder,device)
        head.train(); optimizer.zero_grad(set_to_none=True); prediction,logits=pose_and_logits(head,fine,coarse)
        pose=(prediction-target).square().mean(); kl=location_kl(logits,target[:,:4].reshape(-1,2,2))
        loss=pose+config['location_kl_weight']*kl if not control else pose
        loss.backward(); norm=nn.utils.clip_grad_norm_(head.parameters(),1.)
        if not torch.isfinite(norm) or not torch.isfinite(loss): raise RuntimeError('nonfinite localization update')
        optimizer.step(); step=update
        append_json(out/'training.jsonl',dict(step=step,loss=float(loss.detach()),pose_mse=float(pose.detach()),location_kl=float(kl.detach()),
            grad_norm=float(norm),examples=step*32,sample_indices_sha256=digest({k:v.tolist() for k,v in draws.items()}),elapsed_seconds=elapsed()))
        if step%config['validate_every']==0 or step==config['updates']: final=validate(step)
    if final['step']!=step: final=validate(step)
    if control:
        reference=ROOT/'development/p1'/f'seed_{seed}'/encoder/'last.pt'
        old=torch.load(reference,map_location='cpu',weights_only=False)['heads']['pose']
        check=state_equivalence(head.state_dict(),old); reference_head=copy.deepcopy(head); reference_head.load_state_dict(old)
        output_differences={}; frames=0
        for split in ('train','validation','test'):
            _,actual=evaluate_head(head,data['pusht',split],encoder,device,maps=True)
            _,expected=evaluate_head(reference_head,data['pusht',split],encoder,device,maps=True)
            frames+=len(actual['predictions'])
            for name in ('predictions','locations','orientation_pool'):
                output_differences[split+'_'+name]=float(np.max(np.abs(actual[name]-expected[name])))
        check.update(reference=str(reference),sha256=file_hash(reference),tolerance=2e-5,
            output_differences=output_differences,maximum_output_difference=max(output_differences.values()),frames=frames)
        json_atomic(out/'reference_check.json',check)
        if check['maximum_identifiable_parameters']>check['tolerance'] or check['maximum_output_difference']>check['tolerance']:
            raise RuntimeError('zero-weight development control differs in informative weights or outputs')
    result=dict(status=status,step=step,selected_step=selected['step'],selected=selected,final=final,examples=step*32,
        elapsed_seconds=elapsed(),peak_cuda_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else None)
    json_atomic(out/'curriculum_result.json',result)
    return evaluate_saved(out,encoder,seed,development,device,head,data)


def read(path): return json.loads(Path(path).read_text())


def evaluate_saved(out,encoder,seed,development,device,head=None,data=None):
    if head is None: head=make_heads(64 if encoder=='cnn' else 384,seed,device)['pose']
    if data is None: data,_=load_data(encoder,development)
    sources=[out/n for n in ('curriculum_manifest.json','curriculum_result.json','training.jsonl','validation.jsonl')]
    evaluation={}; start=time.monotonic()
    for choice in ('selected','endpoint'):
        path=out/('best.pt' if choice=='selected' else 'last.pt'); state=torch.load(path,map_location='cpu',weights_only=False)
        head.load_state_dict(state['heads']['pose']); head.eval()
        before=fingerprint_modules({'pose':head}); evaluation[choice]=dict(step=state['step'],checkpoint_sha256=file_hash(path))
        for split in ('train','validation','test'):
            metric,raw=evaluate_head(head,data['pusht',split],encoder,device,limit=512 if split=='train' else None,maps=choice=='selected' and split=='test')
            evaluation[choice][split]=metric
            path=out/f'{choice}_{split}_errors.npz'; np.savez_compressed(path,**raw); sources.append(path)
        if choice=='selected' and not development:
            metric,raw=evaluate_fresh(head,encoder,device); evaluation['fresh']=metric
            path=out/'fresh_errors.npz'; np.savez_compressed(path,**raw); sources.append(path)
        if before!=fingerprint_modules({'pose':head}): raise RuntimeError('localization evaluation changed pose head')
    evaluation['seconds']=time.monotonic()-start; json_atomic(out/'evaluation.json',evaluation); sources.append(out/'evaluation.json')
    result=read(out/'curriculum_result.json'); metric=evaluation['selected']['test']
    analyze(ROOT,out,'Location-distribution supervision',dict(test_q=metric['q'],test_case_pass=metric['per_case_tolerance_pass'],
        **({'fresh_q':evaluation['fresh']['q'],'fresh_case_pass':evaluation['fresh']['per_case_tolerance_pass']} if 'fresh' in evaluation else {})),
        f'## Point-supervised geometry decoder · {encoder} seed{seed}\n\n'+('Development only. ' if development else 'Adaptive objective-only comparison against frozen P1. ')+
        f'{result["status"]}, {result["step"]}updates. Test q={metric["q"]:.4f}. '
        'The encoder/head architecture and orientation readout stay fixed; the added target distribution has the exact supervised coordinate expectation. '
        'Only32PushT frames/update train this head; unused COCO draws preserve pairing. Output probabilities are not calibrated uncertainty.',sources)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--encoder',choices=['cnn','vit'],required=True)
    p.add_argument('--seed',type=int,default=9107); p.add_argument('--development',action='store_true'); p.add_argument('--control',action='store_true')
    p.add_argument('--device',default='cuda'); p.add_argument('--resume',action='store_true'); a=p.parse_args()
    train(a.encoder,a.seed,a.development,a.control,a.device,a.resume)
