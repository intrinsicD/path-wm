"""Read-only attention interventions; no optimizer, target fitting or checkpoint edits."""
import argparse
from contextlib import contextmanager
import json
import time
import numpy as np
import torch
from .encoder_visual_audit import attention_details
from .encoder_probe import source_models
from .perception_cache import ROOT,CNN_SHA
from .perception_heads import make_heads,feature_grids
from .perception_fresh import cohorts
from .pose_accessibility import setup,pose_metrics,analyze
from .data import file_hash
from world_model.pusht.checkpoints import fingerprint_modules,json_atomic


@contextmanager
def attention_override(model,mode,collector=None,*,directions=('fine_from_coarse','coarse_from_fine')):
    if mode not in ('normal','uniform','zero'): raise ValueError('attention intervention')
    handles=[]; base=getattr(model,'base',model)
    def hook(name):
        def apply(module,args,output):
            if collector is not None or mode=='uniform':
                detail=attention_details(module,*args)
                if not torch.allclose(detail['output'],output,rtol=2e-4,atol=2e-5):
                    raise RuntimeError('explicit attention differs from actual SDPA')
                if collector is not None:
                    p=detail['probabilities']
                    collector.setdefault(name,[]).append(dict(entropy=detail['entropy'].cpu().numpy(),
                        maximum_probability=p.amax(-1).cpu().numpy(),
                        output_norm=output.norm(dim=-1).cpu().numpy(),
                        uniform_difference_norm=(output-detail['uniform_output']).norm(dim=-1).cpu().numpy(),
                        reference_max_abs_difference=float((output-detail['output']).abs().max())))
            if mode=='uniform': return detail['uniform_output']
            if mode=='zero': return torch.zeros_like(output)
            return None
        return apply
    try:
        for name in directions:
            if name not in ('fine_from_coarse','coarse_from_fine'): raise ValueError('attention direction')
            handles.append(getattr(base,name).register_forward_hook(hook(name)))
        yield
    finally:
        for handle in handles: handle.remove()


@torch.no_grad()
def evaluate(development=False,device='cpu',directional=False):
    setup(); family='internal_attention_directional' if directional else 'internal_attention'
    out=ROOT/('development/'+family if development else family); out.mkdir(parents=True,exist_ok=True)
    if (out/'evaluation.json').exists():
        if not (out/'curriculum_analysis.json').exists():
            return publish(out,json.loads((out/'evaluation.json').read_text()),[out/'evaluation.json',*sorted(out.glob('*.npz'))])
        raise FileExistsError('preserve attention intervention')
    models,source=source_models('deeper',7107,False,device)
    if file_hash(source)!=CNN_SHA: raise ValueError('source encoder changed')
    encoder=models['E'].eval().requires_grad_(False); del models
    before=fingerprint_modules({'encoder':encoder}); start=time.monotonic()
    frames,targets=cohorts()['fresh']; size=16 if development else len(frames); frames,targets=frames[:size],targets[:size]
    tokens={}; collector={}; sources=[]
    conditions={mode:(mode,('fine_from_coarse','coarse_from_fine')) for mode in ('normal','uniform','zero')}
    if directional:
        conditions={'normal':conditions['normal'],**{side+'_'+mode:(mode,(side+'_from_'+other,))
            for side,other in [('fine','coarse'),('coarse','fine')] for mode in ('uniform','zero')}}
    for label,(mode,directions) in conditions.items():
        chunks=[]
        with attention_override(encoder,mode,collector if label=='normal' else None,directions=directions):
            for offset in range(0,len(frames),32):
                rgb=torch.from_numpy(frames[offset:offset+32].copy()).permute(0,3,1,2).to(device).float()/255
                chunks.append(encoder(rgb).tokens().half().cpu().numpy())
        tokens[label]=np.concatenate(chunks)
    attention={}
    for direction,rows in collector.items():
        raw={k:np.concatenate([r[k] for r in rows]) for k in ('entropy','maximum_probability','output_norm','uniform_difference_norm')}
        attention[direction]=dict(per_head_mean_entropy=raw['entropy'].mean((0,2)).tolist(),
            per_head_mean_maximum_probability=raw['maximum_probability'].mean((0,2)).tolist(),
            mean_output_norm=float(raw['output_norm'].mean()),
            mean_uniform_difference_norm=float(raw['uniform_difference_norm'].mean()),
            relative_uniform_difference=float(np.linalg.norm(raw['uniform_difference_norm'])/np.linalg.norm(raw['output_norm'])),
            reference_max_abs_difference=max(r['reference_max_abs_difference'] for r in rows),
            keys=64 if direction=='fine_from_coarse' else 256,queries=raw['entropy'].shape[-1])
        path=out/(direction+'.npz'); np.savez_compressed(path,**raw); sources.append(path)
    evaluations={}; baseline_difference={}
    for seed in ((9107,) if development else (9107,9108,9109)):
        checkpoint=ROOT/'p1'/f'seed_{seed}'/'cnn'/'best.pt'; state=torch.load(checkpoint,map_location='cpu',weights_only=False)
        heads=make_heads(64,seed,device)
        for name,head in heads.items(): head.load_state_dict(state['heads'][name]); head.eval().requires_grad_(False)
        original=dict(np.load(ROOT/'fresh/evaluations'/f'seed_{seed}'/'cnn'/'fresh_errors.npz'))
        expected,_=pose_metrics(original['predictions'][:size],targets.astype('float64'))
        evaluations[str(seed)]={}
        for mode,z in tokens.items():
            predicted=[]; image_errors=[]
            for offset in range(0,len(frames),32):
                fine,coarse=feature_grids(torch.from_numpy(z[offset:offset+32].copy()).to(device).float(),'cnn')
                rgb=torch.from_numpy(frames[offset:offset+32].copy()).permute(0,3,1,2).to(device).float()/255
                predicted.append(heads['pose'](fine,coarse).cpu().double().numpy())
                image_errors.extend((heads['rgb'](fine,coarse)-rgb).square().mean((1,2,3)).cpu().tolist())
            metric,raw=pose_metrics(np.concatenate(predicted),targets.astype('float64'))
            case_q=np.maximum(raw['position_abs_error'].max(1)/8,raw['angle_abs_error_deg']/10)
            metric.update(image_mse=float(np.mean(image_errors)),per_case_tolerance_pass=float((case_q<=1).mean()),case_q_p95=float(np.percentile(case_q,95)))
            if mode=='normal':
                delta=max(abs(np.asarray(metric['position_mae'])-expected['position_mae']))
                angle_delta=abs(metric['angle_mae_deg']-expected['angle_mae_deg'])
                baseline_difference[str(seed)]=dict(max_coordinate_mae_difference=float(delta),angle_mae_difference=angle_delta)
                if delta>.02 or angle_delta>.02: raise RuntimeError('baseline differs beyond declared CPU/GPU tolerance')
            path=out/f'{mode}_{seed}_errors.npz'; np.savez_compressed(path,**raw,case_q=case_q,image_mse=image_errors); sources.append(path)
            evaluations[str(seed)][mode]=metric
            print('attention',seed,mode,'q',round(metric['q'],4),'case pass',round(metric['per_case_tolerance_pass'],4),flush=True)
    if before!=fingerprint_modules({'encoder':encoder}): raise RuntimeError('attention audit mutated encoder')
    protocol='docs/perception-attention-directional-protocol-2026-09-08.md' if directional else 'docs/perception-inspection-protocol-2026-09-08.md'
    result=dict(status='completed',development=development,directional=directional,frames=size,device=device,attention=attention,
        evaluations=evaluations,baseline_difference=baseline_difference,encoder_sha256=CNN_SHA,
        source_sha256=file_hash(__file__),protocol_sha256=file_hash(protocol),
        seconds=time.monotonic()-start,frozen_fingerprint=before)
    json_atomic(out/'evaluation.json',result); sources.append(out/'evaluation.json')
    return publish(out,result,sources)


def publish(out,result,sources):
    directional=result.get('directional',False); development=result['development']; size=result['frames']
    purpose='Read-only directional attention intervention' if directional else 'Read-only cross-scale attention intervention'
    analyze(ROOT,out,purpose,dict(frames=size,seconds=result['seconds']),
        '## Cross-scale attention reliance\n\n'+('Development16cases only. ' if development else 'All512 fresh cases, three frozen CNN readout seeds. ')+
        'Explicit Q/K/V probabilities match the actual attention operation. Entropy is normalized over keys separately per head. '
        'Uniform weights retain projected values; zero removes the entire attention-branch output. '+
        ('Each named fine/coarse condition changes only that receiving branch. ' if directional else 'Both directions change simultaneously. ')+
        'These are evaluation-time interventions, not trained no-attention architecture controls. Baseline CPU/GPU agreement and raw errors are recorded.',sources)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--development',action='store_true'); p.add_argument('--device',default='cpu')
    p.add_argument('--directional',action='store_true'); a=p.parse_args()
    evaluate(a.development,a.device,a.directional)
