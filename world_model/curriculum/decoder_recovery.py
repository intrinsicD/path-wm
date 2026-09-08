"""Frozen-encoder recovery experiment; each command uses the reporting wrapper."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .data import coco_frames, task_frames, file_hash
from .training import initial_models, train_phase, evaluate
from .inspection import mean_image, mean_image_errors
from world_model.pusht.checkpoints import read_checkpoint, json_atomic, fingerprint_modules

ROOT=Path('runs/decoder_recovery_2026-09-08')
OLD=Path('runs/curriculum_2026-09-07')
PARENTS={
    'warmup':(OLD/'seed_4107/B/warmup/last.pt','975395bbc7882433a30d99340bb4f08e73b1b7dd1d41a7818800cdb97f6bd00a'),
    'adapted':(OLD/'seed_4107/B/supervised/best.pt','38962cf0837f3722fa985487c6313195de1dba1a77577c053cb5f3a3297308d5'),
}
ARMS={'adapted_parent':('adapted','parent'),'adapted_fresh':('adapted','fresh'),'warmup_fresh':('warmup','fresh')}
CONFIG=dict(seed=5107,phase='warmup',decoder_only=True,device='cuda',cpu_threads=4,
            updates=2000,batch_size=128,microbatch=128,validation_batch=128,
            validate_every=100,learning_rate=3e-4,weight_decay=1e-4,grad_clip=1.,max_seconds=1800)


def train(arm,resume=False):
    parent,initialization=ARMS[arm];path,expected=PARENTS[parent]
    if file_hash(path)!=expected:raise ValueError('parent checkpoint bytes changed')
    data=coco_frames('data/curriculum/coco_v1','train')
    val=coco_frames('data/curriculum/coco_v1','validation')
    ids=np.random.default_rng(5107).choice(len(val),2048,replace=False)
    config={**CONFIG,'arm':arm,'decoder_initialization':initialization,
            'parent_sha256':expected,'protocol':'frozen-encoder decoder recovery; image-only COCO'}
    print(json.dumps(train_phase(config,data,val,ids,ROOT/arm,warmup=path,resume=resume)),flush=True)


def paired_group_bootstrap(current,reference,groups,seed,draws=2000):
    """Resample complete source groups; preserve paired errors and frame weighting."""
    current,reference=np.asarray(current),np.asarray(reference)
    groups=np.asarray(groups)
    if current.shape!=reference.shape or groups.shape!=current.shape or current.ndim!=1 or not len(current):
        raise ValueError('aligned nonempty per-frame paired errors and groups required')
    if not np.isfinite(current-reference).all():raise ValueError('finite paired errors required')
    unique,index=np.unique(groups,return_inverse=True)
    sums=np.bincount(index,weights=current-reference);counts=np.bincount(index)
    rng=np.random.default_rng(seed);values=[]
    for start in range(0,draws,64):
        ids=rng.integers(len(unique),size=(min(64,draws-start),len(unique)))
        values.extend((sums[ids].sum(1)/counts[ids].sum(1)).tolist())
    return dict(delta_mse=float((current-reference).mean()),
                ci95=np.quantile(values,[.025,.975]).tolist(),groups=len(unique),
                draws=draws,seed=seed,unit='source-group resampling; frame-weighted MSE difference'),np.asarray(values)


def loaded(path,device):
    saved=read_checkpoint(path);models=initial_models(5107,saved['models'])
    if 'H' in saved['models']:models['H'].load_state_dict(saved['models']['H'])
    for model in models.values():model.to(device).eval().requires_grad_(False)
    return saved,models


def audit():
    rows={};streams=[];fresh=[]
    for arm,(parent,initialization) in ARMS.items():
        path,sha=PARENTS[parent]
        if file_hash(path)!=sha:raise ValueError('reference checkpoint mutated')
        original=read_checkpoint(path)
        initial=read_checkpoint(ROOT/arm/'update_00000000.pt')
        final=read_checkpoint(ROOT/arm/'last.pt');best=read_checkpoint(ROOT/arm/'best.pt')
        if not final['training_complete'] or final['global_update']!=2000:raise ValueError('incomplete recovery arm')
        for saved in (initial,best,final):
            for k in ('E','H'):
                expected=original['models'].get(k,initial['models'][k])
                if any(not torch.equal(t,saved['models'][k][name]) for name,t in expected.items()):
                    raise ValueError('frozen state changed')
            actual=fingerprint_modules({k:saved['models'][k] for k in ('E','H')})
            if actual!=saved['frozen_fingerprint']:raise ValueError('frozen audit fingerprint mismatch')
        stream=[json.loads(line)['sample_indices_sha256'] for line in (ROOT/arm/'training.jsonl').read_text().splitlines()]
        streams.append(stream)
        if initialization=='fresh':fresh.append(fingerprint_modules({'D':initial['models']['D']}))
        rows[arm]=dict(parent_sha256=sha,frozen_state_exact=True,updates=final['global_update'],
                       examples=final['examples_processed'],selected_step=best['global_update'],
                       elapsed_seconds=final['elapsed_seconds'],initial_decoder=fingerprint_modules({'D':initial['models']['D']}))
    if any(s!=streams[0] for s in streams[1:]) or len(set(fresh))!=1:
        raise ValueError('decoder initialization or batch streams do not match')
    return dict(arms=rows,batch_streams_identical=True,fresh_decoders_identical=True)


def run_evaluation(device='cuda'):
    out=ROOT/'evaluation';out.mkdir(parents=True,exist_ok=True)
    if (out/'metrics.json').exists():raise ValueError('preserve completed decoder evaluation')
    torch.set_num_threads(4)
    # Match the declared training precision rather than cuDNN's default TF32.
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    matching=audit();json_atomic(out/'matching_audit.json',matching)
    checkpoints={f'reference_{name}':pair[0] for name,pair in PARENTS.items()}
    checkpoints.update({f'{arm}_{which}':ROOT/arm/filename for arm in ARMS
                        for which,filename in [('selected','best.pt'),('final','last.pt')]})
    data={'COCO':coco_frames('data/curriculum/coco_v1','test'),
          'PushT':task_frames('data/pusht_world_model/cchi_v1','test')}
    training={'COCO':coco_frames('data/curriculum/coco_v1','train'),
              'PushT':task_frames('data/pusht_world_model/cchi_v1','train')}
    ids={'COCO':np.load(OLD/'generic_reconstruction/panel_images.npz')['indices'],
         'PushT':np.random.default_rng(5107).choice(len(data['PushT']),6,replace=False)}
    baselines={k:float(np.mean(mean_image_errors(mean_image(training[k]),v))) for k,v in data.items()}
    rows={};raw={};images={};inputs={};sources=[out/'matching_audit.json'];panels=[]
    for name,path in checkpoints.items():
        saved,models=loaded(path,device);rows[name]={};raw[name]={};images[name]={}
        for domain,ds in data.items():
            metric,record=evaluate(models,ds,np.arange(len(ds)),device=device,labelled=False,return_records=True)
            metric.update(rmse=float(np.sqrt(metric['image_mse'])),mean_image_mse=baselines[domain],
                          ratio_to_mean=metric['image_mse']/baselines[domain])
            rows[name][domain]=dict(checkpoint=str(path),sha256=file_hash(path),step=saved['global_update'],**metric)
            raw[name][domain]=np.asarray(record['image_mse'])
            file=out/f'{name}_{domain}.npz'
            np.savez_compressed(file,indices=record['indices'],image_mse=record['image_mse'],
                                groups=[m['group'] for m in ds.metadata]);sources.append(file)
            with torch.no_grad():
                x,_=ds.batch(ids[domain],device,False)
                images[name][domain]=models['D'](models['E'](x)).permute(0,2,3,1).cpu().numpy()
                inputs[domain]=x.permute(0,2,3,1).cpu().numpy()
            print(name,domain,metric,flush=True)
        del models
    comparisons={};draws={}
    for i,domain in enumerate(data):
        comparisons[domain]={};groups=[m['group'] for m in data[domain].metadata]
        pairs=[('adapted_parent_selected','reference_adapted'),
               ('adapted_parent_selected','reference_warmup'),
               ('adapted_fresh_selected','warmup_fresh_selected')]
        for j,(left,right) in enumerate(pairs):
            key=f'{left}_minus_{right}'
            result,samples=paired_group_bootstrap(raw[left][domain],raw[right][domain],groups,95107+10*i+j)
            result['mse_ratio']=rows[left][domain]['image_mse']/rows[right][domain]['image_mse']
            comparisons[domain][key]=result;draws[f'{domain}_{key}']=samples
    np.savez_compressed(out/'bootstrap_draws.npz',**draws);sources.append(out/'bootstrap_draws.npz')
    report=dict(status='completed',scope='exploratory reused internal test populations; one training seed',
                metrics=rows,comparisons=comparisons,matching_audit=matching,panel_indices={k:v.tolist() for k,v in ids.items()},
                evaluation_runtime=dict(device=device,torch=torch.__version__,
                    cuda_matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                    cudnn_tf32=torch.backends.cudnn.allow_tf32,cudnn_benchmark=torch.backends.cudnn.benchmark))
    json_atomic(out/'metrics.json',report);sources.append(out/'metrics.json')
    shown=['reference_warmup','reference_adapted','adapted_parent_selected','adapted_fresh_selected','warmup_fresh_selected']
    labels=['Original COCO E/D','Task-adapted E/D','Frozen adapted E\nrefitted existing D',
            'Frozen adapted E\nfresh D','Frozen warmup E\nfresh D']
    for domain in data:
        fig,axes=plt.subplots(6,6,figsize=(11,10),layout='constrained')
        for r,(label,array) in enumerate([('Input',inputs[domain])]+[(label,images[n][domain]) for n,label in zip(shown,labels)]):
            for c in range(6):axes[r,c].imshow(array[c]);axes[r,c].set_xticks([]);axes[r,c].set_yticks([])
            axes[r,0].set_ylabel(label,fontsize=8)
        fig.suptitle(f'{domain} decoder recovery · six fixed examples\nRGB64; selected by COCO validation MSE; one seed; exact images in raw ledger',fontsize=12)
        path=out/f'{domain.lower()}_recovery.png';fig.savefig(path,dpi=115);fig.savefig(path.with_suffix('.svg'));plt.close(fig)
        sources.append(path);panels.append(dict(file=str(path),title=f'{domain} decoder recovery',caption='Fixed examples, not a prevalence estimate. Encoder frozen during refitting.',embed=True))
    fig,ax=plt.subplots(figsize=(9,4.5),layout='constrained')
    for arm,color,style in zip(ARMS,['#1f5b99','#1f5b99','#666666'],['-','--',':']):
        curve=[json.loads(s) for s in (ROOT/arm/'validation.jsonl').read_text().splitlines()]
        ax.plot([r['step'] for r in curve],[r['image_mse'] for r in curve],label=arm,color=color,linestyle=style)
    ax.set_yscale('log');ax.set_xlabel('Decoder optimizer updates');ax.set_ylabel('Validation RGB MSE · log scale')
    ax.legend(frameon=False);ax.grid(axis='y',alpha=.2)
    ax.set_title('COCO decoder recovery validation\nSame 2,048 images and batch draws; encoder frozen; seed 5107')
    path=out/'learning_curves.png';fig.savefig(path,dpi=120);fig.savefig(path.with_suffix('.svg'));plt.close(fig)
    sources.append(path);panels.append(dict(file=str(path),title='Decoder validation learning curves',caption='Logarithmic MSE axis; includes untrained decoder at step zero.',embed=True))
    narrative='## Frozen-encoder decoder recovery\n\nExploratory follow-up, one seed, original encoders/readouts byte-identical. '
    narrative+='Each arm: 2,000 decoder-only COCO updates, 256,000 presentations. Selected on the same 2,048 COCO validation images. '
    narrative+='Test populations were already exposed: 4,146 COCO and 2,506 PushT frames. No new control or pose-readiness claim.\n\n'
    narrative+='| Decoder pair | COCO MSE | PushT MSE |\n|---|---:|---:|\n'
    for name in checkpoints:narrative+=f"| {name} | {rows[name]['COCO']['image_mse']:.8f} | {rows[name]['PushT']['image_mse']:.8f} |\n"
    narrative+='\nPaired uncertainty resamples complete duplicate/episode groups and is conditional on these trained models. '
    narrative+='Decoder recovery demonstrates decodability under the tested budget; failure alone does not prove information loss. '
    narrative+='Task conditioning, replay, register tokens and new skip paths are separate, untested interventions.'
    json_atomic(out/'curriculum_analysis.json',dict(status='completed',purpose='Decoder recovery with frozen encoder and matched decoder controls',
        metrics={f'{name}_{domain}_mse':r[domain]['image_mse'] for name,r in rows.items() for domain in data},
        sources={p.relative_to(ROOT).as_posix():file_hash(p) for p in sources},narrative=narrative,panels=panels))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['train','evaluate'])
    parser.add_argument('--arm',choices=list(ARMS));parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if args.command=='train':
        if args.arm is None:parser.error('--arm required for training')
        train(args.arm,args.resume)
    else:run_evaluation()


if __name__=='__main__':main()
