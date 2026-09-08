"""One bounded head-package and training-budget diagnostic on the same frozen E."""
from torch import nn
from torch.nn import functional as F
import torch
from .bottleneck_models import SpatialPoseHead


class IndependentPoseHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk=nn.Sequential(nn.Conv2d(128,64,3,padding=1),nn.GELU())
        self.position=nn.Conv2d(64,2,1)
        self.orientation=nn.Linear(64*16*16,2)
        y,x=torch.meshgrid((torch.arange(16)+.5)/16,(torch.arange(16)+.5)/16,indexing='ij')
        self.register_buffer('grid',torch.stack((x,y),-1).reshape(256,2))

    def details(self,z):
        fine=z.fine.transpose(1,2).reshape(-1,64,16,16)
        coarse=z.coarse.transpose(1,2).reshape(-1,64,8,8)
        feature=self.trunk(torch.cat((fine,F.interpolate(coarse,size=(16,16),mode='nearest')),1))
        probabilities=self.position(feature).flatten(2).softmax(-1)
        points=probabilities@self.grid
        # Angle gradients may affect the shared feature trunk but never flow
        # through a predicted center or the position-map output parameters.
        pose=torch.cat((points.flatten(1),self.orientation(feature.flatten(1))),-1)
        return pose,points,probabilities.reshape(-1,2,16,16)

    def forward(self,z):
        return self.details(z)[0]


def paired_heads(seed):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        coupled=SpatialPoseHead()
        independent=IndependentPoseHead()
    independent.trunk[0].load_state_dict(coupled.net[0].state_dict())
    with torch.no_grad():
        independent.position.weight.copy_(coupled.net[2].weight[:2])
        independent.position.bias.copy_(coupled.net[2].bias[:2])
    return {'coupled':coupled,'independent':independent}


def train(arm,development=False,device='cuda'):
    import json,time
    import numpy as np
    from pathlib import Path
    from .pose_accessibility import features,batch,evaluate_head,setup,analyze,PARENT
    from .data import digest,file_hash
    from world_model.paddle.training import optimizer_for
    from world_model.pusht.checkpoints import atomic_checkpoint,json_atomic,fingerprint_modules
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    setup();root=Path('runs/encoder_study_2026-09-08')
    out=root/('development' if development else 'diagnostic')/arm;out.mkdir(parents=True,exist_ok=True)
    if (out/'last.pt').exists():raise FileExistsError('preserve existing diagnostic')
    cache=Path('runs/bottlenecks_2026-09-08/pose/experiment')
    x,ty=features(cache,'train');v,vy=features(cache,'validation')
    ids=np.random.default_rng(6107).choice(len(v),16 if development else 2048,replace=False)
    model=paired_heads(6107)[arm].to(device)
    config=dict(seed=6107,arm=arm,phase='supervised',updates=3 if development else 6000,
                batch_size=128,learning_rate=.0003,weight_decay=.0001,grad_clip=1.,
                validate_every=1 if development else 100,max_seconds=1800,development=development,
                protocol='encoder-study-A1; frozen E; head package and budget diagnostic')
    manifest=dict(config=config,dataset=digest(ty['groups'].tolist()),parent=str(PARENT),parent_sha256=file_hash(PARENT),
                  feature_manifest_sha256=file_hash(cache/'features/manifest.json'),validation_indices=ids.tolist(),
                  initial=fingerprint_modules({'H':model}),parameters=sum(p.numel() for p in model.parameters()),
                  source_sha256=file_hash(__file__))
    json_atomic(out/'curriculum_manifest.json',manifest)
    opt=optimizer_for({'H':model},config);rng=np.random.default_rng(6108)
    curve=[];best=None;selected=None;begin=time.monotonic();windows={}
    if device=='cuda':torch.cuda.reset_peak_memory_stats()
    def validate(step):
        nonlocal best,selected
        metrics,_=evaluate_head(model,v,vy['targets'],ids,device)
        row=dict(step=step,elapsed_seconds=time.monotonic()-begin,**metrics)
        curve.append(row)
        with (out/'validation.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        saved=dict(head={k:t.detach().cpu().clone() for k,t in model.state_dict().items()},config=config,
                   step=step,metrics=row,parent_sha256=manifest['parent_sha256'])
        key=(metrics['q'],metrics['pose_mse'],step)
        atomic_checkpoint(out/'last.pt',saved)
        if best is None or key<best:
            best=key;selected=row;atomic_checkpoint(out/'best.pt',saved)
        if step in (2000,6000,config['updates']):
            windows[str(step)]={'selected':selected,'final':row}
            chosen=torch.load(out/'best.pt',map_location='cpu',weights_only=False)
            atomic_checkpoint(out/f'best_cap_{step}.pt',chosen)
        print(arm,step,'q',round(metrics['q'],4),'angle',round(metrics['angle_mae_deg'],3),flush=True)
    validate(0)
    for step in range(1,config['updates']+1):
        if time.monotonic()-begin>config['max_seconds']:raise TimeoutError('diagnostic wall budget')
        draw=rng.integers(len(x),size=128);z=batch(x,draw,device);target=torch.from_numpy(ty['targets'][draw].copy()).to(device)
        model.train();opt.zero_grad(set_to_none=True);pred,points,_=model.details(z)
        pose=(pred-target).square().mean()
        points_loss=(points-(SpatialPoseHead.target_points(target) if arm=='coupled' else target[:,:4].reshape(-1,2,2))).square().mean()
        loss=pose+points_loss
        loss.backward();grad=nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not torch.isfinite(loss) or not torch.isfinite(grad):raise RuntimeError('nonfinite loss/gradient')
        opt.step()
        with (out/'training.jsonl').open('a') as f:
            f.write(json.dumps(dict(step=step,loss=float(loss.detach()),pose_mse=float(pose.detach()),point_mse=float(points_loss.detach()),
                                   examples=step*128,grad_norm=float(grad),sample_indices_sha256=digest(draw.tolist()),
                                   elapsed_seconds=time.monotonic()-begin))+'\n')
        if step%config['validate_every']==0 or step==config['updates']:validate(step)
    result=dict(status='completed',step=step,selected_step=selected['step'],selected=selected,final=curve[-1],
                examples=step*128,elapsed_seconds=time.monotonic()-begin,windows=windows,
                peak_cuda_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else None)
    json_atomic(out/'curriculum_result.json',result)
    selected_ckpt=torch.load(out/'best.pt',map_location='cpu',weights_only=False);model.load_state_dict(selected_ckpt['head'])
    evaluation={};sources=[out/'curriculum_result.json',out/'curriculum_manifest.json',out/'training.jsonl',out/'validation.jsonl']
    for split in ('train','validation','test'):
        f,l=features(cache,split)
        rows=np.random.default_rng(6107).choice(len(f),16 if development else min(2048,len(f)),replace=False) if split!='test' or development else np.arange(len(f))
        m,r=evaluate_head(model,f,l['targets'],rows,device);evaluation[split]=m
        path=out/f'{split}_errors.npz';np.savez_compressed(path,**r,indices=rows,groups=l['groups'][rows]);sources.append(path)
    json_atomic(out/'evaluation.json',evaluation);sources.append(out/'evaluation.json')
    fig,ax=plt.subplots(figsize=(7,3.5),layout='constrained')
    ax.plot([r['step'] for r in curve],[r['q'] for r in curve],color='#245a90')
    ax.axhline(1,color='#555555',ls='--');ax.set(xlabel='Optimizer updates',ylabel='Worst normalized pose MAE (q)',ylim=(0,None),title=f'{arm} frozen-E readout · seed6107')
    path=out/'validation.png';fig.savefig(path,dpi=140);plt.close(fig);sources.append(path)
    analyze(root,out,'Frozen-E readout package diagnostic',{'selected_q':selected['q'],'test_q':evaluation['test']['q']},
            f"## {arm} readout diagnostic\n\n{'Development subset. ' if development else 'One seed; reused grouped holdouts. '}Selected q={selected['q']:.4f}; test q={evaluation['test']['q']:.4f}. q≤1 required for numeric readiness. Encoder unchanged. The independent head changes orientation readout and capacity; this is not an isolated gradient-coupling experiment.",
            sources,[dict(file=str(path),title='Frozen-E readout validation',caption='Fixed validation groups; lower q is better.',embed=True)])
    return result
