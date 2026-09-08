"""Inspect real observed latent states; no imagined-state or planning claims."""
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from world_model.curriculum.encoder_factorial import ARMS
from world_model.curriculum.encoder_variants import matched_models
from world_model.curriculum.data import task_frames,file_hash
from world_model.curriculum.pose_accessibility import setup,analyze
from world_model.pusht.checkpoints import read_checkpoint,json_atomic

ROOT=Path('runs/encoder_study_2026-09-08');OUT=ROOT/'evaluation/internals'


def main():
    setup();OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'curriculum_analysis.json').exists():raise FileExistsError('preserve completed first-seed inspection')
    if any(not (ROOT/'factorial/seed_7107'/a/'evaluation.json').exists() for a in ARMS):
        raise RuntimeError('complete all four first-seed evaluations before representative inspection')
    train=task_frames('data/pusht_world_model/cchi_v1','train');test=task_frames('data/pusht_world_model/cchi_v1','test')
    train_ids=np.random.default_rng(8208).choice(len(train),256,replace=False)
    test_ids=np.random.default_rng(8207).choice(len(test),6,replace=False)
    # CPU inspection avoids competing with the measured GPU training schedule.
    for arm,(depth,exchange) in ARMS.items():
        checkpoint=ROOT/'factorial/seed_7107'/arm/'best.pt'
        if not checkpoint.exists() or not (checkpoint.parent/'curriculum_result.json').exists():continue
        state=read_checkpoint(checkpoint);models=matched_models(7107,depth,exchange)
        for k,m in models.items():m.load_state_dict(state['models'][k]);m.eval().requires_grad_(False)
        ztrain=[]
        with torch.no_grad():
            for start in range(0,len(train_ids),32):
                rgb,_=train.batch(train_ids[start:start+32]);ztrain.append(models['E'](rgb).tokens().numpy())
        bank=np.concatenate(ztrain).reshape(-1,64).astype('float64');mean=bank.mean(0);centered=bank-mean
        covariance=centered.T@centered/(len(centered)-1);values,vectors=np.linalg.eigh(covariance);basis=vectors[:,-3:][:,::-1]
        projected=centered@basis;lo,hi=np.percentile(projected,[2,98],axis=0)
        attention={};hooks=[]
        def capture(name):
            def hook(module,args,output):
                q,k,_=args
                def heads(t):return t.reshape(len(t),-1,4,16).transpose(1,2)
                scores=heads(module.query_projection(q))@heads(module.key_projection(k)).transpose(-1,-2)/4
                prob=scores.softmax(-1)
                entropy=-(prob*prob.clamp_min(1e-12).log()).sum(-1)/np.log(prob.shape[-1])
                attention[name]=entropy.mean(1).detach().numpy()
            return hook
        if exchange:
            hooks=[models['E'].fine_from_coarse.register_forward_hook(capture('fine')),
                   models['E'].coarse_from_fine.register_forward_hook(capture('coarse'))]
        rgb,target=test.batch(test_ids)
        with torch.no_grad():z=models['E'](rgb);pose=models['H'](z);reconstruction=models['D'](z)
        for h in hooks:h.remove()
        fine=z.fine.numpy();coarse=z.coarse.numpy()
        def color(tokens,size):
            p=(tokens-mean)@basis
            return np.clip((p-lo)/(hi-lo).clip(1e-9),0,1).reshape(-1,size,size,3)
        fc,cc=color(fine,16),color(coarse,8)
        truth=target.numpy();pred=pose.numpy()
        raw=dict(test_indices=test_ids,train_pca_indices=train_ids,targets=truth,predictions=pred,
                 rgb=rgb.numpy(),reconstruction=reconstruction.numpy(),
                 fine=fine,coarse=coarse,pca_mean=mean,pca_basis=basis,color_low=lo,color_high=hi,
                 fine_entropy=attention.get('fine',np.empty((0,))),coarse_entropy=attention.get('coarse',np.empty((0,))))
        np.savez_compressed(OUT/f'{arm}_states.npz',**raw)
        fig,axes=plt.subplots(6,6,figsize=(11,10.5),layout='constrained')
        for c in range(6):
            ax=axes[0,c];ax.imshow(rgb[c].permute(1,2,0));ax.scatter(truth[c,[0,2]]*64,truth[c,[1,3]]*64,c='#245a90',marker='+',s=55)
            ax.scatter(pred[c,[0,2]]*64,pred[c,[1,3]]*64,c='#ac651f',marker='x',s=35)
            for p,color_ in [(truth[c],'#245a90'),(pred[c],'#ac651f')]:
                angle=np.arctan2(p[4],p[5]);ax.arrow(p[2]*64,p[3]*64,5*np.cos(angle),5*np.sin(angle),color=color_,width=.25,length_includes_head=True)
            ax.set(xlim=(-.5,63.5),ylim=(63.5,-.5),title=f'Test row {test_ids[c]}')
            axes[1,c].imshow(reconstruction[c].permute(1,2,0))
            axes[2,c].imshow(fc[c]);axes[3,c].imshow(cc[c])
            for row,name,size in [(4,'fine',16),(5,'coarse',8)]:
                if exchange:axes[row,c].imshow(attention[name][c].reshape(size,size),cmap='gray',vmin=0,vmax=1)
                else:axes[row,c].text(.5,.5,'Exchange\ndisabled',ha='center',va='center',fontsize=10)
            for row in range(6):axes[row,c].set_xticks([]);axes[row,c].set_yticks([])
        for row,label in enumerate(['Pose (+ truth, × pred)','Reconstruction','Fine features · PCA','Coarse features · PCA','Fine attention entropy','Coarse attention entropy']):axes[row,0].set_ylabel(label,fontsize=9)
        fig.suptitle(f'{arm} · seed7107 · validation-selected observed perception\nPCA fitted on training tokens separately for each encoder; entropy0=concentrated,1=uniform',fontsize=11)
        fig.savefig(OUT/f'{arm}_perception.png',dpi=140);plt.close(fig)
        json_atomic(OUT/f'{arm}_manifest.json',dict(checkpoint=str(checkpoint),checkpoint_sha256=file_hash(checkpoint),step=state['global_update'],
            seed=7107,train_pca_indices=train_ids.tolist(),test_indices=test_ids.tolist(),dataset_fingerprint=train.fingerprint,
            pca_top3_variance_fraction=float(values[-3:].sum()/values.sum()),fine_norm_mean=float(np.linalg.norm(fine,axis=-1).mean()),
            coarse_norm_mean=float(np.linalg.norm(coarse,axis=-1).mean()),
            fine_attention_entropy_mean=float(attention['fine'].mean()) if exchange else None,
            coarse_attention_entropy_mean=float(attention['coarse'].mean()) if exchange else None,
            interpretation='Real observed states only. PCA colors are not aligned across encoders; attention maps are diagnostics, not causal benefit evidence.'))
        print('Inspected',arm,flush=True)
    sources=[p for p in OUT.iterdir() if p.suffix in ('.png','.npz','.json')]
    panels=[dict(file=str(OUT/f'{a}_perception.png'),title=f'{a} observed perception',
        caption='First declared seed; six fixed test frames. PCA colors are fitted separately per encoder; entropy uses a shared 0–1 scale. These are observed states, not imagined futures.',embed=True) for a in ARMS]
    analyze(ROOT,OUT,'Observed perception states in the first paired seed',{'encoders':4,'test_frames_per_encoder':6},
        '## Observed perception states\n\nFirst declared seed7107; same six test frames across all four arms. Training-only PCA shows feature organization, and attention entropy describes the routing distribution. Neither visualization is evidence of causal benefit or planning quality. Disabled exchange is shown as absent, not zero-entropy attention.',sources,panels)

if __name__=='__main__':main()
