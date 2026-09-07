"""Read-only perception evaluation; fitted bases and probes use training rows only."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.nn import functional as F
from .data import FrameSet,task_frames,digest,file_hash
from .training import initial_models,evaluate
from world_model.pusht.checkpoints import read_checkpoint,json_atomic,fingerprint_modules

ROOT=Path('runs/curriculum_2026-09-07')
DATA='data/pusht_world_model/cchi_v1'


def mean_image(training):
    total=np.zeros((64,64,3),np.float64)
    for start in range(0,len(training),256):
        total+=np.asarray(training.frames[training.rows[start:start+256]],dtype=np.float64).sum(0)/255
    return total/len(training)

def mean_image_errors(frame,data):
    errors=[]
    for start in range(0,len(data),256):
        x=np.asarray(data.frames[data.rows[start:start+256]],dtype=np.float64)/255
        errors.extend(np.square(x-frame).mean((1,2,3)).tolist())
    return errors

def pca_fit(training):
    x=np.asarray(training,dtype=np.float64);mean=x.mean(0);x=x-mean
    values,vectors=np.linalg.eigh(x.T@x/max(1,len(x)-1));order=np.argsort(values)[::-1]
    values=np.maximum(values[order],0);vectors=vectors[:,order]
    for i in range(vectors.shape[1]):
        if vectors[np.argmax(np.abs(vectors[:,i])),i]<0:vectors[:,i]*=-1
    projected=x@vectors[:,:3]
    return dict(mean=mean,components=vectors[:,:3],eigenvalues=values,
                low=np.quantile(projected,.01,axis=0),high=np.quantile(projected,.99,axis=0))

def pca_transform(values,basis):
    return (np.asarray(values)-basis['mean'])@basis['components']

def pca_rgb(values,basis):
    z=pca_transform(values,basis)
    return np.clip((z-basis['low'])/np.maximum(basis['high']-basis['low'],1e-12),0,1)

def attention_weights(module,query,key):
    def heads(x):return x.reshape(x.shape[0],x.shape[1],4,16).transpose(1,2)
    q=heads(module.query_projection(query));k=heads(module.key_projection(key))
    return (q@k.transpose(-2,-1)/4).softmax(-1)

def grouped_errors(raw,metadata):
    groups=np.asarray([m['group'] for m in metadata]);unique=np.unique(groups)
    pos=np.asarray(raw['position_abs_error']);angle=np.asarray(raw['angle_abs_error_deg'])
    rows=[dict(group=int(g),frames=int((groups==g).sum()),position_mae=pos[groups==g].mean(0).tolist(),
               angle_mae_deg=float(angle[groups==g].mean())) for g in unique]
    return dict(groups=len(rows),position_mae=np.mean([r['position_mae'] for r in rows],axis=0).tolist(),
                angle_mae_deg=float(np.mean([r['angle_mae_deg'] for r in rows])),records=rows)

def sample_training(data,n=256):
    # Fixed group-balanced draw: cycle shuffled groups, privately draw one frame.
    rng=np.random.default_rng(92501);groups={}
    for i,m in enumerate(data.metadata):groups.setdefault(m['group'],[]).append(i)
    order=rng.permutation(sorted(groups));return np.asarray([rng.choice(groups[int(order[i%len(order)])]) for i in range(min(n,len(data)))])
    
@torch.no_grad()
def encode(models,data,ids,device='cuda'):
    fine=[];coarse=[]
    for start in range(0,len(ids),128):
        x,_=data.batch(ids[start:start+128],device)
        s=models['E'](x);fine.append(s.fine.cpu().numpy());coarse.append(s.coarse.cpu().numpy())
    return dict(fine=np.concatenate(fine),coarse=np.concatenate(coarse))

def gradient_diagnostic(models,data,ids,device):
    """E gradient alignment for the actual two loss terms; no optimizer update."""
    x,y=data.batch(ids,device);s=models['E'](x)
    pixel=(models['D'](s)-x).square().mean();pose=(models['H'](s)-y).square().mean()
    params=list(models['E'].parameters())
    a=torch.autograd.grad(pixel,params,retain_graph=True);b=torch.autograd.grad(pose,params)
    a=torch.cat([g.flatten() for g in a]);b=torch.cat([g.flatten() for g in b])
    return dict(frames=len(ids),image_mse=float(pixel.detach()),pose_mse=float(pose.detach()),
                image_E_grad_l2=float(a.norm()),pose_E_grad_l2=float(b.norm()),
                cosine=float((a@b)/(a.norm()*b.norm()).clamp_min(1e-30)))

def foreground_masks(target):
    # Same diagnostic geometry as the archived perception diagnosis; source is
    # third_party/swm/pusht.py add_tee(scale30), circle radius15. Not training labels.
    import cv2
    x,y,bx,by=target[:4]*512;angle=np.arctan2(target[4],target[5])
    yy,xx=np.mgrid[:512,:512]
    pusher=(((xx+.5-x)**2+(yy+.5-y)**2)<=225).astype(np.float32)
    block=np.zeros((512,512),np.uint8)
    rotation=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    for polygon in (np.array([[-60,30],[60,30],[60,0],[-60,0]]),
                    np.array([[-15,30],[-15,120],[15,120],[15,30]])):
        cv2.fillPoly(block,[np.rint(polygon@rotation.T+[bx,by]).astype(np.int32)],1)
    pusher=cv2.resize(pusher,(64,64),interpolation=cv2.INTER_AREA)
    block=cv2.resize(block.astype(np.float32),(64,64),interpolation=cv2.INTER_AREA)
    return dict(pusher=pusher,block=block,background=1-np.maximum(pusher,block))

@torch.no_grad()
def region_errors(models,data,ids,device,mean):
    total={name:dict(squared_error=0.,white_squared_error=0.,mean_squared_error=0.,scalars=0.) for name in ('pusher','block','background')}
    records=[]
    for start in range(0,len(ids),128):
        part=ids[start:start+128];x,_=data.batch(part,device);d=models['D'](models['E'](x))
        error=(d-x).square().permute(0,2,3,1).cpu().numpy()
        white=(1-x).square().permute(0,2,3,1).cpu().numpy()
        mean_error=np.square(x.permute(0,2,3,1).cpu().numpy()-mean)
        for j,i in enumerate(part):
            row=dict(index=int(i),regions={})
            for name,mask in foreground_masks(data.targets[i]).items():
                values=dict(squared_error=float((error[j]*mask[...,None]).sum()),
                            white_squared_error=float((white[j]*mask[...,None]).sum()),mean_squared_error=float((mean_error[j]*mask[...,None]).sum()),scalars=float(mask.sum()*3))
                row['regions'][name]=values
                for k,v in values.items():total[name][k]+=v
            records.append(row)
    return {name:{**v,'mse':v['squared_error']/v['scalars'],'white_mse':v['white_squared_error']/v['scalars'],'mean_image_mse':v['mean_squared_error']/v['scalars']} for name,v in total.items()},records

def linear_probe(train_z,train_y,held_z,held_y):
    # Full E feature vector, centered on train only. Fixed ridge, no tuning/test fitting.
    a=torch.from_numpy(np.concatenate([train_z[k].reshape(len(train_y),-1) for k in ('fine','coarse')],axis=1)).double()
    b=torch.from_numpy(np.concatenate([held_z[k].reshape(len(held_y),-1) for k in ('fine','coarse')],axis=1)).double()
    mean=a.mean(0);scale=(a-mean).square().mean().sqrt().clamp_min(1e-12)
    a=(a-mean)/scale/math.sqrt(a.shape[1]);b=(b-mean)/scale/math.sqrt(b.shape[1])
    y=torch.from_numpy(train_y).double();ym=y.mean(0);gram=a@a.T
    alpha=torch.linalg.solve(gram+.01*len(a)*torch.eye(len(a)),y-ym)
    pred=(b@a.T@alpha+ym).numpy()
    pos=np.abs(pred[:,:4]-held_y[:,:4])*512
    angle=np.arctan2(pred[:,4],pred[:,5])-np.arctan2(held_y[:,4],held_y[:,5])
    return dict(train_frames=len(a),heldout_frames=len(b),ridge=.01,training_rms_scale=float(scale),
                position_mae=pos.mean(0).tolist(),angle_mae_deg=float(np.abs(np.arctan2(np.sin(angle),np.cos(angle))).mean()*180/np.pi)),pred

@torch.no_grad()
def panels(models,data,ids,encoded,bases,output,label,device):
    ids=ids[:6];x,y=data.batch(ids,device);s=models['E'](x);d=models['D'](s)
    captured={}
    def capture(name):
        def hook(module,args):captured[name]=attention_weights(module,args[0],args[1]).detach().cpu().numpy()
        return hook
    handles=[models['E'].fine_from_coarse.register_forward_pre_hook(capture('fine_from_coarse')),
             models['E'].coarse_from_fine.register_forward_pre_hook(capture('coarse_from_fine'))]
    models['E'](x)
    for handle in handles:handle.remove()
    raw=x.permute(0,2,3,1).cpu().numpy();recon=d.permute(0,2,3,1).cpu().numpy()
    fig,axes=plt.subplots(len(ids),7,figsize=(13.5,1.9*len(ids)),layout='constrained')
    titles=['Input (pusher query marked)','Reconstruction','Mean absolute RGB error','Fine tokens · train PCA','Coarse tokens · train PCA','Fine token L2 norm','Fine→coarse attention']
    norm=np.linalg.norm(encoded['fine'],axis=-1);vmax=max(float(np.quantile(norm,.99)),1e-6)
    for j,i in enumerate(ids):
        images=[raw[j],recon[j],np.abs(raw[j]-recon[j]).mean(-1),
                pca_rgb(encoded['fine'][j],bases['fine']).reshape(16,16,3),
                pca_rgb(encoded['coarse'][j],bases['coarse']).reshape(8,8,3),norm[j].reshape(16,16)]
        for c,im in enumerate(images):
            kwargs={'vmin':0,'vmax':.25,'cmap':'magma'} if c==2 else {'vmin':0,'vmax':vmax,'cmap':'viridis'} if c==5 else {}
            handle=axes[j,c].imshow(im,**kwargs)
            if j==len(ids)-1 and c in (2,5):fig.colorbar(handle,ax=axes[j,c],fraction=.05)
        px,py=data.targets[i,:2]*16;query=int(np.clip(py,0,15))*16+int(np.clip(px,0,15))
        att=captured['fine_from_coarse'][j,:,query].mean(0).reshape(8,8)
        handle=axes[j,6].imshow(att,cmap='viridis',vmin=0,vmax=max(float(att.max()),1/64))
        fig.colorbar(handle,ax=axes[j,6],fraction=.05)
        axes[j,0].plot(px*4,py*4,'+',color='#d62728',markersize=9)
        axes[j,0].set_ylabel(f"ep{data.metadata[i]['source_episode']} frame{data.metadata[i]['frame']}",fontsize=8)
        for c in range(7):
            axes[j,c].set_xticks([]);axes[j,c].set_yticks([])
            if j==0:axes[j,c].set_title(titles[c],fontsize=8)
    population='fixed training diagnostic frames' if 'fixed64_training' in label else 'fixed held-out frames'
    fig.suptitle(label+' · '+population+'; PCA fitted to training tokens\nError range 0–0.25; attention is mean over four heads, descriptive rather than causal',fontsize=12)
    path=output/'perception_states.png';fig.savefig(path,dpi=110);fig.savefig(output/'perception_states.svg');plt.close(fig)
    np.savez_compressed(output/'panel_states.npz',indices=ids,inputs=raw,reconstructions=recon,**captured)
    return dict(file=path.name,title=label+' — perception states',caption=population+'; six predeclared frames; inputs, decoded RGB, errors, train-fit fine/coarse PCA, activation norms and pusher-query cross-scale attention.')


def pose_error_panel(raw,output,label):
    pred=np.asarray(raw['predictions']);target=np.asarray(raw['targets'])
    angle=np.asarray(raw['angle_abs_error_deg']);norm=np.asarray(raw['angle_norm'])
    fig,axes=plt.subplots(2,3,figsize=(11,6),layout='constrained')
    for c,name in enumerate(('pusher x','pusher y','block x','block y')):
        ax=axes.flat[c];ax.scatter(target[:,c]*512,pred[:,c]*512,s=3,alpha=.2,color='#2463a6',rasterized=True)
        lo=min(0.,float(pred[:,c].min()*512),float(target[:,c].min()*512))-8
        hi=max(512.,float(pred[:,c].max()*512),float(target[:,c].max()*512))+8
        outside=int(((pred[:,c]<0)|(pred[:,c]>1)).sum())
        ax.plot([lo,hi],[lo,hi],color='#db7923',lw=1)
        ax.set(title=f'{name} · {outside} estimates outside world',xlabel='True · world units',ylabel='Estimated · world units',xlim=(lo,hi),ylim=(lo,hi))
    axes[1,1].hist(angle,bins=np.arange(0,181,10),color='#2463a6')
    axes[1,1].axvline(10,color='#db7923',label='10° mean target (not per-frame gate)')
    axes[1,1].set(xlabel='Wrapped absolute angle error °',ylabel='Frames',title='Orientation error distribution')
    axes[1,2].scatter(norm,angle,s=3,alpha=.2,color='#2463a6',rasterized=True)
    axes[1,2].set(xlabel='Predicted sin/cos vector norm',ylabel='Angle absolute error °',title='Readout norm versus angle error')
    fig.suptitle(label+' · all evaluated frames; repeated frames within a group are correlated',fontsize=11)
    fig.savefig(output/'pose_readout.png',dpi=120);fig.savefig(output/'pose_readout.svg');plt.close(fig)
    return dict(file='pose_readout.png',title=label+' — pose readout',caption='Identity line for XY, wrapped angle errors and sin/cos output norm. These are descriptive frame distributions, not independent training replicates.')

def inspect_checkpoint(checkpoint,output,*,split='test',diagnostic=False):
    output=Path(output)
    if (output/'inspection_summary.json').exists():raise ValueError('preserve completed inspection')
    output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);device='cuda' if torch.cuda.is_available() else 'cpu'
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    saved=read_checkpoint(checkpoint);models=initial_models(4107,saved['models'])
    labelled='H' in saved['models']
    if labelled:models['H'].load_state_dict(saved['models']['H'])
    for model in models.values():model.to(device).eval()
    before=fingerprint_modules({k:models[k] for k in saved['models'] if k in models})
    train=task_frames(DATA,'train');held=task_frames(DATA,split)
    if diagnostic:
        # Reconstruct exact fixed64 population from the immutable diagnostic sampler.
        rng=np.random.default_rng(4107);episodes={}
        for i,m in enumerate(train.metadata):episodes.setdefault(m['source_episode'],[]).append(i)
        chosen=[int(rng.choice(episodes[int(e)])) for e in rng.choice(sorted(episodes),64,replace=False)]
        train=FrameSet(train.frames,train.rows[chosen],train.targets[chosen],[train.metadata[i] for i in chosen],train.fingerprint)
        held=train;split='fixed64_training'
    fit_ids=sample_training(train);held_ids=np.random.default_rng(92502).choice(len(held),min(256,len(held)),replace=False)
    metrics,raw=evaluate(models,held,np.arange(len(held)),device=device,labelled=labelled,return_records=True)
    mean=mean_image(train);np.save(output/'train_mean_image.npy',mean)
    raw['mean_image_mse']=mean_image_errors(mean,held);metrics['mean_image_mse']=float(np.mean(raw['mean_image_mse']))
    metrics['reconstruction_to_mean_ratio']=metrics['image_mse']/max(metrics['mean_image_mse'],1e-30)
    raw['metadata']=held.metadata;json_atomic(output/'frame_errors.json',raw)
    summary=dict(schema='curriculum-inspection-v1',status='completed',checkpoint=str(checkpoint),
                 checkpoint_sha256=file_hash(checkpoint),model_fingerprint=saved['model_fingerprint'],step=saved['global_update'],
                 split=split,frames=len(held),labelled=labelled,metrics=metrics,
                 train_fit_indices=fit_ids.tolist(),heldout_diagnostic_indices=held_ids.tolist(),
                 fit_source_rows=train.rows[fit_ids].tolist(),heldout_diagnostic_source_rows=held.rows[held_ids].tolist(),
                 dataset_fingerprint=held.fingerprint,panels=[],activation={})
    if labelled:
        summary['group_balanced']=grouped_errors(raw,held.metadata)
        summary['E_gradients']=gradient_diagnostic(models,train,fit_ids[:64],device)
    train_z=encode(models,train,fit_ids,device);held_z=encode(models,held,held_ids,device);bases={}
    for name in ('fine','coarse'):
        basis=pca_fit(train_z[name].reshape(-1,64));bases[name]=basis
        np.savez_compressed(output/f'{name}_train_pca.npz',**basis)
        spectrum=basis['eigenvalues'];p=spectrum/np.maximum(spectrum.sum(),1e-30)
        summary['activation'][name]=dict(train_channel_effective_rank=float(np.exp(-(p*np.log(np.maximum(p,1e-30))).sum())),
            train_top3_variance_fraction=float(p[:3].sum()),heldout_mean_token_norm=float(np.linalg.norm(held_z[name],axis=-1).mean()),
            heldout_mean_std_across_frames=float(held_z[name].std(axis=0).mean()),eigenvalues=spectrum.tolist())
    probe,pred=linear_probe(train_z,train.targets[fit_ids],held_z,held.targets[held_ids]);summary['linear_probe']=probe
    np.savez_compressed(output/'heldout_probe.npz',predictions=pred,targets=held.targets[held_ids],indices=held_ids)
    regions,region_raw=region_errors(models,held,held_ids,device,mean);summary['regions']=regions
    json_atomic(output/'region_errors.json',dict(indices=held_ids.tolist(),records=region_raw,summary=regions))
    label=output.name+f" · update{saved['global_update']} · {split}"
    summary['panels'].append(panels(models,held,held_ids,held_z,bases,output,label,device))
    if labelled:summary['panels'].append(pose_error_panel(raw,output,label))
    fig,axes=plt.subplots(1,2,figsize=(9,3),layout='constrained')
    for name in ('fine','coarse'):
        e=bases[name]['eigenvalues'];axes[0].semilogy(np.arange(1,65),np.maximum(e,1e-14),label=name)
    axes[0].set(xlabel='Channel component',ylabel='Training token covariance eigenvalue',title='Within-checkpoint channel spectrum');axes[0].legend()
    if labelled:
        err=np.asarray(raw['position_abs_error']);axes[1].boxplot(err,whis=(5,95),showfliers=False,tick_labels=['pusher x','pusher y','block x','block y'])
        axes[1].set(ylabel='Absolute error · world units',title='All evaluated frames · 5–95% whiskers')
    else:axes[1].text(.1,.5,'Image-only checkpoint\nNo task head / no task pose result');axes[1].axis('off')
    fig.savefig(output/'spectrum_errors.png',dpi=120);plt.close(fig)
    summary['panels'].append(dict(file='spectrum_errors.png',title=label+' — spectrum and error distribution',
        caption='Channel spectrum includes spatial and across-frame variation; it is not the rank of the full feature vector. Error whiskers show frame distribution, not uncertainty over training seeds.'))
    after=fingerprint_modules({k:models[k] for k in saved['models'] if k in models})
    if before!=after:raise RuntimeError('inspection mutated checkpoint modules')
    summary['checkpoint_unchanged']=True
    json_atomic(output/'inspection_summary.json',summary)
    print(json.dumps({k:summary[k] for k in ('checkpoint','split','frames','metrics','linear_probe','regions')}),flush=True)
    return summary

def main():
    p=argparse.ArgumentParser();p.add_argument('checkpoint',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--split',default='test',choices=['validation','test']);p.add_argument('--diagnostic',action='store_true')
    a=p.parse_args();inspect_checkpoint(a.checkpoint,a.output,split=a.split,diagnostic=a.diagnostic)
if __name__=='__main__':main()
