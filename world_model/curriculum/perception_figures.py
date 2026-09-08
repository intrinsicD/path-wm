"""Scientific panels from fixed held-outs and train-fitted feature projections."""
import json
from pathlib import Path
import time
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from .perception_cache import ROOT,cache_root
from .perception_training import load_data
from .perception_heads import feature_grids
from .encoder_visual_audit import fit_pca,pca_colors,variance_parts
from .pose_accessibility import setup,analyze
from .data import file_hash
from world_model.pusht.checkpoints import json_atomic

OUT=ROOT/'figures'


def save(fig,out,name):
    fig.savefig(out/name,dpi=145,facecolor='white'); plt.close(fig)
    return out/name


def image_axis(ax,values):
    ax.imshow(values,interpolation='nearest'); ax.set_xticks([]); ax.set_yticks([])


def pca_panel():
    out=OUT/'pca'; out.mkdir(parents=True,exist_ok=True)
    if (out/'feature_pca_v2.png').exists(): raise FileExistsError('preserve completed PCA layout revision')
    fits={}; raw={}; statistics={}; samples={}; sample_rgb=[]; identities={}
    for encoder in ('cnn','vit'):
        data,_=load_data(encoder,False); train=[]; test=[]; ids={}
        rng=np.random.default_rng(92011)
        for domain in ('coco','pusht'):
            d=data[domain,'train']; ids[domain]=rng.choice(len(d['ds']),64,replace=False)
            train.append(np.asarray(d['cache'][ids[domain]]).astype('float32'))
            d=data[domain,'test']; test.append(np.asarray(d['cache'][:3]).astype('float32'))
            if encoder=='cnn': sample_rgb.extend(np.asarray(d['ds'].frames[d['ds'].rows[:3]]))
        identities[encoder]=dict(training_indices={k:v.tolist() for k,v in ids.items()},
            cache_manifest_sha256=file_hash(cache_root()/encoder/'manifest.json'))
        training=feature_grids(torch.from_numpy(np.concatenate(train)),encoder)
        testing=feature_grids(torch.from_numpy(np.concatenate(test)),encoder)
        for level,values,sample in zip(('fine','coarse'),training,testing):
            values=values.numpy(); sample=sample.numpy(); samples[encoder,level]=sample
            statistics[encoder+'_'+level]=variance_parts(values)
            for centered in ((False,True) if level=='coarse' else (False,)):
                key=encoder+'_'+level+('_centered' if centered else '')
                fitted=fit_pca(values,image_centered=centered); fits[key]=fitted
                colors,clipping=pca_colors(sample,fitted)
                raw[key+'_colors']=colors
                for name in ('mean','basis','low','high'): raw[key+'_'+name]=fitted[name]
                statistics[key+'_projection']=dict(top3_variance_fraction=fitted['top3_variance_fraction'],
                    test_channel_clipping_fraction=clipping,image_centered=centered)
    columns=[('cnn_fine','CNN fine16×16'),('cnn_coarse','CNN coarse8×8'),
        ('cnn_coarse_centered','CNN coarse\nimage-centered'),('vit_fine','ViT final16×16'),
        ('vit_coarse','ViT pooled8×8'),('vit_coarse_centered','ViT pooled\nimage-centered')]
    fig,axes=plt.subplots(6,7,figsize=(16,13)); fig.subplots_adjust(top=.875,bottom=.055,wspace=.03,hspace=.12)
    fig.suptitle('Feature PCA: global variation and within-image structure',fontsize=17,y=.97)
    fig.text(.5,.935,'Train-fitted bases and color limits. Colors have no shared meaning across columns. ViT coarse is an average pool.',ha='center',fontsize=10)
    for i in range(6):
        image_axis(axes[i,0],sample_rgb[i]); axes[i,0].set_ylabel(('COCO' if i<3 else 'PushT')+f' test {i%3}',fontsize=10)
        for col,(key,title) in enumerate(columns,1):
            colors=raw[key+'_colors'][i]; side=int(np.sqrt(len(colors)))
            image_axis(axes[i,col],colors.reshape(side,side,3))
            if i==0: axes[i,col].set_title(title,fontsize=10)
    axes[0,0].set_title('Observed RGB64',fontsize=10)
    fig.text(.5,.018,'Each basis: 64 COCO + 64 PushT training images. Global PCA can be dominated by image means; centering reveals different variation.',ha='center',fontsize=10)
    path=save(fig,out,'feature_pca_v2.png'); raw['rgb']=np.asarray(sample_rgb)
    np.savez_compressed(out/'projection_data_v2.npz',**raw)
    summary=dict(statistics=statistics,identities=identities,test_indices={'coco':[0,1,2],'pusht':[0,1,2]},
        source_sha256=file_hash(__file__),protocol_sha256=file_hash('docs/perception-inspection-protocol-2026-09-08.md'))
    json_atomic(out/'summary_v2.json',summary)
    sources=[path,out/'projection_data_v2.npz',out/'summary_v2.json']
    analyze(ROOT,out,'Train-fitted feature PCA inspection',{},
        '## Interpreting fine and coarse feature pictures\n\nPCA fits and color limits use only a fixed balanced training sample. '
        'Per-image centering changes the question from total variation to spatial deviations within an image. '
        'Each encoder/scale has an independent color basis; colors are not semantic labels or comparable across columns. '
        'These pictures do not decide representation quality; the paired downstream readouts do.',sources,
        [dict(file=str(path.resolve()),title='Fine and coarse feature PCA',caption='Fixed held-out examples; independent train-only bases. Image-centered coarse views isolate within-image variation.',embed=True)])


def reconstruction_panel():
    out=OUT/'readouts'; out.mkdir(parents=True,exist_ok=True)
    if (out/'curriculum_analysis.json').exists(): raise FileExistsError('preserve readout panels')
    paths={e:ROOT/'p1/seed_9107'/e/'coco_panels.npz' for e in ('cnn','vit')}
    values={e:dict(np.load(p)) for e,p in paths.items()}
    if not np.array_equal(values['cnn']['rgb'],values['vit']['rgb']): raise ValueError('readout panels use different images')
    fig,axes=plt.subplots(6,6,figsize=(13.5,13)); fig.subplots_adjust(top=.9,bottom=.04,wspace=.03,hspace=.10)
    fig.suptitle('The same held-out images, different recoverable information',fontsize=17,y=.97)
    fig.text(.5,.928,'Frozen encoders · independent typed heads · seed9107 · first six COCO test images',ha='center',fontsize=11)
    titles=['Observed RGB','CNN RGB','ViT RGB','Target union mask','CNN mask probability','ViT mask probability']
    for i in range(6):
        c,v=values['cnn'],values['vit']
        inputs=[c['rgb'][i].transpose(1,2,0),c['reconstruction'][i].transpose(1,2,0),v['reconstruction'][i].transpose(1,2,0)]
        for col,data in enumerate(inputs): image_axis(axes[i,col],data)
        for col,data in enumerate((c['mask'][i,0],c['probability'][i,0],v['probability'][i,0]),3):
            masked=np.ma.masked_where(c['valid'][i,0]==0,data)
            cmap=plt.get_cmap('gray').copy(); cmap.set_bad('#ce6fa1')
            axes[i,col].imshow(masked,vmin=0,vmax=1,cmap=cmap,interpolation='nearest'); axes[i,col].set_xticks([]); axes[i,col].set_yticks([])
        axes[i,0].set_ylabel(f'Test {i}',fontsize=10)
    for ax,title in zip(axes[0],titles): ax.set_title(title,fontsize=10)
    fig.text(.5,.014,'White mask probability =1, black =0; pink marks ignored crowd pixels. These fixed examples are not a performance average.',ha='center',fontsize=10)
    path=save(fig,out,'coco_readouts.png')
    analyze(ROOT,out,'Fixed image and mask readout panels',{},
        '## RGB detail and foreground recognition\n\nThe first six COCO test images are shown for the same paired seed. '
        'The RGB and mask heads have independent trainable parameters. All panels use the predeclared pose-selected snapshot. '
        'Full-population metrics, other seeds and endpoints remain separate; these examples do not select a winner.',
        [path,*paths.values()], [dict(file=str(path.resolve()),title='COCO readouts from frozen encoders',caption='Fixed examples; probability scales are identical and ignored pixels are pink.',embed=True)])


def fresh_panel():
    out=OUT/'fresh'; out.mkdir(parents=True,exist_ok=True)
    if (out/'fresh_pose_distributions_v2.png').exists(): raise FileExistsError('preserve completed fresh layout revision')
    roots={e:ROOT/'fresh/evaluations/seed_9107'/e for e in ('cnn','vit')}
    errors={e:dict(np.load(p/'fresh_errors.npz')) for e,p in roots.items()}
    images={e:dict(np.load(p/'fresh_images.npz')) for e,p in roots.items()}
    if not np.array_equal(images['cnn']['rgb'],images['vit']['rgb']): raise ValueError('fresh panel source mismatch')
    ids=list(dict.fromkeys([0,1,2,3,int(errors['cnn']['case_q'].argmax()),int(errors['vit']['case_q'].argmax())]))
    fig,axes=plt.subplots(len(ids),7,figsize=(16,2.15*len(ids)+1.6)); fig.subplots_adjust(top=.90,bottom=.10,wspace=.05,hspace=.25)
    fig.suptitle('Fresh poses: spatial output distributions and difficult cases',fontsize=17,y=.98)
    fig.legend([Line2D([0],[0],color=color,marker='o',linestyle='') for color in ('black','#087fda','#e57909')],
        ['Ground truth','CNN prediction','ViT prediction'],loc='upper center',bbox_to_anchor=(.5,.963),ncol=3,frameon=False)
    titles=['Observed scene','CNN pusher','CNN body origin','CNN orientation pool','ViT pusher','ViT body origin','ViT orientation pool']
    statistics={}
    for row,index in enumerate(ids):
        ax=axes[row,0]; ax.imshow(images['cnn']['rgb'][index],extent=(0,512,512,0),interpolation='nearest')
        for pose,color in [(errors['cnn']['targets'][index],'black'),(errors['cnn']['predictions'][index],'#087fda'),(errors['vit']['predictions'][index],'#e57909')]:
            xy=pose[:4]*512; theta=np.arctan2(pose[4],pose[5])
            ax.plot(xy[0],xy[1],'o',ms=4,mfc='none',mec=color,mew=1.2)
            ax.plot(xy[2],xy[3],'+',ms=7,color=color)
            ax.arrow(xy[2],xy[3],50*np.cos(theta),50*np.sin(theta),width=1.6,head_width=8,color=color,length_includes_head=True)
        ax.set_xticks([]); ax.set_yticks([])
        reason='fixed' if index<4 else '/'.join(e+' worst' for e in ('cnn','vit') if index==int(errors[e]['case_q'].argmax()))
        ax.set_ylabel(f'Case {index}\n{reason}',fontsize=9)
        for group,e in enumerate(('cnn','vit')):
            raw=errors[e]; probability=[raw['locations'][index,0],raw['locations'][index,1],raw['orientation_pool'][index,0]]
            for k,p in enumerate(probability):
                col=1+group*3+k; ax=axes[row,col]
                im=ax.imshow(np.log10(p.clip(1e-6)),vmin=-6,vmax=0,cmap='magma',interpolation='nearest')
                h=float(-(p*np.log(p.clip(1e-30))).sum()/np.log(p.size))
                ax.set_xlabel(f'H/log256={h:.2f}',fontsize=8); ax.set_xticks([]); ax.set_yticks([])
    for ax,title in zip(axes[0],titles): ax.set_title(title,fontsize=9)
    cax=fig.add_axes((.32,.063,.36,.009)); fig.colorbar(im,cax=cax,orientation='horizontal',ticks=[-6,-4,-2,0],label='log10 probability (same scale in every map)')
    fig.text(.5,.013,'Circles=pusher; plus signs=body origin; arrows=orientation. Output distributions are not encoder attention. Error-selected rows are explicitly labeled.',ha='center',fontsize=9)
    path=save(fig,out,'fresh_pose_distributions_v2.png')
    for e,raw in errors.items():
        maps=raw['locations']; h=-(maps*np.log(maps.clip(1e-30))).sum((2,3))/np.log(256)
        statistics[e]=dict(mean_location_entropy=h.mean(0).tolist(),
            correlation_pusher_entropy_max_coordinate_error=float(np.corrcoef(h[:,0],raw['position_abs_error'][:,:2].max(1))[0,1]),
            correlation_body_entropy_max_coordinate_error=float(np.corrcoef(h[:,1],raw['position_abs_error'][:,2:].max(1))[0,1]))
    json_atomic(out/'summary.json',dict(indices=ids,statistics=statistics,seed=9107,scope='Four fixed cases and package-specific worst cases; summary statistics use all512'))
    sources=[path,out/'summary.json',*[p/n for p in roots.values() for n in ('fresh_errors.npz','fresh_images.npz')]]
    analyze(ROOT,out,'Fresh spatial output distribution inspection',{},
        '## What the pose readout attends to\n\nFour fixed cases and explicitly marked package-specific worst cases show the learned pusher/body-location distributions '
        'and orientation pooling. A spatial expectation can fall between competing peaks; the complete distributions therefore matter alongside a single predicted coordinate. '
        'These maps belong to the decoder. They are not cross-scale attention or proof of calibrated uncertainty. Full512-case error and entropy statistics accompany the figures.',
        sources,[dict(file=str(path.resolve()),title='Fresh pose readout distributions',caption='Fixed and labeled error-selected examples; common log-probability scale; source and predictions are paired.',embed=True)])


def main():
    setup(); plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    OUT.mkdir(exist_ok=True); start=time.monotonic()
    for name,filename,fn in [('pca','feature_pca_v2.png',pca_panel),('readouts','coco_readouts.png',reconstruction_panel),('fresh','fresh_pose_distributions_v2.png',fresh_panel)]:
        if not (OUT/name/filename).exists(): fn()
    json_atomic(OUT/'generation.json',dict(seconds=time.monotonic()-start,source_sha256=file_hash(__file__)))


if __name__=='__main__': main()
