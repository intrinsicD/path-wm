"""Fixed-case geometry comparisons from saved P1 and objective-follow-up outputs."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from .perception_cache import ROOT
from .pose_accessibility import setup,analyze
from .data import file_hash
from world_model.pusht.checkpoints import json_atomic


def build():
    setup(); root=ROOT/'figures/localization'; root.mkdir(parents=True,exist_ok=True)
    if (root/'curriculum_analysis.json').exists(): raise FileExistsError('preserve geometry figure evidence')
    for seed in (9107,9108,9109):
        for arm in ('cnn','vit'):
            if not (ROOT/f'localization/seed_{seed}/{arm}/curriculum_analysis.json').exists():
                raise RuntimeError('complete all geometry objective fits first')
    selection=ROOT/'figures/fresh/summary.json'; ids=json.loads(selection.read_text())['indices']
    sources=[selection]; panels=[]
    for arm in ('cnn','vit'):
        old_root=ROOT/f'fresh/evaluations/seed_9107/{arm}'
        paths=[old_root/'fresh_errors.npz',ROOT/f'localization/seed_9107/{arm}/fresh_errors.npz',old_root/'fresh_images.npz']
        old,new=(dict(np.load(p)) for p in paths[:2]); images=dict(np.load(paths[2]))['rgb']; sources+=paths
        for key in ('indices','targets'):
            if not np.array_equal(old[key],new[key]): raise ValueError('geometry figure populations differ')
        for raw in (old,new):
            if not np.allclose(raw['locations'].sum((-2,-1)),1,atol=1e-5) or (raw['locations']<0).any():
                raise ValueError('unnormalized output probability map')
        if len(images)!=len(old['targets']): raise ValueError('geometry figure images unpaired')
        fig,axes=plt.subplots(len(ids),5,figsize=(12,2.05*len(ids)+1.7),dpi=130)
        fig.subplots_adjust(left=.085,right=.99,top=.89,bottom=.105,wspace=.07,hspace=.22)
        fig.suptitle(f'{arm.upper()}: geometry before and after location-distribution supervision',fontsize=15,y=.989)
        fig.text(.085,.965,'Frozen encoder · seed 9107 · independently validation-selected heads · same saved observations',fontsize=9)
        colors=('black','#087fda','#e57909')
        fig.legend([Line2D([0],[0],color=c,marker='o',linestyle='') for c in colors],
            ['Ground truth','Original MSE','MSE + 0.001 KL'],loc='upper center',bbox_to_anchor=(.5,.952),ncol=3,frameon=False,fontsize=9)
        titles=['Observed scene','Pusher: original','Pusher: + KL','Body: original','Body: + KL']
        for row,index in enumerate(ids):
            ax=axes[row,0];ax.imshow(images[index],extent=(0,512,512,0),interpolation='nearest')
            for pose,color in zip((old['targets'][index],old['predictions'][index],new['predictions'][index]),colors):
                xy=pose[:4]*512;theta=np.arctan2(pose[4],pose[5])
                ax.plot(xy[0],xy[1],'o',ms=4,mfc='none',mec=color,mew=1.2)
                ax.plot(xy[2],xy[3],'+',ms=7,color=color)
                ax.arrow(xy[2],xy[3],50*np.cos(theta),50*np.sin(theta),width=1.6,head_width=8,color=color,length_includes_head=True)
            reason='fixed' if index<4 else ('this encoder worst' if index==int(old['case_q'].argmax()) else 'other encoder worst')
            ax.set_ylabel(f'Case {index} · {reason}\ncase q: {old["case_q"][index]:.2f} → {new["case_q"][index]:.2f}',fontsize=8)
            for col,(raw,point) in enumerate(((old,0),(new,0),(old,1),(new,1)),1):
                ax=axes[row,col];p=raw['locations'][index,point]
                im=ax.imshow(np.log10(p.clip(1e-6)),vmin=-6,vmax=0,cmap='magma',interpolation='nearest')
                target=raw['targets'][index,point*2:point*2+2]*15
                ax.plot(*target,'+',color='white',ms=6,mew=1.)
                entropy=float(-(p*np.log(p.clip(1e-30))).sum()/np.log(256))
                ax.set_xlabel(f'H/log256 = {entropy:.2f}',fontsize=8)
            for ax in axes[row]:ax.set_xticks([]);ax.set_yticks([])
        for ax,title in zip(axes[0],titles):ax.set_title(title,fontsize=9)
        cax=fig.add_axes((.33,.064,.34,.009));fig.colorbar(im,cax=cax,orientation='horizontal',ticks=[-6,-4,-2,0],label='log10 probability, common scale')
        fig.text(.5,.012,'Fixed original cases, including labeled baseline worst cases. White map crosses = target points. Maps are readout distributions, not calibrated uncertainty.',ha='center',fontsize=8)
        path=root/f'{arm}_location_comparison.png';fig.savefig(path,facecolor='white');plt.close(fig);sources.append(path)
        panels.append(dict(file=str(path.resolve()),title=f'{arm.upper()} geometry objective comparison',
            caption='Same first-seed cases chosen before the new objective: four fixed rows and the original CNN/ViT worst cases. Common log-probability scale.',embed=True))
    manifest=root/'manifest.json';json_atomic(manifest,dict(seed=9107,indices=ids,source_sha256=file_hash(__file__),
        sources={str(p):file_hash(p) for p in sources},scope='Saved outputs only; no new inference, adaptation or case selection'))
    sources.append(manifest)
    analyze(ROOT,root,'Geometry objective distribution comparison',dict(cases=len(ids),encoders=2),
        '## Location maps before and after distribution supervision\n\nThe case list is inherited from the original P1 inspection. '
        'All original and new predictions use identical observations and targets. White map crosses mark target points, and log-probability ranges are shared. '
        'Sharper maps alone do not establish better coordinates or calibrated uncertainty; paired quantitative errors are reported separately.',sources,panels)
    return root


if __name__=='__main__': build()
