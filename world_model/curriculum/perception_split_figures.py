"""Paired saved RGB and mask outputs with shared versus split decoder trunks."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .perception_cache import ROOT
from .pose_accessibility import setup,analyze
from .data import file_hash
from world_model.pusht.checkpoints import json_atomic


def build():
    setup();root=ROOT/'figures/split';root.mkdir(parents=True,exist_ok=True)
    if (root/'curriculum_analysis.json').exists():raise FileExistsError('preserve split decoder figure')
    sources=[ROOT/'decoders/seed_9107/early/coco_panels.npz',ROOT/'independent/seed_9107/split/coco_panels.npz']
    shared,split=(dict(np.load(p)) for p in sources)
    for key in ('rgb','mask','valid','indices'):
        if not np.array_equal(shared[key],split[key]):raise ValueError('split decoder image populations differ')
    fig,axes=plt.subplots(6,6,figsize=(13.2,12.3),dpi=130)
    fig.subplots_adjust(left=.045,right=.995,bottom=.075,top=.93,wspace=.06,hspace=.14)
    titles=['RGB target','RGB shared','RGB separate','Mask target','Mask shared','Mask separate']
    for row in range(6):
        values=[shared['rgb'][row],shared['reconstruction'][row],split['reconstruction'][row],
            shared['mask'][row,0],shared['probability'][row,0],split['probability'][row,0]]
        for col,value in enumerate(values):
            ax=axes[row,col]
            if col<3:ax.imshow(value.transpose(1,2,0).clip(0,1),interpolation='nearest')
            else:
                ax.imshow(value,cmap='gray',vmin=0,vmax=1,interpolation='nearest')
                invalid=~shared['valid'][row,0].astype(bool);overlay=np.zeros((*invalid.shape,4));overlay[invalid]=[1,.2,.65,.7]
                ax.imshow(overlay,interpolation='nearest')
            ax.set_xticks([]);ax.set_yticks([])
            if row==0:ax.set_title(titles[col],fontsize=10)
            if col==0:ax.set_ylabel(f'Test row {int(shared["indices"][row])}',fontsize=8)
    fig.suptitle('Separate decoder trunks: appearance and foreground tradeoffs',fontsize=16,x=.045,ha='left',y=.982,weight='bold')
    fig.text(.045,.952,'Seed 9107 · fixed 4,000-update endpoints · same frozen early/final/coarse inputs and observed test crops',fontsize=10)
    fig.text(.045,.043,'Common RGB and probability ranges [0,1]. Pink: ignored crowd pixels. Foreground target is a union, not a requested object.',fontsize=9)
    fig.text(.045,.019,'Separate trunks retain joint gradient clipping. This first-seed illustration accompanies three-seed measurements; it does not test imagined futures.',fontsize=9)
    path=root/'decoder_comparison.png';fig.savefig(path,facecolor='white');plt.close(fig);sources.append(path)
    manifest=root/'manifest.json';json_atomic(manifest,dict(seed=9107,indices=shared['indices'].tolist(),source_sha256=file_hash(__file__),sources={str(p):file_hash(p) for p in sources}))
    sources.append(manifest)
    analyze(ROOT,root,'Shared and split decoder output comparison',dict(cases=6,seed=9107),
        '## Shared versus separate dense trunks\n\nSame first six test crops, first paired seed and fixed endpoints. '
        'Both models receive the same local/contextual inputs. RGB and foreground use common ranges, with ignored crowd pixels marked. '
        'The panel visualizes one seed; final tradeoffs are reported across all three.',sources,
        [dict(file=str(path.resolve()),title='Shared and split decoder outputs',caption='First six seed9107 COCO crops, same inputs and endpoints; common ranges.',embed=True)])


if __name__=='__main__':build()
