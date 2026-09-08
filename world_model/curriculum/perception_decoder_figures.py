"""Fixed-case dense decoder comparisons from saved endpoint outputs only."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .perception_cache import ROOT
from .pose_accessibility import analyze,setup
from .data import file_hash
from world_model.pusht.checkpoints import json_atomic


def build():
    setup(); root=ROOT/'figures/decoders'; root.mkdir(parents=True,exist_ok=True)
    if (root/'curriculum_analysis.json').exists(): raise FileExistsError('preserve decoder figures')
    kinds=('late','early','raw','conditioned'); panels={}; sources=[]
    for kind in kinds:
        out=ROOT/'decoders/seed_9107'/kind
        if not (out/'curriculum_analysis.json').exists(): raise RuntimeError('first paired decoder seed is not complete')
        path=out/'coco_panels.npz'; panels[kind]=dict(np.load(path));sources.append(path)
    reference=panels['late']
    for p in panels.values():
        for field in ('rgb','mask','valid','indices'):
            if not np.array_equal(p[field],reference[field]): raise ValueError('decoder panel populations differ')
    pictures=[]
    for task in ('rgb','mask'):
        fig,axes=plt.subplots(6,5,figsize=(10.8,12.3),dpi=150)
        fig.subplots_adjust(left=.045,right=.995,bottom=.07,top=.93,wspace=.07,hspace=.14)
        for row in range(6):
            for col,kind in enumerate(('target',*kinds)):
                ax=axes[row,col]
                if task=='rgb':
                    values=reference['rgb'][row] if kind=='target' else panels[kind]['reconstruction'][row]
                    ax.imshow(np.clip(values.transpose(1,2,0),0,1),interpolation='nearest')
                else:
                    values=reference['mask'][row,0] if kind=='target' else panels[kind]['probability'][row,0]
                    ax.imshow(values,cmap='gray',vmin=0,vmax=1,interpolation='nearest')
                    invalid=~reference['valid'][row,0].astype(bool)
                    overlay=np.zeros((*invalid.shape,4));overlay[invalid]=[1,.2,.65,.7]
                    ax.imshow(overlay,interpolation='nearest')
                ax.set_xticks([]);ax.set_yticks([])
                if row==0: ax.set_title(kind.replace('conditioned','early + task FiLM'),fontsize=10)
                if col==0: ax.set_ylabel(f'Test row {int(reference["indices"][row])}',fontsize=8)
        title='Observed RGB reconstruction' if task=='rgb' else 'COCO foreground probability'
        fig.suptitle(title+' · four decoder inputs',fontsize=16,x=.045,ha='left',y=.983,weight='bold')
        fig.text(.045,.954,'Seed 9107 · fixed 4,000-update endpoints · first six COCO test crops · frozen pretrained ViT',fontsize=10)
        fig.text(.045,.038,('RGB in [0,1], common range; local inputs are available for the present observation.' if task=='rgb' else
            'All probabilities use [0,1]. Pink: ignored crowd-only pixels. Target is union foreground, not an object query.'),fontsize=9)
        fig.text(.045,.017,'Late: duplicated final tokens. Early: patch embedding. Raw: source pixels. Task FiLM: early inputs with output conditioning.',fontsize=8)
        path=root/f'{task}_comparison.png';fig.savefig(path,facecolor='white');plt.close(fig);sources.append(path)
        pictures.append(dict(file=str(path.resolve()),title=title,caption='Fixed first-six COCO cases, all four seed9107 endpoints; common output ranges.',embed=True))
    json_atomic(root/'manifest.json',dict(seed=9107,indices=reference['indices'].tolist(),source_sha256=file_hash(__file__),
        sources={str(p):file_hash(p) for p in sources},scope='Fixed first-six present-observation outputs; no cherry-picked examples or new model inference'))
    sources.append(root/'manifest.json')
    analyze(ROOT,root,'Fixed decoder output comparisons',dict(cases=6,arms=4,seed=9107),
        '## Dense decoder outputs on fixed COCO crops\n\nFirst six test rows, first paired seed, fixed endpoints. '
        'All RGB and probability views use the same ranges; crowd-only ignored pixels are pink. '
        'These panels show observed outputs and do not measure future reconstruction.',sources,pictures)
    return root


if __name__=='__main__': build()
