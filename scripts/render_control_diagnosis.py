"""Scientific contact sheets of actual simulator frames; no image decoder implied."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def render():
    directory=Path('runs/diagnostics/pusht_control_diagnosis/ranking')
    manifest=json.loads((directory/'manifest.json').read_text())
    outputs=[]
    for index in (0,4):
        case=manifest['cases'][index]
        images=np.load(directory/f'case_{index}_images.npz')
        fig,axes=plt.subplots(3,8,figsize=(15,6.2),dpi=130)
        for row,kind in enumerate(('replay','pilot_plan','released_plan')):
            frames=[images['source'],*images[kind],images['goal']]
            for column,(axis,frame) in enumerate(zip(axes[row],frames)):
                axis.imshow(frame);axis.set_xticks([]);axis.set_yticks([])
                for spine in axis.spines.values():spine.set_visible(False)
                if row==0:axis.set_title(['Source start','Simulator reset','Step 5','Step 10','Step 15','Step 20','Step 25','Source goal'][column],fontsize=10)
            axes[row,0].set_ylabel(kind.replace('_',' '),fontsize=10)
        fig.suptitle(f"Actual simulator rollouts · {case['population']} case {index} · source episode {case['source_episode']}, start {case['start']}",fontsize=13)
        fig.text(.5,.02,'All candidates start from the same reset. Images show observed simulator states; latent predictions are plotted in the dashboard.',ha='center',fontsize=9)
        fig.subplots_adjust(left=.06,right=.995,top=.9,bottom=.08,wspace=.02,hspace=.06)
        path=directory/f'case_{index}_rollouts.png'
        fig.savefig(path,facecolor='white');plt.close(fig);outputs.append(str(path))
    print(json.dumps({'panels':outputs}))


if __name__=='__main__':render()
