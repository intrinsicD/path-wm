"""Export per-pair logged gradients and discrete matched prediction controls.

These figures preserve every available source metric row in an immutable snapshot.
No smoothing, pooled seed trajectory, gradient-variance estimate or extra model
execution is involved. Publication figures accompany the native HTML dashboard.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import yaml

from scripts.projection_experiment import DEFAULT_BASE, ROOT, sha
from world_model.train import write_json


def plot(base=DEFAULT_BASE, output=None):
    base=Path(base)
    snapshot=[]
    for config in json.loads((base/'planned_configs.json').read_text())['training']:
        cfg=yaml.safe_load((ROOT/config).read_text())
        run=ROOT/cfg['run_dir'];path=run/'metrics.jsonl'
        content=path.read_bytes() if path.exists() else b''
        rows=[json.loads(line) for line in content.splitlines()]
        ds=yaml.safe_load((ROOT/cfg['dataset']).read_text())['name']
        snapshot.append(dict(dataset=ds,seed=cfg['seed'],projections=cfg['sigreg_projections'],
            run=run.name,source=str(path.relative_to(ROOT)),source_sha256=hashlib.sha256(content).hexdigest(),
            config=cfg,rows=rows))
    if output is None: raise ValueError('Choose a new output directory to preserve earlier snapshots')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    write_json(output/'metrics_snapshot.json',snapshot)
    colors={1024:'#0072B2',4096:'#D55E00'};markers={1024:'o',4096:'^'}
    completed=sum(any(r['kind']=='complete' and r['step']==1500 for r in s['rows']) for s in snapshot)
    plotted=[]
    for metric,title,filename in [('grad_norm','Logged gradient norms before clipping','gradient_norms'),
                                  ('prediction_copy','Saved-buffer one-step prediction relative to copying','prediction_copy')]:
        values=[]
        with plt.rc_context({'font.size':10,'axes.spines.top':False,'axes.spines.right':False}):
            fig,axes=plt.subplots(2,3,figsize=(13,7),sharex=True,sharey=True)
            for i,dataset in enumerate(('pusht','tworoom')):
                for j,seed in enumerate((3072,3073,3074)):
                    ax=axes[i,j];counts=[]
                    for arm in (1024,4096):
                        run=next(s for s in snapshot if s['dataset']==dataset and s['seed']==seed and s['projections']==arm)
                        selected=[r for r in run['rows'] if r['kind']==('train' if metric=='grad_norm' else 'validation') and r['step']>0]
                        x=[r['step'] for r in selected]
                        y=[r['grad_norm'] if metric=='grad_norm' else r['pred_mse']/r['identity_mse'] for r in selected]
                        if any(not np.isfinite(v) or v<=0 for v in y):
                            raise ValueError('Logarithmic diagnostic requires positive finite measurements')
                        values.extend(y);counts.append(len(selected))
                        if y:ax.scatter(x,y,s=15 if metric=='grad_norm' else 40,marker=markers[arm],
                            facecolors='none' if arm==1024 else colors[arm],edgecolors=colors[arm],linewidths=.9,alpha=.8)
                        plotted.extend(dict(metric=metric,dataset=dataset,seed=seed,projections=arm,step=step,value=value)
                                       for step,value in zip(x,y))
                    threshold=1.0 if metric=='prediction_copy' else run['config']['grad_clip']
                    ax.axhline(threshold,color='#555555',linewidth=1,linestyle=':')
                    ax.set(title=f"{'PushT' if dataset=='pusht' else 'TwoRoom'} · seed {seed}",xlim=(0,1550),xticks=[250,750,1500],yscale='log')
                    ax.grid(axis='y',alpha=.2)
                    ax.set_title(f"{'PushT' if dataset=='pusht' else 'TwoRoom'} · seed {seed}\n{counts[0]}/{counts[1]} logged points (1024/4096)",fontsize=10)
                    if not any(counts):ax.text(.5,.5,'No recorded points',ha='center',transform=ax.transAxes,color='#666666')
                    if i==1:ax.set_xlabel('Optimizer updates')
                    if j==0:ax.set_ylabel('Global L2 norm (log scale)' if metric=='grad_norm' else 'Prediction / copy MSE\n(log scale)')
            if values:axes[0,0].set_ylim(min(1,min(values))*.72,max(1,max(values))*1.65)
            handles=[Line2D([0],[0],marker=markers[arm],color=colors[arm],markerfacecolor='none' if arm==1024 else colors[arm],linestyle='none',label=f'{arm} projections') for arm in (1024,4096)]
            handles.append(Line2D([0],[0],color='#555555',linestyle=':',label='Clipping threshold = 1' if metric=='grad_norm' else 'Copying ratio = 1'))
            fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.055),ncol=3,frameon=False)
            fig.suptitle(f'{title}\n{completed}/12 runs recorded complete at 1500 updates',fontsize=14)
            note=('Full-objective norms at sampled steps (normally every 25); clipping between samples and gradient variance are not measured.' if metric=='grad_norm' else 'Saved BN buffers; discrete in-training validation every 250 updates. Initialization stays in the source snapshot. Ratios below 1 beat copying.')
            fig.text(.5,.015,note+'\nMatched within seed and dataset; each model uses its own learned latent space. Missing points remain missing.',ha='center',fontsize=9)
            fig.tight_layout(rect=(0,.14,1,.91))
            for extension in ('png','svg'):fig.savefig(output/f'{filename}.{extension}',dpi=180,facecolor='white')
            plt.close(fig)
    write_json(output/'plot_receipt.json',dict(source_snapshot='metrics_snapshot.json',source_sha256=sha(output/'metrics_snapshot.json'),
        plotted_rows=plotted,code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        figures={p.name:sha(p) for p in sorted(output.glob('*')) if p.suffix in ('.png','.svg')}))
    print(output);return output


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',type=Path,default=DEFAULT_BASE);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();plot(args.base,args.output)
