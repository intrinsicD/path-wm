"""Export source-bound paired control curves on a fixed 0–100% scale.

Each line is one seed/arm, without pooling datasets or BN policies. Preserve
an immutable input snapshot because the live comparison is refreshed by run.py.
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

from scripts.paired_summary import summarize_pairs
from scripts.projection_experiment import DEFAULT_BASE, read, sha
from world_model.train import write_json


def plot(base=DEFAULT_BASE, output=None):
    base=Path(base);source=base/'projection_comparison.json';value=read(source)
    if summarize_pairs(value['rows'])!=value['groups']:
        raise ValueError('Paired statistics disagree with source rows')
    rows=value['rows'];expected=value['expected_outcomes']
    if len(rows)+len(value['missing_outcomes'])!=expected:
        raise ValueError('Measured/missing outcome count differs from plan')
    output=Path(output) if output else base/'figures'/f'outcomes{len(rows):02d}'
    output.mkdir(parents=True,exist_ok=False)
    snapshot=output/'plot_source.json';snapshot.write_bytes(source.read_bytes())
    colors={1024:'#0072B2',4096:'#D55E00'}
    styles={3072:'-',3073:'--',3074:':'}
    with plt.rc_context({'font.size':10,'axes.spines.top':False,'axes.spines.right':False}):
        fig,axes=plt.subplots(2,2,figsize=(11,7.5),sharex=True,sharey=True)
        for i,dataset in enumerate(('pusht','tworoom')):
            for j,variant in enumerate(('calibrated','saved')):
                ax=axes[i,j];sample=[r for r in rows if r['dataset']==dataset and r['variant']==variant]
                for seed,style in styles.items():
                    for count,color in colors.items():
                        points=sorted([r for r in sample if r['seed']==seed and r['projections']==count],key=lambda r:r['step'])
                        if points:
                            ax.plot([r['step'] for r in points],[100*r['successes']/r['cases'] for r in points],
                                    color=color,linestyle=style,marker='o',markersize=5,linewidth=1.7,alpha=.85)
                ax.set(title=f"{'PushT' if dataset=='pusht' else 'TwoRoom'} · {variant} BN",ylim=(0,100),xlim=(650,1600),xticks=[750,1500],yticks=[0,20,40,60,80,100])
                ax.grid(axis='y',alpha=.2)
                if not sample:ax.text(.5,.5,'No completed evaluations',ha='center',transform=ax.transAxes,color='#666666')
                ax.text(.02,.95,f'{len(sample)}/12 planned outcomes',ha='left',va='top',transform=ax.transAxes,fontsize=9)
                if i==1:ax.set_xlabel('Training updates (750 = 96k windows; 1500 = 192k)')
                if j==0:ax.set_ylabel('Successful fixed source goals (%)')
        handles=[Line2D([0],[0],color=color,lw=2,label=f'{count} projections') for count,color in colors.items()]
        handles += [Line2D([0],[0],color='#444444',linestyle=style,lw=2,label=f'Seed {seed}') for seed,style in styles.items()]
        fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.065),ncol=5,frameon=False)
        status='Complete control collection' if len(rows)==expected else 'Incomplete control collection'
        fig.suptitle(f'SIGReg projection count: paired learning screen\n{status} · {len(rows)}/{expected} planned outcomes',fontsize=14)
        fig.text(.5,.025,'Calibrated BN is the prespecified primary policy (512 training windows). Fixed source cases; three paired seeds.\nMissing evaluations are not zero successes. No significance threshold or independent-data generalization claim.',ha='center',fontsize=9)
        fig.tight_layout(rect=(0,.13,1,.91))
        for extension in ('png','svg'):
            fig.savefig(output/f'projection_learning_curves.{extension}',dpi=180,facecolor='white')
        plt.close(fig)
    write_json(output/'plot_receipt.json',dict(source_snapshot=str(snapshot),source_sha256=sha(snapshot),
        measured_outcomes=len(rows),expected_outcomes=expected,missing_outcomes=len(value['missing_outcomes']),
        plotted_rows_sha256=hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest(),
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        figures={extension:sha(output/f'projection_learning_curves.{extension}') for extension in ('png','svg')}))
    print(output);return output


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',type=Path,default=DEFAULT_BASE);p.add_argument('--output',type=Path)
    args=p.parse_args();plot(args.base,args.output)
