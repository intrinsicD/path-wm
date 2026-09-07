"""Post-freeze train/test diagnosis; no optimizer updates or selection."""
from pathlib import Path
import json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .training import initial_models,evaluate
from .data import task_frames,file_hash
from world_model.pusht.checkpoints import read_checkpoint,json_atomic
ROOT=Path('runs/curriculum_2026-09-07')

def main():
    output=ROOT/'perception_analysis';output.mkdir(exist_ok=True)
    if (output/'curriculum_analysis.json').exists():raise ValueError('preserve completed analysis')
    torch.set_num_threads(2);train=task_frames('data/pusht_world_model/cchi_v1','train')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    decision=json.loads((ROOT/'seed_4107/screen_decision.json').read_text())
    results={};sources=[ROOT/'seed_4107/screen_decision.json']
    for arm in ('A','B','C'):
        saved=read_checkpoint(ROOT/f'seed_4107/{arm}/supervised/best.pt')
        models=initial_models(4107,saved['models']);models['H'].load_state_dict(saved['models']['H'])
        for model in models.values():model.eval()
        metrics,raw=evaluate(models,train,np.arange(len(train)),device='cpu',return_records=True)
        target=output/f'{arm}_training_errors.npz'
        np.savez_compressed(target,**{k:np.asarray(v) for k,v in raw.items()})
        path=ROOT/f'inspection/{arm}_selected/inspection_summary.json'
        test=json.loads(path.read_text());sources.extend([target,path])
        final=json.loads((ROOT/f'inspection/{arm}_final/inspection_summary.json').read_text())
        results[arm]=dict(train=metrics,validation_selection=decision['selected'][arm],test=test['metrics'],
                         test_group_balanced=test['group_balanced'],test_regions=test['regions'],
                         test_final=final['metrics'],activation=test['activation'],probe=test['linear_probe'])
        print(arm,'train q',metrics['q'],'test q',test['metrics']['q'],flush=True)
    json_atomic(output/'comparison.json',results);sources.append(output/'comparison.json')
    fig,axes=plt.subplots(1,3,figsize=(12,3.4),layout='constrained')
    x=np.arange(3)
    for j,(population,label,color) in enumerate([('train','Training','#659ac4'),('validation_selection','Fixed validation','#2463a6'),('test','Test','#db7923')]):
        axes[0].bar(x+(j-1)*.24,[results[a][population]['q'] for a in 'ABC'],width=.24,label=label,color=color)
    axes[0].axhline(1,color='black',ls='--',lw=1);axes[0].set(xticks=x,xticklabels=['A task-only','B COCO','C task-image'],ylabel='Worst normalized physical error q',title='Perception readiness · lower is better');axes[0].legend(fontsize=7)
    for j,(region,color) in enumerate([('pusher','#2463a6'),('block','#db7923'),('background','#6d7278')]):
        axes[1].bar(x+(j-1)*.24,[results[a]['test_regions'][region]['mse']/results[a]['test_regions'][region]['mean_image_mse'] for a in 'ABC'],width=.24,label=region,color=color)
    axes[1].axhline(1,color='black',ls='--',lw=1);axes[1].set(xticks=x,xticklabels=list('ABC'),ylabel='Reconstruction MSE / train-mean MSE',title='Object reconstruction beats the mean');axes[1].legend(fontsize=7)
    for j,(scale,color) in enumerate([('fine','#2463a6'),('coarse','#db7923')]):
        axes[2].bar(x+(j-.5)*.3,[results[a]['activation'][scale]['heldout_mean_std_across_frames'] for a in 'ABC'],width=.3,label=scale,color=color)
    axes[2].set(xticks=x,xticklabels=list('ABC'),ylabel='Mean token/channel SD across frames',title='Feature variation · descriptive scale');axes[2].legend(fontsize=7)
    fig.suptitle('Frozen seed4107 screen · selected checkpoints · no pretraining policy adopted',fontsize=12)
    panel=output/'perception_comparison.png';fig.savefig(panel,dpi=130);fig.savefig(output/'perception_comparison.svg');plt.close(fig);sources.append(panel)
    lines=['## Perception outcome','',
        'All three arms completed their4000-update budget. Neither warmup improved the validation physical errors; all numeric readiness gates failed. No PushT U/P expansion or extra confirmation seeds were launched.',
        '', '| Arm | Train q | Validation q | Test q | Test angle MAE ° | Test RGB MSE |',
        '|---|---:|---:|---:|---:|---:|']
    for a,r in results.items():lines.append(f"| {a} | {r['train']['q']:.3f} | {r['validation_selection']['q']:.3f} | {r['test']['q']:.3f} | {r['test']['angle_mae_deg']:.2f} | {r['test']['image_mse']:.6g} |")
    lines.extend(['','q≤1 is the prospective numeric target. Validation selection uses2048fixed frames; test uses every held-out frame. Group-balanced errors, p95/max tails and selected/final controls remain in raw summaries. The small ridge probes are diagnostics, not proof of a representation bottleneck. These are one-training-seed development results; no independent-seed effect is established.'])
    summary=dict(status='completed',purpose='Frozen perception train/test and reconstruction diagnosis',
        metrics={a+'_train_q':r['train']['q'] for a,r in results.items()},
        sources={p.relative_to(ROOT).as_posix():file_hash(p) for p in sources},
        panels=[dict(file=str(panel),embed=True,title='Perception comparison',caption='Readiness, reconstruction against train-only mean images, and across-frame feature variation. No test-based selection.')],
        narrative='\n'.join(lines))
    json_atomic(output/'curriculum_analysis.json',summary)
if __name__=='__main__':main()
