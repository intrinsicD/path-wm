"""Internal COCO test reconstruction before/after task adaptation; no task labels."""
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .data import coco_frames,file_hash
from .training import initial_models,evaluate
from .inspection import mean_image,mean_image_errors
from world_model.pusht.checkpoints import read_checkpoint,json_atomic
ROOT=Path('runs/curriculum_2026-09-07')

def main():
    output=ROOT/'generic_reconstruction';output.mkdir(exist_ok=True)
    if (output/'curriculum_analysis.json').exists():raise ValueError('preserve completed evaluation')
    torch.set_num_threads(2)
    train=coco_frames('data/curriculum/coco_v1','train');test=coco_frames('data/curriculum/coco_v1','test')
    baseline=mean_image(train);base_error=mean_image_errors(baseline,test)
    selected=np.random.default_rng(92504).choice(len(test),6,replace=False)
    rows={};decoded={};sources=[]
    for name,checkpoint in [
        ('COCO_warmup',ROOT/'seed_4107/B/warmup/last.pt'),
        ('COCO_then_task_selected',ROOT/'seed_4107/B/supervised/best.pt'),
        ('COCO_then_task_final',ROOT/'seed_4107/B/supervised/last.pt')]:
        saved=read_checkpoint(checkpoint);models=initial_models(4107,saved['models'])
        for m in models.values():m.eval()
        metrics,raw=evaluate(models,test,np.arange(len(test)),device='cpu',labelled=False,return_records=True)
        metrics['mean_image_mse']=float(np.mean(base_error));metrics['ratio_to_train_mean']=metrics['image_mse']/metrics['mean_image_mse']
        rows[name]=dict(checkpoint=str(checkpoint),sha256=file_hash(checkpoint),metrics=metrics)
        file=output/(name+'.npz');np.savez_compressed(file,indices=raw['indices'],image_mse=raw['image_mse'],mean_image_mse=base_error);sources.append(file)
        with torch.no_grad():
            x,_=test.batch(selected,'cpu',False);decoded[name]=models['D'](models['E'](x)).permute(0,2,3,1).numpy()
        print(name,metrics,flush=True)
    fig,axes=plt.subplots(3,6,figsize=(10,5.2),layout='constrained')
    original=x.permute(0,2,3,1).numpy()
    for r,(label,images) in enumerate([('COCO test input',original),('After COCO warmup',decoded['COCO_warmup']),('After task adaptation',decoded['COCO_then_task_selected'])]):
        for j in range(6):
            axes[r,j].imshow(images[j]);axes[r,j].set_xticks([]);axes[r,j].set_yticks([])
        axes[r,0].set_ylabel(label,fontsize=8)
    fig.suptitle('Generic reconstruction retention · six fixed internal COCO test images\nNo pose head or task labels used for this assessment',fontsize=12)
    panel=output/'coco_retention.png';fig.savefig(panel,dpi=120);fig.savefig(output/'coco_retention.svg');plt.close(fig)
    json_atomic(output/'metrics.json',rows);sources.extend([panel,output/'metrics.json'])
    np.savez_compressed(output/'panel_images.npz',indices=selected,inputs=original,**decoded);sources.append(output/'panel_images.npz')
    narrative='## Generic reconstruction retention\n\nAll4146internal COCO test images; duplicate-group split, RGB64 center crops. '
    narrative+='This is an internal reconstruction assessment, not the official COCO benchmark. '
    for name,r in rows.items():narrative+=f"{name}: RGB MSE{r['metrics']['image_mse']:.6g}, train-mean ratio{r['metrics']['ratio_to_train_mean']:.3f}. "
    narrative+='The immutable COCO warmup checkpoint remains available independently of task adaptation; the standalone image grid shows retention on fixed examples.'
    json_atomic(output/'curriculum_analysis.json',dict(status='completed',purpose='Generic E/D reconstruction retention after task adaptation',
        metrics={name+'_image_mse':r['metrics']['image_mse'] for name,r in rows.items()},
        sources={p.relative_to(ROOT).as_posix():file_hash(p) for p in sources},narrative=narrative,
        panels=[dict(file=str(panel),title='COCO reconstruction retention',caption='Fixed internal test images; same crops and no task labels.',embed=False)]))
if __name__=='__main__':main()
