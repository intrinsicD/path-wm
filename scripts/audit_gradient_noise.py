"""Separate stochastic sources on the existing training batches, via run.py."""
import json
from pathlib import Path
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from scripts.audit_sample_efficiency import probe
from scripts.gradient_audit_math import gradient_geometry
from scripts.inspect_checkpoint import load, inspection_datasets, sha256
from world_model.data import preprocess_pixels, normalize_actions
from world_model.introspection import state_digest
from world_model.train import write_json


def main():
    torch.set_num_threads(4)
    root=Path('runs/diagnostics/sample_efficiency_2026-09-06')
    out=root/'noise_controls';out.mkdir(exist_ok=False)
    result=dict(step=None, datasets={}, checkpoint_unchanged=True, model_state_unchanged=True)
    manifest=dict(dataset={'name':'noise_controls_both_datasets'},population='First declared 128 TRAIN windows per dataset; same frozen data',
                  precision='bf16, float32 and bf16 with float32 SIGReg',sampling_seed=603072,
                  protocol='docs/sample-efficiency-plan-2026-09-06.md',sources=[],optimizer_updates=0)
    tick=time.monotonic()
    for name in ('pusht','tworoom'):
        source=json.loads((root/name/'manifest.json').read_text());checkpoint=Path(source['checkpoint'])
        training=json.loads(Path(source['source_run_manifest']).read_text());digest=sha256(checkpoint)
        model,stats,step,cfg=load(checkpoint,training,False);model.to('cuda')
        model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        before=state_digest(model);train,_,_=inspection_datasets(training)
        batch=next(iter(DataLoader(train,batch_size=128,sampler=source['train_indices'][:128],num_workers=4)))
        pixels=preprocess_pixels(batch['pixels'].to('cuda'),model.encoder.config.image_size)
        actions=normalize_actions(batch['action'].to('cuda'),stats)
        rows=[];vectors={};baseline=None
        for condition in ('dropout_varies','projections_vary','full_float32','sigreg_float32'):
            grads=[]
            for i in range(4 if condition.endswith('varies') else 1):
                c={**cfg,'precision':'float32'} if condition=='full_float32' else cfg
                seed=630001+(i if condition=='dropout_varies' else 0)
                reg_seed=640001+(i if condition=='projections_vary' else 0)
                row,g=probe(model,pixels,actions,c,seed,branches=condition=='full_float32',reg_seed=reg_seed,fp32_sigreg=condition=='sigreg_float32')
                row.update(condition=condition,replicate=i,projection_seed=reg_seed)
                if baseline is None: baseline=g
                row['gradient_cosine_vs_bf16']=float(torch.dot(g,baseline)/(g.norm()*baseline.norm()))
                row['gradient_relative_difference_vs_bf16']=float((g-baseline).norm()/baseline.norm())
                rows.append(row);grads.append(g)
                with (out/'probes.jsonl').open('a') as f:f.write(json.dumps(dict(dataset=name,**row))+'\n')
            if len(grads)>1: vectors[condition]=gradient_geometry(grads)
        result['datasets'][name]=dict(step=step,geometry=vectors,
            precision={r['condition']:r for r in rows if r['condition'] in ('full_float32','sigreg_float32')},
            baseline={k:v for k,v in rows[0].items() if k not in ('modules',)},
            checkpoint_unchanged=sha256(checkpoint)==digest,model_state_unchanged=state_digest(model)==before)
        manifest['sources'].append(dict(checkpoint=str(checkpoint),checkpoint_sha256=digest,source_manifest=str(root/name/'manifest.json'),base_indices_sha256=source['base_indices_sha256']))
        assert result['datasets'][name]['checkpoint_unchanged'] and result['datasets'][name]['model_state_unchanged']
        print(json.dumps(dict(dataset=name,geometry=vectors,precision={r['condition']:{k:r[k] for k in ('decomposition_relative_error','gradient_cosine_vs_bf16','gradient_relative_difference_vs_bf16')} for r in rows if r['condition'] in ('full_float32','sigreg_float32')})),flush=True)
        del model,pixels,actions,grads,baseline
        torch.cuda.empty_cache()
    result['elapsed_seconds']=time.monotonic()-tick
    fig,axes=plt.subplots(1,2,figsize=(10,3.5),dpi=110)
    for ax,(name,value) in zip(axes,result['datasets'].items()):
        modes=value['geometry'];ax.bar(['Dropout varies','SIGReg directions vary'],[v['sample_covariance_trace'] for v in modes.values()],color=['#397b98','#cc6d42'])
        ax.set(title=f'{name}: fixed 128 training windows',ylabel='Gradient covariance trace (4 replicates)')
    fig.suptitle('Independent stochastic controls; original checkpoints preserved',fontsize=11)
    fig.tight_layout();fig.savefig(out/'gradient_geometry.png');plt.close(fig)
    write_json(out/'manifest.json',manifest);write_json(out/'gradient_audit.json',result)


if __name__=='__main__':main()
