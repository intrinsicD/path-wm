"""Reconcile completed encoder experiments and export comparable scientific figures."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from world_model.curriculum.data import file_hash,digest
from world_model.curriculum.encoder_factorial import ARMS
from world_model.curriculum.pose_accessibility import analyze
from world_model.pusht.checkpoints import json_atomic

ROOT=Path('runs/encoder_study_2026-09-08');OUT=ROOT/'evaluation'
COLORS=['#245a90','#ac651f','#555555','#83949c']


def read(p):return json.loads(Path(p).read_text())
def lines(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]


def integrity():
    checks={};counts={'updates':0,'presentations':0,'completed_runs':0}
    paths=[]
    for parent in ('diagnostic','reference','factorial','probes'):
        for p in sorted((ROOT/parent).rglob('curriculum_result.json')):
            r=read(p);m=read(p.parent/'curriculum_manifest.json');cfg=m['config'];train=lines(p.parent/'training.jsonl');val=lines(p.parent/'validation.jsonl')
            if r['status']!='completed' or len(train)!=cfg['updates'] or r['step']!=cfg['updates']:raise AssertionError(f'incomplete {p}')
            if [t['step'] for t in train]!=list(range(1,cfg['updates']+1)):raise AssertionError(f'update gaps {p}')
            if r['examples']!=cfg['updates']*cfg['batch_size']:raise AssertionError(f'exposures {p}')
            if parent=='probes':key=lambda row:(row['loss'],row['step'])
            elif parent=='diagnostic':key=lambda row:(row['q'],row['pose_mse'],row['step'])
            else:key=lambda row:(row['q'],row.get('image_mse',row['pose_mse']),row['step'])
            selected=min(val,key=key)
            if selected['step']!=r['selected_step']:raise AssertionError(f'wrong selection {p}')
            for k,v in r['selected'].items():
                if selected[k]!=v:raise AssertionError(f'selected metric mismatch {p}:{k}')
            for k,v in r['final'].items():
                if val[-1][k]!=v:raise AssertionError(f'final metric mismatch {p}:{k}')
            evaldata=read(p.parent/'evaluation.json');metrics=evaldata.get('metrics',evaldata)
            for split in ('train','validation','test'):
                raw=dict(np.load(p.parent/f'{split}_errors.npz'));metric=metrics[split]
                if 'position_abs_error' in raw:
                    pos=raw['position_abs_error'].mean(0);angle=raw['angle_abs_error_deg'].mean();q=max(*list(pos/8),angle/10)
                    np.testing.assert_allclose(pos,metric['position_mae'],rtol=0,atol=1e-10)
                    np.testing.assert_allclose([angle,q],[metric['angle_mae_deg'],metric['q']],rtol=0,atol=1e-10)
                if 'image_mse' in raw:np.testing.assert_allclose(raw['image_mse'].mean(),metric['image_mse'],rtol=0,atol=1e-12)
                if 'iou' in raw:np.testing.assert_allclose(raw['iou'][raw['has_valid'].astype(bool)].mean(),metric['iou'],rtol=0,atol=1e-12)
            checks[str(p.parent.relative_to(ROOT))]={'selection_and_raw_metrics':'passed','updates':len(train),'result_sha256':file_hash(p)}
            counts['completed_runs']+=1;counts['updates']+=len(train);counts['presentations']+=r['examples'];paths.append(p)
    paired={}
    for seed in (7107,7108,7109):
        roots=[ROOT/'factorial'/f'seed_{seed}'/a for a in ARMS]
        if not all((p/'curriculum_result.json').exists() for p in roots):continue
        streams=[[r['sample_indices_sha256'] for r in lines(p/'training.jsonl')] for p in roots]
        manifests=[read(p/'curriculum_manifest.json') for p in roots]
        if any(s!=streams[0] for s in streams[1:]):raise AssertionError('factorial draws unpaired')
        if any(m['validation_indices']!=manifests[0]['validation_indices'] for m in manifests[1:]):raise AssertionError('factorial selection population unpaired')
        if len({m['initial_h_fingerprint'] for m in manifests})!=1:raise AssertionError('factorial H initialization changed')
        paired[str(seed)]={'draws':'identical','validation':'identical','head_initialization':'identical','updates':len(streams[0])}
    probes=[p.parent for p in paths if '/probes/' in str(p)]
    if probes:
        base=[r['sample_indices_sha256'] for r in lines(probes[0]/'training.jsonl')]
        if any([r['sample_indices_sha256'] for r in lines(p/'training.jsonl')]!=base for p in probes[1:]):raise AssertionError('probe draws unpaired')
        if len({read(p/'curriculum_manifest.json')['initial'] for p in probes})!=1:raise AssertionError('probe D/M initialization changed')
    receipt=dict(status='passed',checks=checks,counts=counts,paired_factorial=paired,paired_probes=len(probes))
    json_atomic(OUT/'integrity.json',receipt);return receipt


def reference_figures():
    entries=[('coupled 6k',ROOT/'diagnostic/coupled'),('independent',ROOT/'diagnostic/independent'),
             ('custom + fresh D/H',ROOT/'reference/custom'),('DINO + adapter',ROOT/'reference/dino'),('DINO native scaled',ROOT/'reference/native_scaled')]
    entries=[(n,p) for n,p in entries if (p/'evaluation.json').exists()];summary={}
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    x=np.arange(len(entries))
    angle=[];q=[]
    for name,p in entries:
        e=read(p/'evaluation.json');r=read(p/'curriculum_result.json');summary[name]=dict(selected_step=r['selected_step'],validation=r['selected'],test=e['test'])
        angle.append(e['test']['angle_mae_deg']);q.append(e['test']['q'])
    for ax,values,label,target in zip(axes,[angle,q],['Orientation MAE (degrees)','Worst normalized pose MAE (q)'],[10,1]):
        ax.barh(x,values,color='#245a90',edgecolor='#163851');ax.set_yticks(x,[n for n,_ in entries],fontsize=9)
        ax.axvline(target,color='#555555',ls='--',lw=1);ax.invert_yaxis();ax.set_xlabel(label);ax.set_xlim(0,max(values)*1.2)
        for y,v in zip(x,values):ax.text(v,y,f' {v:.2f}',va='center',fontsize=9)
    fig.suptitle('Frozen representation and readout diagnostics\nValidation-selected checkpoints · one seed · 2,506 reused test frames',fontsize=12)
    fig.savefig(OUT/'reference_errors.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for i,(name,p) in enumerate(entries[:2]):
        rows=lines(p/'validation.jsonl')
        for ax,metric in zip(axes,['q','angle_mae_deg']):
            ax.plot([r['step'] for r in rows],[r[metric] for r in rows],color=COLORS[i],ls=['-','--'][i],label=name)
    for ax,target,label in zip(axes,[1,10],['Worst normalized pose MAE (q)','Orientation MAE (degrees)']):
        ax.axhline(target,color='#555555',ls=':',lw=1);ax.set(xlabel='Readout optimizer updates',ylabel=label,ylim=(0,None));ax.legend(fontsize=9)
    fig.suptitle('Frozen-E head training budget\nSame encoder and frame draws; head packages differ',fontsize=12)
    fig.savefig(OUT/'head_budget.png',dpi=160);plt.close(fig)
    json_atomic(OUT/'reference_summary.json',summary)


def paired_bootstrap(a,b,seed=8207):
    # Conditional on these trained models, resample whole configuration groups.
    if not np.array_equal(a['groups'],b['groups']):raise AssertionError('unpaired group records')
    groups=np.unique(a['groups']);rng=np.random.default_rng(seed);draws=[]
    for i in range(2000):
        chosen=rng.choice(groups,len(groups),replace=True);ids=np.concatenate([np.flatnonzero(a['groups']==g) for g in chosen])
        def q(r):return max(*list(r['position_abs_error'][ids].mean(0)/8),r['angle_abs_error_deg'][ids].mean()/10)
        draws.append(q(a)-q(b))
    return np.percentile(draws,[2.5,50,97.5]).tolist(),np.array(draws)


def factorial_figures():
    rows=[];contrasts=[];wall=[];interactions=[]
    for seed in (7107,7108,7109):
        roots={a:ROOT/'factorial'/f'seed_{seed}'/a for a in ARMS}
        if not all((p/'evaluation.json').exists() for p in roots.values()):continue
        selected={}
        for arm,p in roots.items():
            r=read(p/'curriculum_result.json');e=read(p/'evaluation.json')['metrics'];selected[arm]=r
            rows.append(dict(seed=seed,arm=arm,selected_step=r['selected_step'],validation_q=r['selected']['q'],test_q=e['test']['q'],
                             test_angle=e['test']['angle_mae_deg'],test_positions=e['test']['position_mae'],
                             seconds=r['elapsed_seconds'],peak_cuda_bytes=r['peak_cuda_bytes'],gate=r['selected']['q']<=1))
        for name,a,b in [('depth_on','deeper','reference'),('depth_off','deeper_no_exchange','no_exchange'),
                         ('exchange_shallow','reference','no_exchange'),('exchange_deep','deeper','deeper_no_exchange')]:
            raw_a=dict(np.load(roots[a]/'test_errors.npz'));raw_b=dict(np.load(roots[b]/'test_errors.npz'))
            interval,draws=paired_bootstrap(raw_a,raw_b,8207+seed)
            np.save(OUT/f'{seed}_{name}_group_bootstrap.npy',draws)
            av,bv=selected[a]['selected']['q'],selected[b]['selected']['q']
            q=lambda x:max(*list(x['position_abs_error'].mean(0)/8),x['angle_abs_error_deg'].mean()/10)
            contrasts.append(dict(seed=seed,contrast=name,candidate=a,comparator=b,validation_delta_q=av-bv,
                                  validation_relative_change=av/bv-1,test_delta_q=q(raw_a)-q(raw_b),conditional_group_95=interval))
        seed_contrasts={r['contrast']:r for r in contrasts if r['seed']==seed}
        interactions.append(dict(seed=seed,
            validation_q_interaction=seed_contrasts['depth_on']['validation_delta_q']-seed_contrasts['depth_off']['validation_delta_q'],
            test_q_interaction=seed_contrasts['depth_on']['test_delta_q']-seed_contrasts['depth_off']['test_delta_q'],
            definition='(deeper-on minus shallow-on) minus (deeper-off minus shallow-off); negative means depth helps more with exchange'))
        for deep,shallow in [('deeper','reference'),('deeper_no_exchange','no_exchange')]:
            bound=selected[shallow]['elapsed_seconds'];eligible=[r for r in lines(roots[deep]/'validation.jsonl') if r['elapsed_seconds']<=bound]
            if eligible:
                r=eligible[-1];wall.append(dict(seed=seed,deep=deep,shallow=shallow,deep_step=r['step'],
                    wall_seconds_bound=bound,deep_validation_q=r['q'],shallow_final_q=selected[shallow]['final']['q'],
                    limitation='validation checkpoint metrics at matched elapsed cap; not a matched-time test evaluation'))
    if not rows:return
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    labels=list(ARMS);seeds=sorted({r['seed'] for r in rows})
    for i,seed in enumerate(seeds):
        seedrows={r['arm']:r for r in rows if r['seed']==seed}
        for ax,key in zip(axes,['validation_q','test_q']):
            ax.plot(np.arange(4),[seedrows[a][key] for a in labels],color=COLORS[i],ls=['-','--',':'][i],marker=['o','s','^'][i],label=f'seed {seed}')
    for ax,title in zip(axes,['Selected validation q','Held-out test q']):
        ax.set(xticks=np.arange(4),xticklabels=['shallow\nexchange','deeper\nexchange','shallow\nno exchange','deeper\nno exchange'],ylabel='Worst normalized pose MAE (q)',title=title,ylim=(0,None))
        ax.axhline(1,color='#555555',ls='--',lw=1);ax.grid(axis='y',alpha=.2)
    axes[0].legend(fontsize=8);fig.suptitle('Depth × cross-scale exchange\nOriginal RGB+pose recipe · paired training seeds · reused configuration holdouts',fontsize=12)
    fig.savefig(OUT/'factorial_q.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4.5),layout='constrained');names=['depth_on','depth_off','exchange_shallow','exchange_deep']
    for i,seed in enumerate(seeds):
        vals=[next(r['validation_relative_change']*100 for r in contrasts if r['seed']==seed and r['contrast']==n) for n in names]
        ax.scatter(vals,np.arange(4)+(i-1)*.15,color=COLORS[i],marker=['o','s','^'][i],label=f'seed {seed}',s=45)
    ax.set(yticks=np.arange(4),yticklabels=['Add depth, exchange on','Add depth, exchange off','Enable exchange, shallow','Enable exchange, deeper'],
           xlabel='Validation q change (%) · negative favors the change',title='Paired intervention effects')
    ax.axvline(0,color='#555555',lw=1);ax.axvline(-10,color='#777777',ls=':',lw=1);ax.legend(fontsize=8)
    fig.savefig(OUT/'factorial_effects.png',dpi=160);plt.close(fig)
    signals={name:all(r['validation_relative_change']<=-.1 for r in contrasts if r['contrast']==name) and len(seeds)==3 for name in names}
    json_atomic(OUT/'factorial_summary.json',dict(rows=rows,contrasts=contrasts,interactions=interactions,wall_matched=wall,candidate_signals=signals,
        inference_limit='Three paired seeds; reused holdouts. Group intervals condition on trained models. No equivalence claim; no planning inference from perception-only results.'))


def probe_figures():
    paths=sorted((ROOT/'probes').glob('*/evaluation.json'))
    if not paths:return
    rows=[]
    for p in paths:
        r=read(p);rows.append(dict(source=p.parent.name,**r['metrics']['test'],baselines=r['baselines'],strata=r['strata']['test']))
    order=sorted(rows,key=lambda r:(r['source'] not in ('custom','warmup','dino'),r['source']));y=np.arange(len(rows))
    fig,axes=plt.subplots(1,2,figsize=(11,max(4,len(rows)*.3)),layout='constrained')
    for ax,key,title in zip(axes,['image_mse','iou'],['RGB reconstruction MSE ↓','Annotated foreground IoU ↑']):
        values=[r[key] for r in order];ax.barh(y,values,color='#245a90',edgecolor='#163851');ax.set(yticks=y,yticklabels=[r['source'] for r in order],xlabel=title);ax.invert_yaxis()
        if key=='iou':
            ax.set_xlim(0,1)
            ax.axvline(order[0]['baselines']['full']['iou'],color='#555555',ls='--',lw=1,label='always foreground')
            ax.axvline(order[0]['baselines']['training_mean']['iou'],color='#777777',ls=':',lw=1,label='training mean mask')
            ax.legend(fontsize=8,loc='lower right')
    fig.suptitle('Fresh readouts on frozen representations\nSame COCO views, initialization and draws · 512 test images · crowd ignored',fontsize=12)
    fig.savefig(OUT/'rgb_mask_audit.png',dpi=160);plt.close(fig)
    json_atomic(OUT/'probe_summary.json',dict(rows=rows))
    for name in ('custom','warmup','dino'):
        p=ROOT/'probes'/name/'panels.npz'
        if not p.exists():continue
        data=dict(np.load(p));fig,axes=plt.subplots(4,6,figsize=(11,7),layout='constrained')
        for c in range(6):
            for row,key in enumerate(('rgb','reconstruction','mask','probability')):
                im=data[key][c];axes[row,c].imshow(im.transpose(1,2,0) if im.shape[0]==3 else im[0],cmap='gray',vmin=0,vmax=1)
                axes[row,c].set_xticks([]);axes[row,c].set_yticks([])
            axes[0,c].set_title(f'COCO row {data["rows"][c]}',fontsize=8)
        for row,title in enumerate(['RGB64','Reconstruction','Annotated union','Mask probability']):axes[row,0].set_ylabel(title,fontsize=9)
        fig.suptitle(f'{name} · frozen representation, fresh output decoders\nSix fixed test views · shared mask probability scale [0,1]',fontsize=12)
        fig.savefig(OUT/f'{name}_output_panels.png',dpi=140);plt.close(fig)


def main():
    p=argparse.ArgumentParser();p.add_argument('--publish',action='store_true');args=p.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    audit=integrity();reference_figures();factorial_figures();probe_figures()
    if args.publish:
        sources=[p for p in OUT.iterdir() if p.suffix in ('.json','.png','.npy') and p.name!='curriculum_analysis.json']
        figures=[('reference_errors.png','Frozen representation errors'),('head_budget.png','Frozen head budget'),('factorial_q.png','Factorial pose comparison'),
                 ('factorial_effects.png','Paired intervention effects'),('rgb_mask_audit.png','RGB and mask audit')]
        panels=[dict(file=str(OUT/f),title=t,caption='Measured raw results; full protocol and exact values retained.',embed=True) for f,t in figures if (OUT/f).exists()]
        analyze(ROOT,OUT,'Reconciled encoder study',audit['counts'],
            '## Encoder study results\n\nRaw per-frame metrics, selected validation checkpoints and paired draws reconcile. All results use declared budgets and previously inspected grouped holdouts. Training completion is distinct from readiness and planning quality. The independent-head change is a package; DINO differs in pretraining and adapter capacity. The raw native head was unstable and its fixed-scaling correction is reported separately.',sources,panels)
    print(json.dumps(audit['counts']),flush=True)

if __name__=='__main__':main()
