"""Reconcile curriculum phase ledgers; share one wide dataset across native charts."""
from __future__ import annotations
import base64
import hashlib
import json
from pathlib import Path
from .ledger import RunResult,DashboardDataError,read_json,read_jsonl,numeric

def collect_curriculum_results(runs_root):
    results=[]
    for path in sorted(runs_root.rglob('curriculum_manifest.json')):
        manifest=read_json(path);root=path.parent;config=manifest['config']
        train=read_jsonl(root/'training.jsonl') if (root/'training.jsonl').exists() else ()
        validation=read_jsonl(root/'validation.jsonl') if (root/'validation.jsonl').exists() else ()
        result_path=root/'curriculum_result.json'
        result=read_json(result_path) if result_path.exists() else {}
        status=result.get('status','running');step=result.get('step',train[-1]['step'] if train else 0)
        if train and [r['step'] for r in train]!=list(range(1,train[-1]['step']+1)):
            raise DashboardDataError(f'{path}: incomplete or duplicated training update ledger')
        if result:
            if len(train)!=step or result['examples']!=step*config['batch_size']:
                raise DashboardDataError(f'{path}: curriculum update/exposure mismatch')
            indexed={r['step']:r for r in validation}
            for key,update in [('selected',result['selected_step']),('final',step)]:
                if update not in indexed:raise DashboardDataError(f'{path}: missing {key} validation')
                for k,v in result[key].items():
                    if k not in indexed[update] or indexed[update][k]!=v:
                        raise DashboardDataError(f'{path}: {key} {k} differs from validation ledger')
        metrics={**numeric(result)}
        for which in ('selected','final'):
            for k,v in result.get(which,{}).items():
                if isinstance(v,(int,float)) and not isinstance(v,bool):metrics[f'{which}_{k}']=v
                elif isinstance(v,list):
                    metrics.update({f'{which}_{k}_{i}':n for i,n in enumerate(v) if isinstance(n,(int,float))})
        sources=[path,*[root/n for n in ('training.jsonl','validation.jsonl','curriculum_result.json') if (root/n).exists()]]
        results.append(RunResult(root.relative_to(runs_root).as_posix(),'curriculum_training',status,step,metrics,
                       {**config,'dataset_fingerprint':manifest['dataset'],'protocol':'accepted curriculum; supervised and warmup populations distinct'},
                       tuple(p.relative_to(runs_root.parent).as_posix() for p in sources),max(p.stat().st_mtime for p in sources),
                       internals={'training':train,'validation':validation,'result':result}))
    return results,[]

def add_curriculum_views(runs,datasets,charts,tables,cards,chart,table,source_id):
    selected=[r for r in runs if r.kind=='curriculum_training']
    if not selected:return []
    blocks=[{'id':'curriculum_intro','type':'markdown','sourceId':source_id,'body':
        '## Perception curriculum\n\nEncoder/decoder reconstruction warmup and labelled PushT adaptation are separate stages. '
        'The physical selector q is the worst position MAE/8 world units or angle MAE/10 degrees; q≤1 is the numeric readiness target. '
        'Training completion does not establish perception readiness, dynamics, or control. Exact selected/final metrics and phase budgets are in the record inventory.'}]
    # One seed per native curve family keeps legends legible; all seeds remain indexed.
    screen=[r for r in selected if r.context.get('arm') in ('A','B','C')]
    focus=screen if screen else selected
    seed=max(r.context['seed'] for r in focus)
    focus=[r for r in focus if r.context['seed']==seed]
    wide={};fields={}
    for run in focus:
        token=f"{run.context.get('arm','run')}_{run.context['phase']}"
        names=[]
        for row in run.internals['validation']:
            output=wide.setdefault(row['step'],{'step':row['step']})
            for metric in ('loss','image_mse','pose_mse','q','angle_mae_deg','angle_norm_mean'):
                if metric in row:
                    name=f'{token}_{metric}';output[name]=row[metric];names.append(name)
            for i,n in enumerate(row.get('position_mae',[])):
                name=f'{token}_position_{i}';output[name]=n;names.append(name)
        fields[token]=set(names)
    name='curriculum_validation';all_fields=set().union(*fields.values())
    datasets[name]=[{'step':step,**{k:row.get(k) for k in sorted(all_fields)}} for step,row in sorted(wide.items())]
    number=lambda field:{'field':field,'type':'quantitative'}
    supervised=[token for token in fields if token.endswith('supervised')]
    specs=[]
    if supervised:
        specs.extend([
            ('q','Physical validation readiness',[f'{t}_q' for t in supervised],'q; target ≤1'),
            ('angle','Validation block orientation',[f'{t}_angle_mae_deg' for t in supervised],'mean absolute degrees'),
            ('image','Task reconstruction',[f'{t}_image_mse' for t in supervised],'mean squared RGB error in [0,1]')])
        for coord,title in enumerate(('pusher x','pusher y','block x','block y')):
            specs.append((f'position{coord}',f'Validation {title}',[f'{t}_position_{coord}' for t in supervised],'mean absolute world units'))
    for token in fields:
        if token.endswith('warmup'):
            specs.append((token,f'Reconstruction warmup: {token}',[f'{token}_image_mse'],'RGB MSE; this phase has its own image population'))
    for key,title,names,unit in specs:
        names=[f for f in names if f in all_fields]
        if not names:continue
        y={'fields':names,'type':'quantitative','label':unit} if len(names)>1 else number(names[0])
        view=chart(f'curriculum_{key}',f'{title} · seed {seed}',
                   f'{unit}. Fixed validation frames; x counts updates within this phase. A has up to4000 supervised updates; B/C up to2000 after separate2000 warmup updates. Exact values; no pooled latent spaces.',
                   name,'line' if len(wide)>1 else 'bar',number('step'),y)
        if len(names)>1:
            view['legend']={'position':'bottom','sort':'spec'}
            view['palette']={'kind':'categorical','name':'PATH-WM blue-orange'}
        charts.append(view)
    return blocks
