"""Source-backed views for the separate CCHI E/U/P task; never relabel LeWM."""
from __future__ import annotations

import base64
import html
import json
from pathlib import Path

from .ledger import DashboardDataError, RunResult, numeric, read_json, read_jsonl
from .paddle import _equal, _flat, _list, _sample


POSITIONS = ('pusher_x', 'pusher_y', 'block_x', 'block_y')
MOTION = (*POSITIONS, 'block_angle')


def _reconcile(report, directory):
    source = directory/'control_records.json'; controls = _list(source)
    if len({(r['case_id'],r['controller']) for r in controls}) != len(controls):
        raise DashboardDataError(f'{source}: duplicate controller/case identity')
    for controller, summary in report['control'].items():
        rows = [r for r in controls if r['controller'] == controller]
        _equal(len(rows),summary['count'],'controller count',source)
        successes = sum(r['success'] for r in rows)
        _equal(successes,summary['successes'],'controller successes',source)
        if rows: _equal(successes/len(rows),summary['success_rate'],'final success fraction',source)
        if 'any_time_successes' in summary:
            _equal(sum(r['any_time_success'] for r in rows),summary['any_time_successes'],'any-time successes',source)
        if 'failure_case_ids' in summary:
            if sorted(summary['failure_case_ids']) != sorted(r['case_id'] for r in rows if not r['success']):
                raise DashboardDataError(f'{source}: failure identities disagree')
    prediction_source = directory/'prediction_records.json'; predictions = _list(prediction_source)
    for method, horizons in report['prediction'].items():
        for horizon, summary in horizons.items():
            rows = [r for r in predictions if r['method'] == method and r['horizon'] == int(horizon)]
            _equal(len(rows),summary['count'],'prediction count',prediction_source)
            if not rows: continue
            for raw, field in [('h_position_abs_error','h_position_mae'),('r_position_abs_error','r_position_mae'),
                               ('h_angle_abs_error_deg','h_angle_mae_deg'),('r_angle_abs_error_deg','r_angle_mae_deg')]:
                values = [r[raw] for r in rows]
                average = [sum(v[i] for v in values)/len(values) for i in range(len(values[0]))] if isinstance(values[0],list) else sum(values)/len(values)
                _equal(average,summary[field],field,prediction_source)
            motions = [r['r_motion_abs_error'] for r in rows if r.get('r_motion_abs_error') is not None]
            _equal(len(motions),summary['motion_count'],'post-warm-up motion count',prediction_source)
            if motions:
                _equal([sum(v[i] for v in motions)/len(motions) for i in range(5)],summary['r_motion_mae'],'physical motion MAE',prediction_source)
            for field in ('latent_error','image_mse'):
                values = [r[field] for r in rows if r.get(field) is not None]
                if values: _equal(sum(values)/len(values),summary[field],field,prediction_source)
    return [source,prediction_source]


def collect_pusht_results(runs_root):
    results, notices = [], []
    def add(path,kind,status,metrics,context,sources,step=None,internals=None):
        results.append(RunResult(path.relative_to(runs_root).as_posix(),kind,status,step,metrics,context,
            tuple(p.relative_to(runs_root.parent).as_posix() for p in sources),
            max(p.stat().st_mtime for p in sources),internals=internals or {}))
    for path in sorted(runs_root.rglob('pusht_manifest.json')):
        manifest = read_json(path); directory = path.parent; sources = [path]
        rows = {}
        for name in ('training','validation'):
            ledger = directory/f'{name}.jsonl'
            rows[name] = read_jsonl(ledger) if ledger.exists() else ()
            if ledger.exists(): sources.append(ledger)
        result_path = directory/'pusht_result.json'
        result = read_json(result_path) if result_path.exists() else {}
        if result_path.exists(): sources.append(result_path)
        for name in ('status.json','failure.json'):
            if (directory/name).exists(): sources.append(directory/name)
        status = result.get('status','failed' if (directory/'failure.json').exists() else 'incomplete')
        selected = result.get('selected_update')
        step = result.get('global_update',rows['training'][-1]['step'] if rows['training'] else None)
        if selected is not None:
            matched = [r for r in rows['validation'] if r['step'] == selected]
            if not matched: raise DashboardDataError(f'{result_path}: selected validation checkpoint missing')
            for key,value in _flat(result.get('metrics',{})).items():
                _equal(value,_flat(matched[-1]).get(key),f'selected validation {key}',result_path)
        if status == 'completed' and rows['training'] and step != rows['training'][-1]['step']:
            raise DashboardDataError(f'{result_path}: completed update differs from raw training')
        metrics = {**numeric(result),**_flat(result.get('metrics',{}),'selected_validation')}
        for name in ('training','validation'):
            if rows[name]: metrics.update(_flat(rows[name][-1],f'latest_{name}'))
        context = {k:manifest.get(k) for k in ('stage','horizon','config','dependencies','dataset_fingerprint','versions','normalization')}
        context.update(task='CCHI PushT E/U/P; primitive10Hz absolute actions',selected_update=selected,
            checkpoint=result.get('checkpoint'),quality_gate=result.get('quality_gate'),smoke=result.get('smoke',manifest.get('config',{}).get('smoke',False)))
        add(directory,'pusht_training',status,metrics,context,sources,step,rows)
        if context['smoke']: notices.append(f'{directory.relative_to(runs_root)}: PushT SMOKE is execution evidence only')
        if status != 'completed': notices.append(f'{directory.relative_to(runs_root)}: PushT stage {status}; quality gates remain separate')
    for path in sorted(runs_root.rglob('metrics.json')):
        report = read_json(path)
        if report.get('schema_version') != 'pusht-eup-evaluation-v1': continue
        sources = [path]; directory = path.parent
        raw_ready = all((directory/name).exists() for name in ('control_records.json','prediction_records.json'))
        if raw_ready: sources += _reconcile(report,directory)
        elif report['status'] == 'completed': raise DashboardDataError(f'{path}: completed PushT raw records missing')
        for name in ('protocol.json','case_manifest.json','case_eligibility_records.json'):
            if (directory/name).exists(): sources.append(directory/name)
        panels=[]
        for visual in report.get('visuals',[]):
            panel=Path(visual['png'])
            if not panel.is_absolute(): panel=runs_root.parent/panel
            if not panel.exists(): raise DashboardDataError(f'{path}: PushT image missing {panel}')
            sources.append(panel); panels.append({**visual,'png':str(panel)})
        # Protocol sections retain their exact values without one oversized cell.
        context={f'protocol.{key}':value for key,value in report.get('protocol',{}).items()}
        context.update(case_selection=report.get('case_selection'),
            smoke=report.get('smoke'),raw_status=report.get('raw_status'),error=report.get('error'),
            goal_task='Block XY <20 world units AND circular angle <pi/9; not95%coverage',
            oracle_budget='Replay is5-interval reachability only, separate from fixed-budget controllers')
        add(directory,'pusht_evaluation' if raw_ready else 'pusht_reporting_failure',report['status'],
            _flat({k:report.get(k,{}) for k in ('prediction','control')}),context,sources,
            internals={'report':report,'panels':panels})
        if report['status'] != 'completed': notices.append(f'{directory.relative_to(runs_root)}: PushT evaluation/report {report["status"]}: {report.get("error","")}')
    return results, notices


def add_pusht_views(runs,datasets,charts,tables,cards,chart,table,source_id):
    training=[r for r in runs if r.kind=='pusht_training']; evaluations=[r for r in runs if r.kind=='pusht_evaluation']
    if not training and not evaluations: return []
    number=lambda field:{'field':field,'type':'quantitative'}
    category=lambda field:{'field':field,'type':'nominal'}
    blocks=[{'id':'pusht_intro','type':'markdown','sourceId':source_id,
        'body':'## CCHI PushT E/U/P\n\nSeparate new-model experiment: absoluteXY actions at10Hz. '
        'Training, prediction, and block-goal control are distinct evidence. Motion labels are backward displacements, not instantaneous velocities. '
        'Final controller success uses the complete fixed budget; any-time success is secondary. The5-action replay oracle only checks reachability. '
        'SMOKE results and the retained LeWM PushT experiments do not establish this model’s control quality.'}]
    latest={}
    for run in training:
        key=run.context['stage']
        if key=='predictor':key+=f'_{run.context["horizon"]}'
        latest[key]=run
    for key,run in latest.items():
        name=f'pusht_training_{key}'
        values=sorted([dict(step=row['step'],value=row['loss'])
                       for row in _sample(run.internals['training']) if 'loss' in row],key=lambda r:r['step'])
        if values:
            datasets[name]=values
            charts.append(chart(name,f'PushT {key}: training objective · {run.label}',
                'Task-specific recorded objective; no LeWM SIGReg meaning. At most50 raw updates; ordered categorical step spacing. Training and validation grids are separate.',
                name,'line' if len(values)>1 else 'bar',number('step'),number('value')))
        # Each validation checkpoint is one wide row, reused by all metric charts.
        # This preserves its exact step grid and keeps the portable dataset budget bounded.
        name=f'pusht_validation_objective_{key}';values=[]
        for row in _sample(run.internals['validation']):
            value={'step':row['step']}
            for field in ('loss','copy_loss'):
                if field in row:value[field]=row[field]
            for field,prefix,coordinates in [('position_mae','position',POSITIONS),('motion_mae','motion',MOTION)]:
                vector=row.get(field)
                if vector is None:continue
                if isinstance(vector[0],list):vector=vector[-1]
                value.update({f'{prefix}_{coordinate}':vector[i] for i,coordinate in enumerate(coordinates)})
            angle=row.get('angle_mae_deg')
            if angle is not None:value['angle_mae_deg']=angle[-1] if isinstance(angle,list) else angle
            values.append(value)
        if not values:continue
        datasets[name]=sorted(values,key=lambda r:r['step'])
        available=set().union(*(row.keys() for row in values))
        kind='line' if len(values)>1 else 'bar'
        groups=[('objective','validation objective',['loss','copy_loss'],'recorded objective'),
                ('position_mae','validation position MAE',[f'position_{c}' for c in POSITIONS],'world units'),
                ('angle_mae_deg','validation block-angle MAE',['angle_mae_deg'],'degrees'),
                ('motion_mae','validation displacement MAE',[f'motion_{c}' for c in POSITIONS],'world units per0.1second'),
                ('angular_motion_mae','validation angular-displacement MAE',['motion_block_angle'],'radians per0.1second')]
        for suffix,label,fields,units in groups:
            fields=[f for f in fields if f in available]
            if not fields:continue
            identifier=name if suffix=='objective' else f'pusht_validation_{suffix}_{key}'
            y={'fields':fields,'type':'quantitative','label':units} if len(fields)>1 else number(fields[0])
            charts.append(chart(identifier,f'PushT {key}: {label} · {run.label}',
                f'Exact validation checkpoints in {units}; predictor physical metrics show the final trained horizon. Ordered categorical step spacing; no interpolation.',
                name,kind,number('step'),y))
            if len(fields)>1:
                charts[-1]['palette']={'kind':'categorical','name':'PATH-WM blue-orange'}
                charts[-1]['legend']={'position':'bottom','sort':'spec'}
                charts[-1]['surface']['interactiveLegend']=True
            if kind=='line':charts[-1]['settings']['showPoints']='always'
    datasets['pusht_control_detail']=[];datasets['pusht_prediction_detail']=[]
    for run in evaluations:
        report=run.internals['report']
        for controller,summary in report['control'].items():
            datasets['pusht_control_detail'].append(dict(run=run.label,controller=controller,
                budget_scope='5-action oracle reachability' if controller=='replay' else 'fixed controller budget',
                **{k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in summary.items() if k!='failure_case_ids'}))
        for method,horizons in report['prediction'].items():
            for horizon,summary in horizons.items():
                datasets['pusht_prediction_detail'].append(dict(run=run.label,method=method,horizon=int(horizon),
                    **{k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in summary.items()}))
    if evaluations:
        run=evaluations[-1];report=run.internals['report']
        control=[row for row in datasets['pusht_control_detail'] if row['run']==run.label and row['controller']!='replay']
        if control:
            name='pusht_final_control';datasets[name]=control
            charts.append(chart(name,f'PushT final block-goal success · {run.label}',
                'Final success after the same fixed primitive-action budget. Excludes the separately reported5-action replay oracle; no95%coverage claim.',
                name,'bar',category('controller'),number('success_rate'),tooltip=[number('successes'),number('count')]))
        name='pusht_matched_horizons'
        values=[]
        for method,horizons in report['prediction'].items():
            if method=='actual':continue
            for h,row in horizons.items():
                value=dict(horizon=int(h),method=method,count=row['count'])
                for field in ('latent_error','h_angle_mae_deg','r_angle_mae_deg','image_mse'):
                    if row.get(field) is not None:value[field]=row[field]
                for prefix in ('h','r'):
                    for i,coordinate in enumerate(POSITIONS):
                        if row.get(f'{prefix}_position_mae') is not None:
                            value[f'{prefix}_{coordinate}']=row[f'{prefix}_position_mae'][i]
                values.append(value)
        if values:
            datasets[name]=sorted(values,key=lambda row:(row['horizon'],row['method']))
            fields=[('latent_error','variance-normalized latent MSE'),('h_angle_mae_deg','H block-angle MAE (degrees)')]
            fields += [(f'h_{c}',f'H {c} MAE (world units)') for c in POSITIONS]
            for field,label in fields:
                if not any(field in row for row in values):continue
                charts.append(chart(f'pusht_horizon_{field}',f'PushT matched rollout: {label}',
                    'Prediction, copy and reset share exact windows/actions. One horizon step is0.1seconds.',name,'line',
                    number('horizon'),number(field),color=category('method'),tooltip=[number('count')]))
        for visual in run.internals['panels'][:1]:
            encoded=base64.b64encode(Path(visual['png']).read_bytes()).decode()
            blocks.append({'id':'pusht_qualitative','type':'html','layout':'full','body':
                '<h3>PushT actual / reconstruction / imagination</h3>'
                f'<img style="max-width:100%;height:auto" alt="Matched PushT observations and imagined frames" src="data:image/png;base64,{encoded}">'
                f'<p>{html.escape(visual.get("action_alignment","Matched recorded actions"))}. Separate real/imagined R appears in the raw GIF.</p>'})
    for name,title,columns,subtitle in [
        ('pusht_control_detail','PushT exact controller and oracle results',
         [('run','Run'),('controller','Controller'),('budget_scope','Budget meaning'),('count','Cases'),('successes','Final successes'),
          ('success_rate','Final fraction'),('any_time_successes','Any-time successes'),('mean_episode_length','Executed intervals'),
          ('planning_failures','Planning failures'),('latency_median_ms','Decision median ms'),('latency_p95_ms','Decision p95 ms')],
         'Replay uses five privileged actions to verify reachable goal construction. All other controllers share the declared fixed budget. Raw case identities remain in control_records.json.'),
        ('pusht_prediction_detail','PushT exact pose and observable-motion errors',
         [('run','Run'),('method','Method'),('horizon','Horizon'),('count','Pose frames'),('motion_count','Motion frames'),
          ('h_position_mae','H position MAE [pusherXY,blockXY]'),('h_angle_mae_deg','H angle MAE degrees'),
          ('r_position_mae','R position MAE'),('r_angle_mae_deg','R angle MAE degrees'),
          ('r_motion_mae','R displacement MAE [dx4,dangle]'),('latent_error','Normalized latent MSE'),('image_mse','RGB MSE')],
         'Actual H/R pose uses all recorded held-out frames; motion excludes the first two. dx has world units and dangle radians per0.1second, not simulator velocity.')]:
        if datasets[name]:tables.append(table(name,title,columns,'run',subtitle))
    return blocks
