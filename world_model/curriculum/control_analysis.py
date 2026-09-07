"""Descriptive matched-case comparisons, with paired members resampled together."""
from pathlib import Path
import argparse
import json
import shutil
import numpy as np
from .data import file_hash
from world_model.pusht.checkpoints import json_atomic
ROOT=Path('runs/curriculum_2026-09-07')
REFERENCE=Path('runs/paddle/continuation_v1/evaluation')

def paired_delta(new,old,seed=93501):
    new=np.asarray(new,dtype=float);old=np.asarray(old,dtype=float)
    if new.shape!=old.shape or new.ndim!=2 or new.shape[1]!=2 or not len(new):
        raise ValueError('matching complete two-member pairs required')
    if not np.isfinite(new).all() or not np.isfinite(old).all():raise ValueError('finite outcomes required')
    difference=(new-old).mean(1)
    rng=np.random.default_rng(seed);indices=rng.integers(len(new),size=(2000,len(new)))
    draws=difference[indices].mean(1)
    return dict(pairs=len(new),members=2*len(new),new_rate=float(new.mean()),reference_rate=float(old.mean()),
                difference=float(difference.mean()),ci95=np.quantile(draws,[.025,.975]).tolist(),seed=seed,draws=2000),draws

def match_rows(new,old,policy,count):
    a={r['case_id']:r for r in new if r['controller']==policy}
    b={r['case_id']:r for r in old if r['controller']==policy}
    if len(a)!=count or a.keys()!=b.keys():raise ValueError('case population mismatch')
    for k,r in a.items():
        if r['seed']!=b[k]['seed'] or r['initial_state']!=b[k]['initial_state']:raise ValueError('initial condition mismatch')
        if r.get('correct_action')!=b[k].get('correct_action'):raise ValueError('paired action label mismatch')
    return [a[k] for k in sorted(a)],[b[k] for k in sorted(a)]

def pair_array(rows,key):
    groups={}
    for r in rows:
        group=groups.setdefault(r['pair_id'],{})
        if r['direction'] in group:raise ValueError('duplicate pair member')
        group[r['direction']]=float(key(r))
    if any(set(g)!=set((-1,1)) for g in groups.values()):raise ValueError('incomplete opposite-direction pair')
    return np.asarray([[groups[g][-1],groups[g][1]] for g in sorted(groups)])

def compare(new_ordinary,old_ordinary,new_pairs,old_pairs):
    result={};replicates={}
    for policy in ('learned','reset','random','tracker','privileged'):
        a,b=match_rows(new_ordinary,old_ordinary,policy,500)
        x=np.asarray([r['first_hit_before_miss'] for r in a],dtype=float)
        y=np.asarray([r['first_hit_before_miss'] for r in b],dtype=float)
        rng=np.random.default_rng(93500);indices=rng.integers(500,size=(2000,500));draws=(x-y)[indices].mean(1)
        ordinary=dict(starts=500,new_successes=int(x.sum()),reference_successes=int(y.sum()),
                      new_rate=float(x.mean()),reference_rate=float(y.mean()),difference=float((x-y).mean()),
                      ci95=np.quantile(draws,[.025,.975]).tolist(),seed=93500,draws=2000,
                      identical_action_trajectories=all(u['actions']==v['actions'] for u,v in zip(a,b)))
        replicates[policy+'_ordinary']=draws
        a,b=match_rows(new_pairs,old_pairs,policy,200)
        x=pair_array(a,lambda r:r['first_hit_before_miss']);y=pair_array(b,lambda r:r['first_hit_before_miss'])
        paired,draws=paired_delta(x,y);replicates[policy+'_paired']=draws
        paired['new_successes']=int(x.sum());paired['reference_successes']=int(y.sum())
        current_ci,_=paired_delta(x,np.zeros_like(x));paired['new_rate_ci95']=current_ci['ci95']
        first,draws=paired_delta(pair_array(a,lambda r:r['first_action']==r['correct_action']),
                                pair_array(b,lambda r:r['first_action']==r['correct_action']))
        replicates[policy+'_first_action']=draws
        result[policy]=dict(ordinary=ordinary,paired=paired,paired_first_action=first)
    return result,replicates

def main():
    p=argparse.ArgumentParser();p.add_argument('--self-check',action='store_true');a=p.parse_args()
    current=REFERENCE if a.self_check else ROOT/'paddle_history/evaluation'
    if json.loads((current/'metrics.json').read_text())['status']!='completed':
        raise ValueError('all controller cases must finish before comparison')
    out=ROOT/('control_comparison_smoke' if a.self_check else 'control_comparison');out.mkdir(exist_ok=True)
    if (out/'curriculum_analysis.json').exists():raise ValueError('preserve completed comparison')
    files={}
    for label,base in [('reference',REFERENCE),('current',current)]:
        for population in ('ordinary','paired'):
            source=base/(population+'.partial.json');target=out/(label+'_'+population+'.json')
            shutil.copy2(source,target);files[label+'_'+population]=target
    rows={k:json.loads(v.read_text()) for k,v in files.items()}
    result,draws=compare(rows['current_ordinary'],rows['reference_ordinary'],rows['current_paired'],rows['reference_paired'])
    json_atomic(out/'comparison.json',result);np.savez_compressed(out/'bootstrap_draws.npz',**draws)
    sources=[*files.values(),out/'comparison.json',out/'bootstrap_draws.npz']
    if a.self_check:
        assert all(r['ordinary']['difference']==r['paired']['difference']==r['paired_first_action']['difference']==0 for r in result.values())
        narrative='## Control comparison software check\n\nDEVELOPMENT ONLY: the complete historical case set compared with itself gives zero success/first-action differences for every controller. It contains no new controller outcome and is not a model quality result.'
        metrics={'self_comparison_max_abs_difference':0.};panels=[]
    else:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(10,3.5),layout='constrained')
        policies=list(result);x=np.arange(len(policies))
        for ax,population in zip(axes,('ordinary','paired')):
            ax.bar(x-.18,[result[k][population]['reference_rate'] for k in policies],width=.36,label='Historical reference',color='#8aa8c7')
            ax.bar(x+.18,[result[k][population]['new_rate'] for k in policies],width=.36,label='New mixed-U follow-up',color='#db7923')
            ax.set(xticks=x,xticklabels=policies,ylim=(0,1),ylabel='First-interception success rate',title=population.capitalize())
            ax.tick_params(axis='x',labelrotation=25)
        axes[0].legend(fontsize=7);fig.suptitle('Matched frozen test cases · source identities verified · one trained model per condition')
        panel=out/'control_comparison.png';fig.savefig(panel,dpi=130);fig.savefig(out/'control_comparison.svg');plt.close(fig);sources.append(panel)
        narrative='## Complete controller comparison\n\n500 ordinary starts and 100 opposite-direction pairs per controller. Historical/current case IDs, initial states, seeds and paired action labels match exactly. '
        narrative+='Paired intervals resample whole two-member pairs, not individual members; ordinary intervals resample matched starts. Both use 2,000 draws. They describe case uncertainty, not variation across training seeds. This is a post-training descriptive analysis, not a new gate.\n\n'
        narrative+='| Controller | Ordinary old → new | Paired old → new | Paired change · 95% interval |\n|---|---:|---:|---:|\n'
        for k,r in result.items():
            o=r['ordinary'];q=r['paired'];narrative+=f"| {k} | {o['reference_successes']}/500 → {o['new_successes']}/500 | {q['reference_successes']}/200 → {q['new_successes']}/200 | {q['difference']*100:.1f}pp [{q['ci95'][0]*100:.1f}, {q['ci95'][1]*100:.1f}] |\n"
        metrics={k+'_'+pop+'_rate':r[pop]['new_rate'] for k,r in result.items() for pop in ('ordinary','paired')}
        panels=[dict(file=str(panel),embed=True,title='Complete controller comparison',caption='Counts use all cases. Whole-pair resampling intervals are in the exact source comparison; bars show point estimates.')]
    json_atomic(out/'curriculum_analysis.json',dict(status='completed',purpose='Reference identity self-check' if a.self_check else 'Complete matched controller comparison',
        metrics=metrics,narrative=narrative,panels=panels,
        sources={p.relative_to(ROOT).as_posix():file_hash(p) for p in sources}))
    print(json.dumps(result if not a.self_check else {'development_identity_check':'passed'}),flush=True)
if __name__=='__main__':main()
