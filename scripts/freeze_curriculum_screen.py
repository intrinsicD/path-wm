"""Freeze validation-only screen decisions and audit matched exposures before tests."""
from pathlib import Path
import json
import subprocess
from world_model.curriculum.data import digest,file_hash
from world_model.pusht.checkpoints import json_atomic
ROOT=Path('runs/curriculum_2026-09-07')

def freeze(seed=4107):
    output=ROOT/f'seed_{seed}'/'screen_decision.json'
    if output.exists():raise ValueError('screen decision already frozen')
    results={};manifests={};draws={}
    for arm in ('A','B','C'):
        phases=['supervised'] if arm=='A' else ['warmup','supervised']
        for phase in phases:
            directory=ROOT/f'seed_{seed}'/arm/phase;key=arm+'_'+phase
            result=json.loads((directory/'curriculum_result.json').read_text())
            if result['status']!='completed':raise ValueError('all prescribed phases must complete')
            results[key]=result;manifests[key]=json.loads((directory/'curriculum_manifest.json').read_text())
            draws[key]=[json.loads(l)['sample_indices_sha256'] for l in (directory/'training.jsonl').read_text().splitlines()]
    initial_ed=[results[k]['initial_ed_fingerprint'] for k in ('A_supervised','B_warmup','C_warmup')]
    initial_heads=[results[k]['initial_h_fingerprint'] for k in ('A_supervised','B_supervised','C_supervised')]
    assert len(set(initial_ed))==len(set(initial_heads))==1
    assert draws['A_supervised'][:2000]==draws['B_supervised']==draws['C_supervised']
    for arm in ('B','C'):
        assert results[arm+'_supervised']['dependencies']['warmup']==results[arm+'_warmup']['model_fingerprint']
        assert sum(results[arm+'_'+p]['step'] for p in ('warmup','supervised'))==4000
        assert sum(results[arm+'_'+p]['elapsed_seconds'] for p in ('warmup','supervised'))<3600
    reference=results['A_supervised']['selected'];comparisons={}
    for arm in ('B','C'):
        m=results[arm+'_supervised']['selected']
        effects=[a<=b for a,b in zip(m['position_mae']+[m['angle_mae_deg']],reference['position_mae']+[reference['angle_mae_deg']])]
        comparisons[arm]=dict(q_ratio=m['q']/reference['q'],constituents_nonworsening=effects,
            passes_numeric_readiness=m['q']<=1,promising=m['q']<=.9*reference['q'] and all(effects))
    all_failed=all(results[a+'_supervised']['selected']['q']>1 for a in ('A','B','C'))
    candidates=[] if all_failed else [a for a,c in comparisons.items() if c['promising'] and c['passes_numeric_readiness']]
    value=dict(schema='curriculum-screen-decision-v1',seed=seed,selection_population='original fixed2048 validation indices',
        all_arms_failed_numeric_readiness=all_failed,confirmation_candidates=candidates,comparisons=comparisons,
        decision='stop PushT U/P expansion; no pretraining policy adopted' if all_failed else 'qualitative review and confirmation required before adoption',
        status='frozen_before_test',selected={a:results[a+'_supervised']['selected'] for a in ('A','B','C')},
        matched_audit=dict(initial_ed=initial_ed[0],fresh_heads=initial_heads[0],supervised_first2000_draws=digest(draws['B_supervised']),
            equal_total_updates=4000,task_supervised_updates={'A':4000,'B':2000,'C':2000},
            actual_seconds={a:sum(v['elapsed_seconds'] for k,v in results.items() if k.startswith(a+'_')) for a in ('A','B','C')}),
        result_hashes={k:digest(v) for k,v in results.items()},
        git_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        test_evaluations=['A_selected','A_final','A_update2000','B_selected','B_final','B_warmup','C_selected','C_final','C_warmup','historical_selected','historical_final'])
    value['fingerprint']=digest(value);json_atomic(output,value)
    print(json.dumps(value,indent=2));return value
if __name__=='__main__':freeze()
