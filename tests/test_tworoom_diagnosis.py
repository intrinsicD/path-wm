"""Protect complete simulator horizons and explicit model/case populations."""
import json
import numpy as np
import pytest
from scripts.diagnose_tworoom_control import select_cases, simulate


def test_frozen_prefix_retains_case_order_and_matching_seeds():
    protocol={'dataset':{'name':'tworoom'},'goal_offset':25,'horizon':5,'action_block':5,
              'cases':[{'row':i*50,'episode':i,'start':0} for i in range(3)],
              'solver_and_reset_seeds':[1234,1235,1236]}
    selected=select_cases(protocol,2)
    assert [(c['row'],c['reset_seed']) for c in selected]==[(0,1234),(50,1235)]
    assert all(c['population']=='source primary' for c in selected)
    with pytest.raises(ValueError):select_cases(protocol,4)
    with pytest.raises(ValueError):select_cases({**protocol,'goal_offset':100},2)


def test_open_loop_retains_actions_after_success_and_reproduces_frames():
    state=np.array([50.,60.],np.float32);target=np.array([70.,60.],np.float32)
    actions=np.zeros((25,2),np.float32);actions[:4,0]=1.;actions[4:,0]=-1.
    before=actions.copy();frames,states,result=simulate(state,target,actions,1234)
    assert frames.shape==(6,224,224,3) and states.shape==(26,2)
    assert result['success_any'] and not result['success_terminal']
    assert not result['initial_success']
    assert result['position_error']==pytest.approx(np.linalg.norm(states[-1]-target))
    np.testing.assert_array_equal(actions,before)
    again=simulate(state,target,actions,1234)
    np.testing.assert_array_equal(frames,again[0]);np.testing.assert_array_equal(states,again[1])


def test_ranking_ledger_accepts_declared_models_and_rejects_missing_pair(tmp_path):
    from viewer.ledger import collect_run_results,DashboardDataError
    run=tmp_path/'runs'/'ranking';run.mkdir(parents=True)
    manifest={'models':['local','released'],'candidates_per_case':2,'cases':[{'row':0}],
              'dataset':{'name':'tworoom'},'checkpoint_sha256s':{'local':'a','released':'b'}}
    (run/'manifest.json').write_text(json.dumps(manifest))
    records=[dict(case_index=0,model=model,candidate_index=i,candidate=kind,
                  predicted_cost=float(i),position_error=float(i+1),angle_error=0.,state_distance=float(i+1),
                  success_terminal=False,success_any=False,initial_success=False)
             for model in manifest['models'] for i,kind in enumerate(['local_plan','replay'])]
    summaries=[dict(case_index=0,model=model,population='source primary',candidates=2,
                    selected_index=0,selected_distance=1.,best_distance=1.,regret=0.) for model in manifest['models']]
    (run/'ranking_records.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    (run/'ranking.json').write_text(json.dumps(dict(records=4,checkpoint_unchanged=True,case_summaries=summaries)))
    results,_=collect_run_results(tmp_path/'runs')
    assert {r.context['model'] for r in results}=={'local','released'}
    manifest['models'].append('missing');(run/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(DashboardDataError,match='population'):
        collect_run_results(tmp_path/'runs')


def test_dashboard_defaults_to_latest_ranking_and_its_actual_plan():
    from viewer.ledger import RunResult
    from viewer.dashboard import build_dashboard_artifact
    results=[]
    for label,model,time in [('old','pilot',1.),('new','local',2.)]:
        record=dict(candidate=model+'_plan',candidate_index=0,model=model,predicted_cost=1.,position_error=2.,
                    rollout_mse_by_step=[1.]*5,copy_mse_by_step=[2.]*5,
                    predicted_cost_by_step=[3.]*5,actual_cost_by_step=[4.]*5)
        results.append(RunResult(label,'ranking','evaluated',None,{},
            {'model':model,'case_index':0,'population':'source primary'},(label+'/ranking.json',),time,
            ranking=(record,)))
    artifact=build_dashboard_artifact(results,[])
    assert {r['run'] for r in artifact['snapshot']['datasets']['ranking_selected']}=={'new'}
    assert len(artifact['snapshot']['datasets']['rollout_error'])==10
    assert {r['candidate'] for r in artifact['snapshot']['datasets']['rollout_error']}=={'local_plan'}
