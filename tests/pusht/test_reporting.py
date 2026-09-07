"""PushT charts require exact raw-record reconciliation, separate from paddle."""
import json
import pytest

from viewer.ledger import collect_run_results, DashboardDataError
from viewer.dashboard import build_dashboard_artifact


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


def test_pusht_completed_control_summary_cannot_exceed_raw_successes(tmp_path):
    root=tmp_path/'runs';path=root/'pusht_world_model'/'evaluation'
    write(path/'control_records.json',[{'case_id':'a','controller':'learned','success':False,'episode_length':2}])
    write(path/'prediction_records.json',[])
    write(path/'metrics.json',{'schema_version':'pusht-eup-evaluation-v1','status':'completed',
        'prediction':{},'control':{'learned':{'count':1,'successes':1,'success_rate':1.}},'visuals':[]})
    with pytest.raises(DashboardDataError,match='success'):
        collect_run_results(root)


def test_pusht_training_keeps_selected_values_and_separate_chronological_grids(tmp_path):
    root=tmp_path/'runs';path=root/'pusht_world_model'/'predictor_1'
    write(path/'pusht_manifest.json',{'stage':'predictor','horizon':1,'config':{'smoke':True}})
    (path/'training.jsonl').write_text(''.join(json.dumps({'step':i,'loss':1/i})+'\n' for i in range(1,101)))
    validation=[{'step':0,'loss':1.,'copy_loss':1.},{'step':50,'loss':.2,'copy_loss':1.},{'step':100,'loss':.3,'copy_loss':1.}]
    (path/'validation.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in validation))
    write(path/'pusht_result.json',{'status':'completed','global_update':100,'selected_update':50,'metrics':{'loss':.2}})
    runs,notices=collect_run_results(root)
    item=next(r for r in runs if r.kind=='pusht_training')
    assert item.metrics['selected_validation.loss']==.2 and item.metrics['latest_validation.loss']==.3
    artifact=build_dashboard_artifact(runs,notices);datasets=artifact['snapshot']['datasets']
    assert len(datasets['pusht_training_predictor_1'])==50
    curve=datasets['pusht_validation_objective_predictor_1']
    assert [r['step'] for r in curve]==sorted(r['step'] for r in curve)
    assert {r['series'] for r in curve}=={'validation objective','matched copy objective'}
    assert not any(r.get('run','').startswith('pusht_world_model') for r in datasets.get('train_loss',[]))
