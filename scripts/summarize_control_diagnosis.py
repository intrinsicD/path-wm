"""Reconcile the completed diagnostic ledgers into a source-bound readout."""
import json
from pathlib import Path
import numpy as np
from scripts.diagnose_pusht_control import ranking_metrics,sha256
from world_model.train import write_json


def summarize():
    root=Path('runs/diagnostics/pusht_control_diagnosis')
    paths=[root/'training_control/summary.json',root/'training_control/action_baselines.json',
           root/'ranking/ranking.json',root/'ranking/ranking_records.jsonl',
           root/'real_batch_parity_expanded.json',root/'real_batch_parity_checkpointed.json',root/'reproduction_memory.json']
    control=json.loads(paths[0].read_text());baselines=json.loads(paths[1].read_text())
    summary=json.loads(paths[2].read_text());records=[json.loads(line) for line in paths[3].read_text().splitlines()]
    populations=[];case_metrics=[]
    for case in summary['case_summaries']:
        rows=[r for r in records if r['case_index']==case['case_index'] and r['model']==case['model']]
        extra=dict(actual_latent_vs_position_spearman=ranking_metrics([r['actual_latent_cost'] for r in rows],[r['position_error'] for r in rows])['spearman'],
                   predicted_vs_actual_latent_spearman=ranking_metrics([r['predicted_cost'] for r in rows],[r['actual_latent_cost'] for r in rows])['spearman'])
        case_metrics.append({**case,**extra})
    for population in ('training','heldout'):
        for model in ('pilot','released'):
            rows=[r for r in case_metrics if r['population']==population and r['model']==model]
            average=lambda key:float(np.mean([r[key] for r in rows]))
            populations.append(dict(population=population,model=model,cases=len(rows),
                selected_successes=sum(r['selected_success'] for r in rows),
                mean_spearman=average('spearman'),mean_selected_position_error=average('selected_distance'),
                mean_regret=average('regret'),mean_actual_latent_vs_position_spearman=average('actual_latent_vs_position_spearman'),
                mean_predicted_vs_actual_latent_spearman=average('predicted_vs_actual_latent_spearman')))
    sequence_metrics=[]
    for model in ('pilot','released'):
        for candidate in ('replay','stationary','pilot_plan','released_plan'):
            rows=[r for r in records if r['model']==model and r['candidate']==candidate]
            sequence_metrics.append(dict(model=model,candidate=candidate,cases=len(rows),
                terminal_successes=sum(r['success_terminal'] for r in rows),
                mean_position_error=float(np.mean([r['position_error'] for r in rows])),
                mean_rollout_mse_by_step=np.mean([r['rollout_mse_by_step'] for r in rows],axis=0).tolist(),
                mean_copy_mse_by_step=np.mean([r['copy_mse_by_step'] for r in rows],axis=0).tolist()))
    result=dict(control=control,action_baselines=baselines['summary'],ranking_by_population=populations,
                ranking_cases=case_metrics,sequence_metrics=sequence_metrics,
                checkpoint_unchanged=all(json.loads(p.read_text()).get('checkpoint_unchanged',True) for p in paths if p.suffix=='.json'),
                sources=[dict(path=str(path),sha256=sha256(path)) for path in paths],
                interpretation_limits='Eight fixed diagnostic cases, one seed. No population estimate or causal attribution to training scale/precision.')
    write_json(root/'diagnosis_summary.json',result)
    print(json.dumps({'control':control,'baselines':baselines['summary'],'ranking':populations}),flush=True)


if __name__=='__main__':summarize()
