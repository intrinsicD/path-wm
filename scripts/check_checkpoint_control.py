"""A few untouched-checkpoint control cases drawn only from held-out episodes."""
import argparse,json,hashlib
from pathlib import Path
import h5py,numpy as np,torch
from world_model.model import build_model
from world_model.evaluation import sample_source_cases
from world_model.eval_pusht import evaluate_case
from world_model.train import write_json

@torch.inference_mode()
def check(run,output,count=5):
    run=Path(run);out=Path(output);out.mkdir(parents=True,exist_ok=False)
    meta=json.loads((run/'manifest.json').read_text())
    checkpoint=run/'checkpoint.pt';digest=hashlib.file_digest(checkpoint.open('rb'),'sha256').hexdigest()
    saved=torch.load(checkpoint,weights_only=True,map_location='cuda')
    model=build_model(saved['model_config']).cuda().eval();model.load_state_dict(saved['model'],strict=True)
    torch.set_num_threads(4)
    results=[]
    with h5py.File(meta['dataset']['path'],'r') as f:
        cases=sample_source_cases(f['ep_len'][:],f['ep_offset'][:],count=count,seed=42,population=meta['val_episodes'])
        write_json(out/'manifest.json',dict(checkpoint=str(checkpoint),checkpoint_sha256=digest,step=saved['step'],
            dataset=meta['dataset'],cases=cases,action_stats=saved['action_stats'],sampling_seed=42,
            solver_and_reset_seeds=[1234+i for i in range(count)],
            population='held-out episodes only; closely related prefix variants',
            batchnorm='saved buffers, unchanged',goal_offset=25,budget=50,samples=300,iterations=30,elites=30))
        for i,case in enumerate(cases):
            row=case['row']
            result=evaluate_case(model,f['pixels'][row],f['pixels'][row+25],f['state'][row],f['state'][row+25],saved['action_stats'],seed=1234+i)
            np.save(out/f'actions_{i}.npy',result.pop('actions'));result.update(case)
            results.append(result)
            with (out/'cases.jsonl').open('a') as log:log.write(json.dumps(result)+'\n')
            print(json.dumps({k:v for k,v in result.items() if k!='plans'}),flush=True)
    assert hashlib.file_digest(checkpoint.open('rb'),'sha256').hexdigest()==digest
    summary=dict(successes=sum(r['success'] for r in results),cases=len(results),
        initial_successes=sum(r['initial_success'] for r in results),
        mean_steps=float(np.mean([r['steps'] for r in results])),checkpoint_unchanged=True)
    write_json(out/'summary.json',summary);print(json.dumps(summary),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run');p.add_argument('output');p.add_argument('--cases',type=int,default=5)
    a=p.parse_args();check(a.run,a.output,a.cases)
