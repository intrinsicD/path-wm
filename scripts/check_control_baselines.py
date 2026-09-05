"""Stationary and recorded-action controls for an existing checkpoint case set."""
import argparse,json
from pathlib import Path
import h5py,numpy as np
from world_model.eval_pusht import PushT
from world_model.train import write_json


def check(directory):
    directory=Path(directory);meta=json.loads((directory/'manifest.json').read_text())
    out=directory/'action_baselines.json'
    if out.exists():raise FileExistsError(out)
    records=[]
    with h5py.File(meta['dataset']['path'],'r') as f:
        for i,case in enumerate(meta['cases']):
            row=case['row'];state=f['state'][row].astype(float);target=f['state'][row+25].astype(float)
            for kind,actions in [('stationary',np.zeros((50,2))),('recorded_replay',f['action'][row:row+25])]:
                env=PushT(resolution=224,relative=True)
                env.reset(seed=meta['solver_and_reset_seeds'][i]);env._set_state(state);env._set_goal_state(target)
                initial=bool(env.eval_state(target,env._get_obs())[0]);success=False
                for step,action in enumerate(actions,1):
                    _,_,success,_,_=env.step(action)
                    if success:break
                env.close();records.append(dict(**case,kind=kind,success=bool(success),initial_success=initial,steps=step))
    result=dict(records=records,summary={kind:dict(successes=sum(r['success'] for r in records if r['kind']==kind),cases=len(meta['cases'])) for kind in ('stationary','recorded_replay')})
    write_json(out,result);print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory');check(p.parse_args().directory)
