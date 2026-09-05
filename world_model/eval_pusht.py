"""Closed-loop PushT goals with saved per-episode outcomes and explicit budgets.

Uses the pinned SWM environment and baseline CEM. A released-checkpoint run on
its training population is a compatibility check, not a generalization claim.
"""
import argparse,json,os,time
from pathlib import Path
os.environ.setdefault('SDL_VIDEODRIVER','dummy')
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT','1')
import h5py
import numpy as np
import torch
import yaml
from third_party.swm.pusht import PushT
from world_model.model import build_model
from world_model.data import preprocess_pixels, action_statistics, split_episodes
from world_model.planning import cem,latent_cost
from world_model.train import write_json


@torch.no_grad()
def evaluate(config_path, checkpoint, output, episodes=50, seed=42, samples=300,
             iterations=30, elites=30, budget=50, goal_offset=25, released=False):
    cfg=yaml.safe_load(Path(config_path).read_text())
    if cfg['name'] not in ('pusht','pusht_cchi'): raise ValueError('PushT evaluation requires PushT data')
    directory=Path(output);directory.mkdir(parents=True,exist_ok=True)
    if (directory/'episodes.jsonl').exists(): raise FileExistsError('Use a new evaluation output directory')
    torch.set_num_threads(4);torch.manual_seed(seed)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weights=torch.load(checkpoint,map_location=device,weights_only=True)
    with h5py.File(cfg['path'],'r') as f:
        lengths,offsets=f['ep_len'][:],f['ep_offset'][:]
        if released:
            if cfg['name']!='pusht': raise ValueError('Released weights require the LeWM dataset action scale')
            model=build_model().to(device)
            model.load_state_dict(weights,strict=True)
            stats=action_statistics(cfg['path'],list(range(len(lengths))))
            population=list(range(len(lengths)))
        else:
            model=build_model(weights['model_config']).to(device)
            model.load_state_dict(weights['model'],strict=True);stats=weights['action_stats']
            manifest=json.loads((Path(checkpoint).parent/'manifest.json').read_text())
            if manifest['dataset']!=cfg:raise ValueError('Checkpoint dataset does not match evaluation dataset')
            population=manifest['val_episodes']
        model.eval()
        rng=np.random.default_rng(seed)
        eligible=[int(e) for e in population if lengths[e]>goal_offset+1]
        if episodes>len(eligible): raise ValueError(f'Only {len(eligible)} eligible episodes; reduce count')
        selected=rng.choice(eligible,episodes,replace=False)
        settings=dict(dataset=cfg,checkpoint=str(Path(checkpoint).resolve()),released=released,
            population='source training population' if released else 'held-out episodes',
            seed=seed,episodes=episodes,samples=samples,iterations=iterations,elites=elites,
            horizon=5,action_block=cfg['frameskip'],receding_horizon=5,budget=budget,
            goal_offset=goal_offset,environment_commit='6f1e499e9cc0c898d326112f485c1062c3d20f24',
            protocol='local compatibility/development evaluation; not an exact paper benchmark')
        write_json(directory/'config.json',settings)
        results=[]
        for i,ep in enumerate(selected):
            local=int(rng.integers(0,int(lengths[ep])-goal_offset))
            row=int(offsets[ep])+local
            state=np.array(f['state'][row],dtype=np.float64)
            target=np.array(f['state'][row+goal_offset],dtype=np.float64)
            relative=cfg['name']=='pusht'
            # Original CCHI stores five state coordinates, omitting velocity.
            # SWM reset sets missing velocity to zero; this is a stated mismatch.
            env=PushT(resolution=224 if relative else 96,relative=relative)
            observation,info=env.reset(seed=seed+i,options={'state':state,'goal_state':target})
            goal_pixels=torch.from_numpy(info['goal'].copy()).permute(2,0,1)[None,None].to(device)
            goal=model.encode(preprocess_pixels(goal_pixels,224))
            success,_=env.eval_state(target,observation['state'])
            initial_success=bool(success);used=0;calls=0;plans=[];tick=time.monotonic()
            while used<budget and not success:
                pixels=torch.from_numpy(env.render().copy()).permute(2,0,1)[None,None].to(device)
                context=model.encode(preprocess_pixels(pixels,224))
                actions,details=cem(latent_cost(model,context,goal),5,2*cfg['frameskip'],device,
                    samples=samples,iterations=iterations,elites=elites,seed=seed+i*1000+used)
                calls+=details['predictor_steps'];plans.append(details)
                raw=actions.reshape(-1,2).cpu().numpy()*np.asarray(stats['std'])+np.asarray(stats['mean'])
                for action in raw[:budget-used]:
                    observation,_,terminated,_,info=env.step(action)
                    used+=1
                    if terminated:success=True;break
            distance=float(env.eval_state(target,observation['state'])[1]);env.close()
            result=dict(episode=int(ep),start=local,success=bool(success),initial_success=initial_success,
                steps=used,state_distance=distance,predictor_steps=calls,seconds=time.monotonic()-tick,plans=plans)
            results.append(result)
            with (directory/'episodes.jsonl').open('a') as log:log.write(json.dumps(result)+'\n')
            print(json.dumps({k:v for k,v in result.items() if k!='plans'}),flush=True)
        summary=dict(success_rate=float(np.mean([r['success'] for r in results])),episodes=len(results),
            initial_successes=sum(r['initial_success'] for r in results),
            mean_steps=float(np.mean([r['steps'] for r in results])),protocol=settings['protocol'])
        write_json(directory/'summary.json',summary);print(json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('dataset');p.add_argument('checkpoint');p.add_argument('output')
    p.add_argument('--episodes',type=int,default=50);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--samples',type=int,default=300);p.add_argument('--iterations',type=int,default=30)
    p.add_argument('--elites',type=int,default=30);p.add_argument('--budget',type=int,default=50)
    p.add_argument('--goal-offset',type=int,default=25);p.add_argument('--released',action='store_true')
    a=p.parse_args();evaluate(a.dataset,a.checkpoint,a.output,a.episodes,a.seed,a.samples,a.iterations,
        a.elites,a.budget,a.goal_offset,a.released)
