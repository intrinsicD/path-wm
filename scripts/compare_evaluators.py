"""Paired native SWM and local control on identical official-data source cases.

Uses unmodified upstream World.evaluate, WorldModelPolicy, CEMSolver and LeWM
JEPA from pinned sources. A thin runner replaces version-specific CLI wiring.
The optional prefix dataset is explicitly a compatibility subset, not a paper
score; both sides use the same population-standard-deviation normalization.
"""
import argparse,copy,json,os,sys,time
from pathlib import Path
os.environ.setdefault('SDL_VIDEODRIVER','dummy')
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
sys.path.insert(0,str(next((root/'.runtime/reference_swm').iterdir())))
import h5py,hdf5plugin,numpy as np,torch
from sklearn.preprocessing import StandardScaler
from torchvision.transforms import v2
import stable_worldmodel as swm
from stable_worldmodel.planning.solver.cem import CEMSolver
from third_party.lewm.jepa import JEPA
from world_model.model import build_model
from world_model.data import preprocess_pixels
from world_model.planning import cem,latent_cost
from world_model.train import write_json
from world_model.eval_pusht import initial_observations
from third_party.swm.pusht import PushT


@torch.no_grad()
def compare(path,output,count):
    torch.set_num_threads(4);torch.manual_seed(42)
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    model=build_model().cuda().eval()
    model.load_state_dict(torch.load(root/'data/reference/lewm-pusht/weights.pt',weights_only=True,map_location='cuda'),strict=True)
    native=JEPA(*(copy.deepcopy(getattr(model,k)) for k in ['encoder','predictor','action_encoder','projector','pred_proj'])).eval()
    ds=swm.data.HDF5Dataset(path=path,keys_to_load=['pixels','action','state','proprio'])
    with h5py.File(path,'r',swmr=True) as f:
        actions=f['action'][:];actions=actions[np.isfinite(actions).all(1)]
        scaler=StandardScaler().fit(actions)
        lengths,offsets=f['ep_len'][:],f['ep_offset'][:]
        rng=np.random.default_rng(42)
        cases=[]
        while len(cases)<count:
            ep=int(rng.integers(len(lengths)));start=int(rng.integers(int(lengths[ep])-25))
            row=int(offsets[ep])+start
            if np.linalg.norm(f['state'][row,:4]-f['state'][row+25,:4])<=30:continue
            if any(c['episode']==ep and c['start']==start for c in cases):continue
            cases.append(dict(episode=ep,start=start,seed=1234+len(cases)))
        write_json(out/'manifest.json',dict(path=str(path),cases=cases,action_mean=scaler.mean_.tolist(),action_std=scaler.scale_.tolist(),
            upstream_commit='6f1e499e9cc0c898d326112f485c1062c3d20f24',model_commit='8edfeb336732b5f3ce7b8b210d0ba370a09e2cac',
            subset='prefix' in str(path),normalization_population='all finite actions in supplied dataset',
            horizon=5,frameskip=5,budget=50,samples=300,iterations=30,elites=30))
        transform=v2.Compose([v2.ToImage(),v2.ToDtype(torch.float32,scale=True),
            v2.Normalize(mean=[.485,.456,.406],std=[.229,.224,.225]),v2.Resize(224)])
        callables=[{'method':'_set_state','args':{'state':{'value':'state'}}},
                   {'method':'_set_goal_state','args':{'goal_state':{'value':'goal_state'}}}]
        records=[]
        for i,case in enumerate(cases):
            tick=time.monotonic()
            world=swm.World(env_name='swm/PushT-v1',num_envs=1,image_shape=(224,224),max_episode_steps=60)
            solver=CEMSolver(cost=native,num_samples=300,n_steps=30,topk=30,device='cuda',seed=case['seed'])
            policy=swm.policy.WorldModelPolicy(solver,swm.PlanConfig(horizon=5,receding_horizon=5,action_block=5),
                process={'action':scaler},transform={'pixels':transform,'goal':transform})
            # Capture actions emitted by the original policy without changing them.
            native_actions=[]
            original=policy.get_action
            def traced(info,**kwargs):
                action=original(info,**kwargs);native_actions.append(action.copy());return action
            policy.get_action=traced
            world.set_policy(policy)
            result=world.evaluate(dataset=ds,episodes_idx=[case['episode']],start_steps=[case['start']],
                goal_offset=25,eval_budget=50,callables=callables)
            native_success=bool(result['episode_successes'][0]);world.close()
            row=int(offsets[case['episode']])+case['start']
            state,target=f['state'][row].astype(float),f['state'][row+25].astype(float)
            env=PushT();env.reset(seed=case['seed']);env._set_state(state);env._set_goal_state(target)
            initial,goal_pixels=initial_observations(f['pixels'][row],f['pixels'][row+25])
            goal=model.encode(preprocess_pixels(goal_pixels.cuda()))
            generator=torch.Generator(device='cuda').manual_seed(case['seed'])
            used=0;success=False;local_actions=[];plans=[]
            while used<50 and not success:
                pixels=initial.cuda() if used==0 else torch.from_numpy(env.render().copy()).permute(2,0,1)[None,None].cuda()
                z=model.encode(preprocess_pixels(pixels))
                plan,detail=cem(latent_cost(model,z,goal),5,10,torch.device('cuda'),generator=generator)
                raw=scaler.inverse_transform(plan.reshape(-1,2).cpu().numpy())
                plans.append(detail)
                for action in raw[:50-used]:
                    obs,_,success,_,_=env.step(action);used+=1;local_actions.append(action.copy())
                    if success:break
            env.close()
            np.savez(out/f'actions_{i}.npz',native=np.asarray(native_actions),local=np.asarray(local_actions))
            na=np.asarray(native_actions)[:,0];la=np.asarray(local_actions)
            common=min(len(na),len(la))
            rec={**case,'native_success':native_success,'local_success':bool(success),
                 'native_steps':len(na),'local_steps':len(la),
                 'action_max_difference':float(np.max(np.abs(na[:common]-la[:common]))),
                 'seconds':time.monotonic()-tick}
            records.append(rec)
            with (out/'cases.jsonl').open('a') as log:log.write(json.dumps(rec)+'\n')
            print(json.dumps(rec),flush=True)
        write_json(out/'summary.json',dict(native_successes=sum(r['native_success'] for r in records),
            local_successes=sum(r['local_success'] for r in records),cases=len(records),
            outcome_agreement=sum(r['native_success']==r['local_success'] for r in records),
            max_action_difference=max(r['action_max_difference'] for r in records)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('path');p.add_argument('output');p.add_argument('--cases',type=int,default=10)
    a=p.parse_args();compare(a.path,a.output,a.cases)
