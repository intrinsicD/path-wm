"""Released LeWM through the unmodified SWM evaluator on the full source data.

Thin wiring for the pinned APIs; sampling, scalers and solver settings follow
LeWM eval.py/config. Environment reset seed arguments are recorded without overriding them;
null means the upstream environment chooses its own entropy. Require a verified extraction receipt before full evaluation.
"""
import argparse,copy,hashlib,json,os,sys,time,subprocess
from pathlib import Path
os.environ.setdefault('SDL_VIDEODRIVER','dummy')
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
sys.path.insert(0,str(next((root/'.runtime/reference_swm').iterdir())))
import h5py,hdf5plugin,numpy as np,torch,yaml
from sklearn.preprocessing import StandardScaler
from torchvision.transforms import v2
import stable_worldmodel as swm
from stable_worldmodel.planning.solver.cem import CEMSolver
from third_party.lewm.jepa import JEPA
from world_model.model import build_model
from world_model.evaluation import sample_source_cases
from world_model.train import write_json

@torch.inference_mode()
def evaluate(config,output,count=50):
    cfg=yaml.safe_load(Path(config).read_text());path=Path(cfg['path'])
    if cfg['name']!='pusht':raise ValueError('This reference checkpoint is for PushT')
    receipt=json.loads((path.parent/'extraction.json').read_text())
    if any(receipt[k]!=cfg[k] for k in ('repo','revision','sha256')):
        raise ValueError('Extraction receipt does not match the pinned source')
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.manual_seed(42)
    checkpoint=root/'data/reference/lewm-pusht/weights.pt'
    local=build_model().cuda().eval()
    local.load_state_dict(torch.load(checkpoint,weights_only=True,map_location='cuda'),strict=True)
    native=JEPA(*(getattr(local,k) for k in ('encoder','predictor','action_encoder','projector','pred_proj'))).eval()
    native.requires_grad_(False)
    process={};stats={}
    with h5py.File(path,'r') as f:
        lengths,offsets=f['ep_len'][:],f['ep_offset'][:]
        cases=sample_source_cases(lengths,offsets,count=count,seed=42)
        for col in ('action','proprio','state'):
            values=f[col][:];values=values[~np.isnan(values).any(axis=1)]
            scaler=StandardScaler().fit(values);process[col]=scaler
            stats[col]=dict(mean=scaler.mean_.tolist(),std=scaler.scale_.tolist(),count=len(values))
            if col!='action':process['goal_'+col]=scaler
        # Confirm offset-derived sampling equals actual source episode/step IDs.
        ep_col='episode_idx' if 'episode_idx' in f else 'ep_idx'
        for case in cases:
            assert int(f[ep_col][case['row']])==case['episode']
            assert int(f['step_idx'][case['row']])==case['start']
    manifest=dict(dataset=cfg,source_verification=receipt,cases=cases,normalization=stats,
        checkpoint_sha256=hashlib.file_digest(checkpoint.open('rb'),'sha256').hexdigest(),
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        model_commit='8edfeb336732b5f3ce7b8b210d0ba370a09e2cac',
        swm_commit='6f1e499e9cc0c898d326112f485c1062c3d20f24',
        source_episodes=len(lengths),source_frames=int(lengths.sum()),
        protocol='Full source, authors sampling/scalers/config, pinned current SWM evaluator; thin API wiring; no video export',
        num_envs=count,solver_batch_size=1,samples=300,iterations=30,elites=30,
        horizon=5,action_block=5,receding_horizon=5,budget=50,goal_offset=25,
        sampling_seed=42,solver_seed=42,reset_seed_policy='upstream default; per-environment reset seed arguments recorded; null means unseeded')
    write_json(out/'manifest.json',manifest)
    transform=v2.Compose([v2.ToImage(),v2.ToDtype(torch.float32,scale=True),
        v2.Normalize(mean=[.485,.456,.406],std=[.229,.224,.225]),v2.Resize(224)])
    ds=swm.data.HDF5Dataset(path=path,keys_to_load=['pixels','action','state','proprio'])
    world=swm.World(env_name='swm/PushT-v1',num_envs=count,image_shape=(224,224),max_episode_steps=100)
    solver=CEMSolver(cost=native,batch_size=1,num_samples=300,n_steps=30,topk=30,device='cuda',seed=42)
    policy=swm.policy.WorldModelPolicy(solver,swm.PlanConfig(horizon=5,receding_horizon=5,action_block=5),
        process=process,transform={'pixels':transform,'goal':transform})
    world.set_policy(policy)
    original_reset=world.reset
    def recorded_reset(*args,**kwargs):
        result=original_reset(*args,**kwargs)
        seed=kwargs.get('seed',args[0] if args else None)
        if seed is None:seeds=[None]*count
        elif np.isscalar(seed):seeds=[int(seed)+i for i in range(count)]
        else:seeds=[None if value is None else int(value) for value in seed]
        write_json(out/'reset_seeds.json',seeds)
        return result
    world.reset=recorded_reset
    tick=time.monotonic()
    try:
        result=world.evaluate(dataset=ds,episodes_idx=[c['episode'] for c in cases],
            start_steps=[c['start'] for c in cases],goal_offset=25,eval_budget=50,
            callables=[{'method':'_set_state','args':{'state':{'value':'state'}}},
                       {'method':'_set_goal_state','args':{'goal_state':{'value':'goal_state'}}}])
        summary=dict(successes=int(result['episode_successes'].sum()),cases=count,
            success_rate=float(result['episode_successes'].mean()),
            episode_successes=result['episode_successes'].tolist(),seconds=time.monotonic()-tick,
            protocol=manifest['protocol'])
        write_json(out/'summary.json',summary);print(json.dumps(summary),flush=True)
    finally:world.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('output');p.add_argument('--cases',type=int,default=50)
    a=p.parse_args();evaluate(a.config,a.output,a.cases)
