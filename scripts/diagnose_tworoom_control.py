"""Frozen TwoRoom action ranking against complete simulator trajectories.

This measures a fixed 25-action horizon, including motion after any success;
closed-loop 50-action scores remain separate. Run stages through run.py.
"""
import argparse
import json
from pathlib import Path
import subprocess
import time
import h5py
import numpy as np
import torch
from scripts.diagnose_pusht_control import action_blocks, raw_actions, ranking_metrics, sha256
from scripts.inspect_checkpoint import load
from world_model.data import preprocess_pixels
from world_model.eval_tworoom import _environment, _distance
from world_model.introspection import state_digest
from world_model.planning import cem, latent_cost
from world_model.train import write_json


def select_cases(protocol, count):
    if (protocol['dataset']['name'] != 'tworoom' or protocol.get('goal_offset') != 25
            or protocol.get('horizon') != 5 or protocol.get('action_block') != 5):
        raise ValueError('Ranking requires the TwoRoom primary five-block goal protocol')
    cases, seeds = protocol['cases'], protocol['solver_and_reset_seeds']
    if not isinstance(count, int) or not 0 < count <= len(cases) or len(cases) != len(seeds):
        raise ValueError('Invalid frozen case/seed count')
    return [dict(**case, reset_seed=seeds[i], population='source primary')
            for i, case in enumerate(cases[:count])]


def simulate(state, target, actions, seed):
    actions = np.asarray(actions, dtype=np.float32)
    if actions.shape != (25, 2) or not np.isfinite(actions).all():
        raise ValueError('Ranking requires 25 finite raw two-dimensional actions')
    env = _environment(state, target, seed)
    try:
        initial = _distance(env) < 16.
        frames, states, successes = [env.render().copy()], [env.agent_position.numpy().copy()], []
        for step, action in enumerate(actions, 1):
            _, _, success, _, _ = env.step(action)
            states.append(env.agent_position.numpy().copy()); successes.append(bool(success))
            if step % 5 == 0: frames.append(env.render().copy())
        distance = _distance(env)
        outcome = dict(initial_success=initial, success_any=any(successes),
                       success_terminal=successes[-1], position_error=distance,
                       state_distance=distance, angle_error=None)
        return np.stack(frames), np.stack(states), outcome
    finally:
        env.close()


def prepare(run, case_file, output, count=8):
    run, case_file, output = Path(run), Path(case_file), Path(output)
    training = json.loads((run/'manifest.json').read_text())
    protocol = json.loads(case_file.read_text())
    if training['dataset'] != protocol['dataset']: raise ValueError('Training/control source mismatch')
    cases = select_cases(protocol, count)
    with h5py.File(protocol['dataset']['path'], 'r') as source:
        for case in cases:
            ep, start = case['episode'], case['start']
            if not 0 <= ep < len(source['ep_len']) or not 0 <= start < int(source['ep_len'][ep])-25:
                raise ValueError('Case crosses episode boundary')
            if case['row'] != int(source['ep_offset'][ep])+start: raise ValueError('Case row mismatch')
    checkpoints = {'local':str(run/'checkpoint.pt'), 'released':protocol['released_reference']['checkpoint']}
    hashes = {name:sha256(path) for name,path in checkpoints.items()}
    if hashes['released'] != protocol['released_reference']['checkpoint_sha256']:
        raise ValueError('Released checkpoint mismatch')
    manifest = dict(dataset=protocol['dataset'], cases=cases, models=list(checkpoints),
        checkpoints=checkpoints, checkpoint_sha256s=hashes, training_manifest=str(run/'manifest.json'),
        training_manifest_sha256=sha256(run/'manifest.json'), reference_manifest=str(case_file),
        reference_manifest_sha256=sha256(case_file), random_seed_base=9000, random_sequences=16,
        candidates_per_case=20, goal_offset=25, horizon=5, action_block=5, open_loop_actions=25,
        samples=300,iterations=30,elites=30,precision='float32',batchnorm='saved buffers, unchanged',
        protocol='Fixed first eight primary cases; full open-loop horizon including after success; not a closed-loop score',
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'manifest.json',manifest)
    print(json.dumps(dict(stage='prepared',cases=count,checkpoint_sha256s=hashes)),flush=True)


def check_sources(manifest):
    for key in ('training_manifest','reference_manifest'):
        if sha256(manifest[key]) != manifest[key+'_sha256']: raise ValueError(f'{key} changed')
    for name,path in manifest['checkpoints'].items():
        if sha256(path) != manifest['checkpoint_sha256s'][name]: raise ValueError(f'{name} checkpoint changed')


def encode(model, frames, device):
    pixels=torch.from_numpy(np.asarray(frames).copy()).permute(0,3,1,2)[None].to(device)
    return model.encode(preprocess_pixels(pixels,model.encoder.config.image_size))


@torch.inference_mode()
def ranking(output, device='cuda'):
    output=Path(output);manifest=json.loads((output/'manifest.json').read_text());check_sources(manifest)
    torch.set_num_threads(4);torch.manual_seed(42);device=torch.device(device)
    training=json.loads(Path(manifest['training_manifest']).read_text())
    models={}
    for name in manifest['models']:
        model,stats,_,_=load(manifest['checkpoints'][name],training,name=='released',manifest['reference_manifest'])
        models[name]=(model.to(device).eval(),stats)
    states_before={name:state_digest(model) for name,(model,_) in models.items()}
    directory=output/'ranking';directory.mkdir(exist_ok=False)
    write_json(directory/'manifest.json',{**manifest, 'execution_code_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()})
    all_records=[];summaries=[];started=time.monotonic()
    with h5py.File(manifest['dataset']['path'],'r') as data:
        for case_index,case in enumerate(manifest['cases']):
            row=case['row'];source,goal=data['pixels'][row],data['pixels'][row+25]
            state,target=data['proprio'][row],data['proprio'][row+25]
            rng=np.random.default_rng(manifest['random_seed_base']+case_index)
            candidates=[('replay',data['action'][row:row+25]),('stationary',np.zeros((25,2),np.float32))]
            candidates += [(f'random_{i:02d}',raw_actions(rng.standard_normal((5,10)),models['local'][1]))
                           for i in range(manifest['random_sequences'])]
            contexts={};goals={};plans={}
            for name,(model,stats) in models.items():
                contexts[name]=encode(model,source[None],device);goals[name]=encode(model,goal[None],device)
                blocks,plans[name]=cem(latent_cost(model,contexts[name],goals[name]),5,10,device,
                    seed=case['reset_seed'],samples=manifest['samples'],iterations=manifest['iterations'],elites=manifest['elites'])
                candidates.append((f'{name}_plan',raw_actions(blocks.cpu().numpy(),stats)))
            raw=np.stack([actions for _,actions in candidates]).astype(np.float32)
            if len(raw)!=manifest['candidates_per_case']:raise ValueError('Candidate population changed')
            predictions={}
            for name,(model,stats) in models.items():
                blocks=torch.from_numpy(np.stack([action_blocks(actions,stats) for actions in raw])).to(device)
                predictions[name]=model.rollout(contexts[name].expand(len(raw),-1,-1),blocks)
            records=[];sim_states=[];images=dict(source=source,goal=goal)
            for candidate_index,(kind,_) in enumerate(candidates):
                frames,trajectory,outcome=simulate(state,target,raw[candidate_index],case['reset_seed'])
                sim_states.append(trajectory)
                if not kind.startswith('random'):images[kind]=frames
                for name,(model,_) in models.items():
                    actual=encode(model,frames,device);pred=predictions[name][candidate_index:candidate_index+1]
                    errors=(pred-actual[:,1:]).square().mean(-1)[0]
                    copying=(contexts[name]-actual[:,1:]).square().mean(-1)[0]
                    costs=(pred-goals[name]).square().sum(-1)[0]
                    actual_costs=(actual[:,1:]-goals[name]).square().sum(-1)[0]
                    records.append(dict(case_index=case_index,**case,model=name,candidate_index=candidate_index,
                        candidate=kind,**outcome,predicted_cost=float(costs[-1]),actual_latent_cost=float(actual_costs[-1]),
                        predicted_cost_by_step=costs.cpu().tolist(),actual_cost_by_step=actual_costs.cpu().tolist(),
                        rollout_mse_by_step=errors.cpu().tolist(),copy_mse_by_step=copying.cpu().tolist(),
                        initial_source_sim_mse=float((contexts[name]-actual[:,:1]).square().mean())))
            np.savez_compressed(directory/f'case_{case_index}_trajectories.npz',actions=raw,states=np.stack(sim_states),
                                candidate_names=np.array([name for name,_ in candidates]))
            np.savez_compressed(directory/f'case_{case_index}_images.npz',**images)
            for name in models:
                selected=[r for r in records if r['model']==name]
                measured=ranking_metrics([r['predicted_cost'] for r in selected],[r['position_error'] for r in selected])
                oracle=ranking_metrics([r['actual_latent_cost'] for r in selected],[r['position_error'] for r in selected])
                choice=selected[measured['selected_index']]
                summary=dict(case_index=case_index,population=case['population'],model=name,candidates=len(selected),
                    **measured,selected_candidate=choice['candidate'],selected_success=choice['success_terminal'],
                    actual_latent_spearman=oracle['spearman'],actual_latent_selected_success=selected[oracle['selected_index']]['success_terminal'],
                    plan=plans[name])
                summaries.append(summary);print(json.dumps(summary),flush=True)
            all_records.extend(records)
            with (directory/'ranking_records.jsonl').open('a') as stream:
                for record in records:stream.write(json.dumps(record)+'\n')
    check_sources(manifest)
    if any(state_digest(model)!=states_before[name] for name,(model,_) in models.items()):
        raise RuntimeError('Model buffers changed during ranking')
    result=dict(schema_version=1,records=len(all_records),case_summaries=summaries,
                checkpoint_unchanged=True,seconds=time.monotonic()-started)
    write_json(directory/'ranking.json',result);print(json.dumps(dict(stage='complete',records=len(all_records),seconds=result['seconds'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','ranking'])
    p.add_argument('--run',default='runs/overnight_2026-09-06/tworoom')
    p.add_argument('--cases',default='runs/overnight_2026-09-06/tworoom_preparation/control_cases.json')
    p.add_argument('--output',default='runs/diagnostics/tworoom_followup_2026-09-06/action_ranking')
    p.add_argument('--count',type=int,default=8);p.add_argument('--device',default='cuda',choices=['cpu','cuda'])
    args=p.parse_args()
    if args.stage=='prepare':prepare(args.run,args.cases,args.output,args.count)
    else:ranking(args.output,args.device)
