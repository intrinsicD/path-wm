"""Freeze TwoRoom source goals and verify simulator alignment/action controls."""
import argparse
import json
import subprocess
from pathlib import Path
import h5py
import numpy as np
import torch
import yaml
from sklearn.preprocessing import StandardScaler
from world_model.evaluation import sample_source_cases
from world_model.eval_tworoom import _environment, run_actions
from world_model.protocol import prepare_training_data
from world_model.train import write_json
from scripts.diagnose_pusht_control import sha256


def prepare(config, output, goal_offset=25, budget=50, count=50):
    if (goal_offset, budget) not in ((25,50),(100,150)):
        raise ValueError('Use an explicit released-config (25/50) or paper (100/150) protocol')
    cfg = yaml.safe_load(Path(config).read_text())
    ds = yaml.safe_load(Path(cfg['dataset']).read_text())
    if ds['name'] != 'tworoom': raise ValueError('TwoRoom dataset required')
    receipt = json.loads((Path(ds['path']).parent/'extraction.json').read_text())
    if any(receipt[k] != ds[k] for k in ('repo','revision','sha256')):
        raise ValueError('Pinned source extraction receipt differs')
    train, val, _, _, stats, split = prepare_training_data(cfg, ds)
    steps = len(train)//cfg['batch_size']*cfg['epochs']
    if len(train)!=cfg['expected_train_windows'] or steps!=cfg['expected_total_steps']:
        raise ValueError('Prepared TwoRoom source/update budget differs')
    checkpoint = Path('data/reference/lewm-tworooms/weights.pt')
    digest = sha256(checkpoint)
    if digest != '566f223624ea4bfb39dbfe6ae731198dd6ea73b7b8919fed6b1ecafca810f7dd':
        raise ValueError('Released TwoRoom checkpoint hash differs from Hub LFS metadata')
    released = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if tuple(released['predictor.pos_embedding'].shape) != (1,3,192):
        raise ValueError('Released model history/dimension changed')
    torch.set_num_threads(2)
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    with h5py.File(ds['path'],'r') as data:
        lengths, offsets = data['ep_len'][:], data['ep_offset'][:]
        cases = sample_source_cases(lengths, offsets, count=count, seed=42, goal_offset=goal_offset)
        raw = data['action'][:]; finite = raw[np.isfinite(raw).all(1)]
        scaler = StandardScaler().fit(finite)
        reference_stats = dict(mean=scaler.mean_.tolist(), std=scaler.scale_.tolist(), count=len(finite))
        control = dict(dataset=ds, cases=cases, sampling_seed=42,
            solver_and_reset_seeds=list(range(1234,1234+count)),
            population='Full TwoRoom source; shared with training and released-model data',
            protocol=('Pinned released evaluation config' if goal_offset==25 else 'Paper Appendix F goal/budget') + '; deterministic paired local evaluation',
            samples=300, iterations=30, elites=30, horizon=5, action_block=5,
            receding_horizon=5, goal_offset=goal_offset, budget=budget, precision='float32',
            success_threshold=None, environment_commit='6f1e499e9cc0c898d326112f485c1062c3d20f24',
            simulator_success='Post-action Euclidean agent distance <16 pixels; initial successes retained and counted',
            released_reference=dict(checkpoint=str(checkpoint), checkpoint_sha256=digest,
                model_revision='77adaae0bc31deab21c93740d1f8bb947cd0bdec',
                model_config=dict(history=3), action_stats=reference_stats,
                normalization='Full-source finite-action sklearn StandardScaler population statistics'))
        write_json(output/'control_cases.json', control)
        write_json(output/'manifest.json', {**control, 'source_verification':receipt,
            'training_config':cfg, 'data_protocol':split, 'action_stats':stats,
            'train_windows':len(train), 'val_windows':len(val), 'total_steps':steps,
            'control_cases_sha256':sha256(output/'control_cases.json'),
            'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()})
        alignment = []; records = []
        for i, case in enumerate(cases):
            row = case['row']; state = data['proprio'][row]; target = data['proprio'][row+goal_offset]
            actions = data['action'][row:row+goal_offset]
            env = _environment(state, target, 1234+i)
            error = np.abs(env.render().astype(float)-data['pixels'][row].astype(float))
            differences = []
            for t, action in enumerate(actions):
                env.step(action)
                differences.append(float(np.max(np.abs(env.agent_position.numpy()-data['proprio'][row+t+1]))))
            env._set_state(target)
            goal_error = np.abs(env.render().astype(float)-data['pixels'][row+goal_offset].astype(float))
            env.close()
            alignment.append(dict(**case, initial_render_max=float(error.max()),
                goal_render_max=float(goal_error.max()), transition_max=max(differences), transitions=len(actions)))
            for kind, sequence in [('stationary', np.zeros((budget,2),np.float32)), ('recorded_replay',actions)]:
                result = run_actions(state,target,sequence,budget=budget,seed=1234+i)
                records.append(dict(**case,kind=kind,**result))
        write_json(output/'source_alignment.json',dict(records=alignment,
            cases=count,transitions=sum(x['transitions'] for x in alignment),
            all_exact=all(x['initial_render_max']==x['goal_render_max']==x['transition_max']==0 for x in alignment)))
        summary = {kind:dict(successes=sum(x['success'] for x in records if x['kind']==kind),
            cases=count,initial_successes=sum(x['initial_success'] for x in records if x['kind']==kind))
            for kind in ('stationary','recorded_replay')}
        write_json(output/'action_baselines.json',dict(records=records,summary=summary))
    exact = all(x['initial_render_max']==x['goal_render_max']==x['transition_max']==0 for x in alignment)
    print(json.dumps(dict(cases=count,goal_offset=goal_offset,budget=budget,alignment_exact=exact,
        train_windows=len(train),val_windows=len(val),schedule_steps=steps,action_baselines=summary)),flush=True)
    if not exact: raise RuntimeError('Source geometry/dynamics differ; do not interpret learned control')


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('config');p.add_argument('output')
    p.add_argument('--goal-offset',type=int,default=25);p.add_argument('--budget',type=int,default=50)
    p.add_argument('--cases',type=int,default=50)
    a=p.parse_args();prepare(a.config,a.output,a.goal_offset,a.budget,a.cases)
