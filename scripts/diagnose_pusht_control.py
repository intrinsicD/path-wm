"""Frozen saved-checkpoint control and simulator-grounded action-plan diagnosis.

No training, buffer replacement or filtering by outcomes. Each completed command
must run through run.py so its raw ledger refreshes the offline dashboard.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import h5py
import numpy as np
import torch
from scipy.stats import rankdata

from world_model.data import preprocess_pixels
from world_model.eval_pusht import PushT, evaluate_case
from world_model.model import build_model
from world_model.planning import cem, latent_cost
from world_model.train import write_json

PILOT = Path('runs/diagnostics/pusht_broader_pilot')
DEFAULT_OUTPUT = Path('runs/diagnostics/pusht_control_diagnosis')
SETTINGS = dict(samples=300, iterations=30, elites=30, budget=50, frameskip=5)


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def freeze_training_cases(lengths, offsets, population, source_ids, count=20, seed=42):
    lengths, offsets = np.asarray(lengths), np.asarray(offsets)
    eligible = [int(e) for e in population if lengths[e] > 25]
    if len(set(eligible)) != len(eligible) or len(eligible) < count:
        raise ValueError('Need enough distinct eligible training episodes')
    rng = np.random.default_rng(seed)
    cases = []
    for episode in rng.choice(eligible, count, replace=False):
        episode = int(episode)
        start = int(rng.integers(0, int(lengths[episode]) - 25))
        cases.append(dict(episode=episode, source_episode=int(source_ids[episode]),
                          start=start, row=int(offsets[episode]) + start))
    return cases


def action_blocks(raw, stats):
    raw = np.asarray(raw, dtype=np.float32)
    if raw.shape != (25, 2) or not np.isfinite(raw).all():
        raise ValueError('Expected 25 finite 2D actions')
    mean, std = np.asarray(stats['mean'], np.float32), np.asarray(stats['std'], np.float32)
    if np.any(std <= 0):
        raise ValueError('Invalid action scale')
    return ((raw - mean) / std).reshape(5, 10)


def raw_actions(blocks, stats):
    raw = np.asarray(blocks, dtype=np.float32).reshape(25, 2).copy()
    raw *= np.asarray(stats['std'])
    raw += np.asarray(stats['mean'])
    return raw


def ranking_metrics(costs, distances):
    costs, distances = np.asarray(costs), np.asarray(distances)
    if costs.ndim != 1 or costs.shape != distances.shape or not len(costs):
        raise ValueError('Rankings require matched nonempty candidate vectors')
    if not np.isfinite(costs).all() or not np.isfinite(distances).all():
        raise ValueError('Nonfinite ranking values')
    a, b = rankdata(costs), rankdata(distances)
    correlation = float(np.corrcoef(a, b)[0, 1]) if np.std(a) and np.std(b) else None
    selected = int(np.argmin(costs))
    return dict(spearman=correlation, selected_index=selected,
                selected_distance=float(distances[selected]), best_distance=float(distances.min()),
                regret=float(distances[selected] - distances.min()))


def prepare(output):
    meta = json.loads((PILOT / 'manifest.json').read_text())
    released = json.loads(Path('runs/diagnostics/pusht_broader_pilot_control_released/manifest.json').read_text())
    with h5py.File(meta['dataset']['path'], 'r') as data:
        cases = freeze_training_cases(data['ep_len'][:], data['ep_offset'][:],
                                      meta['train_episodes'], meta['episode_split']['source_episode_ids'])
    with h5py.File(meta['episode_split']['source']['path'], 'r') as source:
        for case in cases:
            case['source_row'] = int(source['ep_offset'][case['source_episode']]) + case['start']
    checkpoints = dict(pilot=str(PILOT / 'checkpoint_001000.pt'), released=released['checkpoint'])
    digests = {name: sha256(path) for name, path in checkpoints.items()}
    expected = dict(pilot='1988e4646c81376d065c513a8c3ac33bae6dd77cce8a342b4b35f38f78348d96',
                    released='48938400ae3464c9680731287f583a9cb516f55a8ec64ea13a91be47fb15b607')
    if digests != expected:
        raise ValueError('Saved checkpoint hash differs from completed pilot evidence')
    if sha256(meta['dataset']['path']) != meta['dataset']['subset_sha256']:
        raise ValueError('Pilot source subset digest changed')
    ranking_cases = [dict(**case, population=population, reset_seed=1234+i)
                     for population, selected in [('training', cases), ('heldout', meta['episode_split']['cases'])]
                     for i, case in enumerate(selected[:4])]
    manifest = dict(dataset=meta['dataset'], cases=cases, sampling_seed=42,
                    solver_and_reset_seeds=list(range(1234, 1254)), ranking_cases=ranking_cases,
                    checkpoints=checkpoints, checkpoint_sha256s=digests,
                    random_seed_base=9000, random_sequences=16, candidates_per_case=20,
                    precision='float32', batchnorm='saved buffers, unchanged',
                    goal_offset=25, horizon=5, action_block=5, receding_horizon=5,
                    **SETTINGS, protocol='docs/control-diagnosis-plan.md',
                    code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                    source_manifest_sha256=sha256(PILOT / 'manifest.json'),
                    frozen_protocol_sha256=sha256('docs/control-diagnosis-plan.md'))
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'manifest.json', manifest)
    print(json.dumps(dict(stage='frozen', training_cases=len(cases), ranking_cases=len(ranking_cases),
                          checkpoint_sha256s=digests)), flush=True)


def load_models(manifest, names):
    torch.set_num_threads(4)
    torch.manual_seed(42)
    device = torch.device('cuda')
    models = {}
    for name in names:
        path = manifest['checkpoints'][name]
        if sha256(path) != manifest['checkpoint_sha256s'][name]:
            raise ValueError(f'{name} checkpoint hash changed')
        saved = torch.load(path, weights_only=True, map_location='cpu')
        if name == 'pilot':
            meta = json.loads((PILOT / 'manifest.json').read_text())
            if saved['fingerprint'] != meta['fingerprint']:
                raise ValueError('Pilot checkpoint fingerprint mismatch')
            model, stats = build_model(saved['model_config']), saved['action_stats']
            model.load_state_dict(saved['model'], strict=True)
        else:
            ref = json.loads(Path('runs/diagnostics/reference_full_source/manifest.json').read_text())
            model, stats = build_model(), ref['normalization']['action']
            model.load_state_dict(saved, strict=True)
        models[name] = (model.to(device).eval(), stats)
    return models


def check_hashes(manifest):
    if any(sha256(path) != manifest['checkpoint_sha256s'][name]
           for name, path in manifest['checkpoints'].items()):
        raise RuntimeError('Checkpoint changed during diagnostic')


@torch.inference_mode()
def control(output):
    manifest = json.loads((output / 'manifest.json').read_text())
    model, stats = load_models(manifest, ['pilot'])['pilot']
    directory = output / 'training_control'
    directory.mkdir(exist_ok=False)
    settings = {**manifest, 'checkpoint': manifest['checkpoints']['pilot'],
                'checkpoint_sha256': manifest['checkpoint_sha256s']['pilot'], 'step': 1000,
                'population': '20 distinct training initial-configuration groups', 'action_stats': stats,
                'normalization': 'Pilot saved training-only unbiased action statistics'}
    write_json(directory / 'manifest.json', settings)
    records = []
    with h5py.File(manifest['dataset']['path'], 'r') as data:
        for index, case in enumerate(manifest['cases']):
            row = case['row']
            result = evaluate_case(model, data['pixels'][row], data['pixels'][row+25],
                                   data['state'][row], data['state'][row+25], stats,
                                   seed=manifest['solver_and_reset_seeds'][index], **SETTINGS)
            np.save(directory / f'actions_{index}.npy', result.pop('actions'))
            result.update(case)
            records.append(result)
            with (directory / 'cases.jsonl').open('a') as stream:
                stream.write(json.dumps(result) + '\n')
            print(json.dumps(dict(case=index+1, **{k:v for k,v in result.items() if k != 'plans'})), flush=True)
    check_hashes(manifest)
    summary = dict(successes=sum(r['success'] for r in records), cases=len(records),
                   initial_successes=sum(r['initial_success'] for r in records),
                   mean_steps=float(np.mean([r['steps'] for r in records])),
                   control_seconds=sum(r['seconds'] for r in records), checkpoint_unchanged=True)
    write_json(directory / 'summary.json', summary)
    print(json.dumps(summary), flush=True)


def simulate(state, target, actions, seed):
    env = PushT(resolution=224, relative=True)
    try:
        env.reset(seed=seed)
        env._set_state(np.asarray(state, float))
        env._set_goal_state(np.asarray(target, float))
        initial = bool(env.eval_state(target, env._get_obs())[0])
        frames, states, successes = [env.render().copy()], [env._get_obs().copy()], []
        for step, action in enumerate(actions, 1):
            observation, _, success, _, _ = env.step(action)
            states.append(observation['state'].copy())
            successes.append(bool(success))
            if step % 5 == 0:
                frames.append(env.render().copy())
        final = np.asarray(states[-1])
        angle = abs(float(final[4] - target[4])) % (2*np.pi)
        result = dict(initial_success=initial, success_any=any(successes),
                      success_terminal=successes[-1],
                      position_error=float(np.linalg.norm(final[:4]-target[:4])),
                      angle_error=float(min(angle, 2*np.pi-angle)),
                      state_distance=float(env.eval_state(target, final)[1]))
        return np.stack(frames), np.stack(states), result
    finally:
        env.close()


def encode_frames(model, frames):
    pixels = torch.from_numpy(np.asarray(frames).copy()).permute(0,3,1,2)[None].cuda()
    return model.encode(preprocess_pixels(pixels))


@torch.inference_mode()
def ranking(output):
    manifest = json.loads((output / 'manifest.json').read_text())
    models = load_models(manifest, ['pilot', 'released'])
    directory = output / 'ranking'
    directory.mkdir(exist_ok=False)
    write_json(directory / 'manifest.json', {**manifest, 'cases': manifest['ranking_cases']})
    all_records, all_summaries = [], []
    started = time.monotonic()
    with h5py.File(manifest['dataset']['path'], 'r') as data:
        for case_index, case in enumerate(manifest['ranking_cases']):
            row = case['row']
            source, goal = data['pixels'][row], data['pixels'][row+25]
            state, target = data['state'][row], data['state'][row+25]
            seeds = np.random.default_rng(manifest['random_seed_base'] + case_index)
            candidates = [('replay', data['action'][row:row+25]), ('stationary', np.zeros((25,2), np.float32))]
            candidates += [(f'random_{i:02d}', raw_actions(seeds.standard_normal((5,10)), models['pilot'][1]))
                           for i in range(manifest['random_sequences'])]
            contexts, goals, plan_details = {}, {}, {}
            for name, (model, stats) in models.items():
                contexts[name], goals[name] = encode_frames(model, source[None]), encode_frames(model, goal[None])
                blocks, details = cem(latent_cost(model, contexts[name], goals[name]), 5, 10, torch.device('cuda'),
                                      seed=case['reset_seed'], samples=300, iterations=30, elites=30)
                candidates.append((f'{name}_plan', raw_actions(blocks.cpu().numpy(), stats)))
                plan_details[name] = details
            if len(candidates) != manifest['candidates_per_case']:
                raise ValueError('Candidate count differs from frozen protocol')
            raw = np.stack([actions for _, actions in candidates]).astype(np.float32)
            predictions = {}
            for name, (model, stats) in models.items():
                blocks = torch.from_numpy(np.stack([action_blocks(actions, stats) for actions in raw])).cuda()
                predictions[name] = model.rollout(contexts[name].expand(len(raw),-1,-1), blocks)
            case_records, sim_states, images = [], [], dict(source=source, goal=goal)
            for candidate_index, (kind, actions) in enumerate(candidates):
                frames, states, outcome = simulate(state, target, raw[candidate_index], case['reset_seed'])
                sim_states.append(states)
                if not kind.startswith('random'):
                    images[kind] = frames
                for name, (model, _) in models.items():
                    actual = encode_frames(model, frames)
                    predicted = predictions[name][candidate_index:candidate_index+1]
                    error = (predicted-actual[:,1:]).square().mean(-1)[0]
                    copying = (contexts[name]-actual[:,1:]).square().mean(-1)[0]
                    costs = (predicted-goals[name]).square().sum(-1)[0]
                    actual_costs = (actual[:,1:]-goals[name]).square().sum(-1)[0]
                    record = dict(case_index=case_index, **case, model=name,
                                  candidate_index=candidate_index, candidate=kind, **outcome,
                                  predicted_cost=float(costs[-1]), actual_latent_cost=float(actual_costs[-1]),
                                  predicted_cost_by_step=costs.cpu().tolist(), actual_cost_by_step=actual_costs.cpu().tolist(),
                                  rollout_mse_by_step=error.cpu().tolist(), copy_mse_by_step=copying.cpu().tolist(),
                                  initial_source_sim_mse=float((contexts[name]-actual[:,:1]).square().mean()))
                    case_records.append(record)
            np.savez_compressed(directory / f'case_{case_index}_trajectories.npz', actions=raw,
                                states=np.stack(sim_states), candidate_names=np.array([c[0] for c in candidates]))
            np.savez_compressed(directory / f'case_{case_index}_images.npz', **images)
            for name in models:
                records = [r for r in case_records if r['model']==name]
                metrics = ranking_metrics([r['predicted_cost'] for r in records], [r['position_error'] for r in records])
                selected = records[metrics['selected_index']]
                summary = dict(case_index=case_index, population=case['population'], model=name,
                               candidates=len(records), **metrics, selected_candidate=selected['candidate'],
                               selected_success=selected['success_terminal'], plan=plan_details[name])
                all_summaries.append(summary)
                print(json.dumps(summary), flush=True)
            all_records.extend(case_records)
            with (directory / 'ranking_records.jsonl').open('a') as stream:
                for record in case_records:
                    stream.write(json.dumps(record) + '\n')
    check_hashes(manifest)
    result = dict(schema_version=1, records=len(all_records), case_summaries=all_summaries,
                  checkpoint_unchanged=True, seconds=time.monotonic()-started)
    write_json(directory / 'ranking.json', result)
    print(json.dumps(dict(stage='ranking_complete', records=len(all_records), seconds=result['seconds'])), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare','control','ranking'])
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    globals()[args.stage](args.output)


if __name__ == '__main__':
    main()
