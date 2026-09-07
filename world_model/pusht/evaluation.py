"""Matched PushT predictions and image-goal control, with immutable raw evidence.

The harness alone owns reset poses and future replay actions. Learned CEM receives
only encoded images, observed-action memory, and an image-derived GoalPose.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch

from world_model.paddle.types import ObservationLatent, PlanningState


SCHEMA = 'pusht-eup-evaluation-v1'
CASE_PROTOCOL = 'current-simulator-replay-goal-k5-block-pose-v1'
CONTROLLERS = ('learned', 'reset', 'hold', 'repeat_last', 'random', 'replay')
EVALUATION_DEFAULTS = dict(split='test', control_cases=20, prediction_windows=256,
    control_max_steps=50, candidates=64, iterations=4, elites=8, horizon=5,
    controllers=list(CONTROLLERS))


def _plain(value):
    if isinstance(value, dict): return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [_plain(v) for v in value]
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, torch.Tensor): return value.detach().cpu().tolist()
    if isinstance(value, Path): return str(value)
    return value


def _hash(value):
    return hashlib.sha256(json.dumps(_plain(value), sort_keys=True, allow_nan=False).encode()).hexdigest()


def _file_hash(path):
    with Path(path).open('rb') as source: return hashlib.file_digest(source, 'sha256').hexdigest()


def _json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(_plain(value), sort_keys=True, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def _journal(path, identity, produce):
    path = Path(path)
    if path.exists():
        saved = json.loads(path.read_text())
        if saved['identity'] != identity or saved['checksum'] != _hash(saved['payload']):
            raise ValueError(f'Incompatible or corrupted PushT journal: {path}')
        return saved['payload']
    payload = _plain(produce())
    _json(path, dict(identity=identity, checksum=_hash(payload), payload=payload))
    return payload


def _export(directory, name, rows):
    directory = Path(directory); _json(directory / f'{name}.json', rows)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = directory / f'{name}.csv.tmp'
    with temporary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in row.items()})
    temporary.replace(directory / f'{name}.csv')


def _rgb(frames, device):
    return torch.as_tensor(np.asarray(frames).copy(), device=device).permute(0, 3, 1, 2).float() / 255


def _start(batch, device):
    return torch.full((batch, 2), -1., device=device)


def _sync(device):
    if torch.device(device).type == 'cuda': torch.cuda.synchronize(device)


@torch.inference_mode()
def assimilate_history(system, frames, actions, device='cpu'):
    if len(frames) != len(actions) + 1 or not len(frames):
        raise ValueError('History needs T+1 frames and T previous actions')
    memory = torch.zeros(1, 128, device=device)
    for i, frame in enumerate(frames):
        observation = system['E'](_rgb(frame[None], device))
        action = _start(1, device) if i == 0 else torch.as_tensor(actions[i-1], device=device).float()[None]
        memory = system['U'](memory, observation, action)
    return PlanningState(observation, memory)


@torch.inference_mode()
def matched_rollout(system, initial, actions):
    """Target-free P→U rollout; copy and reset receive identical actual actions."""
    if actions.ndim != 3 or actions.shape[-1] != 2:
        raise ValueError('Actions need [batch,horizon,2]')
    if not torch.isfinite(actions).all() or ((actions < 0) | (actions > 1)).any():
        raise ValueError('Invalid executable actions')
    fresh = system['U'](torch.zeros_like(initial.memory), initial.observation,
                        _start(len(actions), initial.memory.device))
    current = dict(prediction=initial.clone(), copy=initial.clone(),
                   reset=PlanningState(initial.observation.clone(), fresh))
    result = {name: [] for name in current}
    for action in actions.unbind(1):
        for name, state in current.items():
            observation = initial.observation.clone() if name == 'copy' else system['P'](state.observation, state.memory, action)
            current[name] = PlanningState(observation, system['U'](state.memory, observation, action))
            result[name].append(current[name])
    return result


def _angle_error(a, b):
    return abs(float(np.arctan2(np.sin(a-b), np.cos(a-b))))


def _block_error(pose, goal):
    distance = float(np.linalg.norm(np.asarray(pose)[2:4] - np.asarray(goal)[2:4]))
    angle = _angle_error(pose[4], goal[4])
    return dict(block_distance=distance, angle_error=angle,
                pusher_distance=float(np.linalg.norm(np.asarray(pose)[:2] - np.asarray(goal)[:2])),
                success=bool(distance < 20 and angle < math.pi/9))


def physical_errors(h_pose, r_state, truth_pose, truth_motion, motion_scales=None):
    h_pose, r_state, truth_pose = [np.asarray(x, dtype=float) for x in (h_pose, r_state, truth_pose)]
    if h_pose.shape != (6,) or r_state.shape != (11,) or truth_pose.shape != (5,):
        raise ValueError('Physical readouts require H6, R11 and truth pose5')
    if not all(np.isfinite(x).all() for x in (h_pose, r_state, truth_pose)):
        raise FloatingPointError('Nonfinite PushT readout')
    result = {}
    for label, value in (('h', h_pose), ('r', r_state)):
        theta = math.atan2(value[4], value[5])
        result.update({f'{label}_position_abs_error': np.abs(value[:4]*512 - truth_pose[:4]).tolist(),
                       f'{label}_angle_abs_error_deg': math.degrees(_angle_error(theta, truth_pose[4])),
                       f'{label}_orientation_norm': float(np.linalg.norm(value[4:6]))})
    scale = np.asarray(motion_scales if motion_scales is not None else [512., 512., 512., 512., math.pi])
    result['r_motion_abs_error'] = None if truth_motion is None else np.abs((r_state[6:]-np.asarray(truth_motion))*scale).tolist()
    return result


def summarize_prediction_records(records):
    result = {}
    for method in sorted({row['method'] for row in records}):
        result[method] = {}
        for horizon in sorted({row['horizon'] for row in records if row['method'] == method}):
            rows = [r for r in records if r['method'] == method and r['horizon'] == horizon]
            summary = dict(count=len(rows))
            for raw, output in (('h_position_abs_error', 'h_position'), ('r_position_abs_error', 'r_position'),
                                ('h_angle_abs_error_deg', 'h_angle_deg'), ('r_angle_abs_error_deg', 'r_angle_deg')):
                values = np.asarray([r[raw] for r in rows])
                for suffix, value in (('mae', values.mean(0)), ('p95', np.quantile(values, .95, axis=0)), ('max', values.max(0))):
                    key = f'{output}_{suffix}'
                    if output.endswith('_deg'): key = output[:-4] + '_' + suffix + '_deg'
                    summary[key] = _plain(value)
            motions = [r['r_motion_abs_error'] for r in rows if r.get('r_motion_abs_error') is not None]
            summary['motion_count'] = len(motions)
            summary['r_motion_mae'] = np.asarray(motions).mean(0).tolist() if motions else None
            for field in ('latent_error', 'image_mse'):
                values = [r[field] for r in rows if r.get(field) is not None]
                summary[field] = float(np.mean(values)) if values else None
            result[method][str(horizon)] = summary
    return result


def select_case_indices(candidates, *, split='test', count=20, seed=3107):
    rows = [r for r in candidates if r['split'] == split]
    eligible = [r for r in rows if r['block_distance'] >= 40 or r['angle_error'] >= 2*math.pi/9]
    groups = {}
    for row in sorted(eligible, key=lambda r: (r['index'], r['episode'])):
        groups.setdefault(row['group_id'], row)
    ids = sorted(groups)
    chosen = np.random.default_rng(seed).choice(ids, min(count, len(ids)), replace=False) if ids else []
    selected = [groups[int(group)] for group in sorted(chosen)]
    return selected, dict(split=split, candidate_windows=len(rows), eligible_windows=len(eligible),
        eligible_groups=len(ids), selected_cases=len(selected), requested_cases=count,
        selection='seeded groups; earliest index at least twice a success tolerance per group',
        initially_solved_windows=sum(r['block_distance'] < 20 and r['angle_error'] < math.pi/9 for r in rows),
        selected_initially_solved=sum(r['block_distance'] < 20 and r['angle_error'] < math.pi/9 for r in selected),
        block_distance_threshold=40, angle_threshold=2*math.pi/9)


def prepare_cases(dataset, output, *, count=20, seed=3107, split='test', env_factory=None):
    """Replay each episode once; future poses/frames remain oracle preparation."""
    if env_factory is None:
        from .env import PushTEnv
        env_factory = PushTEnv
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    candidates, histories = [], {}
    for ep in range(len(dataset)):
        episode = dataset[ep]; env = env_factory()
        try:
            frame, _ = env.reset(pose5_world=episode['poses_world'][0], seed=seed + int(episode['source_episode']))
            frames, poses = [frame], [env.pose]
            for action in episode['actions']:
                frame, _, _, _, _ = env.step(action); frames.append(frame); poses.append(env.pose)
            histories[ep] = (frames, poses)
            for index in range(2, len(frames)-5):
                candidates.append(dict(episode=ep, group_id=int(episode['group_id']), index=index,
                    split=split, **_block_error(poses[index], poses[index+5])))
        finally:
            env.close()
    chosen, audit = select_case_indices(candidates, split=split, count=count, seed=seed)
    public, oracle = [], []
    for row in chosen:
        ep, index = row['episode'], row['index']; episode = dataset[ep]
        frames, poses = histories[ep]
        identity = f'{split}-e{int(episode["source_episode"]):04d}-i{index:04d}'
        goal_path = output / f'{identity}-goal.npy'
        with goal_path.open('wb') as stream: np.save(stream, frames[index+5], allow_pickle=False)
        public.append(dict(case_id=identity, source_episode=int(episode['source_episode']),
            group_id=row['group_id'], start_index=index, seed=seed+int(episode['source_episode']),
            reset_pose=episode['poses_world'][0].tolist(), prefix_actions=episode['actions'][:index].tolist(),
            goal_frame=str(goal_path.resolve()), goal_sha256=_file_hash(goal_path)))
        oracle.append(dict(case_id=identity, goal_pose=np.asarray(poses[index+5]).tolist(),
            initial_pose=np.asarray(poses[index]).tolist(), future_actions=episode['actions'][index:index+5].tolist(),
            expected_goal_frame_sha256=_hash(frames[index+5]), initial_error=_block_error(poses[index], poses[index+5])))
    return dict(public=public, oracle=oracle, audit=audit, eligibility_records=candidates)


@torch.inference_mode()
def run_control_case(system, public_case, oracle_case, controller, config, *, env_factory=None,
                     planner_factory=None, device='cpu'):
    if controller not in CONTROLLERS: raise ValueError('Unknown PushT controller')
    if env_factory is None:
        from .env import PushTEnv
        env_factory = PushTEnv
    if planner_factory is None:
        from .planner import CEMPlanner
        planner_factory = CEMPlanner
    from .planner import PlanningFailure
    env = env_factory(); poses = []; traces = []; actions = []; latencies = []; failures = []
    prefix = public_case['prefix_actions']; goal_pose = oracle_case['goal_pose']
    try:
        frame, _ = env.reset(pose5_world=public_case['reset_pose'], seed=public_case['seed'])
        frames = [frame]
        for action in prefix:
            frame, _, _, _, _ = env.step(action); frames.append(frame)
        if 'initial_pose' in oracle_case and not np.allclose(env.pose, oracle_case['initial_pose'], rtol=0, atol=1e-7):
            raise ValueError('Reconstructed physical prefix differs from frozen case')
        env.set_goal(goal_pose)
        actual = assimilate_history(system, frames, prefix, device)
        initial = _block_error(env.pose, goal_pose)
        poses.append(np.asarray(env.pose).tolist())
        goal_frame = public_case['goal_frame']
        if isinstance(goal_frame, (str, Path)):
            if public_case.get('goal_sha256') and _file_hash(goal_frame) != public_case['goal_sha256']:
                raise ValueError('Goal image checksum changed')
            goal_frame = np.load(goal_frame, allow_pickle=False)
        proposal = {'initial_std': np.clip(system['action_offset_rms'], .01, .2).tolist()} if 'action_offset_rms' in system else {}
        planner = planner_factory(system['P'], system['U'], system['H'],
            candidates=int(config.get('candidates', 64)), iterations=int(config.get('iterations', 4)),
            elites=int(config.get('elites', 8)), horizon=5, seed=int(public_case['seed']), **proposal)
        goal = None; success = initial['success']; any_success = success; predictor_transitions = 0
        if controller in ('learned', 'reset'):
            try: goal = planner.prepare_goal(system['E'](_rgb(goal_frame[None], device)))
            except PlanningFailure as error: failures.append(str(error))
        rng = np.random.default_rng(public_case['seed'])
        budget = int(config.get('control_max_steps', 50))
        if controller == 'replay': budget = min(budget, len(oracle_case['future_actions']))
        hold = np.asarray(_plain(system['H'](actual.observation)[0]), dtype=np.float32)[:2].clip(0, 1)
        for step in range(budget):
            if failures: break
            state = actual
            if controller == 'reset':
                state = PlanningState(actual.observation, system['U'](torch.zeros_like(actual.memory),
                    actual.observation, _start(1, device)))
            _sync(device); tick = time.perf_counter(); imagined_states = []
            try:
                if controller in ('learned', 'reset'):
                    plan = planner.plan(state, goal)
                    action = np.asarray(_plain(plan.action), dtype=np.float32)
                    predictor_transitions += int(plan.stats.get('predictor_transitions', 0))
                    imagined_states = plan.states
                elif controller == 'random': action = rng.uniform(0, 1, 2).astype(np.float32)
                elif controller == 'hold': action = hold.copy()
                elif controller == 'repeat_last': action = np.asarray(prefix[-1],dtype=np.float32)
                else: action = np.asarray(oracle_case['future_actions'][step], dtype=np.float32)
            except PlanningFailure as error:
                failures.append(str(error)); predictor_transitions += int(error.stats.get('predictor_transitions', 0)); break
            _sync(device); latencies.append((time.perf_counter()-tick)*1000)
            # Every policy crosses the same explicit FP32 observer boundary.
            action = np.asarray(action, dtype=np.float32)
            if action.shape != (2,): raise ValueError('Controller must execute exactly one primitive XY action')
            imagined = [dict(h_pose=_plain(system['H'](s.observation)[0]),
                             r_state=_plain(system['R'](s.memory)[0])) for s in imagined_states]
            traces.append(dict(step=step, action=action.tolist(), h_pose=_plain(system['H'](actual.observation)[0]),
                               real_r=_plain(system['R'](actual.memory)[0]), imagined=imagined))
            frame, _, _, truncated, _ = env.step(action)
            actions.append(action.tolist()); poses.append(np.asarray(env.pose).tolist())
            observation = system['E'](_rgb(frame[None], device))
            actual = PlanningState(observation, system['U'](actual.memory, observation,
                                  torch.as_tensor(action, device=device, dtype=torch.float32)[None]))
            success = _block_error(env.pose, goal_pose)['success']
            any_success = any_success or success
            if truncated: break
        final = _block_error(env.pose, goal_pose)
        replay_verified = None
        if controller == 'replay':
            replay_verified = bool(success)
            if not replay_verified: raise ValueError('Privileged replay failed a constructed reachable goal')
        return dict(case_id=public_case['case_id'], controller=controller, success=bool(success),
            any_time_success=bool(any_success), evaluated_budget=budget,
            budget_scope='five-interval privileged reachability' if controller=='replay' else 'fixed-budget final controller state',
            initially_solved=initial['success'], episode_length=len(actions), final=final,
            predictor_transitions=predictor_transitions, planning_failures=len(failures), failure_messages=failures,
            decision_latency_ms=latencies, actions=actions, poses=poses, observations=traces,
            replay_verified=replay_verified, source_episode=public_case['source_episode'],
            group_id=public_case['group_id'], start_index=public_case['start_index'])
    finally: env.close()


def summarize_control_records(records):
    result = {}
    for name in sorted({r['controller'] for r in records}):
        rows = [r for r in records if r['controller'] == name]
        latency = [v for row in rows for v in row['decision_latency_ms']]
        unsolved = [r for r in rows if not r['initially_solved']]
        result[name] = dict(count=len(rows), successes=sum(r['success'] for r in rows),
            success_rate=sum(r['success'] for r in rows)/len(rows), initially_solved=sum(r['initially_solved'] for r in rows),
            any_time_successes=sum(r['any_time_success'] for r in rows),
            any_time_success_rate=sum(r['any_time_success'] for r in rows)/len(rows),
            initially_unsolved_count=len(unsolved), newly_solved=sum(r['success'] for r in unsolved),
            mean_episode_length=float(np.mean([r['episode_length'] for r in rows])),
            planning_failures=sum(r['planning_failures'] for r in rows),
            latency_median_ms=float(np.median(latency)) if latency else None,
            latency_p95_ms=float(np.quantile(latency,.95)) if latency else None,
            failure_case_ids=[r['case_id'] for r in rows if not r['success']])
    return result


@torch.inference_mode()
def _encode_episode(system, episode, device):
    frames = episode['frames']
    observation = ObservationLatent.cat([system['E'](_rgb(frames[i:i+128], device)) for i in range(0, len(frames), 128)])
    memory = torch.zeros(1, 128, device=device); memories = []
    for index in range(len(frames)):
        previous = _start(1, device) if index == 0 else torch.as_tensor(episode['actions'][index-1], device=device)[None]
        memory = system['U'](memory, observation[index], previous)
        memories.append(memory)
    return observation, torch.cat(memories)


@torch.inference_mode()
def prediction_diagnostics(system, dataset, directory, identity, *, window_limit=256, seed=3107,
                           episode_limit=None, device='cpu'):
    available = min(len(dataset), int(episode_limit)) if episode_limit is not None else len(dataset)
    candidates = [(int(ep), int(i)) for ep, i in dataset.window_indices(5, min_history=2) if ep < available]
    if not candidates: raise ValueError('No held-out five-transition window after warm-up')
    indices = np.random.default_rng(seed).choice(len(candidates), min(window_limit, len(candidates)), replace=False)
    chosen = [candidates[int(i)] for i in sorted(indices)]
    vf = max(float(system['statistics']['v_fine']), 1e-6)
    vc = max(float(system['statistics']['v_coarse']), 1e-6)
    records = []
    for ep in range(available):
        episode = dataset[ep]
        selected = [i for e, i in chosen if e == ep]
        def produce():
            observations, memories = _encode_episode(system, episode, device)
            local = []
            def error(state, target_index, method, horizon, source_index):
                target = observations[target_index]
                h = _plain(system['H'](state.observation)[0]); r = _plain(system['R'](state.memory)[0])
                rgb = system['D'](state.observation)
                truth_motion = episode['motion_targets'][target_index] if episode['motion_mask'][target_index].all() else None
                latent = .5*((state.observation.fine-target.fine).square().mean()/vf
                              +(state.observation.coarse-target.coarse).square().mean()/vc)
                image_mse = (rgb-_rgb(episode['frames'][target_index:target_index+1], device)).square().mean()
                if not torch.isfinite(latent) or not torch.isfinite(image_mse):
                    raise FloatingPointError('Nonfinite PushT prediction diagnostic')
                return dict(episode=int(episode['source_episode']), group_id=int(episode['group_id']),
                    source_index=source_index, frame_index=target_index, method=method, horizon=horizon,
                    h_pose=h, r_state=r, truth_pose=episode['poses_world'][target_index].tolist(),
                    latent_error=float(latent), image_mse=float(image_mse),
                    **physical_errors(h, r, episode['poses_world'][target_index], truth_motion, system.get('motion_scales')))
            for t in range(len(episode['frames'])):
                local.append(error(PlanningState(observations[t], memories[t:t+1]), t, 'actual', 0, t))
            for source in selected:
                initial = PlanningState(observations[source], memories[source:source+1])
                actions = torch.as_tensor(episode['actions'][source:source+5], device=device)[None]
                rollouts = matched_rollout(system, initial, actions)
                for method, states in rollouts.items():
                    local.extend(error(state, source+k, method, k, source) for k, state in enumerate(states, 1))
            return local
        records.extend(_journal(Path(directory)/'diagnostics'/f'episode_{ep:04d}.json',
                                _hash(dict(context=identity, episode=ep, windows=selected)), produce))
    return records, chosen


@torch.inference_mode()
def render_prediction_visuals(system, dataset, windows, output, device='cpu'):
    """Actual, decoded-real and imagined rows use the same recorded actions."""
    from PIL import Image, ImageDraw
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    panels = []
    for ep, start in windows[:2]:
        episode = dataset[ep]; observation, memory = _encode_episode(system, episode, device)
        initial = PlanningState(observation[start], memory[start:start+1])
        states = matched_rollout(system, initial,
            torch.as_tensor(episode['actions'][start:start+5], device=device)[None])['prediction']
        imagined = [initial, *states]
        def decoded(s):
            return Image.fromarray((system['D'](s)[0].clamp(0,1).permute(1,2,0).cpu().numpy()*255).round().astype(np.uint8))
        canvas = Image.new('RGB', (6*128+100, 3*128+90), 'white'); draw = ImageDraw.Draw(canvas)
        for row, label in enumerate(('Actual', 'D(real S)', 'D(imagined S)')):
            draw.text((5, 30+row*128), label, fill='black')
        gif = []
        scale = np.asarray(system.get('motion_scales', [512]*4+[math.pi]))
        for k in range(6):
            actual = Image.fromarray(episode['frames'][start+k])
            reconstruction = decoded(observation[start+k]); prediction = decoded(imagined[k].observation)
            draw.text((100+k*128, 5), f'h={k}, frame {start+k}', fill='black')
            for row, picture in enumerate((actual, reconstruction, prediction)):
                canvas.paste(picture.resize((128,128)), (100+k*128, 25+row*128))
            real_r = np.asarray(_plain(system['R'](memory[start+k:start+k+1])[0]))[6:]*scale
            imagined_r = np.asarray(_plain(system['R'](imagined[k].memory)[0]))[6:]*scale
            slide = Image.new('RGB', (576,270), 'white'); sd = ImageDraw.Draw(slide)
            for column, (label, picture) in enumerate(zip(('Actual', 'Reconstruction', 'Imagined'), (actual,reconstruction,prediction))):
                sd.text((column*192+5,5), label, fill='black'); slide.paste(picture.resize((192,192)),(column*192,24))
            sd.text((5,221), f'h={k}; real R motion {np.array2string(real_r,precision=2)}',fill='black')
            sd.text((5,237), f'imagined R motion {np.array2string(imagined_r,precision=2)}',fill='black')
            sd.text((5,253), 'dx pusher/block: world units; dangle: rad, per 0.1s interval',fill='black')
            gif.append(slide)
        draw.text((5,418), 'Matched source/actions. R after actual and imagined observations is separate in the GIF.', fill='black')
        action_caption = ', '.join(f'({float(x):.2f}, {float(y):.2f})'
                                   for x,y in episode['actions'][start:start+5])
        draw.text((5,438), f'source episode {int(episode["source_episode"])}; start {start}; actions {action_caption}', fill='black')
        name=f'episode_{int(episode["source_episode"]):04d}_frame_{start:04d}'
        png, animation = output/f'{name}.png', output/f'{name}.gif'
        canvas.save(png); gif[0].save(animation,save_all=True,append_images=gif[1:],duration=600,loop=0)
        panels.append(dict(episode=int(episode['source_episode']), start=start, png=str(png.resolve()),gif=str(animation.resolve()),
            action_alignment='actual/reconstruction/prediction share identical recorded five-action prefix'))
    return panels


def evaluate(config, data, perception, memory, predictor, output):
    """Run or resume raw cases/diagnostics; plot only after authoritative exports."""
    if isinstance(config, (str, Path)):
        import yaml
        config = yaml.safe_load(Path(config).read_text())
    config = dict(config); settings = {**EVALUATION_DEFAULTS, **config.get('evaluation', {})}
    if settings['horizon'] != 5 or settings['split'] not in ('validation', 'test'):
        raise ValueError('PushT evaluation requires K5 and a held-out split')
    if set(settings['controllers'])-set(CONTROLLERS): raise ValueError('Unknown evaluation controller')
    from .data import EpisodeDataset
    from .checkpoints import load_system, read_checkpoint, TENSOR_SCHEMA
    dataset = EpisodeDataset(data, settings['split'])
    # Validate the requested population before constructing any models or physics.
    # A mutually compatible checkpoint trio may still belong to different data.
    a = read_checkpoint(perception, stage='perception', dataset_fingerprint=dataset.fingerprint)
    b = read_checkpoint(memory, stage='memory', dataset_fingerprint=dataset.fingerprint,
                        dependencies={'perception':a['model_fingerprint']})
    c = read_checkpoint(predictor, stage='predictor', dataset_fingerprint=dataset.fingerprint,
                        dependencies={'perception':a['model_fingerprint'],'memory':b['model_fingerprint']})
    if any(saved.get('normalization') != dataset.manifest['normalization'] for saved in (a,b,c)):
        raise ValueError('Evaluation normalization differs from trained checkpoint normalization')
    if c.get('horizon') != 5:
        raise ValueError('This evaluation requires a trained five-step K5 predictor')
    statistics = c.get('statistics',{})
    expected_statistics = dict(encoder_fingerprint=a['model_fingerprint'],dataset_fingerprint=dataset.fingerprint,
                               tensor_schema=TENSOR_SCHEMA,split='train',precision='float32')
    if any(statistics.get(k) != v for k,v in expected_statistics.items()):
        raise ValueError('Evaluation latent statistics have incompatible encoder, dataset, split or schema')
    count, lengths = statistics.get('count'),statistics.get('lengths',[])
    if type(count) is not int or count <= 0 or not lengths or any(type(n) is not int or n<=0 for n in lengths) or count != sum(lengths):
        raise ValueError('Evaluation latent statistic count and training lengths disagree')
    if any(not isinstance(statistics.get(k),(int,float)) or not math.isfinite(statistics[k]) or statistics[k]<0
           for k in ('v_fine','v_coarse')):
        raise ValueError('Evaluation latent variances must be finite and nonnegative')
    requested = config.get('device', 'auto')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if requested == 'auto' else torch.device(requested)
    torch.set_num_threads(int(config.get('cpu_threads', 4)))
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[2]
    dependencies = ('world_model/pusht/evaluation.py','world_model/pusht/models.py','world_model/pusht/env.py',
                    'world_model/pusht/planner.py','world_model/pusht/data.py','world_model/paddle/models.py',
                    'world_model/paddle/types.py','world_model/paddle/rollout.py','third_party/swm/pusht.py',
                    'third_party/swm/spaces.py','third_party/swm/draw.py')
    code = {name:_file_hash(root/name) for name in dependencies}
    normalization = dataset.manifest.get('normalization', {})
    protocol = dict(schema_version=SCHEMA, task=CASE_PROTOCOL, config=config, settings=settings,
        dataset_fingerprint=dataset.fingerprint, checkpoints={k:dict(path=str(Path(v).resolve()),sha256=_file_hash(v))
        for k,v in [('perception',perception),('memory',memory),('predictor',predictor)]}, code=code,
        device=str(device), dtype='float32', normalization=dataset.manifest.get('normalization'),
        proposal_std_raw=normalization.get('action_offset_rms'),
        proposal_std_resolved=np.clip(normalization['action_offset_rms'],.01,.2).tolist() if normalization.get('action_offset_rms') is not None else [.2,.2],
        latency_scope='synchronized decision including all CEM candidates; excludes rendering and real-frame encoding',
        goal_protocol='current simulator five-step replay; earliest block goal at least2x one success tolerance per held-out group; final fixed-budget outcome primary, any-time secondary; replay5-only separate',
        motion_units='causal backward displacements: four world units and wrapped radians per 0.1-second interval; not instantaneous velocity')
    identity = _hash(protocol)
    existing = output/'protocol.json'
    if existing.exists() and _hash(json.loads(existing.read_text())) != identity:
        raise ValueError('Evaluation output belongs to different dependencies/configuration; use a new directory')
    _json(existing, protocol)
    report = dict(schema_version=SCHEMA,status='running',raw_status='running',smoke=bool(config.get('smoke',False)),
                  protocol=protocol,prediction={},control={},visuals=[])
    try:
        system = load_system(perception,memory,predictor,device)
        normalization = dataset.manifest.get('normalization', {})
        if normalization.get('motion_scales') is not None: system['motion_scales'] = normalization['motion_scales']
        if normalization.get('action_offset_rms') is not None: system['action_offset_rms'] = normalization['action_offset_rms']
        for module in ('E','D','H','U','R','P'): system[module].eval()
        cases = _journal(output/'case_manifest.json',identity,
            lambda:prepare_cases(dataset,output/'goals',count=int(settings['control_cases']),
                                 seed=int(config.get('seed',3107)),split=settings['split']))
        for case in cases['public']:
            if _file_hash(case['goal_frame']) != case['goal_sha256']: raise ValueError('Frozen goal image changed')
        _export(output,'case_eligibility_records',cases['eligibility_records'])
        report['case_selection'] = cases['audit']
        prediction, windows = prediction_diagnostics(system,dataset,output,identity,
            window_limit=int(settings['prediction_windows']),seed=int(config.get('seed',3107)),
            episode_limit=settings.get('diagnostic_episodes',2 if config.get('smoke') else None),device=device)
        _export(output,'prediction_records',prediction)
        report['prediction'] = summarize_prediction_records(prediction)
        controls=[]
        for public, oracle in zip(cases['public'],cases['oracle']):
            if public['case_id'] != oracle['case_id']: raise ValueError('Public/oracle case identities disagree')
            for controller in settings['controllers']:
                key=_hash(dict(context=identity,public=public,oracle=oracle,controller=controller))
                row=_journal(output/'cases'/f'{public["case_id"]}-{controller}.json',key,
                    lambda:run_control_case(system,public,oracle,controller,settings,device=device))
                controls.append(row)
        _export(output,'control_records',controls)
        report['control']=summarize_control_records(controls)
        report.update(raw_status='completed',status='report_pending')
        _json(output/'metrics.json',report)
        report['visuals']=render_prediction_visuals(system,dataset,windows,output/'visuals',device)
        report.update(status='completed')
        _json(output/'metrics.json',report)
        return report
    except Exception as error:
        report.update(status='report_failed' if report['raw_status']=='completed' else 'failed',error=f'{type(error).__name__}: {error}')
        _json(output/'metrics.json',report)
        raise
