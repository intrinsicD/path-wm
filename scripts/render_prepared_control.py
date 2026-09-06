"""Render the first frozen case from saved executed actions, checking exact replay."""
import argparse
import json
from pathlib import Path
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from world_model.eval_pusht import PushT
from world_model.eval_tworoom import _environment, _distance
from world_model.train import write_json


def render(local, released, output, index=0):
    directories = [Path(local), Path(released)]
    manifests = [json.loads((d / 'manifest.json').read_text()) for d in directories]
    fields = ('cases', 'solver_and_reset_seeds', 'dataset', 'goal_offset', 'budget',
              'horizon', 'action_block', 'receding_horizon', 'samples', 'iterations', 'elites')
    if any(manifests[0].get(key) != manifests[1].get(key) for key in fields):
        raise ValueError('Qualitative comparison requires identical cases, resets, source and planning protocol')
    dataset_name = manifests[0]['dataset']['name']
    if dataset_name not in ('pusht', 'tworoom'):
        raise ValueError('Unsupported qualitative dataset')
    goal_offset, budget = manifests[0]['goal_offset'], manifests[0]['budget']
    case = manifests[0]['cases'][index]
    row = case['row']; seed = manifests[0]['solver_and_reset_seeds'][index]
    out = Path(output); out.mkdir(parents=True, exist_ok=False)
    images, measurements = {}, {}
    with h5py.File(manifests[0]['dataset']['path'], 'r') as source:
        start, goal = source['pixels'][row], source['pixels'][row + goal_offset]
        state_key = 'state' if dataset_name == 'pusht' else 'proprio'
        state, target = source[state_key][row].astype(float), source[state_key][row + goal_offset].astype(float)
        sequences = [('Recorded replay', source['action'][row:row + goal_offset], None)]
        labels = ['Released model' if m.get('released') else f"{'Calibrated local' if m.get('diagnostic_only') else 'Local model'} · step {m.get('step', 'unrecorded')}" for m in manifests]
        for label, directory in zip(labels, directories):
            recorded = [json.loads(line) for line in (directory / 'cases.jsonl').read_text().splitlines()][index]
            sequences.append((label, np.load(directory / f'actions_{index}.npy'), recorded))
        for label, actions, recorded in sequences:
            if dataset_name == 'pusht':
                env = PushT(resolution=start.shape[0], relative=True)
                env.reset(seed=seed); env._set_state(state); env._set_goal_state(target)
                initial = bool(env.eval_state(target, env._get_obs())[0])
            else:
                env = _environment(state, target, seed)
                initial = _distance(env) < 16.
            try:
                frames = [env.render().copy()]; success = False; step = 0
                # Upstream tests success after an action, including initially satisfied goals.
                for step, action in enumerate(actions[:budget], 1):
                    _, _, success, _, _ = env.step(action)
                    frames.append(env.render().copy())
                    if success:
                        break
                distance = float(env.eval_state(target, env._get_obs())[1]) if dataset_name == 'pusht' else _distance(env)
                final_state = None if dataset_name == 'pusht' else env.agent_position.tolist()
                if recorded is not None:
                    matches = (step == recorded['steps'] and bool(success) == recorded['success']
                               and initial == recorded['initial_success'] and len(actions) == step
                               and np.isclose(distance, recorded['state_distance'], rtol=1e-8, atol=1e-8))
                    if dataset_name == 'tworoom':
                        matches = matches and np.allclose(final_state, recorded['final_state'], rtol=0, atol=1e-6)
                    if not matches:
                        raise ValueError('Saved actions do not reproduce the recorded control outcome')
                images[label] = np.stack(frames)
                measurements[label] = dict(steps=step, success=bool(success), initial_success=initial,
                    state_distance=distance, final_state=final_state, verified_recorded_outcome=recorded is not None)
            finally:
                env.close()
        np.savez_compressed(out / 'frames.npz', source=start, goal=goal, **images)
        ticks = [0, 5, 10, goal_offset, budget]
        fig, axes = plt.subplots(3, 7, figsize=(14, 6.4), dpi=130)
        for i, (label, frames) in enumerate(images.items()):
            selected = [start, *[frames[min(t, len(frames) - 1)] for t in ticks], goal]
            for axis, frame in zip(axes[i], selected):
                axis.imshow(frame); axis.set_xticks([]); axis.set_yticks([])
                for spine in axis.spines.values(): spine.set_visible(False)
            axes[i, 0].set_ylabel(label, fontsize=10)
            axes[i, -1].text(0.5, -0.06, f"{'success' if measurements[label]['success'] else 'failed'} · {len(frames)-1} steps", transform=axes[i, -1].transAxes, ha='center', fontsize=9)
        for axis, title in zip(axes[0], ['Source start', 'Simulator reset', *[f'Step {t}' for t in ticks[1:]], 'Source goal']):
            axis.set_title(title, fontsize=10)
        fig.suptitle(f"{dataset_name} · goal {goal_offset}, budget {budget} · first frozen case · episode {case['episode']}, start {case['start']} · actual simulator frames", fontsize=12)
        fig.text(0.5, 0.015, 'Frames after termination repeat the final frame. Saved model actions reproduce the recorded outcomes; no decoder is used.', ha='center', fontsize=9)
        fig.tight_layout(rect=(0, 0.035, 1, 0.95)); fig.savefig(out / f'case_{index}_rollouts.png', facecolor='white'); plt.close(fig)
    write_json(out / 'manifest.json', dict(case_index=index, case=case, dataset=manifests[0]['dataset'], goal_offset=goal_offset, budget=budget, reset_seed=seed, source_manifests=[str(d / 'manifest.json') for d in directories], selection='First frozen case, no outcome-based selection', measurements=measurements))
    print(json.dumps(measurements), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('local'); p.add_argument('released'); p.add_argument('output')
    a = p.parse_args(); render(a.local, a.released, a.output)
