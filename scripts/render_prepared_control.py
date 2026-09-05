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
from world_model.train import write_json


def render(local, released, output, index=0):
    directories = [Path(local), Path(released)]
    manifests = [json.loads((d / 'manifest.json').read_text()) for d in directories]
    if any(manifests[0][key] != manifests[1][key] for key in ('cases', 'solver_and_reset_seeds', 'dataset')):
        raise ValueError('Qualitative comparison requires identical cases, resets and source')
    case = manifests[0]['cases'][index]
    row = case['row']; seed = manifests[0]['solver_and_reset_seeds'][index]
    out = Path(output); out.mkdir(parents=True, exist_ok=False)
    images, measurements = {}, {}
    with h5py.File(manifests[0]['dataset']['path'], 'r') as source:
        start, goal = source['pixels'][row], source['pixels'][row + 25]
        state, target = source['state'][row].astype(float), source['state'][row + 25].astype(float)
        sequences = [('recorded replay', source['action'][row:row + 25], None)]
        for label, directory in zip(('10-minute model', 'released model'), directories):
            recorded = [json.loads(line) for line in (directory / 'cases.jsonl').read_text().splitlines()][index]
            sequences.append((label, np.load(directory / f'actions_{index}.npy'), recorded))
        for label, actions, recorded in sequences:
            env = PushT(resolution=start.shape[0], relative=True)
            env.reset(seed=seed); env._set_state(state); env._set_goal_state(target)
            frames = [env.render().copy()]; success = False
            for step, action in enumerate(actions, 1):
                _, _, success, _, _ = env.step(action)
                frames.append(env.render().copy())
                if success:
                    break
            _, distance = env.eval_state(target, env._get_obs())
            if recorded is not None:
                if step != recorded['steps'] or bool(success) != recorded['success'] or not np.isclose(distance, recorded['state_distance'], rtol=1e-8, atol=1e-8):
                    raise ValueError('Saved actions do not reproduce the recorded control outcome')
            env.close()
            images[label] = np.stack(frames)
            measurements[label] = dict(steps=step, success=bool(success), state_distance=float(distance), verified_recorded_outcome=recorded is not None)
        np.savez_compressed(out / 'frames.npz', source=start, goal=goal, **images)
        ticks = [0, 5, 10, 25, 50]
        fig, axes = plt.subplots(3, 7, figsize=(14, 6.4), dpi=130)
        for i, (label, frames) in enumerate(images.items()):
            selected = [start, *[frames[min(t, len(frames) - 1)] for t in ticks], goal]
            for axis, frame in zip(axes[i], selected):
                axis.imshow(frame); axis.set_xticks([]); axis.set_yticks([])
                for spine in axis.spines.values(): spine.set_visible(False)
            axes[i, 0].set_ylabel(label, fontsize=10)
            axes[i, -1].text(0.5, -0.06, f"{'success' if measurements[label]['success'] else 'failed'} · {len(frames)-1} steps", transform=axes[i, -1].transAxes, ha='center', fontsize=9)
        for axis, title in zip(axes[0], ['Source start', 'Simulator reset', 'Step 5', 'Step 10', 'Step 25', 'Step 50', 'Source goal']):
            axis.set_title(title, fontsize=10)
        fig.suptitle(f"First frozen control case · episode {case['episode']}, start {case['start']} · actual simulator frames", fontsize=12)
        fig.text(0.5, 0.015, 'Frames after termination repeat the final frame. Saved model actions reproduce the recorded outcomes; no decoder is used.', ha='center', fontsize=9)
        fig.tight_layout(rect=(0, 0.035, 1, 0.95)); fig.savefig(out / 'case_0_rollouts.png', facecolor='white'); plt.close(fig)
    write_json(out / 'manifest.json', dict(case_index=index, case=case, reset_seed=seed, source_manifests=[str(d / 'manifest.json') for d in directories], selection='First frozen case, no outcome-based selection', measurements=measurements))
    print(json.dumps(measurements), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('local'); p.add_argument('released'); p.add_argument('output')
    a = p.parse_args(); render(a.local, a.released, a.output)
