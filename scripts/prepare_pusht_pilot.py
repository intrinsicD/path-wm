"""Freeze a small, configuration-disjoint PushT pilot from the verified archive.

Initial-state labels serve only the split audit; training inputs remain pixels
and actions. One trajectory per configuration avoids weighting repeated variants.
"""
import argparse
import hashlib
import json
from pathlib import Path
import h5py
import hdf5plugin  # source HDF5 filters
import numpy as np
import yaml
from world_model.train import write_json


def configuration_groups(states, position_tolerance=5., angle_tolerance=.05):
    """Connected components of exact/near initial states, with circular angles."""
    unique, inverse = np.unique(np.asarray(states)[:, :5], axis=0, return_inverse=True)
    agent = np.linalg.norm(unique[:, None, :2] - unique[None, :, :2], axis=-1)
    block = np.linalg.norm(unique[:, None, 2:4] - unique[None, :, 2:4], axis=-1)
    angle = np.abs((unique[:, None, 4] - unique[None, :, 4] + np.pi) % (2*np.pi) - np.pi)
    distance = np.maximum.reduce([agent/position_tolerance, block/position_tolerance, angle/angle_tolerance])
    parent = np.arange(len(unique))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for a, b in np.argwhere(np.triu(distance <= 1, k=1)):
        parent[root(b)] = root(a)
    labels = np.array([root(i) for i in range(len(unique))])
    _, labels = np.unique(labels, return_inverse=True)
    np.fill_diagonal(distance, np.inf)
    return labels[inverse], dict(exact_initial_configurations=len(unique),
        near_configuration_groups=int(labels.max()+1),
        position_tolerance_pixels=position_tolerance, angle_tolerance_radians=angle_tolerance,
        minimum_distinct_configuration_scaled_distance=float(distance.min()))


def prepare(source_config, output, dataset_config, seed=3072):
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=False)
    source = yaml.safe_load(Path(source_config).read_text())
    rng = np.random.default_rng(seed)
    destination = directory/'trajectories.h5'
    with h5py.File(source['path'], 'r') as src:
        lengths, offsets = src['ep_len'][:], src['ep_offset'][:]
        groups, audit = configuration_groups(src['state'][offsets])
        available = np.unique(groups)
        if len(available) < 160: raise ValueError('Need 160 independent initial-configuration groups')
        chosen_groups = rng.choice(available, 160, replace=False)
        selected = [int(rng.choice(np.flatnonzero(groups == g))) for g in chosen_groups]
        training_sources = set(selected[:128])
        selected.sort()
        train = [i for i, e in enumerate(selected) if e in training_sources]
        val = [i for i, e in enumerate(selected) if e not in training_sources]
        new_lengths = lengths[selected]
        new_offsets = np.r_[0, np.cumsum(new_lengths[:-1])].astype(np.int64)
        frames = int(new_lengths.sum())
        case_rng = np.random.default_rng(42)
        cases = []
        for e in case_rng.choice(val, 20, replace=False):
            e = int(e)
            start = int(case_rng.integers(0, int(new_lengths[e])-25))
            cases.append(dict(episode=e, source_episode=selected[e], start=start,
                row=int(new_offsets[e])+start, source_row=int(offsets[selected[e]])+start))
        receipt = dict(seed=seed, source=source, source_episode_ids=selected,
            group_ids=groups[selected].tolist(), train_episodes=train, val_episodes=val,
            audit=audit, selection='160 random initial-state groups, one random episode per group; first 128 groups train',
            source_episode_count=len(lengths), source_frames=int(lengths.sum()),
            frames=frames, cases=cases, sampling_seed=42, goal_offset=25,
            solver_and_reset_seeds=[1234+i for i in range(20)],
            limitation='Initial-configuration grouping does not prove independence of later shared trajectory segments.')
        write_json(directory/'split.json', receipt)
        print(json.dumps(dict(stage='split_frozen', train=len(train), heldout=len(val), frames=frames,
            source_id_range=[min(selected), max(selected)], audit=audit)), flush=True)
        with h5py.File(destination.with_suffix('.h5.tmp'), 'w') as dst:
            dst['ep_len'] = new_lengths
            dst['ep_offset'] = new_offsets
            frame_keys = [k for k in src if k not in ('ep_len', 'ep_offset')]
            for key in frame_keys:
                dst.create_dataset(key, shape=(frames, *src[key].shape[1:]), dtype=src[key].dtype)
            for i, ep in enumerate(selected):
                old = slice(int(offsets[ep]), int(offsets[ep]+lengths[ep]))
                new = slice(int(new_offsets[i]), int(new_offsets[i]+new_lengths[i]))
                for key in frame_keys:
                    values = src[key][old] if key != 'episode_idx' else np.full(int(new_lengths[i]), i, np.int64)
                    dst[key][new] = values
                if (i+1) % 20 == 0: print(json.dumps(dict(stage='extracting', episodes=i+1, total=160)), flush=True)
            dst.attrs['source_revision'] = source['revision']
            dst.attrs['episode_idx_mapping'] = 'local index; original IDs in split.json'
        destination.with_suffix('.h5.tmp').replace(destination)
        # Every selected pixel/action/state block must exactly match the source.
        with h5py.File(destination, 'r') as dst:
            for i, ep in enumerate(selected):
                old = slice(int(offsets[ep]), int(offsets[ep]+lengths[ep]))
                new = slice(int(new_offsets[i]), int(new_offsets[i]+new_lengths[i]))
                for key in ('pixels', 'action', 'state', 'proprio', 'step_idx'):
                    if not np.array_equal(dst[key][new], src[key][old], equal_nan=True):
                        raise ValueError(f'Extraction mismatch: {key}, source episode {ep}')
    with destination.open('rb') as f: digest = hashlib.file_digest(f, 'sha256').hexdigest()
    subset_config = {**source, 'path':str(destination), 'subset_sha256':digest,
        'subset_receipt':str(directory/'split.json'), 'subset_protocol':'128 train / 32 held-out initial-configuration groups'}
    Path(dataset_config).write_text(yaml.safe_dump(subset_config, sort_keys=False))
    write_json(directory/'verification.json', dict(sha256=digest, bytes=destination.stat().st_size,
        verified_all_selected_pixels_actions_states=True, episodes=160, frames=frames))
    print(json.dumps(dict(stage='complete', sha256=digest, frames=frames)), flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--source', default='configs/datasets/pusht.yaml')
    p.add_argument('--output', default='data/pusht/pilot_128_32')
    p.add_argument('--dataset-config', default='configs/datasets/pusht_pilot.yaml')
    a=p.parse_args(); prepare(a.source, a.output, a.dataset_config)
