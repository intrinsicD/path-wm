"""Static coverage observations for perception only; no source pose/image access.

Requested poses are independent. Labels always describe the actual post-reset
simulator world, including contact corrections. Completed data and committed
partial prefixes are verified before reuse; no learned outcome filters samples.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import torch

from .data import PREPROCESSING, _atomic_json, _file_hash, _fingerprint
from .env import ENVIRONMENT_SCHEMA_VERSION, PushTEnv


SCHEMA_VERSION = 'pusht-static-coverage-actual-pose6-v1'
CHUNK_SIZE = 128
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_MANIFEST = ROOT / 'data/pusht_world_model/cchi_v1/manifest.json'
LOW = np.array([16., 16., 96., 96., 0.])
HIGH = np.array([496., 496., 416., 416., 2*np.pi])


def _read(path):
    try:
        value = json.loads(Path(path).read_text())
    except (OSError, ValueError) as error:
        raise ValueError(f'Missing or invalid supplement metadata: {path}') from error
    if not isinstance(value, dict):
        raise ValueError(f'Supplement metadata must be a mapping: {path}')
    return value


def _training_provenance(manifest):
    """Read only split identities/counts, never source frames or numeric poses."""
    if not manifest.get('complete') or not isinstance(manifest.get('fingerprint'), str):
        raise ValueError('Source training manifest must have a complete fingerprint')
    entries = [r for r in manifest['episodes'] if r['split'] == 'train']
    ids = sorted(int(r['source_episode']) for r in entries)
    groups = sorted({int(r['group_id']) for r in entries})
    if not ids or len(ids) != len(set(ids)) or any(type(r['frames']) is not int or r['frames'] < 1 for r in entries):
        raise ValueError('Invalid source training episode counts')
    declared = manifest['splits']['train']
    if ids != sorted(declared['source_episode_ids']) or groups != sorted(declared['group_ids']):
        raise ValueError('Source training provenance disagrees with declared split')
    return dict(dataset_fingerprint=manifest['fingerprint'], source_episode_ids=ids,
                group_ids=groups, frame_count=sum(r['frames'] for r in entries))


def _protocol(count, seed, source_manifest):
    paths = [Path(__file__), ROOT/'world_model/pusht/env.py', ROOT/'world_model/pusht/data.py',
             *[ROOT/'third_party/swm'/name for name in ('pusht.py', 'draw.py', 'spaces.py', '__init__.py')]]
    versions = {'numpy': np.__version__}
    for name in ('pymunk', 'pygame-ce', 'gymnasium', 'opencv-python'):
        try: versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name] = None
    return dict(schema_version=SCHEMA_VERSION, count=count, seed=seed,
                sampling=dict(rng='numpy-PCG64/default_rng', order=['pusher_x','pusher_y','block_x','block_y','block_angle'],
                              low=LOW.tolist(), high_exclusive=HIGH.tolist(), independent_requested_components=True,
                              reset_seed='seed + zero-based sample index', filtering='none'),
                labels='actual post-reset pusherXY/blockXY world coordinates and angle; target positions/512,sin(angle),cos(angle)',
                reset_convention='wrapper default zero pusher velocity; no source velocities; fixed native green overlay',
                environment_schema=ENVIRONMENT_SCHEMA_VERSION, preprocessing=PREPROCESSING,
                source_training=_training_provenance(_read(source_manifest)),
                code_hashes={str(p.relative_to(ROOT)): _file_hash(p) for p in paths}, versions=versions)


def _specs(count):
    return {'frames.npy': ((count,64,64,3), np.dtype('uint8')),
            'poses.npy': ((count,5), np.dtype('float64')),
            'pose_targets.npy': ((count,6), np.dtype('float32'))}


def _arrays(output, count, mode='r'):
    result = {}
    for name, (shape, dtype) in _specs(count).items():
        try: value = np.load(output/name, mmap_mode=mode, allow_pickle=False)
        except (ValueError, OSError) as error:
            raise ValueError(f'Corrupt supplement array: {name}') from error
        if value.shape != shape or value.dtype != dtype:
            raise ValueError(f'Supplement array schema mismatch: {name}')
        result[name] = value
    return result


def _targets(poses):
    targets = np.empty((len(poses),6), np.float32)
    targets[:,:4] = poses[:,:4]/512
    targets[:,4] = np.sin(poses[:,4]); targets[:,5] = np.cos(poses[:,4])
    return targets


def _chunk_hash(arrays, start, stop):
    digest = hashlib.sha256()
    for name, array in arrays.items():
        digest.update(name.encode()); digest.update(array[start:stop].tobytes(order='C'))
    return digest.hexdigest()


def _verify_prefix(arrays, progress, count):
    cursor = 0
    for chunk in progress.get('chunks', []):
        start, stop = chunk['start'], chunk['stop']
        if type(start) is not int or type(stop) is not int or start != cursor or not start < stop <= count:
            raise ValueError('Corrupt committed supplement prefix indices')
        if _chunk_hash(arrays, start, stop) != chunk['sha256']:
            raise ValueError('Corrupt committed supplement prefix checksum')
        cursor = stop
    if cursor != progress.get('completed'):
        raise ValueError('Corrupt committed supplement prefix count')
    return cursor


def verify_supplement(output):
    """Verify immutable completed bytes and exact actual-pose target formulas."""
    output = Path(output); manifest = _read(output/'manifest.json')
    if manifest.get('schema_version') != SCHEMA_VERSION or manifest.get('complete') is not True:
        raise ValueError('Incomplete or unsupported supplement schema')
    count = manifest.get('count')
    if type(count) is not int or count < 1:
        raise ValueError('Invalid supplement count')
    fingerprint = manifest.get('fingerprint')
    if fingerprint != _fingerprint({k:v for k,v in manifest.items() if k != 'fingerprint'}):
        raise ValueError('Supplement manifest fingerprint mismatch')
    arrays = _arrays(output, count)
    for name, array in arrays.items():
        expected = manifest.get('arrays',{}).get(name,{})
        if expected.get('shape') != list(array.shape) or expected.get('dtype') != str(array.dtype):
            raise ValueError(f'Supplement manifest array schema mismatch: {name}')
        if expected.get('bytes') != (output/name).stat().st_size or expected.get('sha256') != _file_hash(output/name):
            raise ValueError(f'Corrupt supplement file hash: {name}')
    for start in range(0, count, CHUNK_SIZE):
        poses = arrays['poses.npy'][start:start+CHUNK_SIZE]
        if not np.isfinite(poses).all() or not np.array_equal(_targets(poses), arrays['pose_targets.npy'][start:start+CHUNK_SIZE]):
            raise ValueError('Supplement actual pose and target labels disagree')
    return dict(passed=True, fingerprint=fingerprint, count=count,
                bytes=sum((output/name).stat().st_size for name in arrays))


def prepare_supplement(output, count=20493, seed=73107, *, source_manifest=None):
    """Stream private-RNG static resets, resume only a verified committed prefix."""
    if type(count) is not int or count < 1 or type(seed) is not int or seed < 0:
        raise ValueError('Supplement count must be positive and seed nonnegative integers')
    output = Path(output)
    protocol = _protocol(count, seed, source_manifest or DEFAULT_SOURCE_MANIFEST)
    if (output/'manifest.json').exists():
        verify_supplement(output)
        existing = _read(output/'manifest.json')
        if {key:existing.get(key) for key in protocol} != protocol:
            raise ValueError('Incompatible completed supplement seed/count/protocol identity')
        return existing
    output.mkdir(parents=True, exist_ok=True)
    progress_path = output/'progress.json'
    if progress_path.exists():
        progress = _read(progress_path)
        if progress.get('protocol') != protocol:
            raise ValueError('Incompatible partial supplement protocol identity')
        arrays = _arrays(output, count, 'r+')
        start = _verify_prefix(arrays, progress, count)
    else:
        if any(output.iterdir()):
            raise ValueError('Unrecognized or corrupted supplement files without a progress receipt')
        arrays = {name:np.lib.format.open_memmap(output/name,mode='w+',dtype=dtype,shape=shape)
                  for name,(shape,dtype) in _specs(count).items()}
        for array in arrays.values(): array.flush()
        progress = dict(protocol=protocol, completed=0, chunks=[])
        _atomic_json(progress_path, progress); start = 0
    rng = np.random.default_rng(seed)
    for _ in range(start): rng.uniform(LOW, HIGH)
    env = None
    try:
        if start < count: env = PushTEnv()
        for chunk_start in range(start, count, CHUNK_SIZE):
            stop = min(count, chunk_start+CHUNK_SIZE)
            for index in range(chunk_start, stop):
                requested = rng.uniform(LOW, HIGH)
                frame, _ = env.reset(pose5_world=requested, seed=seed+index)
                pose = np.asarray(env.pose, dtype=np.float64)
                if np.asarray(frame).shape != (64,64,3) or np.asarray(frame).dtype != np.uint8:
                    raise ValueError('Wrapper must return canonical RGB64 uint8')
                if pose.shape != (5,) or not np.isfinite(pose).all():
                    raise ValueError('Wrapper must return a finite actual post-reset pose')
                arrays['frames.npy'][index] = frame
                arrays['poses.npy'][index] = pose
                arrays['pose_targets.npy'][index] = _targets(pose[None])[0]
            for array in arrays.values(): array.flush()
            progress['chunks'].append(dict(start=chunk_start, stop=stop, sha256=_chunk_hash(arrays,chunk_start,stop)))
            progress['completed'] = stop
            _atomic_json(progress_path, progress)
    finally:
        if env is not None: env.close()
    manifest = dict(protocol, complete=True, arrays={name:dict(shape=list(array.shape),dtype=str(array.dtype),
                        bytes=(output/name).stat().st_size,sha256=_file_hash(output/name)) for name,array in arrays.items()})
    # Formula checks happen before publishing the final receipt as well as on reuse.
    for start in range(0,count,CHUNK_SIZE):
        if not np.array_equal(_targets(arrays['poses.npy'][start:start+CHUNK_SIZE]),arrays['pose_targets.npy'][start:start+CHUNK_SIZE]):
            raise ValueError('Actual pose labels changed during supplement preparation')
    manifest['fingerprint'] = _fingerprint(manifest)
    _atomic_json(output/'manifest.json', manifest)
    verify_supplement(output)
    return manifest


class SupplementFrames:
    def __init__(self, output, expected_fingerprint):
        output = Path(output); report = verify_supplement(output)
        if report['fingerprint'] != expected_fingerprint:
            raise ValueError('Supplement fingerprint differs from the expected training identity')
        self.manifest = _read(output/'manifest.json')
        arrays = _arrays(output, self.manifest['count'])
        self.frames, self.poses, self.targets = (arrays[name] for name in ('frames.npy','poses.npy','pose_targets.npy'))


class MixedPerceptionSamples:
    """Uniform concatenation only; existing source sampler chooses the indices."""
    def __init__(self, source_samples, supplement):
        if source_samples.dataset.split != 'train':
            raise ValueError('Perception supplement can only mix with source training frames')
        provenance = _training_provenance(source_samples.dataset.manifest)
        if source_samples.dataset.fingerprint != provenance['dataset_fingerprint'] or provenance != supplement.manifest['source_training']:
            raise ValueError('Supplement and source training provenance/fingerprint disagree')
        self.source, self.supplement = source_samples, supplement
        self.source_count = len(source_samples.frames)
        if self.source_count != len(supplement.frames) or self.source_count != provenance['frame_count']:
            raise ValueError('Mixture requires equal full source and supplement frame counts')
        self.dataset = source_samples.dataset
        self.lengths = list(source_samples.lengths)
        self.frames = range(2*self.source_count)

    def frame_batch(self, indices, device):
        indices = np.asarray(indices)
        if indices.ndim != 1 or not np.issubdtype(indices.dtype,np.integer) or np.any(indices<0) or np.any(indices>=len(self.frames)):
            raise ValueError('Mixture indices must be a one-dimensional array of valid frame indices')
        images = torch.empty((len(indices),3,64,64),device=device,dtype=torch.float32)
        targets = torch.empty((len(indices),6),device=device,dtype=torch.float32)
        source_positions = np.flatnonzero(indices<self.source_count)
        synth_positions = np.flatnonzero(indices>=self.source_count)
        if len(source_positions):
            x,y = self.source.frame_batch(indices[source_positions],device)
            positions = torch.as_tensor(source_positions,device=device)
            images[positions],targets[positions] = x,y
        if len(synth_positions):
            rows = indices[synth_positions]-self.source_count
            positions = torch.as_tensor(synth_positions,device=device)
            images[positions] = torch.from_numpy(self.supplement.frames[rows].copy()).to(device).permute(0,3,1,2).float()/255
            targets[positions] = torch.from_numpy(self.supplement.targets[rows].copy()).to(device)
        return images,targets
