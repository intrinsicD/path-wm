"""Atomic, versioned checkpoints with exact observer dependencies."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import shutil
import tempfile

import numpy as np
import torch

from .models import MODEL_SCHEMA_VERSION, ACTION_SCHEMA_VERSION
SCHEMA_VERSION = 1
TENSOR_SCHEMA = MODEL_SCHEMA_VERSION


def code_fingerprint():
    digest = hashlib.sha256()
    for path in sorted([*Path(__file__).parent.glob('*.py'), *Path(__file__).parent.parent.joinpath('paddle').glob('*.py')]):
        digest.update(path.name.encode()); digest.update(path.read_bytes())
    return digest.hexdigest()


def versions():
    result = {'python': platform.python_version(), 'torch': torch.__version__,
              'cuda_runtime': torch.version.cuda, 'platform': platform.platform()}
    for name in ('numpy', 'PyYAML', 'Pillow', 'matplotlib', 'pytest'):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    if torch.cuda.is_available():
        result['gpu'] = torch.cuda.get_device_name(0)
    return result


def fingerprint_modules(modules):
    digest = hashlib.sha256(TENSOR_SCHEMA.encode())
    for name, module in sorted(modules.items()):
        states = module.state_dict() if hasattr(module, 'state_dict') else module
        for key, value in sorted(states.items()):
            t = value.detach().cpu().contiguous()
            digest.update(f'{name}.{key}:{t.dtype}:{tuple(t.shape)}'.encode())
            digest.update(t.numpy().tobytes())
    return digest.hexdigest()


def json_atomic(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as f:
        temporary = Path(f.name)
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    temporary.replace(path)


def atomic_checkpoint(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as f:
        temporary = Path(f.name)
        torch.save(value, f); f.flush(); os.fsync(f.fileno())
    temporary.replace(path)


def atomic_checkpoint_copy(source, target):
    """Recover an alias without changing CUDA storage tags or checkpoint bytes."""
    target = Path(target); target.parent.mkdir(parents=True,exist_ok=True)
    with Path(source).open('rb') as original, tempfile.NamedTemporaryFile(dir=target.parent,delete=False) as f:
        temporary = Path(f.name)
        shutil.copyfileobj(original,f); f.flush(); os.fsync(f.fileno())
    temporary.replace(target)


def read_checkpoint(path, *, dependencies=None, dataset_fingerprint=None, stage=None):
    # These are locally generated trusted checkpoints, including Python RNG state.
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if checkpoint.get('schema_version') != SCHEMA_VERSION:
        raise ValueError(f'{path}: incompatible checkpoint schema')
    if checkpoint.get('tensor_schema') != TENSOR_SCHEMA or checkpoint.get('action_schema') != ACTION_SCHEMA_VERSION:
        raise ValueError(f'{path}: incompatible tensor/action ordering')
    if stage and checkpoint.get('stage') != stage:
        raise ValueError(f'{path}: expected stage {stage}')
    for key, identity in (dependencies or {}).items():
        if checkpoint.get('dependencies', {}).get(key) != identity:
            raise ValueError(f'{path}: incompatible {key} dependency')
    if dataset_fingerprint and checkpoint.get('dataset_fingerprint') != dataset_fingerprint:
        raise ValueError(f'{path}: incompatible dataset fingerprint')
    if 'model_fingerprint' in checkpoint:
        if fingerprint_modules(checkpoint['models']) != checkpoint['model_fingerprint']:
            raise ValueError(f'{path}: model weight fingerprint mismatch')
    return checkpoint


def rng_state(rng):
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'sampler': rng.bit_generator.state, 'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state, rng):
    random.setstate(state['python']); np.random.set_state(state['numpy'])
    rng.bit_generator.state = state['sampler']; torch.set_rng_state(state['torch'])
    if state['cuda'] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])


def resolve_device(device='auto'):
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but inaccessible. On this machine run outside the sandbox.')
    return torch.device(device)


def _construct():
    from .models import Encoder, Decoder, PoseReadout, MemoryUpdater, StateReadout, Predictor
    return {'E': Encoder, 'D': Decoder, 'H': PoseReadout,
            'U': MemoryUpdater, 'R': StateReadout, 'P': Predictor}


def _load_models(checkpoint, device):
    classes = _construct(); result = {}
    for key, weights in checkpoint['models'].items():
        model = classes[key]().to(device)
        model.load_state_dict(weights)
        model.eval().requires_grad_(False); result[key] = model
    return result


def load_observer(perception, memory=None, device='auto'):
    device = resolve_device(device)
    a = read_checkpoint(perception, stage='perception')
    result = _load_models(a, device)
    if memory is not None:
        b = read_checkpoint(memory, stage='memory', dependencies={'perception': a['model_fingerprint']},
                            dataset_fingerprint=a['dataset_fingerprint'])
        if a['normalization'] != b['normalization']:
            raise ValueError('Observer normalization mismatch')
        result.update(_load_models(b, device))
    return result


def load_system(perception, memory, predictor, device='auto'):
    result = load_observer(perception, memory, device)
    a = read_checkpoint(perception); b = read_checkpoint(memory)
    c = read_checkpoint(predictor, stage='predictor', dependencies={
        'perception': a['model_fingerprint'], 'memory': b['model_fingerprint']},
        dataset_fingerprint=a['dataset_fingerprint'])
    if c['normalization'] != a['normalization']:
        raise ValueError('Predictor normalization mismatch')
    result.update(_load_models(c, resolve_device(device)))
    result['statistics'] = c['statistics']
    result['normalization'] = c['normalization']
    return result


def export_bundle(perception, memory, predictor, output):
    system = load_system(perception, memory, predictor, 'cpu')
    c = read_checkpoint(predictor)
    models = {k: m.state_dict() for k, m in system.items() if k not in ('statistics', 'normalization')}
    bundle = {'schema_version': SCHEMA_VERSION, 'tensor_schema': TENSOR_SCHEMA,
              'action_schema': ACTION_SCHEMA_VERSION, 'normalization': system['normalization'],
              'stage': 'inference', 'models': models, 'model_fingerprint': fingerprint_modules(models),
              'statistics': system['statistics'], 'config': c['config'], 'versions': versions(),
              'dataset_fingerprint': c['dataset_fingerprint'], 'dependencies': c['dependencies']}
    atomic_checkpoint(output, bundle)
    return str(output)


def load_bundle(path, device='auto'):
    bundle = read_checkpoint(path, stage='inference')
    result = _load_models(bundle, resolve_device(device))
    result['statistics'] = bundle['statistics']
    result['normalization'] = bundle['normalization']
    return result
