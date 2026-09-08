"""Provenance-checked caches for two frozen packages and identical source views."""
from __future__ import annotations
import json
import shutil
import time
from pathlib import Path
import numpy as np
import torch

from .data import FrameSet, task_frames, file_hash, digest
from .encoder_probe import source_models, masks
from .encoder_reference import pinned_backbone
from .pose_accessibility import setup, analyze
from world_model.pusht.checkpoints import fingerprint_modules, json_atomic, versions

ROOT = Path('runs/perception_overnight_2026-09-08')
OLD = Path('runs/encoder_study_2026-09-08')
CNN_SHA = 'aecc2a7abe545c673324a9ea53094729c69f00ed6d56c5dae036e52ffa820345'


def validate_alignment(expected_rows, expected_targets, rows, targets):
    if not np.array_equal(expected_rows, rows):
        raise ValueError('cache source row order mismatch')
    if expected_targets is not None and not np.array_equal(expected_targets, targets):
        raise ValueError('cache pose target mismatch')


def populations(development=False):
    _, mask_manifest, mask_data = masks(False)
    rgb = np.load('data/curriculum/coco_v1/frames.npy', mmap_mode='r')
    coco_manifest = json.loads(Path('data/curriculum/coco_v1/manifest.json').read_text())
    result = {}
    for split in ('train', 'validation', 'test'):
        p = task_frames('data/pusht_world_model/cchi_v1', split)
        c = mask_data[split]
        n = (32 if split == 'train' else 16) if development else None
        idx = np.arange(len(p))[:n]
        result['pusht', split] = (FrameSet(p.frames, p.rows[idx], p.targets[idx],
            [p.metadata[i] for i in idx], p.fingerprint, p.normalization),
            dict(indices=idx, rows=p.rows[idx], targets=p.targets[idx],
                 groups=np.array([p.metadata[i]['group'] for i in idx])))
        c = {k: v[:n] for k, v in c.items()}
        result['coco', split] = (FrameSet(rgb, c['rows'], fingerprint=coco_manifest['fingerprint']),
                                  dict(c, indices=np.arange(len(c['rows']))))
    return result


def cache_root(development=False):
    return ROOT / ('development/cache' if development else 'cache')


def check_storage(additional_bytes=0):
    if shutil.disk_usage(ROOT).free - additional_bytes < 6 * 1024**3:
        raise RuntimeError('overnight storage reserve would fall below 6 GiB')
    used = sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())
    if used + additional_bytes > 10 * 1024**3:
        raise RuntimeError('overnight artifacts would exceed 10 GiB budget')


@torch.no_grad()
def prepare(encoder, development=False, device='cuda'):
    setup()
    out = cache_root(development) / encoder
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'manifest.json').exists():
        return load_manifest(encoder, development)
    data = populations(development)
    if encoder == 'cnn':
        models, source = source_models('deeper', 7107, False, device)
        if file_hash(source) != CNN_SHA:
            raise ValueError('fixed CNN checkpoint changed')
        model = models['E']
        source_identity = dict(checkpoint=str(source), sha256=CNN_SHA)
    elif encoder == 'vit':
        model = pinned_backbone(device)
        source_identity = dict(upstream_manifest_sha256=file_hash(OLD / 'upstream/manifest.json'))
    else:
        raise ValueError('encoder package')
    model.eval().requires_grad_(False)
    before = fingerprint_modules({'E': model})
    entries = {}; begin = time.monotonic()
    for (domain, split), (ds, labels) in data.items():
        destination = out / domain; destination.mkdir(exist_ok=True)
        reused = False; old_proof = None
        if encoder == 'cnn' and domain == 'coco':
            old = OLD / 'probes/7107_deeper'
            old_proof = json.loads((old / 'features.json').read_text())
            if old_proof['source_checkpoint_sha256'] != CNN_SHA:
                raise ValueError('old CNN cache source changed')
            path = old / f'{split}_features.npy'
            expected = old_proof['splits'][split]['sha256']
            if not development and old_proof['splits'][split]['rows_sha256'] != digest(labels['rows'].tolist()):
                raise ValueError('COCO cache rows differ')
            reused = True
        elif encoder == 'vit' and domain == 'pusht':
            old = OLD / 'dino_features'
            old_proof = json.loads((old / 'manifest.json').read_text())
            if old_proof['upstream_manifest_sha256'] != source_identity['upstream_manifest_sha256']:
                raise ValueError('DINO source identity differs')
            path = old / f'{split}.npy'; expected = old_proof['splits'][split]['features_sha256']
            old_labels = dict(np.load(old / f'{split}_labels.npz'))
            if file_hash(old / f'{split}_labels.npz') != old_proof['splits'][split]['labels_sha256']:
                raise ValueError('old DINO labels changed')
            validate_alignment(labels['rows'], labels['targets'], old_labels['source_rows'][:len(ds)],
                               old_labels['targets'][:len(ds)])
            reused = True
        else:
            path = destination / f'{split}.npy'
        def forward(ids):
            rgb, _ = ds.batch(ids, device)
            z = model(rgb)
            return z.tokens() if encoder == 'cnn' else z
        if reused:
            if file_hash(path) != expected:
                raise ValueError('reused feature bytes changed')
            quantization = (old_proof['splits'][split].get('relative_mse') or
                            old_proof['splits'][split].get('relative_cache_mse'))
        else:
            shape = (len(ds), 320, 64) if encoder == 'cnn' else (len(ds), 256, 384)
            check_storage(int(np.prod(shape)) * 2)
            partial = path.with_suffix('.partial.npy')
            array = np.lib.format.open_memmap(partial, mode='w+', dtype='float16', shape=shape)
            num = den = 0.
            for start in range(0, len(ds), 32):
                ids = np.arange(start, min(start + 32, len(ds)))
                z = forward(ids)
                array[start:start + len(ids)] = z.cpu().half().numpy()
                num += float((z - z.half().float()).square().sum())
                den += float(z.square().sum())
            array.flush(); del array
            partial.replace(path); quantization = num / max(den, 1e-30)
        array = np.load(path, mmap_mode='r')
        ids = np.unique([0, len(ds) // 2, len(ds) - 1])
        live = forward(ids).cpu().numpy(); stored = array[ids].astype('float32')
        relative = float(np.square(live - stored).mean() / np.square(live).mean())
        if relative > 1e-6 or quantization > 1e-6 or not np.allclose(live, stored, rtol=6e-4, atol=3e-4):
            raise RuntimeError('live/cache identity or precision gate failed')
        label_path = destination / f'{split}_labels.npz'
        np.savez_compressed(label_path, **labels)
        entries[f'{domain}/{split}'] = dict(features=str(path), features_sha256=file_hash(path),
            labels=str(label_path), labels_sha256=file_hash(label_path), frames=len(ds),
            shape=list(array.shape), used_prefix=len(ds), dataset=ds.fingerprint,
            source_rows_sha256=digest(labels['rows'].tolist()), reused=reused,
            quantization_relative_mse=quantization, live_relative_mse=relative)
        print('cache', encoder, domain, split, len(ds), 'reused', reused,
              'seconds', round(time.monotonic() - begin, 1), flush=True)
    if before != fingerprint_modules({'E': model}):
        raise RuntimeError('frozen encoder weights or buffers changed')
    manifest = dict(schema='perception-cache-v1', status='completed', encoder=encoder,
        development=development, entries=entries, source=source_identity, frozen_fingerprint=before,
        seconds=time.monotonic() - begin, versions=versions(), precision='FP32 / TF32 off / FP16 cache')
    json_atomic(out / 'manifest.json', manifest)
    analyze(ROOT, out, 'Audited frozen perception features',
        {'frames': sum(e['frames'] for e in entries.values())},
        f'## {encoder} feature audit\n\n{"Development prefixes. " if development else ""}'
        'Frozen weights and buffers unchanged; source-row, label and live-cache checks passed. '
        'No optimizer updates or model-quality claim.', [out / 'manifest.json'])
    return manifest


def load_manifest(encoder, development=False, verify=True):
    path = cache_root(development) / encoder / 'manifest.json'
    manifest = json.loads(path.read_text())
    if manifest['encoder'] != encoder or manifest['development'] != development:
        raise ValueError('feature manifest population mismatch')
    if verify:
        for entry in manifest['entries'].values():
            for key in ('features', 'labels'):
                if file_hash(entry[key]) != entry[key + '_sha256']:
                    raise ValueError('cache source mutated')
    return manifest
