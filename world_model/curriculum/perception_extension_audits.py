"""Independent category readouts and prospective-pose audits of selected extensions."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F
from .perception_cache import ROOT, load_manifest
from .perception_continuation import system
from .perception_training import load_data
from .perception_semantics import LABEL_ROOT, train
from .perception_fresh import evaluate_frozen
from .pose_accessibility import setup
from .data import file_hash, digest
from world_model.pusht.checkpoints import fingerprint_modules, json_atomic


def selected(kind, seed, device, development=False):
    checkpoint = ROOT / ('development/extensions' if development else 'extensions') / f'seed_{seed}' / kind / 'best.pt'
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model, heads = system(kind, seed, device)
    for name, module in {'encoder': model, **heads}.items():
        module.load_state_dict(state['heads'][name]); module.eval().requires_grad_(False)
    return model, heads, checkpoint, state['step']


@torch.no_grad()
def pool(kind, seed, device, development=False):
    setup(); base = ROOT / ('development/extension_audits' if development else 'extension_audits') / f'seed_{seed}' / kind
    out = base / 'pooled'; out.mkdir(parents=True, exist_ok=True)
    model, _, checkpoint, step = selected(kind, seed, device, development)
    identity = dict(labels_sha256=file_hash(LABEL_ROOT / 'manifest.json'), checkpoint_sha256=file_hash(checkpoint))
    if (out / 'manifest.json').exists():
        m = json.loads((out / 'manifest.json').read_text())
        if any(m[k] != v for k, v in identity.items()): raise ValueError('selected pool source changed')
        for split, entry in m['splits'].items():
            if file_hash(out / f'{split}.npy') != entry['sha256']: raise ValueError('selected pool changed')
        return out
    # Even development uses full prepared COCO identities; fitting prefixes only.
    data, _ = load_data('cnn', False); splits = {}; started = time.monotonic()
    before = fingerprint_modules({'encoder': model})
    for split in ('train', 'validation', 'test'):
        ds = data['coco', split]['ds']; labels = dict(np.load(LABEL_ROOT / f'{split}.npz'))
        if not np.array_equal(ds.rows, labels['rows']): raise ValueError('category source row order changed')
        values = []
        for start in range(0, len(ds), 32):
            rgb, _ = ds.batch(np.arange(start, min(start + 32, len(ds))), device)
            z = model(rgb)
            values.append(torch.cat([F.layer_norm(t, (t.shape[-1],)).mean(1) for t in (z.fine, z.coarse)], 1).cpu().numpy())
        p = out / f'{split}.npy'; np.save(p, np.concatenate(values))
        splits[split] = dict(sha256=file_hash(p), frames=len(ds), rows_sha256=digest(ds.rows.tolist()))
    if before != fingerprint_modules({'encoder': model}): raise RuntimeError('pooling changed the selected encoder')
    json_atomic(out / 'manifest.json', dict(status='completed', **identity, splits=splits, encoder=kind,
        seed=seed, selected_step=step, frozen_fingerprint=before, precision='FP32 live features',
        normalization='fixed per-token LayerNorm eps1e-5 before spatial mean; learned affine after mean',
        seconds=time.monotonic()-started, source_sha256=file_hash(__file__)))
    return out


def main():
    p = argparse.ArgumentParser(); p.add_argument('command', choices=['pool', 'semantics', 'fresh'])
    p.add_argument('--kind', choices=['joint', 'conv', 'transformer'], required=True)
    p.add_argument('--seed', type=int, required=True); p.add_argument('--device', default='cuda')
    p.add_argument('--development', action='store_true'); a = p.parse_args(); setup()
    root = ROOT / ('development/extension_audits' if a.development else 'extension_audits') / f'seed_{a.seed}' / a.kind
    if a.command == 'pool': pool(a.kind, a.seed, a.device, a.development)
    elif a.command == 'semantics':
        train(a.kind, a.seed, a.development, a.device, pool_override=root/'pooled',
              output_override=root/'semantics', protocol_path='docs/perception-extension-protocol-2026-09-08.md')
    else:
        model, heads, checkpoint, step = selected(a.kind, a.seed, a.device, a.development)
        evaluate_frozen(model, heads, a.kind, a.seed, root/'fresh', checkpoint, step, a.device,
                        feature_family='cnn', cache_precision=False)


if __name__ == '__main__': main()
