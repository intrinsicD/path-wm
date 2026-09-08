"""Crop-visible category labels and a matched, cached semantic-accessibility probe."""
from __future__ import annotations
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .encoder_masks import ANNOTATIONS, decode_union, transform_mask
from .encoder_probe import masks
from .perception_cache import ROOT, load_manifest, cache_root
from .perception_heads import feature_grids
from .perception_training import append_json, TRAIN_CUTOFF
from .pose_accessibility import setup, analyze
from .data import file_hash, digest
from world_model.pusht.checkpoints import json_atomic, atomic_checkpoint, versions

LABEL_ROOT = ROOT / 'semantics/labels'


def category_labels(annotations, height, width, category_ids):
    """A cropped-away object is absent; a visible crowd is unknown, not negative."""
    lookup = {category: i for i, category in enumerate(category_ids)}
    positive = np.zeros(len(category_ids), dtype='uint8')
    crowd = positive.copy()
    for ann in annotations:
        mask, _ = decode_union([dict(ann, iscrowd=0)], height, width)
        if transform_mask(mask).any():
            target = crowd if ann.get('iscrowd', 0) else positive
            target[lookup[ann['category_id']]] = 1
    known = ((crowd == 0) | (positive > 0)).astype('uint8')
    return positive, known


def average_precision(target, scores, known):
    target = np.asarray(target)[np.asarray(known, dtype=bool)].astype('float64')
    scores = np.asarray(scores)[np.asarray(known, dtype=bool)]
    if not len(target) or target.sum() == 0:
        return None
    order = np.argsort(-scores, kind='stable')
    scores = scores[order]; target = target[order]
    ends = np.r_[np.flatnonzero(scores[:-1] != scores[1:]), len(scores) - 1]
    true_positives = np.cumsum(target)[ends]
    precision = true_positives / (ends + 1)
    recall = true_positives / target.sum()
    return float((precision * np.diff(np.r_[0., recall])).sum())


def known_bce(logits, target, known):
    return (F.binary_cross_entropy_with_logits(logits, target, reduction='none') * known).sum() / known.sum().clamp_min(1)


def prepare():
    LABEL_ROOT.mkdir(parents=True, exist_ok=True)
    if (LABEL_ROOT / 'manifest.json').exists():
        raise FileExistsError('preserve prepared semantic labels')
    begin = time.monotonic()
    source = json.loads(ANNOTATIONS.read_text())
    annotations = defaultdict(list)
    for ann in source['annotations']:
        annotations[ann['image_id']].append(ann)
    images = {row['id']: row for row in source['images']}
    categories = sorted(source['categories'], key=lambda row: row['id'])
    ids = [row['id'] for row in categories]
    _, mask_manifest, populations = masks(False)
    splits = {}; sources = []
    for split, population in populations.items():
        targets = []; known = []
        for index, image_id in enumerate(population['image_ids']):
            if time.monotonic() - begin > 1800:
                raise TimeoutError('category preparation budget')
            image = images[int(image_id)]
            target, valid = category_labels(annotations[int(image_id)], image['height'], image['width'], ids)
            targets.append(target); known.append(valid)
            if (index + 1) % 512 == 0:
                print('category labels', split, index + 1, 'seconds', round(time.monotonic() - begin, 1), flush=True)
        target = np.asarray(targets); valid = np.asarray(known)
        path = LABEL_ROOT / f'{split}.npz'
        np.savez_compressed(path, targets=target, known=valid, rows=population['rows'],
                            groups=population['groups'], image_ids=population['image_ids'])
        sources.append(path)
        splits[split] = dict(frames=len(target), sha256=file_hash(path),
            positive=(target * valid).sum(0).tolist(), negative=((1 - target) * valid).sum(0).tolist(),
            unknown=(1 - valid).sum(0).tolist(), rows_sha256=digest(population['rows'].tolist()))
    retained = [i for i in range(len(ids)) if splits['train']['positive'][i] >= 20 and splits['train']['negative'][i] >= 20]
    manifest = dict(schema='crop-category-v1', status='completed', categories=categories,
        retained_indices=retained, retained_category_ids=[ids[i] for i in retained],
        excluded_categories=[row for i, row in enumerate(categories) if i not in retained],
        splits=splits, source_sha256=file_hash(ANNOTATIONS),
        mask_manifest_sha256=digest(mask_manifest), source_code_sha256=file_hash(__file__),
        seconds=time.monotonic() - begin, semantics='>=1 visible non-crowd mask pixel after RGB-aligned nearest crop; visible crowd unknown unless positive',
        support='>=20 positive and20 known negative training images; no held-out selection')
    json_atomic(LABEL_ROOT / 'manifest.json', manifest); sources.append(LABEL_ROOT / 'manifest.json')
    analyze(ROOT, LABEL_ROOT, 'Crop-aware category labels',
        dict(classes=len(retained), train_frames=splits['train']['frames']),
        f'## Category-accessibility population\n\n{len(retained)} of80 categories meet the training-only support rule. '
        'Labels follow the actual RGB64 crop. Visible crowds are unknown unless a non-crowd positive remains. '
        'This defines category presence, not localization or arbitrary object coverage.', sources)


def pooled_features(encoder):
    out = ROOT / 'semantics' / 'pooled' / encoder; out.mkdir(parents=True, exist_ok=True)
    label_manifest = json.loads((LABEL_ROOT / 'manifest.json').read_text())
    feature_manifest = load_manifest(encoder, False)
    manifest_path = out / 'manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest['labels_sha256'] != file_hash(LABEL_ROOT / 'manifest.json') or manifest['features_sha256'] != file_hash(cache_root() / encoder / 'manifest.json'):
            raise ValueError('pooled source identity changed')
        for split, entry in manifest['splits'].items():
            if file_hash(out / f'{split}.npy') != entry['sha256']:
                raise ValueError('pooled features changed')
        return out, manifest
    begin = time.monotonic(); splits = {}
    for split in ('train', 'validation', 'test'):
        source = feature_manifest['entries'][f'coco/{split}']
        labels = dict(np.load(LABEL_ROOT / f'{split}.npz'))
        if file_hash(LABEL_ROOT / f'{split}.npz') != label_manifest['splits'][split]['sha256']:
            raise ValueError('semantic labels changed')
        if digest(labels['rows'].tolist()) != source['source_rows_sha256']:
            raise ValueError('semantic feature and label rows differ')
        features = np.load(source['features'], mmap_mode='r'); chunks = []
        for start in range(0, source['frames'], 64):
            z = torch.from_numpy(np.asarray(features[start:start + 64]).copy()).float()
            fine, coarse = feature_grids(z, encoder)
            chunks.append(torch.cat([F.layer_norm(level, (level.shape[-1],)).mean(1)
                                     for level in (fine, coarse)], 1).numpy())
        path = out / f'{split}.npy'; np.save(path, np.concatenate(chunks))
        splits[split] = dict(sha256=file_hash(path), frames=source['frames'])
    manifest = dict(status='completed', encoder=encoder, splits=splits,
        labels_sha256=file_hash(LABEL_ROOT / 'manifest.json'), features_sha256=file_hash(cache_root() / encoder / 'manifest.json'),
        normalization='fixed per-token LayerNorm eps1e-5 before spatial mean; learned affine after mean is equivalent',
        seconds=time.monotonic() - begin)
    json_atomic(manifest_path, manifest)
    return out, manifest


class CategoryHead(nn.Module):
    def __init__(self, width, classes, seed):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(width))
        self.bias = nn.Parameter(torch.zeros(width))
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed + 310019)
            self.classifier = nn.Linear(128, classes)
            torch.random.default_generator.manual_seed(seed + 410023)
            self.projection = nn.Linear(width, 128)

    def forward(self, x):
        return self.classifier(F.gelu(self.projection(x * self.scale + self.bias)))


def metrics(scores, targets, known, prevalence):
    aps = [average_precision(targets[:, i], scores[:, i], known[:, i]) for i in range(targets.shape[1])]
    baseline = [average_precision(targets[:, i], np.full(len(targets), prevalence[i]), known[:, i]) for i in range(targets.shape[1])]
    kept = [value for value in aps if value is not None]
    return dict(macro_ap=float(np.mean(kept)), per_class_ap=aps,
        baseline_macro_ap=float(np.mean([v for v in baseline if v is not None])), baseline_per_class_ap=baseline,
        covered_classes=len(kept), classes=len(aps), positive=(targets * known).sum(0).tolist(),
        negative=((1 - targets) * known).sum(0).tolist(), unknown=(1 - known).sum(0).tolist(), frames=len(targets))


def train(encoder, seed, development=False, device='cpu', *, pool_override=None,
          output_override=None, protocol_path='docs/perception-semantic-protocol-2026-09-08.md'):
    setup(); out = Path(output_override) if output_override else ROOT / ('development/semantics' if development else 'semantics/fits') / f'seed_{seed}' / encoder
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'semantic_manifest.json').exists():
        raise FileExistsError('preserve semantic run')
    if pool_override is None:
        pool, pool_manifest = pooled_features(encoder)
    else:
        pool = Path(pool_override); pool_manifest = json.loads((pool / 'manifest.json').read_text())
        if pool_manifest['labels_sha256'] != file_hash(LABEL_ROOT / 'manifest.json'):
            raise ValueError('pooled category labels changed')
        for split, entry in pool_manifest['splits'].items():
            if file_hash(pool / f'{split}.npy') != entry['sha256']:
                raise ValueError('pooled category features changed')
    lm = json.loads((LABEL_ROOT / 'manifest.json').read_text()); classes = lm['retained_indices']
    data = {}
    for split in ('train', 'validation', 'test'):
        labels = dict(np.load(LABEL_ROOT / f'{split}.npz'))
        if file_hash(LABEL_ROOT / f'{split}.npz') != lm['splits'][split]['sha256']:
            raise ValueError('category labels mutated')
        size = (128 if split == 'train' else 64) if development else None
        data[split] = dict(features=np.load(pool / f'{split}.npy')[:size],
            targets=labels['targets'][:size, classes], known=labels['known'][:size, classes],
            rows=labels['rows'][:size], groups=labels['groups'][:size])
    config = dict(encoder=encoder, seed=seed, development=development, device=device,
        updates=30 if development else 2000, batch_size=64, validate_every=10 if development else 100,
        learning_rate=.0003, weight_decay=.0001, grad_clip=1., max_seconds=600,
        selector='maximum validation macro AP; earliest exact tie', classes=[lm['categories'][i] for i in classes])
    x = data['train']['features']; labels = data['train']; head = CategoryHead(x.shape[1], len(classes), seed).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    rng = np.random.default_rng(seed + 530029)
    prevalence = (labels['targets'] * labels['known']).sum(0) / labels['known'].sum(0).clip(1)
    manifest = dict(config=config, pooled_manifest_sha256=file_hash(pool / 'manifest.json'),
        source_sha256=file_hash(__file__), protocol_sha256=file_hash(protocol_path),
        parameters=sum(p.numel() for p in head.parameters()), versions=versions(), prevalence=prevalence.tolist())
    json_atomic(out / 'semantic_manifest.json', manifest)
    begin = time.monotonic(); best = None; selected = None; step = 0; status = 'completed'
    def evaluate(split):
        head.eval(); d = data[split]
        with torch.no_grad():
            scores = head(torch.from_numpy(d['features']).to(device)).sigmoid().cpu().numpy()
        return metrics(scores, d['targets'], d['known'], prevalence), scores
    def validate(update):
        nonlocal best, selected
        metric, _ = evaluate('validation')
        row = dict(step=update, elapsed_seconds=time.monotonic() - begin, **metric)
        append_json(out / 'validation.jsonl', row)
        state = dict(schema='category-head-v1', config=config, step=update,
                     head={n: p.detach().cpu().clone() for n, p in head.state_dict().items()}, metrics=metric)
        atomic_checkpoint(out / 'last.pt', state)
        if best is None or metric['macro_ap'] > best:
            selected = row; best = metric['macro_ap']; atomic_checkpoint(out / 'best.pt', state)
        print('semantic', encoder, seed, update, 'AP', round(metric['macro_ap'], 4), flush=True)
        return row
    final = validate(0)
    for step in range(1, config['updates'] + 1):
        if time.monotonic() - begin > config['max_seconds'] or datetime.now(timezone.utc) >= TRAIN_CUTOFF:
            status = 'stopped_wall_budget'; step -= 1; break
        ids = rng.integers(len(x), size=64)
        target = torch.from_numpy(labels['targets'][ids].copy()).float().to(device)
        known = torch.from_numpy(labels['known'][ids].copy()).float().to(device)
        head.train(); opt.zero_grad(set_to_none=True)
        logits = head(torch.from_numpy(x[ids].copy()).to(device))
        loss = known_bce(logits, target, known)
        loss.backward(); norm = nn.utils.clip_grad_norm_(head.parameters(), 1.)
        if not torch.isfinite(loss) or not torch.isfinite(norm):
            raise RuntimeError('nonfinite category head')
        opt.step()
        append_json(out / 'training.jsonl', dict(step=step, loss=float(loss.detach()), grad_norm=float(norm),
            sample_indices_sha256=digest(ids.tolist()), examples=step * 64, elapsed_seconds=time.monotonic() - begin))
        if step % config['validate_every'] == 0 or step == config['updates']:
            final = validate(step)
    if final['step'] != step:
        final = validate(step)
    result = dict(status=status, step=step, examples=step * 64, selected=selected, final=final,
                  seconds=time.monotonic() - begin, peak_cuda_bytes=None, device=device)
    sources = [out / name for name in ('semantic_manifest.json', 'training.jsonl', 'validation.jsonl')]
    evaluation = {}
    for checkpoint in ('selected', 'endpoint'):
        path = out / ('best.pt' if checkpoint == 'selected' else 'last.pt')
        state = torch.load(path, map_location='cpu', weights_only=False); head.load_state_dict(state['head'])
        evaluation[checkpoint] = dict(step=state['step'], checkpoint_sha256=file_hash(path))
        for split in ('train', 'validation', 'test'):
            metric, scores = evaluate(split); evaluation[checkpoint][split] = metric
            path = out / f'{checkpoint}_{split}_scores.npz'
            np.savez_compressed(path, scores=scores, **{k: v for k, v in data[split].items() if k != 'features'})
            sources.append(path)
    result['evaluation'] = evaluation
    json_atomic(out / 'result.json', result); sources.append(out / 'result.json')
    test = evaluation['selected']['test']
    analyze(ROOT, out, 'Generic category-accessibility readout',
        dict(test_macro_ap=test['macro_ap'], selected_validation_ap=selected['macro_ap'],
             baseline_test_ap=test['baseline_macro_ap'], updates=step, classes=len(classes), seconds=result['seconds']),
        f'## {encoder} category accessibility · seed{seed}\n\n'
        f'{"Development prefixes only. " if development else "Reused COCO grouped holdouts. "}'
        f'{step}updates; selected validation AP={selected["macro_ap"]:.4f}, test AP={test["macro_ap"]:.4f}. '
        f'Test covers{test["covered_classes"]}/{len(classes)} training-supported classes. '
        'Unknown crowd labels are excluded from loss and AP. This tests category presence, not localization or software competence.', sources)
    return result


def main():
    p = argparse.ArgumentParser(); p.add_argument('command', choices=['prepare', 'pool', 'train'])
    p.add_argument('--encoder', choices=['cnn', 'vit']); p.add_argument('--seed', type=int, default=9107)
    p.add_argument('--development', action='store_true'); p.add_argument('--device', default='cpu')
    args = p.parse_args(); setup()
    if args.command == 'prepare': prepare()
    elif args.command == 'pool': pooled_features(args.encoder)
    else: train(args.encoder, args.seed, args.development, args.device)


if __name__ == '__main__':
    main()
