"""Matched independent readout fits, complete raw ledgers and exact RNG resume."""
from __future__ import annotations
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .perception_cache import ROOT, cache_root, load_manifest, populations, validate_alignment, check_storage
from .perception_heads import make_heads, feature_grids
from .data import file_hash, digest
from .encoder_masks import masked_bce, mask_statistics
from .pose_accessibility import setup, pose_metrics, analyze
from world_model.pusht.checkpoints import (json_atomic, atomic_checkpoint, fingerprint_modules,
                                           rng_state, restore_rng, versions)

TRAIN_CUTOFF = datetime(2026, 9, 9, 4, 0, tzinfo=timezone.utc)


def checkpoint_state(heads, optimizers, rng, **metadata):
    return dict(schema='perception-readouts-v1',
        heads={k: {n: t.detach().cpu().clone() for n, t in h.state_dict().items()} for k, h in heads.items()},
        optimizers={k: o.state_dict() for k, o in optimizers.items()}, rng=rng_state(rng), **metadata)


def restore_checkpoint(state, heads, optimizers, rng):
    if state['schema'] != 'perception-readouts-v1':
        raise ValueError('incompatible readout checkpoint')
    for name, head in heads.items():
        head.load_state_dict(state['heads'][name])
        optimizers[name].load_state_dict(state['optimizers'][name])
    restore_rng(state['rng'], rng)


def selected_better(candidate, selected):
    # Earliest exact tie: other task metrics must not silently alter selection.
    return selected is None or candidate['q'] < selected['q']


def load_data(encoder, development):
    manifest = load_manifest(encoder, development)
    data = populations(development)
    result = {}
    for key, (ds, expected) in data.items():
        entry = manifest['entries']['/'.join(key)]
        labels = dict(np.load(entry['labels']))
        validate_alignment(expected['rows'], expected.get('targets'), labels['rows'], labels.get('targets'))
        result[key] = dict(ds=ds, labels=labels, cache=np.load(entry['features'], mmap_mode='r'))
    for domain in ('coco', 'pusht'):
        train = result[domain, 'train']; ds = train['ds']
        mean = np.zeros((64, 64, 3), dtype='float64')
        for start in range(0, len(ds), 128):
            mean += np.asarray(ds.frames[ds.rows[start:start + 128]], dtype='float64').sum(0)
        mean = (mean / (len(ds) * 255)).transpose(2, 0, 1).astype('float32')
        for split in ('train', 'validation', 'test'):
            result[domain, split]['train_mean_rgb'] = mean
    labels = result['coco', 'train']['labels']
    mean_mask = (labels['masks'] * labels['valid']).sum(0) / labels['valid'].sum(0).clip(1)
    for split in ('train', 'validation', 'test'):
        result['coco', split]['train_mean_mask'] = mean_mask.astype('float32')
    return result, manifest


def batch(data, ids, encoder, device):
    tokens = torch.from_numpy(np.asarray(data['cache'][ids]).copy()).to(device).float()
    fine, coarse = feature_grids(tokens, encoder)
    rgb, target = data['ds'].batch(ids, device)
    return fine, coarse, rgb, target


def optimizer_step(heads, optimizers, data, encoder, draws, device, pose_weights=None):
    for h in heads.values():
        h.train()
    for o in optimizers.values():
        o.zero_grad(set_to_none=True)
    stats = {}; rgb_loss = 0.
    for domain in ('coco', 'pusht'):
        d = data[domain, 'train']; ids = draws[domain]
        fine, coarse, rgb, targets = batch(d, ids, encoder, device)
        image_loss = F.mse_loss(heads['rgb'](fine, coarse), rgb)
        (image_loss * .5).backward()
        rgb_loss += float(image_loss.detach()) * .5
        if domain == 'coco':
            labels = d['labels']
            mask = torch.from_numpy(labels['masks'][ids].copy()).to(device).float().unsqueeze(1)
            valid = torch.from_numpy(labels['valid'][ids].copy()).to(device).float().unsqueeze(1)
            loss = masked_bce(heads['mask'](fine, coarse), mask, valid)
            stats['mask_bce'] = float(loss.detach())
        else:
            delta = (heads['pose'](fine, coarse) - targets).square()
            stats['pose_mse'] = float(delta.detach().mean())
            loss = delta.mean() if pose_weights is None else (delta * delta.new_tensor(pose_weights)).mean()
            stats['pose_objective'] = float(loss.detach())
        loss.backward()
    stats['image_mse'] = rgb_loss
    stats['loss'] = rgb_loss + stats['mask_bce'] + stats['pose_objective']
    for name, head in heads.items():
        norm = nn.utils.clip_grad_norm_(head.parameters(), 1.)
        if not torch.isfinite(norm) or not np.isfinite(stats['loss']):
            raise RuntimeError(f'nonfinite {name} update')
        stats['grad_norm_' + name] = float(norm)
    for o in optimizers.values():
        o.step()
    return stats


@torch.no_grad()
def evaluate_domain(heads, data, encoder, domain, split, device, panels=False, limit=None):
    for head in heads.values():
        head.eval()
    d = data[domain, split]; labels = d['labels']
    ids_all = np.arange(len(d['ds']))[:limit]
    raw = {}; chunks = {}; panel_data = {}
    def append(key, values):
        chunks.setdefault(key, []).append(np.asarray(values))
    for start in range(0, len(ids_all), 64):
        ids = ids_all[start:start + 64]
        fine, coarse, rgb, targets = batch(d, ids, encoder, device)
        reconstruction = heads['rgb'](fine, coarse)
        append('image_mse', (reconstruction - rgb).square().mean((1, 2, 3)).cpu().numpy())
        rgb_baseline = torch.from_numpy(d['train_mean_rgb']).to(device).unsqueeze(0)
        append('train_mean_image_mse', (rgb_baseline - rgb).square().mean((1, 2, 3)).cpu().numpy())
        if domain == 'pusht':
            prediction, locations, orientation = heads['pose'](fine, coarse, return_maps=True)
            append('predictions', prediction.cpu().double().numpy())
            if panels and start == 0:
                panel_data.update(pose=prediction[:6].cpu().numpy(), target=targets[:6].cpu().numpy(),
                                  locations=locations[:6].cpu().numpy(), orientation=orientation[:6].cpu().numpy())
        else:
            mask = torch.from_numpy(labels['masks'][ids].copy()).to(device).float().unsqueeze(1)
            valid = torch.from_numpy(labels['valid'][ids].copy()).to(device).float().unsqueeze(1)
            logits = heads['mask'](fine, coarse); probability = logits.sigmoid()
            bce = (F.binary_cross_entropy_with_logits(logits, mask, reduction='none') * valid).sum((1, 2, 3))
            append('mask_bce', (bce / valid.sum((1, 2, 3)).clamp_min(1)).cpu().numpy())
            for key, value in mask_statistics(probability, mask, valid).items():
                append(key, value)
            for name, constant in (('empty', 0.), ('full', 1.)):
                for key, value in mask_statistics(torch.full_like(probability, constant), mask, valid).items():
                    if key in ('iou', 'dice'):
                        append(name + '_' + key, value)
            mean_probability = torch.from_numpy(d['train_mean_mask']).to(device)[None, None].expand_as(probability)
            for key, value in mask_statistics(mean_probability, mask, valid).items():
                if key in ('iou', 'dice'):
                    append('train_mean_' + key, value)
            if panels and start == 0:
                panel_data.update(mask=mask[:6].cpu().numpy(), valid=valid[:6].cpu().numpy(),
                                  probability=probability[:6].cpu().numpy())
        if panels and start == 0:
            panel_data.update(rgb=rgb[:6].cpu().numpy(), reconstruction=reconstruction[:6].cpu().numpy(),
                              indices=ids[:6])
    raw.update({k: np.concatenate(v) for k, v in chunks.items()})
    metric = dict(frames=len(ids_all), image_mse=float(raw['image_mse'].mean()),
                  train_mean_image_mse=float(raw['train_mean_image_mse'].mean()))
    if domain == 'pusht':
        pose, pose_raw = pose_metrics(raw['predictions'], labels['targets'][ids_all].astype('float64'))
        metric.update(pose); raw.update(pose_raw)
        groups = labels['groups'][ids_all]
        group_positions = np.array([raw['position_abs_error'][groups == g].mean(0) for g in np.unique(groups)])
        group_angles = np.array([raw['angle_abs_error_deg'][groups == g].mean() for g in np.unique(groups)])
        metric.update(group_position_mae=group_positions.mean(0).tolist(),
                      group_angle_mae_deg=float(group_angles.mean()), groups=len(np.unique(groups)))
    else:
        keep = raw['has_valid'].astype(bool)
        metric['mask_frames'] = int(keep.sum())
        for key in ('mask_bce', 'iou', 'dice', 'empty_iou', 'full_iou', 'empty_dice', 'full_dice',
                    'train_mean_iou', 'train_mean_dice'):
            metric[key] = float(raw[key][keep].mean())
    raw.update(indices=ids_all, source_rows=labels['rows'][ids_all], groups=labels['groups'][ids_all])
    return metric, raw, panel_data


def validation(heads, data, encoder, device):
    p, _, _ = evaluate_domain(heads, data, encoder, 'pusht', 'validation', device)
    c, _, _ = evaluate_domain(heads, data, encoder, 'coco', 'validation', device)
    return dict(p, image_mse=(p['image_mse'] + c['image_mse']) / 2,
                pusht_image_mse=p['image_mse'], coco_image_mse=c['image_mse'], mask_bce=c['mask_bce'],
                mask_iou=c['iou'], mask_dice=c['dice'], loss=p['pose_mse'] + c['mask_bce'] +
                (p['image_mse'] + c['image_mse']) / 2)


def append_json(path, row):
    with Path(path).open('a') as f:
        f.write(json.dumps(row, allow_nan=False) + '\n')


def recover_ledgers(out, step):
    """Preserve, then replay only work beyond the last atomic checkpoint."""
    for name in ('training', 'validation'):
        path = out / f'{name}.jsonl'
        if not path.exists():
            continue
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        tail = [r for r in rows if r['step'] > step]
        if tail:
            stamp = time.time_ns()
            (out / f'{name}.discarded-{stamp}.json').write_text(json.dumps(tail, indent=2) + '\n')
            path.write_text(''.join(json.dumps(r) + '\n' for r in rows if r['step'] <= step))


def train(encoder, seed, development=False, device='cuda', resume=False):
    setup(); check_storage(300 * 1024**2)
    out = ROOT / ('development/p1' if development else 'p1') / f'seed_{seed}' / encoder
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'curriculum_result.json').exists():
        if resume and not (out / 'curriculum_analysis.json').exists():
            return evaluate_saved(out, encoder, development, device)
        raise FileExistsError('preserve completed/stopped run; explicit new arm required')
    if (out / 'curriculum_manifest.json').exists() and not resume:
        raise FileExistsError('use explicit resume for interrupted run')
    config = dict(seed=seed, encoder=encoder, arm=encoder, phase='supervised',
        updates=50 if development else 4000, batch_size=64, microbatch_size=32,
        validate_every=10 if development else 100, learning_rate=.0003, weight_decay=.0001,
        grad_clip=1., max_seconds=1200 if development else 2400, development=development, access='both',
        protocol='overnight P1: frozen packages; independent spatial heads; full validation q selection',
        precision='FP32 compute, TF32 off, audited FP16 cache', selector='minimum q; earliest exact tie')
    heads = make_heads(64 if encoder == 'cnn' else 384, seed, device)
    optimizers = {k: torch.optim.AdamW(h.parameters(), lr=config['learning_rate'],
        weight_decay=config['weight_decay']) for k, h in heads.items()}
    rng = np.random.default_rng(seed + 230003)
    data, cache_manifest = load_data(encoder, development)
    manifest = dict(config=config, dataset=digest({k: e['dataset'] for k, e in cache_manifest['entries'].items()}),
        cache_manifest_sha256=file_hash(cache_root(development) / encoder / 'manifest.json'),
        initial=fingerprint_modules(heads), parameters={k: sum(p.numel() for p in h.parameters()) for k, h in heads.items()},
        source_hashes={str(p): file_hash(p) for p in [Path(__file__), Path('world_model/curriculum/perception_heads.py'),
                                                    Path('world_model/curriculum/perception_cache.py')]}, versions=versions())
    begin = time.monotonic(); previous_seconds = 0.; selected = None; step = 0; selected_heads = None
    if resume:
        prior = json.loads((out / 'curriculum_manifest.json').read_text())
        if prior != manifest:
            raise ValueError('resume identity changed')
        state = torch.load(out / 'last.pt', map_location='cpu', weights_only=False)
        restore_checkpoint(state, heads, optimizers, rng)
        step = state['step']; selected = state['selected']; previous_seconds = state['elapsed_seconds']
        selected_heads = state['selected_heads']
        recover_ledgers(out, step)
        best_state = dict(state, heads=selected_heads, step=selected['step'])
        atomic_checkpoint(out / 'best.pt', best_state)
    else:
        json_atomic(out / 'curriculum_manifest.json', manifest)
        (out / 'training.jsonl').touch()
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    def elapsed():
        return previous_seconds + time.monotonic() - begin
    def validate(update):
        nonlocal selected, selected_heads
        row = dict(step=update, elapsed_seconds=elapsed(), **validation(heads, data, encoder, device))
        improved = selected_better(row, selected)
        if improved:
            selected = row
            selected_heads = {k: {n: t.detach().cpu().clone() for n, t in h.state_dict().items()} for k, h in heads.items()}
        state = checkpoint_state(heads, optimizers, rng, step=update, selected=selected,
                                 selected_heads=selected_heads, elapsed_seconds=elapsed(), config=config,
                                 manifest_sha256=file_hash(out / 'curriculum_manifest.json'))
        # Atomic checkpoint is authoritative on resume; a ledger tail is preserved.
        append_json(out / 'validation.jsonl', row)
        atomic_checkpoint(out / 'last.pt', state)
        if improved:
            atomic_checkpoint(out / 'best.pt', state)
        print(encoder, seed, 'step', update, 'q', round(row['q'], 4), 'angle', round(row['angle_mae_deg'], 3),
              'maskIoU', round(row['mask_iou'], 4), 'seconds', round(elapsed(), 1), flush=True)
        return row
    if not resume:
        final = validate(0)
    else:
        final = json.loads((out / 'validation.jsonl').read_text().splitlines()[-1])
    status = 'completed'
    try:
        for update in range(step + 1, config['updates'] + 1):
            if elapsed() >= config['max_seconds'] or datetime.now(timezone.utc) >= TRAIN_CUTOFF:
                status = 'stopped_wall_budget'; break
            draws = {domain: rng.integers(len(data[domain, 'train']['ds']), size=32) for domain in ('coco', 'pusht')}
            stats = optimizer_step(heads, optimizers, data, encoder, draws, device)
            step = update
            append_json(out / 'training.jsonl', dict(step=step, **stats, examples=step * 64,
                sample_indices_sha256=digest({k: v.tolist() for k, v in draws.items()}), elapsed_seconds=elapsed()))
            if step % config['validate_every'] == 0 or step == config['updates']:
                final = validate(step)
    except (KeyboardInterrupt, RuntimeError) as error:
        # Preserve the last atomic valid state rather than describing a partial
        # optimizer step as a valid checkpoint. Explicit resume remains possible.
        json_atomic(out / 'interruption.json', dict(type=type(error).__name__, error=str(error),
                                                   last_completed_update=step, elapsed_seconds=elapsed()))
        raise
    if final['step'] != step:
        final = validate(step)
    result = dict(status=status, step=step, selected_step=selected['step'], selected=selected,
        final=final, examples=step * 64, elapsed_seconds=elapsed(),
        peak_cuda_bytes=torch.cuda.max_memory_allocated() if device == 'cuda' else None,
        readiness_pass=bool(selected['q'] <= 1))
    json_atomic(out / 'curriculum_result.json', result)
    return evaluate_saved(out, encoder, development, device, heads=heads, data=data)


def evaluate_saved(out, encoder, development, device, heads=None, data=None):
    """Evaluation can be repaired independently without repeating optimizer work."""
    started = time.monotonic()
    result = json.loads((out / 'curriculum_result.json').read_text())
    config = json.loads((out / 'curriculum_manifest.json').read_text())['config']
    seed = config['seed']; selected = result['selected']; status = result['status']; step = result['step']
    if heads is None:
        heads = make_heads(64 if encoder == 'cnn' else 384, seed, device)
    if data is None:
        data, _ = load_data(encoder, development)
    sources = [out / name for name in ('curriculum_manifest.json', 'curriculum_result.json', 'training.jsonl', 'validation.jsonl')]
    evaluation = {}
    for checkpoint in ('selected', 'endpoint'):
        state = torch.load(out / ('best.pt' if checkpoint == 'selected' else 'last.pt'), map_location='cpu', weights_only=False)
        for k, h in heads.items():
            h.load_state_dict(state['heads'][k])
        evaluation[checkpoint] = dict(step=state['step'], checkpoint_sha256=file_hash(out / ('best.pt' if checkpoint == 'selected' else 'last.pt')))
        for split in ('train', 'validation', 'test'):
            evaluation[checkpoint][split] = {}
            for domain in ('coco', 'pusht'):
                metric, raw, panels = evaluate_domain(heads, data, encoder, domain, split, device,
                    panels=checkpoint == 'selected' and split == 'test', limit=512 if split == 'train' else None)
                evaluation[checkpoint][split][domain] = metric
                path = out / f'{checkpoint}_{domain}_{split}_errors.npz'
                np.savez_compressed(path, **raw); sources.append(path)
                if panels:
                    path = out / f'{domain}_panels.npz'; np.savez_compressed(path, **panels); sources.append(path)
    evaluation['scope'] = 'Reused grouped held-outs; fixed encoder checkpoints; three head seeds only. No dynamics/control evidence.'
    evaluation['seconds'] = time.monotonic() - started
    json_atomic(out / 'evaluation.json', evaluation); sources.append(out / 'evaluation.json')
    test = evaluation['selected']['test']
    analyze(ROOT, out, 'Frozen representation and independent typed readouts',
        dict(test_q=test['pusht']['q'], selected_q=selected['q'], test_mask_iou=test['coco']['iou'],
             test_coco_mse=test['coco']['image_mse'], test_pusht_mse=test['pusht']['image_mse']),
        f'## {encoder} independent readouts · seed{seed}\n\n'
        f'{"Development only. " if development else "Exploratory checkpoint-package comparison. "}'
        f'Status {status}, {step} updates. Pose-selected validation q={selected["q"]:.4f}; '
        f'test q={test["pusht"]["q"]:.4f}, COCO mask IoU={test["coco"]["iou"]:.4f}. '
        'RGB and mask heads are reported at the pose-selected snapshot, with their endpoint recorded separately. '
        'The encoder is frozen; this does not establish prediction or control quality.', sources)
    return result
