"""Mixed-domain training of matched encoder extensions and independent heads."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .perception_cache import ROOT, CNN_SHA, check_storage
from .perception_extensions import ContinuationEncoder, encoder_optimizer_groups
from .perception_heads import make_heads
from .perception_training import (load_data, validation, evaluate_domain, append_json, selected_better,
    checkpoint_state, restore_checkpoint, recover_ledgers, TRAIN_CUTOFF)
from .encoder_masks import masked_bce
from .data import file_hash, digest
from .pose_accessibility import setup, analyze
from world_model.pusht.checkpoints import read_checkpoint, atomic_checkpoint, json_atomic, fingerprint_modules, versions

SOURCE = Path('runs/encoder_study_2026-09-08/factorial/seed_7107/deeper/best.pt')


class LiveFeatures:
    """Read-only evaluator adapter; use the trained encoder, not its old cache."""
    def __init__(self, encoder, dataset, device):
        self.encoder, self.dataset, self.device = encoder, dataset, device

    def __getitem__(self, ids):
        with torch.no_grad():
            rgb, _ = self.dataset.batch(ids, self.device)
            return self.encoder(rgb).tokens().cpu().numpy()


def system(kind, seed, device):
    if file_hash(SOURCE) != CNN_SHA: raise ValueError('fixed source CNN changed')
    source = read_checkpoint(SOURCE)
    encoder = ContinuationEncoder(kind, seed, source['models']['E']).to(device)
    return encoder, make_heads(64, seed, device)


def step_update(encoder, heads, optimizers, data, draws, device, audit):
    encoder.train()
    for h in heads.values(): h.train()
    for o in optimizers.values(): o.zero_grad(set_to_none=True)
    stats = {}; pixels = 0.; generic_grad = None
    parameters = list(encoder.parameters())
    for domain in ('coco', 'pusht'):
        d = data[domain, 'train']; ids = draws[domain]
        rgb, targets = d['ds'].batch(ids, device)
        z = encoder(rgb); image = F.mse_loss(heads['rgb'](z.fine, z.coarse), rgb)
        pixels += float(image.detach()) * .5
        if domain == 'coco':
            labels = d['labels']
            mask = torch.from_numpy(labels['masks'][ids].copy()).to(device).float().unsqueeze(1)
            valid = torch.from_numpy(labels['valid'][ids].copy()).to(device).float().unsqueeze(1)
            task_loss = masked_bce(heads['mask'](z.fine, z.coarse), mask, valid)
            stats['mask_bce'] = float(task_loss.detach())
        else:
            task_loss = F.mse_loss(heads['pose'](z.fine, z.coarse), targets)
            stats['pose_mse'] = float(task_loss.detach())
        (.5 * image + task_loss).backward()
        if audit and domain == 'coco':
            generic_grad = [torch.zeros_like(p) if p.grad is None else p.grad.detach().clone() for p in parameters]
    stats.update(image_mse=pixels, loss=pixels + stats['mask_bce'] + stats['pose_mse'])
    if audit:
        manipulation = [(torch.zeros_like(p) if p.grad is None else p.grad.detach()) - g
                        for p, g in zip(parameters, generic_grad)]
        a2 = torch.stack([g.square().sum() for g in generic_grad]).sum()
        b2 = torch.stack([g.square().sum() for g in manipulation]).sum()
        dot = torch.stack([(a * b).sum() for a, b in zip(generic_grad, manipulation)]).sum()
        stats.update(encoder_generic_grad_norm=float(a2.sqrt()), encoder_manipulation_grad_norm=float(b2.sqrt()),
                     encoder_domain_gradient_cosine=float(dot / (a2 * b2).sqrt().clamp_min(1e-20)))
        for level, modules in encoder.extensions.items():
            values = [p.grad.square().sum() for name, p in modules.named_parameters() if '.gate' not in name and p.grad is not None]
            stats['extension_grad_norm_' + level] = float(torch.stack(values).sum().sqrt()) if values else 0.
    for name, module in {'encoder': encoder, **heads}.items():
        norm = nn.utils.clip_grad_norm_(module.parameters(), 1.)
        if not torch.isfinite(norm) or not np.isfinite(stats['loss']): raise RuntimeError('nonfinite continuation update')
        stats['grad_norm_' + name] = float(norm)
    for o in optimizers.values(): o.step()
    if audit:
        stats['gates'] = {name: float(p.detach()) for name, p in encoder.named_parameters() if '.gate' in name}
    return stats


def train(kind, seed, development=False, device='cuda', resume=False):
    setup(); check_storage(300 * 1024**2)
    out = ROOT / ('development/extensions' if development else 'extensions') / f'seed_{seed}' / kind
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'curriculum_result.json').exists():
        if resume and not (out / 'curriculum_analysis.json').exists(): return evaluate_saved(out, kind, seed, development, device)
        raise FileExistsError('preserve completed/stopped continuation')
    if (out / 'curriculum_manifest.json').exists() and not resume: raise FileExistsError('explicit resume required')
    encoder, heads = system(kind, seed, device); all_models = {'encoder': encoder, **heads}
    optimizers = {name: torch.optim.AdamW(h.parameters(), lr=3e-4, weight_decay=1e-4) for name, h in heads.items()}
    optimizers['encoder'] = torch.optim.AdamW(encoder_optimizer_groups(encoder))
    config = dict(seed=seed, kind=kind, arm=kind, phase='supervised', updates=50 if development else 4000,
        batch_size=64, microbatch_size=32, validate_every=10 if development else 100,
        max_seconds=1200, base_learning_rate=3e-5, new_learning_rate=3e-4, weight_decay=1e-4,
        grad_clip=1., development=development, protocol='overnight extensions: joint mixed supervision and pre-exchange modules',
        precision='FP32, TF32 off; live features, no quantized training cache',
        selector='minimum validation q; earliest exact tie', access='both')
    data, cache_manifest = load_data('cnn', development)
    for item in data.values(): item['cache'] = LiveFeatures(encoder, item['ds'], device)
    rng = np.random.default_rng(seed + 230003)
    manifest = dict(config=config, dataset=digest({k: e['dataset'] for k, e in cache_manifest['entries'].items()}),
        source_checkpoint_sha256=CNN_SHA, initial=fingerprint_modules(all_models),
        parameters={k: sum(p.numel() for p in m.parameters()) for k, m in all_models.items()},
        source_hashes={str(p): file_hash(p) for p in [Path(__file__), Path('world_model/curriculum/perception_extensions.py'),
            Path('world_model/curriculum/perception_training.py'), Path('world_model/curriculum/perception_heads.py')]}, versions=versions())
    begin = time.monotonic(); previous = 0.; selected = None; selected_models = None; step = 0
    if resume:
        if json.loads((out / 'curriculum_manifest.json').read_text()) != manifest: raise ValueError('continuation resume identity changed')
        state = torch.load(out / 'last.pt', map_location='cpu', weights_only=False)
        restore_checkpoint(state, all_models, optimizers, rng)
        previous = state['elapsed_seconds']; selected = state['selected']; selected_models = state['selected_models']; step = state['step']
        recover_ledgers(out, step)
        atomic_checkpoint(out / 'best.pt', dict(state, heads=selected_models, step=selected['step']))
    else:
        json_atomic(out / 'curriculum_manifest.json', manifest); (out / 'training.jsonl').touch()
    if device == 'cuda': torch.cuda.reset_peak_memory_stats()
    def elapsed(): return previous + time.monotonic() - begin
    def validate(update):
        nonlocal selected, selected_models
        encoder.eval()
        row = dict(step=update, elapsed_seconds=elapsed(), **validation(heads, data, 'cnn', device))
        improved = selected_better(row, selected)
        if improved:
            selected = row
            selected_models = {k: {n: p.detach().cpu().clone() for n, p in m.state_dict().items()} for k, m in all_models.items()}
        state = checkpoint_state(all_models, optimizers, rng, step=update, selected=selected,
            selected_models=selected_models, elapsed_seconds=elapsed(), config=config,
            manifest_sha256=file_hash(out / 'curriculum_manifest.json'))
        append_json(out / 'validation.jsonl', row); atomic_checkpoint(out / 'last.pt', state)
        if improved: atomic_checkpoint(out / 'best.pt', state)
        print(kind, seed, update, 'q', round(row['q'], 4), 'IoU', round(row['mask_iou'], 4), 'seconds', round(elapsed(), 1), flush=True)
        return row
    final = json.loads((out / 'validation.jsonl').read_text().splitlines()[-1]) if resume else validate(0)
    status = 'completed'
    try:
        for update in range(step + 1, config['updates'] + 1):
            if elapsed() >= config['max_seconds'] or datetime.now(timezone.utc) >= TRAIN_CUTOFF:
                status = 'stopped_wall_budget'; break
            draws = {d: rng.integers(len(data[d, 'train']['ds']), size=32) for d in ('coco', 'pusht')}
            stats = step_update(encoder, heads, optimizers, data, draws, device, update <= 100 or update % 100 == 0)
            step = update
            append_json(out / 'training.jsonl', dict(step=step, **stats, examples=step * 64,
                sample_indices_sha256=digest({k: v.tolist() for k, v in draws.items()}), elapsed_seconds=elapsed()))
            if step % config['validate_every'] == 0 or step == config['updates']: final = validate(step)
    except (KeyboardInterrupt, RuntimeError) as error:
        json_atomic(out / 'interruption.json', dict(type=type(error).__name__, error=str(error), step=step, elapsed_seconds=elapsed()))
        raise
    if final['step'] != step: final = validate(step)
    result = dict(status=status, step=step, selected_step=selected['step'], selected=selected, final=final,
        examples=step * 64, elapsed_seconds=elapsed(), readiness_pass=bool(selected['q'] <= 1),
        peak_cuda_bytes=torch.cuda.max_memory_allocated() if device == 'cuda' else None)
    json_atomic(out / 'curriculum_result.json', result)
    return evaluate_saved(out, kind, seed, development, device, encoder, heads, data)


def evaluate_saved(out, kind, seed, development, device, encoder=None, heads=None, data=None):
    started = time.monotonic(); result = json.loads((out / 'curriculum_result.json').read_text())
    if encoder is None: encoder, heads = system(kind, seed, device)
    if data is None:
        data, _ = load_data('cnn', development)
        for item in data.values(): item['cache'] = LiveFeatures(encoder, item['ds'], device)
    sources = [out / n for n in ('curriculum_manifest.json', 'curriculum_result.json', 'training.jsonl', 'validation.jsonl')]
    evaluation = {}
    for choice in ('selected', 'endpoint'):
        path = out / ('best.pt' if choice == 'selected' else 'last.pt')
        state = torch.load(path, map_location='cpu', weights_only=False)
        for k, model in {'encoder': encoder, **heads}.items(): model.load_state_dict(state['heads'][k]); model.eval()
        before = fingerprint_modules({'encoder': encoder, **heads})
        evaluation[choice] = dict(step=state['step'], checkpoint_sha256=file_hash(path))
        for split in ('train', 'validation', 'test'):
            evaluation[choice][split] = {}
            for domain in ('coco', 'pusht'):
                metric, raw, panels = evaluate_domain(heads, data, 'cnn', domain, split, device,
                    panels=choice == 'selected' and split == 'test', limit=512 if split == 'train' else None)
                evaluation[choice][split][domain] = metric
                path = out / f'{choice}_{domain}_{split}_errors.npz'; np.savez_compressed(path, **raw); sources.append(path)
                if panels:
                    path = out / f'{domain}_panels.npz'; np.savez_compressed(path, **panels); sources.append(path)
        if before != fingerprint_modules({'encoder': encoder, **heads}): raise RuntimeError('evaluation mutated encoder or readouts')
    evaluation.update(seconds=time.monotonic() - started,
        scope='Matched mixed-supervision continuation; fixed source base, three head/new-module seeds; no architecture/pretraining causal claim')
    json_atomic(out / 'evaluation.json', evaluation); sources.append(out / 'evaluation.json')
    test = evaluation['selected']['test']
    analyze(ROOT, out, 'Mixed-supervision encoder continuation',
        dict(test_q=test['pusht']['q'], selected_q=result['selected']['q'], test_mask_iou=test['coco']['iou'],
             test_coco_mse=test['coco']['image_mse'], test_pusht_mse=test['pusht']['image_mse']),
        f'## {kind} encoder continuation · seed{seed}\n\n'
        f'{"Development only. " if development else "Adaptive matched study. "}'
        f'Status {result["status"]}, {result["step"]}updates. Selected validation q={result["selected"]["q"]:.4f}, '
        f'test q={test["pusht"]["q"]:.4f}, mask IoU={test["coco"]["iou"]:.4f}. '
        'The encoder now updates under mixed supervision. Added modules preserve its initial function; '
        'parameter/compute counts differ. No prediction or control claim.', sources)
    return result


def main():
    p = argparse.ArgumentParser(); p.add_argument('--kind', choices=['joint', 'conv', 'transformer'], required=True)
    p.add_argument('--seed', type=int, default=9107); p.add_argument('--development', action='store_true')
    p.add_argument('--resume', action='store_true'); p.add_argument('--device', default='cuda'); a = p.parse_args()
    train(a.kind, a.seed, a.development, a.device, a.resume)


if __name__ == '__main__': main()
