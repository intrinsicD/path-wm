"""Prospective simulator observations and explicit source-renderer calibration."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch

from .perception_cache import ROOT, CNN_SHA
from .data import task_frames, file_hash, digest
from .pose_accessibility import setup, analyze, pose_metrics
from .encoder_probe import source_models
from .encoder_reference import pinned_backbone
from .perception_heads import make_heads, feature_grids
from world_model.pusht.perception_data import prepare_supplement, verify_supplement, _targets
from world_model.pusht.env import PushTEnv
from world_model.pusht.checkpoints import json_atomic, fingerprint_modules

DATA = Path('data/pusht_world_model/perception_fresh_2026-09-08')
FRESH = ROOT / 'fresh'


def world_pose(target):
    target = np.asarray(target)
    return np.column_stack((target[:, :4] * 512, np.arctan2(target[:, 4], target[:, 5])))


def nearest_pose_distance(candidates, reference):
    """Minimum L-infinity distance in declared position/angle tolerance units."""
    result = []
    for start in range(0, len(candidates), 32):
        current = candidates[start:start + 32]
        xy = np.abs(current[:, None, :4] - reference[None, :, :4]).max(-1) / 8
        angle = current[:, None, 4] - reference[None, :, 4]
        angle = np.abs(np.arctan2(np.sin(angle), np.cos(angle))) / np.deg2rad(10)
        result.extend(np.maximum(xy, angle).min(1))
    return np.asarray(result)


def prepare():
    out = FRESH / 'population'; out.mkdir(parents=True, exist_ok=True)
    if (out / 'manifest.json').exists():
        raise FileExistsError('preserve fresh cohort')
    begin = time.monotonic()
    manifest = prepare_supplement(DATA, count=512, seed=20260909)
    verified = verify_supplement(DATA)
    poses = np.load(DATA / 'poses.npy'); frames = np.load(DATA / 'frames.npy', mmap_mode='r')
    overlap = {}; sources = []; distances = {}
    for split in ('train', 'validation', 'test'):
        ds = task_frames('data/pusht_world_model/cchi_v1', split)
        distance = nearest_pose_distance(poses, world_pose(ds.targets))
        distances[split] = distance
        overlap[split] = dict(reference_frames=len(ds), near_count=int((distance <= 1).sum()),
                              minimum_distance=float(distance.min()), dataset=ds.fingerprint)
    old = Path('data/pusht_world_model/pose_supplement_v1')
    if (old / 'manifest.json').exists():
        verify_supplement(old)
        distance = nearest_pose_distance(poses, np.load(old / 'poses.npy'))
        distances['old_supplement'] = distance
        overlap['old_supplement'] = dict(reference_frames=len(np.load(old / 'poses.npy')),
            near_count=int((distance <= 1).sum()), minimum_distance=float(distance.min()),
            manifest_sha256=file_hash(old / 'manifest.json'))
    else:
        overlap['old_supplement'] = dict(status='unavailable')
    path = out / 'nearest_distances.npz'; np.savez_compressed(path, **distances); sources.append(path)
    json_atomic(out / 'overlap.json', overlap); sources.append(out / 'overlap.json')
    ds = task_frames('data/pusht_world_model/cchi_v1', 'validation')
    groups = np.array([row['group'] for row in ds.metadata]); rng = np.random.default_rng(20260910)
    ids = np.array([i for group in np.unique(groups) for i in np.sort(rng.choice(np.flatnonzero(groups == group), 2, replace=False))])
    requested = world_pose(ds.targets[ids]); actual = []; rendered = []
    env = PushTEnv()
    try:
        for i, pose in enumerate(requested):
            frame, _ = env.reset(pose5_world=pose, seed=20260910 + i)
            actual.append(env.pose); rendered.append(frame)
    finally:
        env.close()
    actual = np.asarray(actual); rendered = np.asarray(rendered)
    original = np.asarray(ds.frames[ds.rows[ids]]).copy()
    drift, drift_raw = pose_metrics(_targets(actual).astype('float64'), ds.targets[ids].astype('float64'))
    rgb_mse = np.square((rendered.astype('float64') - original) / 255).mean((1, 2, 3))
    path = out / 'calibration.npz'
    np.savez_compressed(path, original=original, rendered=rendered, source_targets=ds.targets[ids],
        actual_targets=_targets(actual), requested=requested, actual=actual, indices=ids,
        source_rows=ds.rows[ids], groups=groups[ids], image_mse=rgb_mse, **drift_raw)
    sources.append(path)
    report = dict(status='completed', frames=512, generator_manifest_sha256=file_hash(DATA / 'manifest.json'),
        dataset=str(DATA), verified=verified, overlap=overlap,
        out_of_image_coordinate_frames=int(((poses[:, :4] < 0) | (poses[:, :4] > 512)).any(1).sum()),
        calibration=dict(frames=len(ids), groups=len(np.unique(groups[ids])), requested_actual_drift=drift,
            source_render_image_mse=float(rgb_mse.mean()), source_render_image_mse_p95=float(np.percentile(rgb_mse, 95))),
        seconds=time.monotonic() - begin, source_sha256=file_hash(__file__),
        protocol_sha256=file_hash('docs/perception-fresh-protocol-2026-09-08.md'),
        files={str(p): file_hash(p) for p in sources})
    json_atomic(out / 'manifest.json', report); sources.append(out / 'manifest.json')
    analyze(ROOT, out, 'Fresh simulator cohort and renderer calibration',
        dict(frames=512, calibration_frames=len(ids), calibration_image_mse=float(rgb_mse.mean())),
        '## Fresh simulator perception cohort\n\n512 independently requested resets; actual post-contact labels retained. '
        'Near-match audits do not filter cases. Forty source-versus-render calibration pairs measure renderer and reset shifts. '
        'This is a new simulator stress population, not CCHI test data or a control evaluation.', sources)
    return report


def cohorts():
    manifest = json.loads((FRESH / 'population/manifest.json').read_text())
    verify_supplement(DATA)
    if file_hash(DATA / 'manifest.json') != manifest['generator_manifest_sha256']:
        raise ValueError('fresh population changed')
    for path, h in manifest['files'].items():
        if file_hash(path) != h: raise ValueError('fresh audit/calibration changed')
    cal = dict(np.load(FRESH / 'population/calibration.npz'))
    return dict(fresh=(np.load(DATA / 'frames.npy'), np.load(DATA / 'pose_targets.npy')),
        calibration_original=(cal['original'], cal['source_targets']),
        calibration_rendered=(cal['rendered'], cal['actual_targets']))


@torch.no_grad()
def evaluate(encoder, seed, device='cuda'):
    setup(); out = FRESH / 'evaluations' / f'seed_{seed}' / encoder; out.mkdir(parents=True, exist_ok=True)
    if (out / 'evaluation.json').exists(): raise FileExistsError('preserve fresh evaluation')
    checkpoint = ROOT / 'p1' / f'seed_{seed}' / encoder / 'best.pt'
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    heads = make_heads(64 if encoder == 'cnn' else 384, seed, device)
    for key, head in heads.items(): head.load_state_dict(state['heads'][key]); head.eval().requires_grad_(False)
    if encoder == 'cnn':
        models, source = source_models('deeper', 7107, False, device)
        if file_hash(source) != CNN_SHA: raise ValueError('source encoder changed')
        model = models['E']
    else:
        model = pinned_backbone(device)
    before = fingerprint_modules({'E': model, **heads}); begin = time.monotonic(); results = {}; sources = []
    for name, (frames, targets) in cohorts().items():
        prediction = []; pixels = []; maps = []; pools = []; reconstruction = []
        for start in range(0, len(frames), 32):
            rgb = torch.from_numpy(frames[start:start + 32].copy()).permute(0, 3, 1, 2).to(device).float() / 255
            tokens = model(rgb); tokens = tokens.tokens() if encoder == 'cnn' else tokens
            # Match the audited cache precision used to fit these heads.
            fine, coarse = feature_grids(tokens.half().float(), encoder)
            pose, location, orientation = heads['pose'](fine, coarse, return_maps=True)
            image = heads['rgb'](fine, coarse)
            prediction.append(pose.cpu().double().numpy()); maps.append(location.cpu().numpy())
            pools.append(orientation.cpu().numpy()); reconstruction.append(image.cpu().numpy())
            pixels.extend((image - rgb).square().mean((1, 2, 3)).cpu().tolist())
        metric, raw = pose_metrics(np.concatenate(prediction), targets.astype('float64'))
        case_q = np.maximum(raw['position_abs_error'].max(1) / 8, raw['angle_abs_error_deg'] / 10)
        metric.update(image_mse=float(np.mean(pixels)), case_q_median=float(np.median(case_q)),
            case_q_p95=float(np.percentile(case_q, 95)), per_case_tolerance_pass=float((case_q <= 1).mean()),
            worst_case_q=float(case_q.max()))
        path = out / f'{name}_errors.npz'
        np.savez_compressed(path, **raw, image_mse=np.array(pixels), case_q=case_q,
            indices=np.arange(len(frames)), locations=np.concatenate(maps), orientation_pool=np.concatenate(pools))
        sources.append(path); results[name] = metric
        path = out / f'{name}_images.npz'
        np.savez_compressed(path, rgb=frames, reconstruction=np.concatenate(reconstruction)); sources.append(path)
    if before != fingerprint_modules({'E': model, **heads}): raise RuntimeError('evaluation changed frozen state')
    report = dict(encoder=encoder, seed=seed, selected_step=state['step'], checkpoint=str(checkpoint),
        checkpoint_sha256=file_hash(checkpoint), frozen_fingerprint=before, metrics=results,
        seconds=time.monotonic() - begin, population_sha256=file_hash(FRESH / 'population/manifest.json'),
        source_sha256=file_hash(__file__), scope='Fresh simulator stress and paired renderer calibration; no adaptation or control')
    json_atomic(out / 'evaluation.json', report); sources.append(out / 'evaluation.json')
    m = results['fresh']
    analyze(ROOT, out, 'Fresh frozen perception evaluation',
        dict(fresh_q=m['q'], fresh_case_pass=m['per_case_tolerance_pass'], fresh_case_q_p95=m['case_q_p95']),
        f'## Fresh simulator · {encoder} seed{seed}\n\nMean q={m["q"]:.4f}; '
        f'{100*m["per_case_tolerance_pass"]:.1f}% of cases meet all coordinate/angle tolerances. '
        'All512 cases retained, no adaptation. Source/render calibration and raw per-case errors remain separate. '
        'Decoder heatmaps are output distributions, not encoder attention.', sources)
    return report


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('command', choices=['prepare', 'evaluate'])
    parser.add_argument('--encoder', choices=['cnn', 'vit']); parser.add_argument('--seed', type=int, default=9107)
    parser.add_argument('--device', default='cuda'); args = parser.parse_args()
    if args.command == 'prepare': prepare()
    else: evaluate(args.encoder, args.seed, args.device)


if __name__ == '__main__': main()
