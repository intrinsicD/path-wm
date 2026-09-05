"""Known physical audiovisual correspondence through the existing normalized data boundary.

Four emitters move in separate quadrants; their eight oscillator coordinates modulate eight tones.
Pixels and zero-order-held amplitudes share one physical grid. Independent recording seeds set
motion/background/phase; audit truth stays outside RepresentationBatch and every model objective.
This source and its raw readout isolate the timing question in E1_common_base (DDR §38).
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import io
import json
import math
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F
import yaml

from training.av_data import AVClipRecord, _integer_count, _resolve, _sha256, _validate_shard


def _settings(cfg: Mapping[str, Any]) -> tuple[Mapping[str, Any], int, int, int]:
    data = cfg['data']
    if data.get('dataset') != 'controlled_oscillators_v1' or data.get('kind') != 'synchronized_av_manifest':
        raise ValueError('controlled source requires its named synchronized manifest config')
    source = data['source']
    fps, rate, resolution = (int(data['video']['frames_per_second']),
                             int(data['audio']['sample_rate']), int(data['video']['resolution']))
    if fps < 1 or rate % fps or resolution < 16 or resolution % 2 or data['audio']['channels'] != 1:
        raise ValueError('controlled source needs even resolution >=16 and integral mono samples per frame')
    for key in ('train_groups', 'eval_groups', 'clips_per_group'):
        if not isinstance(source[key], int) or source[key] < 1:
            raise ValueError(f'source {key} must be a positive integer')
    if source['motion'] not in ('dynamic', 'constant'):
        raise ValueError('source motion must be dynamic or constant')
    low, high = source['frequency_hz']
    if not 0 < low <= high < fps / 2:
        raise ValueError('physical frequencies must lie below the video Nyquist rate')
    center, amplitude = float(source['coordinate_center']), float(source['coordinate_amplitude'])
    sigma, background = float(source['blob_sigma_fraction']), float(source['background_max'])
    base, gain = float(source['audio_base_amplitude']), float(source['audio_coordinate_gain'])
    if not (0 < amplitude < min(center, 1 - center) and 0 < sigma < .15 and 0 <= background < .2):
        raise ValueError('source coordinates, blob size or background are outside the supported range')
    carriers = source['carrier_hz']
    if len(carriers) != 8 or len(set(carriers)) != 8 or any(
        not 0 < f < rate / 2 or not math.isclose(f / fps, round(f / fps)) for f in carriers
    ):
        raise ValueError('eight distinct carriers must complete integer cycles per frame below audio Nyquist')
    if not (base >= 0 and gain > 0 and 8 * (base + gain * (center + amplitude)) <= 1):
        raise ValueError('source audio amplitudes must remain bounded without clipping')
    frames = _integer_count(float(source['clip_duration_seconds']), fps, 'source.clip_duration_seconds')
    needed = _integer_count(float(data['window']['duration_seconds']), fps, 'window.duration_seconds')
    needed += max(_integer_count(float(data['window'][key]), fps, key)
                  for key in ('future_offset_seconds', 'shifted_offset_seconds'))
    if frames < needed:
        raise ValueError('source clip is too short for the configured views')
    return source, fps, rate, resolution


def _group(cfg: Mapping[str, Any], group_id: str) -> tuple[dict[str, Any], torch.Tensor]:
    source, fps, rate, resolution = _settings(cfg)
    seed = int(hashlib.sha256(f"controlled-av-v1:{source['seed']}:{group_id}".encode()).hexdigest()[:16], 16)
    generator = torch.Generator().manual_seed(seed)
    frames = _integer_count(float(source['clip_duration_seconds']), fps, 'duration') * source['clips_per_group']
    low, high = source['frequency_hz']
    frequencies = low + (high - low) * torch.rand(8, generator=generator, dtype=torch.float64)
    phases = 2 * math.pi * torch.rand(8, generator=generator, dtype=torch.float64)
    times = torch.arange(frames, dtype=torch.float64) / fps
    if source['motion'] == 'constant':
        times.zero_()
    coordinates = (source['coordinate_center'] + source['coordinate_amplitude'] *
                   torch.sin(2 * math.pi * times[:, None] * frequencies + phases)).float()
    side = resolution // 2
    grid = torch.arange(side, dtype=torch.float32) + .5
    background = source['background_max'] * torch.rand(resolution, resolution, generator=generator)
    pixels = background[None].expand(frames, -1, -1).clone()
    for emitter in range(4):
        top, left = (emitter // 2) * side, (emitter % 2) * side
        x, y = (coordinates[:, 2 * emitter + axis, None, None] * side for axis in (0, 1))
        radius2 = (grid[None, None, :] - x).square() + (grid[None, :, None] - y).square()
        blob = torch.exp(-radius2 / (2 * (source['blob_sigma_fraction'] * side) ** 2))
        patch = pixels[:, top:top + side, left:left + side]
        patch.add_((1 - patch) * blob)
    video = (255 * pixels).round().to(torch.uint8)[:, None].expand(-1, 3, -1, -1).contiguous()
    # Every carrier completes integral cycles in a grid cell. Reusing the waveform cell preserves
    # continuous carrier phase modulo 2pi and makes constant-signal controls exactly stationary.
    sample_times = torch.arange(rate // fps, dtype=torch.float64) / rate
    carrier_phases = 2 * math.pi * torch.rand(8, generator=generator, dtype=torch.float64)
    waves = torch.sin(2 * math.pi * torch.tensor(source['carrier_hz'])[:, None] * sample_times + carrier_phases[:, None])
    amplitudes = source['audio_base_amplitude'] + source['audio_coordinate_gain'] * coordinates.double()
    audio = (amplitudes @ waves).float().reshape(1, -1)
    return {'video': video, 'audio': audio, 'video_fps': fps, 'audio_sample_rate': rate,
            'duration_seconds': frames / fps}, coordinates


def _atomic_bytes(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_bytes(raw)
    temporary.replace(path)


def _tensor_bytes(value: Any) -> bytes:
    buffer = io.BytesIO()
    torch.save(value, buffer)
    return buffer.getvalue()


def generate_controlled_av(cfg: Mapping[str, Any], root: Path) -> tuple[AVClipRecord, ...]:
    """Publish deterministic complete shards; an existing identity is immutable and reverified."""
    source, fps, rate, _ = _settings(cfg)
    manifest = _resolve(root, cfg['data']['manifest'])
    shards = _resolve(root, cfg['data']['shard_root'])
    manifest.parent.mkdir(parents=True, exist_ok=True)
    receipt = manifest.with_suffix('.generation.json')
    identity = {'data_config': dict(cfg['data']), 'renderer_sha256': _sha256(Path(__file__))}
    with (manifest.parent / '.generation.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if manifest.exists():
            previous = json.loads(receipt.read_text())
            if previous['identity'] != identity or previous['manifest_sha256'] != _sha256(manifest):
                raise ValueError('published controlled source has a different config or renderer identity')
            records = tuple(AVClipRecord.from_dict(json.loads(line)) for line in manifest.read_text().splitlines())
            for record in records:
                if (_sha256(shards / record.shard) != record.shard_sha256 or
                    _sha256(shards / record.source['audit_truth']) != record.source['audit_truth_sha256']):
                    raise ValueError('published controlled source bytes differ from their identity')
            return records
        records = []
        frames = _integer_count(float(source['clip_duration_seconds']), fps, 'duration')
        for split in ('train', 'eval'):
            for group in range(source[split + '_groups']):
                group_id = f'{split}-group-{group:04d}'
                payload, coordinates = _group(cfg, group_id)
                for clip in range(source['clips_per_group']):
                    name = f'{group_id}-clip-{clip:02d}'
                    first = clip * frames
                    chunk = {**payload, 'video': payload['video'][first:first + frames].clone(),
                             'audio': payload['audio'][:, first * (rate // fps):(first + frames) * (rate // fps)].clone(),
                             'duration_seconds': frames / fps}
                    _validate_shard(chunk, cfg['data'], name)
                    raw = _tensor_bytes(chunk)
                    truth = _tensor_bytes({'coordinates': coordinates[first:first + frames].clone()})
                    shard_name, truth_name = name + '.pt', 'truth/' + name + '.pt'
                    _atomic_bytes(shards / shard_name, raw)
                    _atomic_bytes(shards / truth_name, truth)
                    records.append(AVClipRecord(int(cfg['data']['manifest_version']), name, split, group_id,
                        shard_name, frames / fps,
                        {'renderer': 'training/controlled_av.py', 'renderer_sha256': identity['renderer_sha256'],
                         'audit_truth': truth_name, 'audit_truth_sha256': hashlib.sha256(truth).hexdigest()},
                        hashlib.sha256(raw).hexdigest()))
        raw_manifest = ''.join(json.dumps(vars(record), sort_keys=True) + '\n' for record in records).encode()
        # Publish the identity before the complete manifest; interruption cannot expose a partial cohort.
        _atomic_bytes(receipt, (json.dumps({'identity': identity, 'manifest_sha256': hashlib.sha256(raw_manifest).hexdigest(),
                                          'clips': len(records)}, indent=2, sort_keys=True) + '\n').encode())
        _atomic_bytes(manifest, raw_manifest)
        return tuple(records)


def read_physical_coordinates(payload: Mapping[str, Any], cfg: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    """Read coordinates from pixels and waveform alone; never consult an audit truth sidecar."""
    source, fps, rate, resolution = _settings(cfg)
    side = resolution // 2
    grid = (torch.arange(side, dtype=torch.float64) + .5) / side
    grayscale = payload['video'].double().mean(1) / 255
    positions = []
    for emitter in range(4):
        top, left = (emitter // 2) * side, (emitter % 2) * side
        mask = grayscale[:, top:top + side, left:left + side] > .5
        mass = mask.sum((1, 2))
        if (mass == 0).any():
            raise ValueError('raw source readout cannot locate an emitter')
        positions.extend(((mask * grid[None, None, :]).sum((1, 2)) / mass,
                          (mask * grid[None, :, None]).sum((1, 2)) / mass))
    video = torch.stack(positions, -1).float()
    waveform = payload['audio'].double().reshape(-1, rate // fps)
    times = torch.arange(rate // fps, dtype=torch.float64) / rate
    phase = 2 * math.pi * torch.tensor(source['carrier_hz'])[:, None] * times
    sine = waveform @ phase.sin().T * (2 / times.numel())
    cosine = waveform @ phase.cos().T * (2 / times.numel())
    amplitude = (sine.square() + cosine.square()).sqrt()
    audio = ((amplitude - source['audio_base_amplitude']) / source['audio_coordinate_gain']).float()
    return video, audio


def audit_controlled_av(cfg: Mapping[str, Any], root: Path) -> dict[str, Any]:
    """Fixed all-eval raw-sensor matching and coordinate error, conditional on the source definition."""
    from training.av_data import build_representation_data
    source, fps, _, _ = _settings(cfg)
    data = build_representation_data(cfg, root)
    records = sorted(data.records['eval'], key=lambda record: record.clip_id)
    groups = sorted({record.group_id for record in records})
    results, errors, starts = [], [], []
    for record in records:
        payload = torch.load(data.shard_root / record.shard, map_location='cpu', weights_only=True)
        video, audio = read_physical_coordinates(payload, cfg)
        errors.append((video - audio).abs().mean())
        available = video.shape[0] - data.window_frames - data.shifted_frames + 1
        first = int(hashlib.sha256(('controlled-av-audit-v1:' + record.clip_id).encode()).hexdigest()[:16], 16) % available
        starts.append(first)
        values = [F.normalize(x[start:start + data.window_frames].mean(0) - source['coordinate_center'], dim=0)
                  for start in (first, first + data.shifted_frames) for x in (video, audio)]
        v, a, vs, ass = values
        margin = ((v - vs) * (a - ass)).sum().double()
        results.append(torch.stack(((margin > 0).double() + .5 * (margin == 0),
                                    (margin < 0).double() + .5 * (margin == 0), (margin == 0).double())))
    scores = torch.stack(results)
    group_ids = torch.tensor([groups.index(record.group_id) for record in records])
    sums = torch.zeros(len(groups), 3, dtype=torch.float64).index_add_(0, group_ids, scores)
    counts = torch.bincount(group_ids, minlength=len(groups)).double()
    draws = torch.randint(len(groups), (10000, len(groups)), generator=torch.Generator().manual_seed(950103))
    resamples = sums[draws].sum(1) / counts[draws].sum(1)[:, None]
    interval = torch.quantile(resamples[:, 0], torch.tensor([.025, .975], dtype=torch.float64)).tolist()
    return {'scope': 'Raw physical-source control, not a trained representation gate',
            'manifest_sha256': data.fingerprint, 'motion': source['motion'], 'eval_clips': len(records),
            'recording_groups': len(groups), 'raw_balanced_accuracy': float(scores[:, 0].mean()),
            'raw_accuracy_group_bootstrap_95': interval, 'swapped_balanced_accuracy': float(scores[:, 1].mean()),
            'tie_fraction': float(scores[:, 2].mean()), 'coordinate_mean_absolute_error': float(torch.stack(errors).mean()),
            'window_seconds': data.window_frames / fps, 'shift_seconds': data.shifted_frames / fps,
            'clip_ids': [record.clip_id for record in records], 'group_ids': [record.group_id for record in records],
            'starts_frames': starts, 'renderer_sha256': _sha256(Path(__file__)), 'labels_used': False,
            'bootstrap': '10000 recording-group resamples; fixed all-eval source, clip-weighted means, seed 950103'}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('spec', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load(args.spec.read_text())
    records = generate_controlled_av(cfg, root)
    audit = audit_controlled_av(cfg, root)
    path = _resolve(root, cfg['data']['manifest']).with_suffix('.raw_audit.json')
    _atomic_bytes(path, (json.dumps(audit, indent=2, sort_keys=True) + '\n').encode())
    print(json.dumps({'generated_clips': len(records), 'audit': {k: v for k, v in audit.items()
                     if k not in ('clip_ids', 'group_ids', 'starts_frames')}}))


if __name__ == '__main__':
    main()
