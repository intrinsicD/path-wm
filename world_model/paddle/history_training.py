"""Separately recorded 50/50 full-episode/complete-suffix observer experiment.

The recurrent computation and supervised objective are the original memory_batch.
Only the initial-history distribution and prospective validation selector change.
"""
from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from .checkpoints import (SCHEMA_VERSION, TENSOR_SCHEMA, atomic_checkpoint,
    atomic_checkpoint_copy, code_fingerprint, fingerprint_modules, json_atomic,
    load_observer, read_checkpoint, resolve_device, restore_rng, rng_state, versions)
from .data import history_pairs
from .models import MemoryUpdater, StateReadout
from .training import (Samples, frozen_encode, memory_batch, optimizer_for,
                       paired_memory_validation, seed_all)
from .types import action_one_hot

HISTORY_SCHEMA = 'paddle-history-starts-v1'
STATE_SCALE = np.asarray([64., 64., 6., 6., 64.])
SELECTION_CRITERION = '0.5 ordinary SSE/valid-scalars + 0.5 suffix SSE/valid-scalars; earliest exact tie'


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def slice_episode(episode, start):
    """Copy an aligned complete suffix; its unavailable preceding action is absent."""
    length = len(episode['frames'])
    if not _integer(start) or start < 0 or start > length - 3:
        raise ValueError('suffix start must be an integer leaving at least three frames')
    if len(episode['states']) != length or len(episode['actions']) != length - 1:
        raise ValueError('suffix source frame/state/action alignment is invalid')
    result = {}
    for key, value in episode.items():
        if isinstance(value, np.ndarray):
            aligned = value.ndim > 0 and len(value) in (length, length - 1)
            result[key] = np.array(value[start:] if aligned else value, copy=True)
        else:
            result[key] = copy.deepcopy(value)
    return result


class HistoryStartSampler:
    """Independent uniform episode and uniform global eligible-start streams."""
    def __init__(self, lengths, seed=1701, suffix_seed=38802):
        self.lengths = list(lengths)
        if not self.lengths or any(not _integer(n) or n < 3 for n in self.lengths):
            raise ValueError('sampler lengths must contain integer episodes with at least three frames')
        self.lengths = [int(n) for n in self.lengths]
        self.seed, self.suffix_seed = int(seed), int(suffix_seed)
        self.offsets = np.concatenate(([0], np.cumsum(np.asarray(self.lengths) - 3)))
        if self.offsets[-1] == 0:
            raise ValueError('sampler population has no eligible augmented suffix starts')
        self.full_rng = np.random.default_rng(self.seed)
        self.suffix_rng = np.random.default_rng(self.suffix_seed)

    def suffix_pair(self, index):
        if not _integer(index) or not 0 <= index < self.offsets[-1]:
            raise ValueError('suffix index outside eligible population')
        episode = int(np.searchsorted(self.offsets, index, side='right') - 1)
        return episode, int(index - self.offsets[episode] + 1)

    def sample(self, batch_size=8):
        if not _integer(batch_size) or batch_size <= 0 or batch_size % 2:
            raise ValueError('batch size must be a positive even integer')
        full = self.full_rng.integers(0, len(self.lengths), size=batch_size // 2)
        suffix = self.suffix_rng.integers(0, self.offsets[-1], size=batch_size // 2)
        return [pair for e, s in zip(full, suffix)
                for pair in ((int(e), 0), self.suffix_pair(int(s)))]

    def state_dict(self):
        return copy.deepcopy({'schema': HISTORY_SCHEMA, 'lengths': self.lengths,
            'seed': self.seed, 'suffix_seed': self.suffix_seed,
            'full_rng': self.full_rng.bit_generator.state,
            'suffix_rng': self.suffix_rng.bit_generator.state})

    def load_state_dict(self, state):
        identity = ('schema', 'lengths', 'seed', 'suffix_seed')
        expected = self.state_dict()
        if any(state.get(k) != expected[k] for k in identity):
            raise ValueError('sampler identity, lengths or seeds differ from checkpoint')
        # Validate both streams before mutating either live generator.
        full, suffix = np.random.default_rng(), np.random.default_rng()
        full.bit_generator.state = copy.deepcopy(state['full_rng'])
        suffix.bit_generator.state = copy.deepcopy(state['suffix_rng'])
        self.full_rng.bit_generator.state = full.bit_generator.state
        self.suffix_rng.bit_generator.state = suffix.bit_generator.state


def joint_validation_score(ordinary, suffix):
    values = []
    for population in (ordinary, suffix):
        count, total = population['supervised_scalar_count'], population['squared_error_sum']
        if not _integer(count) or count <= 0:
            raise ValueError('validation supervised scalar count must be a positive integer')
        if not math.isfinite(float(total)) or total < 0:
            raise ValueError('validation squared error must be finite and nonnegative')
        values.append(float(total) / int(count))
    return .5 * values[0] + .5 * values[1]


class _SuffixSamples:
    def __init__(self, samples, pairs):
        self.samples, self.pairs = samples, pairs

    def episode(self, index):
        episode, start = self.pairs[int(index)]
        return slice_episode(self.samples.episode(episode), start)


def _source(samples, episode, start):
    entries = getattr(samples.dataset, 'entries', [])
    return {'episode': int(episode), 'start': int(start),
            'source': copy.deepcopy(entries[episode]) if entries else {'index': int(episode)}}


def _summary(errors, valid):
    """Physical coordinate errors, with independent ragged coordinate denominators."""
    values = [np.asarray(errors)[np.asarray(valid)[:, i], i] for i in range(5)]
    return {'coordinate_count': [len(x) for x in values],
            **{key: [float(function(x)) if len(x) else None for x in values]
               for key, function in (('mae', np.mean), ('p95', lambda x: np.quantile(x, .95)), ('max', np.max))}}


def _record_summary(records):
    errors, masks, ages = [], [], []
    for row in records:
        prediction, target = np.asarray(row['predicted_state']), np.asarray(row['target_state'])
        errors.append(np.abs(prediction - target))
        mask = np.ones_like(prediction, dtype=bool); mask[:2, 2:4] = False
        masks.append(mask); ages.extend(range(len(prediction)))
    errors, mask, age = np.concatenate(errors), np.concatenate(masks), np.asarray(ages)
    return {'all_valid': _summary(errors, mask),
            'age2': _summary(errors, mask & (age[:, None] == 2)),
            'later': _summary(errors, mask & (age[:, None] > 2)),
            'post_warmup': _summary(errors, mask & (age[:, None] >= 2))}


@torch.no_grad()
def _validate_population(models, samples, pairs, device, batch_size, encoder_batch_size):
    sliced = _SuffixSamples(samples, pairs)
    records, squared_error_sum, scalars = [], 0., 0
    for begin in range(0, len(pairs), batch_size):
        indices = list(range(begin, min(begin + batch_size, len(pairs))))
        captured = []
        hook = models['R'].register_forward_hook(lambda module, inputs, output: captured.append(output.detach().cpu()))
        try:
            loss, metrics = memory_batch(models, sliced, indices, device, encoder_batch_size)
        finally:
            hook.remove()
        count = int(metrics['supervised_scalar_count'])
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite observer validation loss')
        squared_error_sum += float(loss) * count; scalars += count
        prediction = torch.stack(captured, 1).numpy().astype(np.float64) * STATE_SCALE
        for slot, index in enumerate(indices):
            e, start = pairs[index]; episode = sliced.episode(index); length = len(episode['frames'])
            record = {**_source(samples, e, start), 'length': length,
                      'source_start_state': episode['states'][0].tolist(),
                      'predicted_state': prediction[slot, :length].tolist(),
                      'target_state': episode['states'].tolist(),
                      'supervised_scalar_count': 5 * length - 4}
            record['physical'] = _record_summary([record])
            records.append(record)
    summary = {'squared_error_sum': squared_error_sum, 'supervised_scalar_count': scalars,
               'loss': squared_error_sum / scalars, 'sequences': len(records),
               'observations': sum(row['length'] for row in records), **_record_summary(records)}
    return summary, records


@torch.no_grad()
def _paired_diagnostics(models, count, device, encoder_batch_size):
    if not count:
        return {}, []
    # Preserve the baseline helper/mean definition alongside richer physical rows.
    summary = paired_memory_validation(models, count, device)
    pairs = history_pairs(count, seed=7000)
    members = [member for pair in pairs for member in pair['members']]
    encodings = frozen_encode(models['E'], np.concatenate([m['frames'] for m in members]), device, encoder_batch_size)
    memory = torch.zeros(len(members), 128, device=device)
    for t in range(3):
        observation = encodings[torch.arange(len(members), device=device) * 3 + t]
        previous = torch.zeros(len(members), 3, device=device) if t == 0 else action_one_hot(
            torch.as_tensor([m['actions'][t-1] for m in members], device=device))
        memory = models['U'](memory, observation, previous)
    prediction = models['R'](memory).cpu().numpy() * STATE_SCALE
    reset = models['R'](models['U'](torch.zeros_like(memory), observation,
                                  torch.zeros(len(members), 3, device=device))).cpu().numpy() * STATE_SCALE
    truth = np.stack([m['states'][-1] for m in members])
    if not np.isfinite(prediction).all() or not np.isfinite(reset).all():
        raise RuntimeError('Nonfinite paired observer diagnostic')
    records = [{'pair_seed': pairs[i // 2]['pair_seed'], 'direction': m['direction'],
                'correct_action': m['correct_action'], 'target_state': truth[i].tolist(),
                'predicted_state': prediction[i].tolist(), 'reset_predicted_state': reset[i].tolist(),
                'absolute_error': np.abs(prediction[i]-truth[i]).tolist(),
                'reset_absolute_error': np.abs(reset[i]-truth[i]).tolist(),
                'vx_sign_correct': bool(np.sign(prediction[i, 2]) == np.sign(truth[i, 2])),
                'reset_vx_sign_correct': bool(np.sign(reset[i, 2]) == np.sign(truth[i, 2]))}
               for i, m in enumerate(members)]
    valid = np.ones_like(truth, dtype=bool)
    summary.update({'pairs': count, 'members': len(members), 'pair_seed': 7000,
                    'physical': _summary(np.abs(prediction - truth), valid),
                    'reset_physical': _summary(np.abs(reset - truth), valid),
                    'vx_sign_accuracy': float(np.mean([r['vx_sign_correct'] for r in records])),
                    'reset_vx_sign_accuracy': float(np.mean([r['reset_vx_sign_correct'] for r in records]))})
    return summary, records


@torch.no_grad()
def _matched_diagnostics(models, samples, observations, ordinary_records, device, encoder_batch_size):
    """Replay exactly the pre-existing diagnostic triples; never select new rows."""
    records = []
    full = {row['episode']: row for row in ordinary_records}
    for episode_index, frame in observations:
        episode = samples.episode(episode_index)
        triple = {'frames': episode['frames'][frame-2:frame+1].copy(),
                  'states': episode['states'][frame-2:frame+1].copy(),
                  'actions': episode['actions'][frame-2:frame].copy()}
        observation = frozen_encode(models['E'], triple['frames'], device, encoder_batch_size)
        memory = torch.zeros(1, 128, device=device)
        for t in range(3):
            previous = torch.zeros(1, 3, device=device) if t == 0 else action_one_hot(
                torch.as_tensor([triple['actions'][t-1]], device=device))
            memory = models['U'](memory, observation[t:t+1], previous)
        cold = models['R'](memory)[0].cpu().numpy() * STATE_SCALE
        actual = np.asarray(full[episode_index]['predicted_state'][frame])
        target = episode['states'][frame]
        records.append({'episode': episode_index, 'frame': frame, 'target_state': target.tolist(),
                        'full_predicted_state': actual.tolist(), 'three_frame_predicted_state': cold.tolist(),
                        'full_absolute_error': np.abs(actual-target).tolist(),
                        'three_frame_absolute_error': np.abs(cold-target).tolist()})
    if not records:
        return {}, []
    valid = np.ones((len(records), 5), dtype=bool)
    return {'observations': len(records),
            'full': _summary(np.asarray([r['full_absolute_error'] for r in records]), valid),
            'three_frame': _summary(np.asarray([r['three_frame_absolute_error'] for r in records]), valid)}, records


def _hash_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _populations(config, settings, train, validation, perception_identity):
    options = config.get('history_starts', {})
    if options.get('schema_version', 1) != 1 or options.get('normalization', STATE_SCALE.tolist()) != STATE_SCALE.tolist():
        raise ValueError('history schema/normalization differs from the reviewed protocol')
    if options.get('dataset_fingerprint', train.dataset.fingerprint) != train.dataset.fingerprint:
        raise ValueError('history source dataset fingerprint differs')
    if train.dataset.fingerprint != validation.dataset.fingerprint:
        raise ValueError('training and validation dataset fingerprints differ')
    ordinary_source = {}
    if 'ordinary_validation_indices' in options:
        if not config.get('smoke', False):
            raise ValueError('explicit ordinary indices are restricted to declared smoke configurations')
        ordinary = options['ordinary_validation_indices']
    else:
        path = Path(options['ordinary_validation_manifest'])
        reference = json.loads(path.read_text())
        if (reference.get('dataset_fingerprint') != train.dataset.fingerprint or
            reference.get('dependencies', {}).get('perception') != perception_identity):
            raise ValueError('ordinary validation manifest has a different dataset/perception identity')
        ordinary = reference['validation_indices']
        ordinary_source = {'path': str(path), 'sha256': _hash_file(path)}
    if not _integer(settings['validation_examples']) or settings['validation_examples'] <= 0:
        raise ValueError('validation example count must be a positive integer')
    if not _integer(settings.get('validation_pairs', 50)) or settings.get('validation_pairs', 50) < 0:
        raise ValueError('paired validation count must be a nonnegative integer')
    count = int(settings['validation_examples'])
    if config.get('smoke', False):
        ordinary = ordinary[:count]
    if len(ordinary) != count or len(set(ordinary)) != count or any(
            not _integer(e) or not 0 <= e < len(validation.lengths) for e in ordinary):
        raise ValueError('fixed ordinary validation population/count is invalid')
    eligible = HistoryStartSampler(validation.lengths)
    if count > eligible.offsets[-1]:
        raise ValueError('too few eligible validation suffixes for distinct sampling')
    suffix_seed = int(options.get('validation_suffix_seed', 38920))
    suffix = [eligible.suffix_pair(int(i)) for i in np.random.default_rng(suffix_seed).choice(
        eligible.offsets[-1], size=count, replace=False)]
    matched, diagnostic_source = [], {}
    if options.get('matched_diagnostic'):
        path = Path(options['matched_diagnostic']); previous = json.loads(path.read_text())
        # These exact observations generated the prospective hypothesis; no re-selection.
        matched = [(int(r['episode']), int(r['frame'])) for r in previous['ordinary_records']
                   if 'three_frame_suffix_r_abs_error' in r]
        if any(e not in ordinary or not 2 <= t < validation.lengths[e] for e, t in matched):
            raise ValueError('matched diagnostic rows are outside the fixed ordinary validation population')
        diagnostic_source = {'path': str(path), 'sha256': _hash_file(path)}
    return {'ordinary': [[int(e), 0] for e in ordinary], 'suffix': [list(p) for p in suffix],
            'ordinary_manifest': ordinary_source, 'suffix_seed': suffix_seed,
            'paired_count': int(settings.get('validation_pairs', 50)), 'paired_seed': 7000,
            'matched_observations': [list(p) for p in matched], 'matched_source': diagnostic_source,
            'source_train_lengths': train.lengths, 'source_validation_lengths': validation.lengths,
            'eligible_train_starts': int(sum(n-3 for n in train.lengths)),
            'eligible_validation_starts': int(eligible.offsets[-1]),
            'ordinary_sources': [_source(validation, e, 0) for e in ordinary],
            'suffix_sources': [_source(validation, e, s) for e, s in suffix]}


def _prune_uncommitted_ledgers(run, update):
    for name in ('training', 'validation'):
        path = run / f'{name}.jsonl'
        if not path.exists():
            continue
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        suffix = [row for row in rows if row['step'] > update]
        if suffix:
            json_atomic(run / f'{name}_interrupted_{time.time_ns()}.json', {'checkpoint_update': update, 'rows': suffix})
            temporary = path.with_suffix('.tmp')
            temporary.write_text(''.join(json.dumps(row)+'\n' for row in rows if row['step'] <= update))
            temporary.replace(path)
    for path in (run / 'checkpoints').glob('update_*.pt'):
        if int(path.stem.split('_')[1]) > update:
            path.replace(path.with_name(f'orphan_{time.time_ns()}_{path.name}'))
    for path in (run / 'diagnostics').glob('validation_*.json'):
        if int(path.stem.split('_')[1]) > update:
            path.replace(path.with_name(f'orphan_{time.time_ns()}_{path.name}'))


def train_history_memory(config, data, perception, run, resume=False):
    """Train fresh U/R with compatible, atomic memory checkpoints and exact resume."""
    run = Path(run); run.mkdir(parents=True, exist_ok=True)
    with (run / '.stage.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f'Another trainer owns {run}; do not launch a duplicate') from error
        try:
            return _train_impl(config, data, perception, run, resume)
        except BaseException as error:
            value = {'stage': 'memory', 'experiment': HISTORY_SCHEMA, 'status': 'failed',
                     'error': str(error), 'type': type(error).__name__, 'training_complete': False}
            # Invalid reuse attempts must never replace a completed experiment's status.
            if not (run / 'paddle_result.json').exists():
                json_atomic(run / 'failure.json', value); json_atomic(run / 'status.json', value)
            raise


def _train_impl(config, data, perception, run, resume):
    if (run / 'last.pt').exists() and not resume:
        raise ValueError('history run already has a checkpoint; use --resume or a new directory')
    if not resume and (run / 'paddle_manifest.json').exists():
        raise ValueError('history run already has immutable provenance; use a new directory after an uncommitted initialization')
    settings, common = config['training']['memory'], config['training']
    options = config.get('history_starts', {})
    batch_size, budget, validate_every = settings['batch_size'], settings['updates'], common['validate_every']
    if (not _integer(batch_size) or batch_size <= 0 or batch_size % 2 or
        not _integer(budget) or budget <= 0 or not _integer(validate_every) or validate_every <= 0):
        raise ValueError('positive integer update/check intervals and an even batch size are required')
    seed = int(config['seed']); seed_all(seed)
    device = resolve_device(config.get('device', 'auto'))
    torch.set_num_threads(int(config.get('cpu_threads', 4)))
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    train, validation = Samples(data, 'train'), Samples(data, 'validation')
    perception_checkpoint = read_checkpoint(perception, stage='perception', dataset_fingerprint=train.dataset.fingerprint)
    deps = {'perception': perception_checkpoint['model_fingerprint']}
    # This order deliberately matches the original memory stage's fresh initialization.
    models = load_observer(perception, None, device)
    models.update({'U': MemoryUpdater(), 'R': StateReadout()})
    for key, model in models.items():
        model.to(device).requires_grad_(key in ('U', 'R')); model.train(key in ('U', 'R'))
    trainable = {k: models[k] for k in ('U', 'R')}
    opt = optimizer_for(trainable, common)
    frozen_identity = fingerprint_modules({k: models[k] for k in ('E', 'D', 'H')})
    initial_identity = fingerprint_modules(trainable)
    sampler = HistoryStartSampler(train.lengths, seed, options.get('suffix_seed', 38802))
    populations = _populations(config, settings, train, validation, deps['perception'])
    identity = {'schema': HISTORY_SCHEMA, 'dataset_fingerprint': train.dataset.fingerprint,
                'dependencies': deps, 'perception_sha256': _hash_file(perception),
                'resolved_device': str(device),
                'normalization': STATE_SCALE.tolist(), 'mixture': [0.5, 0.5],
                'sampler_seeds': [seed, sampler.suffix_seed], 'populations': populations,
                'initial_model_fingerprint': initial_identity}
    counters = {'sequences': 0, 'full_sequences': 0, 'suffix_sequences': 0,
                'observations': 0, 'supervised_scalar_count': 0,
                'full_observations': 0, 'suffix_observations': 0,
                'full_supervised_scalar_count': 0, 'suffix_supervised_scalar_count': 0}
    seen_episodes, seen_starts = set(), set()
    start, best, best_checkpoint, elapsed_previous, peak_previous = 0, float('inf'), None, 0., 0
    saved = None
    if resume:
        saved = read_checkpoint(run / 'last.pt', stage='memory', dependencies=deps,
                                dataset_fingerprint=train.dataset.fingerprint)
        if saved.get('config') != config or saved.get('history_identity') != identity:
            raise ValueError('history resume requires identical config, source, seeds, normalization and frozen E identity')
        for key in trainable:
            trainable[key].load_state_dict(saved['models'][key])
        opt.load_state_dict(saved['optimizer'])
        start, best, best_checkpoint = saved['global_update'], saved['best_validation'], saved['best_checkpoint']
        counters = copy.deepcopy(saved['history_counters'])
        seen_episodes = set(saved['seen_source_episodes'])
        seen_starts = {tuple(pair) for pair in saved['seen_source_starts']}
        elapsed_previous, peak_previous = saved['elapsed_seconds'], saved.get('peak_memory_bytes', 0)
        selected = read_checkpoint(run / best_checkpoint, stage='memory', dependencies=deps,
                                   dataset_fingerprint=train.dataset.fingerprint)
        if selected['history_identity'] != identity or selected['metrics']['loss'] != best:
            raise ValueError('selected history snapshot identity/objective differs from committed last checkpoint')
        if not (run / 'best.pt').exists() or _hash_file(run / best_checkpoint) != _hash_file(run / 'best.pt'):
            atomic_checkpoint_copy(run / best_checkpoint, run / 'best.pt')
        _prune_uncommitted_ledgers(run, start)
        restore_rng(saved['rng'], sampler.full_rng); sampler.load_state_dict(saved['history_sampler'])
    else:
        manifest = {'stage': 'memory', 'experiment': HISTORY_SCHEMA, 'horizon': 1, 'config': config,
                    'dependencies': deps, 'dataset_fingerprint': train.dataset.fingerprint,
                    'versions': versions(), 'tensor_schema': TENSOR_SCHEMA,
                    'validation_indices': [p[0] for p in populations['ordinary']],
                    'history_identity': identity, 'best_validation_criterion': SELECTION_CRITERION,
                    'precision': 'float32; TF32 disabled', 'observer_early_stopping': False,
                    'input_storage': {split: {'kind': 'read-only raw RGB/state/action memory map',
                        'fingerprint': samples.raw_cache.fingerprint, 'path': str(samples.raw_cache.directory)}
                        for split, samples in (('train', train), ('validation', validation))
                        if getattr(samples, 'raw_cache', None) is not None}}
        json_atomic(run / 'resolved_config.json', config)
        json_atomic(run / 'paddle_manifest.json', manifest)
    begin = time.monotonic(); source_hash = code_fingerprint()
    encoder_batch_size = int(common.get('encoder_batch_size', 128))

    def validate(update):
        for model in trainable.values(): model.eval()
        ordinary, ordinary_rows = _validate_population(models, validation, populations['ordinary'], device, batch_size, encoder_batch_size)
        suffix, suffix_rows = _validate_population(models, validation, populations['suffix'], device, batch_size, encoder_batch_size)
        paired, paired_rows = _paired_diagnostics(models, populations['paired_count'], device, encoder_batch_size)
        matched, matched_rows = _matched_diagnostics(models, validation, populations['matched_observations'], ordinary_rows,
                                                   device, encoder_batch_size)
        metrics = {'loss': joint_validation_score(ordinary, suffix), 'ordinary': ordinary, 'suffix': suffix,
                   'r_mae': ordinary['post_warmup']['mae'], 'paired': paired, 'matched_history': matched,
                   'selection_criterion': SELECTION_CRITERION,
                   **{k: v for k, v in paired.items() if k.startswith('paired_validation_')}}
        raw = {'schema': HISTORY_SCHEMA, 'step': update, 'model_fingerprint': fingerprint_modules(trainable),
               'metrics': metrics, 'ordinary_records': ordinary_rows, 'suffix_records': suffix_rows,
               'paired_records': paired_rows, 'matched_history_records': matched_rows,
               'units': ['pixels', 'pixels', 'pixels/interval', 'pixels/interval', 'pixels'],
               'coordinate_order': ['x', 'y', 'vx', 'vy', 'paddle_x'],
               'velocity_mask': 'relative ages 0 and 1 excluded; all position ages included'}
        json_atomic(run / 'diagnostics' / f'validation_{update:08d}.json', raw)
        for model in trainable.values(): model.train()
        return metrics

    def save(update, metrics):
        nonlocal best, best_checkpoint
        improved = metrics['loss'] < best
        snapshot = f'checkpoints/update_{update:08d}.pt'
        if improved:
            best, best_checkpoint = metrics['loss'], snapshot
        if fingerprint_modules({k: models[k] for k in ('E', 'D', 'H')}) != frozen_identity:
            raise RuntimeError('Frozen perception parameters/buffers changed during history training')
        elapsed = elapsed_previous + time.monotonic() - begin
        peak = max(peak_previous, torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0)
        value = {'schema_version': SCHEMA_VERSION, 'tensor_schema': TENSOR_SCHEMA, 'stage': 'memory', 'horizon': 1,
                 'experiment': HISTORY_SCHEMA, 'global_update': update,
                 'models': {k: m.state_dict() for k, m in trainable.items()},
                 'model_fingerprint': fingerprint_modules(trainable), 'optimizer': opt.state_dict(),
                 'rng': rng_state(sampler.full_rng), 'history_sampler': sampler.state_dict(),
                 'config': config, 'versions': versions(), 'code_fingerprint': source_hash,
                 'dataset_fingerprint': train.dataset.fingerprint, 'dependencies': deps, 'statistics': None,
                 'history_identity': identity, 'history_counters': copy.deepcopy(counters),
                 'seen_source_episodes': sorted(seen_episodes), 'seen_source_starts': sorted(seen_starts),
                 'distinct_source_episodes': len(seen_episodes), 'distinct_source_starts': len(seen_starts),
                 'metrics': metrics, 'best_validation': best, 'best_validation_criterion': SELECTION_CRITERION,
                 'best_checkpoint': best_checkpoint, 'examples_processed': counters['sequences'],
                 'elapsed_seconds': elapsed, 'peak_memory_bytes': peak, 'quality_gate': {},
                 'training_complete': update == budget, 'stop_reason': 'maximum_updates' if update == budget else None}
        atomic_checkpoint(run / snapshot, value)
        # Immutable state commits before last; best alias can be recovered from last.
        atomic_checkpoint(run / 'last.pt', value)
        if improved:
            atomic_checkpoint_copy(run / snapshot, run / 'best.pt')
        json_atomic(run / 'status.json', {'stage': 'memory', 'experiment': HISTORY_SCHEMA,
            'global_update': update, 'status': 'running', 'best_validation': best, 'metrics': metrics,
            'history_counters': counters})

    if saved is None:
        initial = validate(0)
        with (run / 'validation.jsonl').open('a') as stream: stream.write(json.dumps({'step': 0, **initial}) + '\n')
        save(0, initial)
        print(f'history memory initial validation: {initial["loss"]:.8g}', flush=True)
    for update in range(start + 1, (start if saved and saved['training_complete'] else budget) + 1):
        tick = time.monotonic(); pairs = sampler.sample(batch_size)
        sliced = _SuffixSamples(train, pairs)
        opt.zero_grad(set_to_none=True)
        loss, batch_metrics = memory_batch(models, sliced, list(range(batch_size)), device, encoder_batch_size)
        if not torch.isfinite(loss):
            raise RuntimeError(f'Nonfinite history memory loss at update {update}')
        loss.backward()
        norm = nn.utils.clip_grad_norm_([p for m in trainable.values() for p in m.parameters()], common['grad_clip'])
        if not torch.isfinite(norm):
            raise RuntimeError(f'Nonfinite history memory gradient at update {update}')
        opt.step()
        for episode_index, source_start in pairs:
            length = train.lengths[episode_index] - source_start
            kind = 'full' if source_start == 0 else 'suffix'
            counters['sequences'] += 1; counters[f'{kind}_sequences'] += 1
            counters['observations'] += length; counters[f'{kind}_observations'] += length
            counters['supervised_scalar_count'] += 5 * length - 4
            counters[f'{kind}_supervised_scalar_count'] += 5 * length - 4
            seen_episodes.add(episode_index); seen_starts.add((episode_index, source_start))
        if device.type == 'cuda': torch.cuda.synchronize(device)
        row = {'step': update, 'loss': float(loss.detach()), 'grad_norm': float(norm),
               'examples_processed': counters['sequences'], 'seconds': time.monotonic() - tick,
               'source_starts': [list(p) for p in pairs], 'history_counters': copy.deepcopy(counters),
               'distinct_source_episodes': len(seen_episodes), 'distinct_source_starts': len(seen_starts), **batch_metrics}
        with (run / 'training.jsonl').open('a') as stream: stream.write(json.dumps(row) + '\n')
        if update % validate_every == 0 or update == budget:
            metrics = validate(update)
            with (run / 'validation.jsonl').open('a') as stream: stream.write(json.dumps({'step': update, **metrics}) + '\n')
            save(update, metrics)
            print(f'history memory update {update}/{budget}: joint {metrics["loss"]:.8g}, '
                  f'ordinary {metrics["ordinary"]["loss"]:.8g}, suffix {metrics["suffix"]["loss"]:.8g}', flush=True)
    final = read_checkpoint(run / 'last.pt'); selected = read_checkpoint(run / best_checkpoint)
    result = {'stage': 'memory', 'experiment': HISTORY_SCHEMA, 'status': 'completed',
              'training_complete': True, 'stop_reason': 'maximum_updates',
              'smoke': bool(config.get('smoke', False)), 'global_update': final['global_update'],
              'selected_update': selected['global_update'], 'metrics': selected['metrics'],
              'final_metrics': final['metrics'], 'checkpoint': str(run / 'best.pt'),
              'selected_checkpoint': str(run / best_checkpoint),
              'final_checkpoint': str(run / f'checkpoints/update_{final["global_update"]:08d}.pt'),
              'initial_checkpoint': str(run / 'checkpoints/update_00000000.pt'),
              'initial_model_fingerprint': initial_identity, 'quality_gate': {},
              'best_validation_criterion': SELECTION_CRITERION,
              'elapsed_seconds': final['elapsed_seconds'], 'peak_memory_bytes': final['peak_memory_bytes'],
              'examples_processed': counters['sequences'], 'history_counters': counters,
              'distinct_source_episodes': len(seen_episodes), 'distinct_source_starts': len(seen_starts),
              'dataset_fingerprint': train.dataset.fingerprint, 'dependencies': deps}
    json_atomic(run / 'paddle_result.json', result); json_atomic(run / 'status.json', result)
    if (run / 'failure.json').exists():
        (run / 'failure.json').replace(run / f'failure_recovered_{time.time_ns()}.json')
    return result
