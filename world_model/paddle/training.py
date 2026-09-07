"""Staged FP32 baseline with fixed validation, complete histories and resumable RNG."""
from __future__ import annotations

from collections import OrderedDict
import fcntl
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn

from .checkpoints import (SCHEMA_VERSION, TENSOR_SCHEMA, atomic_checkpoint, atomic_checkpoint_copy, json_atomic, code_fingerprint,
                          fingerprint_modules, load_observer, read_checkpoint,
                          resolve_device, restore_rng, rng_state, versions)


class CoordinateVariance:
    """Parallel Welford combination; only the frame axis is the sample axis."""
    def __init__(self):
        self.count = 0; self.mean = None; self.m2 = None

    def update(self, values):
        values = values.detach().to(device='cpu', dtype=torch.float64)
        n = len(values)
        if not n:
            return
        mean = values.mean(0); m2 = ((values - mean) ** 2).sum(0)
        if self.count == 0:
            self.count, self.mean, self.m2 = n, mean, m2
        else:
            delta = mean - self.mean; total = self.count + n
            self.m2 += m2 + delta.square() * (self.count * n / total)
            self.mean += delta * (n / total); self.count = total

    def variance(self):
        if not self.count:
            raise ValueError('Cannot estimate variance without training frames')
        return float((self.m2 / self.count).mean())


def supervised_memory_loss(estimate, target, valid):
    mask = valid[..., None].expand_as(estimate).clone()
    mask[:, :2, 2:4] = False
    return ((estimate - target).square() * mask).sum() / mask.sum().clamp_min(1)


def images(frames, device):
    return torch.as_tensor(np.asarray(frames).copy(), device=device).permute(0, 3, 1, 2).float().div_(255)


def normalized_states(states, device):
    return torch.as_tensor(np.asarray(states), dtype=torch.float32, device=device) / torch.tensor(
        [64, 64, 6, 6, 64], device=device)


class Samples:
    """Exact ordered samples with bounded disk-backed raw-frame access."""
    def __init__(self, data, split, use_frame_cache=True):
        from .data import EpisodeDataset
        self.dataset = EpisodeDataset(data, split)
        self.cache = OrderedDict()
        self.raw_cache = None
        self.lengths = [int(entry['frames']) for entry in self.dataset.entries]
        if use_frame_cache:
            from .frame_cache import FrameCache, build_frame_cache
            directory = Path(data).parent/'frame_cache'/self.dataset.fingerprint/split
            build_frame_cache(self.dataset,directory)
            self.raw_cache = FrameCache(self.dataset,directory)
        self.frames = [(e, t) for e, n in enumerate(self.lengths) for t in range(n)]

    def episode(self, index):
        index = int(index)
        if self.raw_cache is not None:
            return self.raw_cache.episode(index)
        if index not in self.cache:
            self.cache[index] = self.dataset[index]
            if len(self.cache) > 32:
                self.cache.popitem(last=False)
        self.cache.move_to_end(index)
        return self.cache[index]

    def frame_batch(self, indices, device):
        if self.raw_cache is not None:
            frames,states = self.raw_cache.frame_batch(indices)
        else:
            pairs = [self.frames[int(i)] for i in indices]
            selected = [(self.episode(e),t) for e,t in pairs]
            frames = np.stack([episode['frames'][t] for episode,t in selected])
            states = np.stack([episode['states'][t] for episode,t in selected])
        return images(frames, device), normalized_states(states, device)

    def windows(self, horizon):
        return [(e, t) for e, n in enumerate(self.lengths) for t in range(2, n - horizon)]


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False


def optimizer_for(models, config):
    decay, no_decay = [], []
    for model in models.values():
        for name, p in model.named_parameters():
            if p.requires_grad:
                (no_decay if p.ndim < 2 or name.endswith('bias') else decay).append(p)
    return torch.optim.AdamW([{'params': decay, 'weight_decay': config['weight_decay']},
                              {'params': no_decay, 'weight_decay': 0.0}],
                             lr=config['learning_rate'], betas=(0.9, 0.999), eps=1e-8)


def frozen_encode(encoder, frames, device, batch_size=128):
    from .types import ObservationLatent
    pieces = []
    with torch.no_grad():
        for start in range(0, len(frames), batch_size):
            pieces.append(encoder(images(frames[start:start + batch_size], device)))
    return ObservationLatent.cat(pieces)


def memory_batch(models, samples, indices, device, encoder_batch_size=128):
    from .types import ObservationLatent, action_one_hot
    episodes = [samples.episode(i) for i in indices]
    lengths = [len(e['frames']) for e in episodes]; length = max(lengths); batch = len(episodes)
    encodings = [frozen_encode(models['E'], e['frames'], device, encoder_batch_size) for e in episodes]
    memory = torch.zeros(batch, 128, device=device); estimates = []
    target = torch.zeros(batch, length, 5, device=device)
    valid = torch.zeros(batch, length, dtype=torch.bool, device=device)
    actions = torch.ones(batch, length, dtype=torch.long, device=device)
    for b, e in enumerate(episodes):
        target[b, :lengths[b]] = normalized_states(e['states'], device)
        valid[b, :lengths[b]] = True
        actions[b, 1:lengths[b]] = torch.as_tensor(e['actions'], device=device)
    for t in range(length):
        # Padded frames never update the carried memory. Reusing a final feature
        # in masked computations avoids invalid tensor/action placeholders.
        observation = ObservationLatent.cat([s[min(t, lengths[b]-1):min(t, lengths[b]-1)+1]
                                              for b, s in enumerate(encodings)])
        previous = torch.zeros(batch, 3, device=device) if t == 0 else action_one_hot(actions[:, t])
        candidate = models['U'](memory, observation, previous)
        memory = torch.where(valid[:, t, None], candidate, memory)
        estimates.append(models['R'](memory))
    estimate = torch.stack(estimates, dim=1)
    loss = supervised_memory_loss(estimate, target, valid)
    errors = (estimate.detach() - target).abs() * torch.tensor([64,64,6,6,64], device=device)
    after = valid.clone(); after[:, :2] = False
    mae = (errors * after[..., None]).sum((0,1)) / after.sum().clamp_min(1)
    return loss, {'r_mae': mae.tolist(), 'observations': sum(lengths),
                  'supervised_scalar_count': sum(5*n-4 for n in lengths),
                  'post_warmup_observations': int(after.sum())}


def latent_loss(prediction, target, statistics):
    return .5 * ((prediction.fine - target.fine).square().mean() / max(statistics['v_fine'], 1e-6)
                 + (prediction.coarse - target.coarse).square().mean() / max(statistics['v_coarse'], 1e-6))


def compute_statistics(encoder, samples, run, fingerprint, batch_size=128, device='cpu'):
    path = Path(run) / 'statistics.json'
    identity = {'encoder_fingerprint': fingerprint, 'dataset_fingerprint': samples.dataset.fingerprint,
                'tensor_schema': TENSOR_SCHEMA, 'precision': 'float32'}
    if path.exists():
        result = json.loads(path.read_text())
        if all(result.get(k) == v for k, v in identity.items()):
            return result
    fine, coarse = CoordinateVariance(), CoordinateVariance()
    for e in range(len(samples.lengths)):
        frames = samples.episode(e)['frames']
        for start in range(0, len(frames), batch_size):
            with torch.no_grad():
                s = encoder(images(frames[start:start+batch_size], device))
            fine.update(s.fine); coarse.update(s.coarse)
        if (e+1) % 500 == 0:
            print(f'latent statistics: {e+1}/{len(samples.lengths)} episodes', flush=True)
    result = {**identity, 'v_fine': fine.variance(), 'v_coarse': coarse.variance(), 'count': fine.count}
    json_atomic(path, result)
    return result


def observer_cache(models, samples, root, dependencies, device, encoder_batch_size=128):
    from .types import action_one_hot
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    identity = {**{k:v for k,v in dependencies.items() if k in ('perception','memory')},
                'dataset': samples.dataset.fingerprint, 'split': samples.dataset.split,
                'schema': TENSOR_SCHEMA, 'precision': 'float32'}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    cache = []
    # Cache only 128-value memories, never the 20480-value frame features.
    for e in range(len(samples.lengths)):
        path = root / f'{e:06d}.npz'
        memories = None
        if path.exists():
            with np.load(path, allow_pickle=False) as saved:
                if str(saved['fingerprint']) == fingerprint:
                    candidate = saved['memories']
                    if candidate.shape == (samples.lengths[e], 128) and np.isfinite(candidate).all():
                        memories = candidate
        if memories is None:
            episode = samples.episode(e)
            s = frozen_encode(models['E'], episode['frames'], device, encoder_batch_size)
            memory = torch.zeros(1, 128, device=device); result = []
            with torch.no_grad():
                for t in range(len(episode['frames'])):
                    action = (torch.zeros(1,3,device=device) if t == 0 else
                              action_one_hot(torch.tensor([episode['actions'][t-1]], device=device)))
                    memory = models['U'](memory, s[t:t+1], action)
                    result.append(memory.cpu().numpy()[0])
            memories = np.stack(result)
            temporary = path.with_suffix('.tmp')
            with temporary.open('wb') as f:
                np.savez_compressed(f, memories=memories, fingerprint=np.array(fingerprint))
            temporary.replace(path)
        cache.append(memories)
        if (e+1) % 500 == 0:
            print(f'observer cache: {e+1}/{len(samples.lengths)} episodes', flush=True)
    json_atomic(root / 'manifest.json', {**identity, 'fingerprint': fingerprint,
                                       'frames': sum(samples.lengths), 'bytes': sum(a.nbytes for a in cache)})
    return cache


def predictor_batch(models, samples, memories, windows, indices, horizon, statistics, device):
    from .types import PlanningState, action_one_hot
    from .rollout import rollout_step
    pairs = [windows[int(i)] for i in indices]
    frames = np.stack([samples.episode(e)['frames'][t:t+horizon+1] for e,t in pairs])
    target_states = np.stack([samples.episode(e)['states'][t+1:t+horizon+1] for e,t in pairs])
    actions = torch.as_tensor(np.stack([samples.episode(e)['actions'][t:t+horizon] for e,t in pairs]), device=device)
    encoded = frozen_encode(models['E'], frames.reshape(-1,64,64,3), device)
    from .types import ObservationLatent
    fine = encoded.fine.reshape(len(pairs), horizon+1, 256,64)
    coarse = encoded.coarse.reshape(len(pairs), horizon+1,64,64)
    source = ObservationLatent(fine[:,0],coarse[:,0])
    state = PlanningState(source, torch.as_tensor(np.stack([memories[e][t] for e,t in pairs]), device=device))
    loss = torch.zeros((), device=device); copy_loss = torch.zeros((), device=device)
    position_error, copy_error = [], []
    targets = torch.as_tensor(target_states[:,:,[0,1,4]], dtype=torch.float32, device=device)
    for j in range(horizon):
        state = rollout_step(state, action_one_hot(actions[:,j]), models['P'], models['U'])
        target = ObservationLatent(fine[:,j+1],coarse[:,j+1])
        loss = loss + latent_loss(state.observation, target, statistics) / horizon
        with torch.no_grad():
            copy_loss += latent_loss(source, target, statistics) / horizon
            position_error.append((models['H'](state.observation)*64 - targets[:,j]).abs().mean(0))
            copy_error.append((models['H'](source)*64 - targets[:,j]).abs().mean(0))
    return loss, {'copy_loss': float(copy_loss), 'h_mae': torch.stack(position_error).tolist(),
                  'copy_h_mae': torch.stack(copy_error).tolist(), 'windows': len(pairs)}


def _mean_metrics(rows):
    result = {}
    for key in rows[0][1]:
        values = [np.asarray(row[1][key], dtype=float) for row in rows]
        if key in ('observations','supervised_scalar_count','post_warmup_observations','windows'):
            result[key] = sum(values).tolist()
        else:
            weights = [r[1]['post_warmup_observations'] for r in rows] if key == 'r_mae' else [r[0] for r in rows]
            result[key] = np.average(values, axis=0, weights=weights).tolist()
    return result


@torch.no_grad()
def paired_memory_validation(models, count, device):
    from .data import history_pairs
    from .types import ObservationLatent, action_one_hot
    members = [member for pair in history_pairs(count,seed=7000) for member in pair['members']]
    frames = np.stack([m['frames'] for m in members])
    encoded = frozen_encode(models['E'],frames.reshape(-1,64,64,3),device)
    fine = encoded.fine.reshape(len(members),3,256,64)
    coarse = encoded.coarse.reshape(len(members),3,64,64)
    memory = torch.zeros(len(members),128,device=device)
    for t in range(3):
        s = ObservationLatent(fine[:,t],coarse[:,t])
        previous = torch.zeros(len(members),3,device=device) if t == 0 else action_one_hot(torch.ones(len(members),device=device,dtype=torch.long))
        memory = models['U'](memory,s,previous)
    reset = models['U'](torch.zeros_like(memory),s,torch.zeros(len(members),3,device=device))
    target = torch.as_tensor(np.stack([m['states'][-1] for m in members]),device=device)
    scale = torch.tensor([64,64,6,6,64],device=device)
    return {'paired_validation_r_mae':(models['R'](memory)*scale-target).abs().mean(0).tolist(),
            'paired_validation_reset_r_mae':(models['R'](reset)*scale-target).abs().mean(0).tolist()}


@torch.no_grad()
def perception_debug(models, samples, indices, run, device, selected_update):
    from PIL import Image, ImageDraw
    indices = indices[:8]
    x, target = samples.frame_batch(indices,device)
    s = models['E'](x); recon = models['D'](s); position = models['H'](s)*64
    canvas = Image.new('RGB',(560,170*len(indices)), '#17212b'); draw = ImageDraw.Draw(canvas)
    for row in range(len(indices)):
        for column, values in enumerate((x[row],recon[row])):
            pixels = (values.clamp(0,1).permute(1,2,0).cpu().numpy()*255).round().astype(np.uint8)
            canvas.paste(Image.fromarray(pixels).resize((128,128)),(column*140,row*170))
        draw.text((284,row*170+10),f'Validation row {int(indices[row])}\nActual | reconstructed\nH: {np.round(position[row].cpu().numpy(),2)}\nTruth: {np.round(target[row,[0,1,4]].cpu().numpy()*64,2)}',fill='white')
    canvas.save(Path(run)/'perception_debug.png')
    json_atomic(Path(run)/'perception_debug.json',{'validation_frame_indices':indices.tolist(),
        'selected_update':selected_update,'dataset_fingerprint':samples.dataset.fingerprint,
        'readout_positions':position.tolist(),'target_positions':(target[:,[0,1,4]]*64).tolist()})


def _stage_impl(config, data, run, stage, perception=None, memory=None, horizon=1, initialize_from=None, resume=False):
    from .models import Encoder, Decoder, PositionReadout, MemoryUpdater, StateReadout, Predictor
    run = Path(run); run.mkdir(parents=True, exist_ok=True)
    if (run/'last.pt').exists() and not resume:
        raise ValueError(f'{run} already has a checkpoint; use --resume or a new run directory')
    if not resume:
        for name in ('training','validation'):
            path = run/f'{name}.jsonl'
            if path.exists():
                path.replace(run/f'{name}_interrupted_before_checkpoint_{time.time_ns()}.jsonl')
    seed = int(config['seed']); seed_all(seed); rng = np.random.default_rng(seed)
    device = resolve_device(config.get('device','auto'))
    torch.set_num_threads(int(config.get('cpu_threads',4)))
    train, validation = Samples(data,'train'), Samples(data,'validation')
    setting_key = stage if stage != 'predictor' else f'predictor_{horizon}'
    settings = config['training'][setting_key]; common = config['training']
    deps = {}; statistics = None; models = {}; cache_train = cache_val = None
    if stage == 'perception':
        models = {'E': Encoder(), 'D': Decoder(), 'H': PositionReadout()}
    else:
        a = read_checkpoint(perception, dataset_fingerprint=train.dataset.fingerprint, stage='perception')
        deps['perception'] = a['model_fingerprint']
        models = load_observer(perception, memory if stage == 'predictor' else None, device)
        if stage == 'memory':
            models.update({'U': MemoryUpdater(), 'R': StateReadout()})
        else:
            b = read_checkpoint(memory); deps['memory'] = b['model_fingerprint']
            models['P'] = Predictor()
    train_keys = {'perception': ('E','D','H'), 'memory': ('U','R'), 'predictor': ('P',)}[stage]
    for key, model in models.items():
        model.to(device).requires_grad_(key in train_keys); model.train(key in train_keys)
    trainable = {k: models[k] for k in train_keys}
    opt = optimizer_for(trainable, common)
    start = 0; best = float('inf'); no_improvement = 0; examples = 0; elapsed_previous = 0.
    best_checkpoint = None; training_complete = False; continuation = None
    if initialize_from and not resume:
        initial = read_checkpoint(initialize_from, dependencies=deps, dataset_fingerprint=train.dataset.fingerprint, stage='predictor')
        if initial.get('horizon') != 1 or horizon != 5:
            raise ValueError('Five-step initialization requires a one-step predictor')
        if not config.get('smoke',False) and not initial.get('quality_gate',{}).get('passed',False):
            raise ValueError('One-step improvement gate failed; retain diagnostics before five-step training')
        models['P'].load_state_dict(initial['models']['P'])
        deps['initial_predictor'] = initial['model_fingerprint']
        statistics = initial['statistics']
    if resume:
        saved = read_checkpoint(run/'last.pt', dependencies=deps, dataset_fingerprint=train.dataset.fingerprint, stage=stage)
        if saved['config'] != config or saved.get('horizon',1) != horizon:
            raise ValueError('Resume requires the identical resolved config and horizon')
        for key in train_keys:
            models[key].load_state_dict(saved['models'][key])
        opt.load_state_dict(saved['optimizer']); start = saved['global_update']; best = saved['best_validation']
        examples = saved['examples_processed']; elapsed_previous = saved['elapsed_seconds']
        no_improvement = saved.get('no_improvement',0)
        training_complete = saved.get('training_complete',False)
        continuation = saved.get('continuation')
        if stage == 'predictor':
            statistics = saved['statistics']
        best_checkpoint = saved.get('best_checkpoint')
        if best_checkpoint:
            selected_snapshot = read_checkpoint(run/best_checkpoint,dependencies=deps,
                dataset_fingerprint=train.dataset.fingerprint,stage=stage)
            alias = run/'best.pt'; source = run/best_checkpoint
            if not alias.exists() or hashlib.sha256(alias.read_bytes()).digest() != hashlib.sha256(source.read_bytes()).digest():
                atomic_checkpoint_copy(source,alias)
        if stage == 'predictor' and horizon == 5:
            deps['initial_predictor'] = saved['dependencies']['initial_predictor']
            statistics = saved['statistics']
            if initialize_from and read_checkpoint(initialize_from)['model_fingerprint'] != deps['initial_predictor']:
                raise ValueError('Resume one-step initialization fingerprint differs')
        # Preserve the interrupted suffix separately, then replay from the last
        # committed RNG/optimizer state without duplicate scientific ledger rows.
        for name in ('training','validation'):
            path = run/f'{name}.jsonl'
            if path.exists():
                rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
                suffix = [row for row in rows if row['step'] > start]
                if suffix:
                    json_atomic(run/f'{name}_interrupted_{time.time_ns()}.json', {'checkpoint_update':start,'rows':suffix})
                    temporary = path.with_suffix('.tmp')
                    temporary.write_text(''.join(json.dumps(row)+'\n' for row in rows if row['step'] <= start))
                    temporary.replace(path)
        result_path = run/'paddle_result.json'
        if result_path.exists():
            prior = json.loads(result_path.read_text())
            if prior.get('global_update') == start and prior.get('status') in ('completed','failed_quality_gate'):
                if prior['status'] == 'failed_quality_gate':
                    raise RuntimeError('Completed one-step stage failed its quality gate; resume cannot extend its budget. Use a separately declared follow-up.')
                return prior
    if stage == 'predictor':
        if statistics is None:
            statistics = compute_statistics(models['E'], train, run, deps['perception'], device=device)
        else:
            if statistics.get('encoder_fingerprint') != deps['perception'] or statistics.get('dataset_fingerprint') != train.dataset.fingerprint:
                raise ValueError('Initial predictor statistics belong to a different encoder or dataset')
            json_atomic(run/'statistics.json',statistics)
        cache_train = observer_cache(models,train,run.parent/'observer_cache_train',deps,device)
        cache_val = observer_cache(models,validation,run.parent/'observer_cache_validation',deps,device)
        train_windows, val_windows = train.windows(horizon), validation.windows(horizon)
        if not train_windows or not val_windows:
            raise ValueError('No valid predictor windows with complete history and requested future horizon')
        train_count, val_count = len(train_windows), len(val_windows)
    elif stage == 'memory':
        train_count, val_count = len(train.lengths),len(validation.lengths)
    else:
        train_count,val_count = len(train.frames),len(validation.frames)
    val_size = min(val_count, int(settings['validation_examples']))
    val_indices = np.random.default_rng(seed+9817).choice(val_count,val_size,replace=False)
    json_atomic(run/'resolved_config.json',config)
    json_atomic(run/'paddle_manifest.json', {'stage': stage,'horizon': horizon,'config':config,
         'dependencies':deps,'dataset_fingerprint':train.dataset.fingerprint,'versions':versions(),
         'validation_indices':val_indices.tolist(),'tensor_schema':TENSOR_SCHEMA,
         **({'continuation':continuation} if continuation else {}),
         'input_storage': {split: {'kind':'read-only raw RGB/state/action memory map',
             'fingerprint':samples.raw_cache.fingerprint, 'path':str(samples.raw_cache.directory)}
             for split,samples in (('train',train),('validation',validation))
             if getattr(samples,'raw_cache',None) is not None}})
    if resume:
        restore_rng(saved['rng'],rng)
    begin = time.monotonic(); last_metrics = {}; gate = {}; source_hash = code_fingerprint()

    def continuation_work(update, processed, elapsed):
        if not continuation:
            return {}
        return {'continuation':continuation,
                'additional_updates':update-continuation['parent_global_update'],
                'additional_examples_processed':processed-continuation['parent_examples_processed'],
                'additional_elapsed_seconds':elapsed-continuation['parent_elapsed_seconds']}

    def batch_loss(samples, indices, validation_mode=False):
        if stage == 'perception':
            x,y = samples.frame_batch(indices,device); s = models['E'](x)
            reconstruction = models['D'](s); position = models['H'](s)
            pixel = (reconstruction-x).square().mean(); pos = (position-y[:,[0,1,4]]).square().mean()
            return pixel+pos, {'image_mse':float(pixel.detach()),'position_mse':float(pos.detach()),
                               'h_mae':((position-y[:,[0,1,4]]).detach().abs().mean(0)*64).tolist()}
        if stage == 'memory':
            return memory_batch(models,samples,indices,device,common.get('encoder_batch_size',128))
        return predictor_batch(models,samples,cache_val if validation_mode else cache_train,
                               val_windows if validation_mode else train_windows,indices,horizon,statistics,device)

    def validate():
        for m in trainable.values(): m.eval()
        rows = []; objectives = []
        with torch.no_grad():
            for i in range(0,len(val_indices),settings['batch_size']):
                indices = val_indices[i:i+settings['batch_size']]
                loss,metrics = batch_loss(validation,indices,True)
                rows.append((len(indices),metrics))
                objectives.append((metrics.get('supervised_scalar_count',len(indices)),float(loss)))
        metrics = _mean_metrics(rows)
        metrics['loss'] = sum(n*l for n,l in objectives)/sum(n for n,_ in objectives)
        if stage == 'memory' and settings.get('validation_pairs',0):
            metrics.update(paired_memory_validation(models,settings['validation_pairs'],device))
        for m in trainable.values(): m.train()
        return metrics

    def save(update, metrics, improved, stop_reason=None):
        nonlocal gate, best_checkpoint
        if stage == 'predictor':
            # Moving-object readout means both ball coordinates and paddle x;
            # require each coordinate improve, not just a pooled average.
            pred = np.asarray(metrics['h_mae']).mean(0); copy = np.asarray(metrics['copy_h_mae']).mean(0)
            gate = {'passed': bool(metrics['loss'] < metrics['copy_loss'] and np.all(pred < copy)),
                    'latent_improved':bool(metrics['loss'] < metrics['copy_loss']),
                    'coordinates_improved':(pred < copy).tolist(), 'smoke_bypass': bool(config.get('smoke',False))}
        model_states = {k:m.state_dict() for k,m in trainable.items()}
        if improved:
            best_checkpoint = f'checkpoints/best_{update:08d}.pt'
        value = {'schema_version':SCHEMA_VERSION,'tensor_schema':TENSOR_SCHEMA,'stage':stage,
                 'horizon':horizon,'global_update':update,'models':model_states,
                 'model_fingerprint':fingerprint_modules(trainable),'optimizer':opt.state_dict(),
                 'rng':rng_state(rng),'config':config,'versions':versions(), 'code_fingerprint':source_hash,
                 'dataset_fingerprint':train.dataset.fingerprint,'dependencies':deps,
                 'statistics':statistics,'metrics':metrics,'best_validation':best,
                 'best_validation_criterion':'minimum fixed-validation objective',
                 'no_improvement':no_improvement,'examples_processed':examples,
                 'elapsed_seconds':elapsed_previous+time.monotonic()-begin,'quality_gate':gate,
                 'best_checkpoint':best_checkpoint,'training_complete':stop_reason is not None,
                 'stop_reason':stop_reason}
        value.update(continuation_work(update,examples,value['elapsed_seconds']))
        # Immutable snapshot precedes the committed last state. If interrupted
        # between replacing last and best aliases, resume restores best from
        # the snapshot referenced by last, never from an uncommitted update.
        if improved: atomic_checkpoint(run/best_checkpoint,value)
        atomic_checkpoint(run/'last.pt',value)
        if improved: atomic_checkpoint(run/'best.pt',value)
        json_atomic(run/'status.json', {'stage':stage,'global_update':update,'best_validation':best,
                                      'status':'running','quality_gate':gate,'metrics':metrics})

    try:
        if start == 0:
            initial_metrics = validate()
            with (run/'validation.jsonl').open('a') as f:
                f.write(json.dumps({'step':0,**initial_metrics})+'\n')
            print(f'{setting_key} initial validation: {initial_metrics}',flush=True)
        completed_update = start
        for update in range(start+1,(start if training_complete else int(settings['updates']))+1):
            tick = time.monotonic()
            indices = rng.integers(0,train_count,size=int(settings['batch_size']))
            opt.zero_grad(set_to_none=True); loss,metrics = batch_loss(train,indices)
            if not torch.isfinite(loss): raise RuntimeError(f'Nonfinite {stage} loss at update {update}')
            loss.backward(); norm = nn.utils.clip_grad_norm_(
                [p for m in trainable.values() for p in m.parameters()],common['grad_clip'])
            opt.step(); examples += int(settings['batch_size']); completed_update = update
            row = {'step':update,'loss':float(loss.detach()),'grad_norm':float(norm),
                   'examples_processed':examples,'seconds':time.monotonic()-tick,**metrics}
            with (run/'training.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
            if update % int(common['validate_every']) == 0 or update == int(settings['updates']):
                last_metrics = validate(); old_best = best
                improved = last_metrics['loss'] < best
                if improved: best = last_metrics['loss']
                no_improvement = 0 if last_metrics['loss'] < old_best * .999 else no_improvement+1
                with (run/'validation.jsonl').open('a') as f:
                    f.write(json.dumps({'step':update,**last_metrics})+'\n')
                early_stop = common.get('early_stopping',True) and update >= 1000 and no_improvement >= 8
                stop_reason = 'maximum_updates' if update == int(settings['updates']) else 'early_stopping' if early_stop else None
                save(update,last_metrics,improved,stop_reason)
                print(f'{setting_key} update {update}/{settings["updates"]}: {last_metrics}',flush=True)
                if early_stop:
                    break
        if completed_update == start and (run/'last.pt').exists():
            last_metrics = read_checkpoint(run/'last.pt')['metrics']
        selected = read_checkpoint(run/'best.pt')
        if stage == 'perception':
            for key in train_keys:
                models[key].load_state_dict(selected['models'][key]); models[key].eval()
            perception_debug(models,validation,val_indices,run,device,selected['global_update'])
        result = {'stage':stage,'horizon':horizon,'status':'completed',
                  'smoke':bool(config.get('smoke',False)),'global_update':completed_update,
                  'selected_update':selected['global_update'],'metrics':selected['metrics'],
                  'quality_gate':selected.get('quality_gate',{}),'checkpoint':str(run/'best.pt'),
                  'elapsed_seconds':elapsed_previous+time.monotonic()-begin,
                  'examples_processed':examples,'dataset_fingerprint':train.dataset.fingerprint}
        result.update(continuation_work(completed_update,examples,result['elapsed_seconds']))
        json_atomic(run/'paddle_result.json',result); json_atomic(run/'status.json',result)
        if stage == 'predictor' and horizon == 1 and not config.get('smoke',False) and not result['quality_gate']['passed']:
            result['status'] = 'failed_quality_gate'; json_atomic(run/'paddle_result.json',result)
            json_atomic(run/'status.json',result)
            raise RuntimeError('One-step prediction did not beat copying on latent loss and each moving-position coordinate. Diagnostics retained; five-step stage blocked.')
        if (run/'failure.json').exists():
            (run/'failure.json').replace(run/f'failure_recovered_{time.time_ns()}.json')
        return result
    except BaseException as error:
        json_atomic(run/'failure.json',{'stage':stage,'error':str(error),'type':type(error).__name__})
        raise


def _stage(config, data, run, *args, **kwargs):
    """Refuse concurrent writers rather than corrupt optimizer state or ledgers."""
    run = Path(run); run.mkdir(parents=True,exist_ok=True)
    with (run/'.stage.lock').open('a') as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f'Another trainer owns {run}; do not launch a duplicate') from error
        return _stage_impl(config,data,run,*args,**kwargs)


def train_perception(config, data, run, resume=False):
    return _stage(config,data,run,'perception',resume=resume)


def train_memory(config, data, perception, run, resume=False):
    return _stage(config,data,run,'memory',perception=perception,resume=resume)


def train_predictor(config, data, perception, memory, horizon, run, initialize_from=None, resume=False):
    if horizon not in (1,5): raise ValueError('Baseline predictor horizon must be 1 or 5')
    if horizon == 5 and not initialize_from and not resume:
        raise ValueError('Five-step training must initialize from the selected one-step checkpoint')
    return _stage(config,data,run,'predictor',perception,memory,horizon,initialize_from,resume)
