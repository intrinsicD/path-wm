"""Bounded PushT stages with complete observed histories and atomic recovery."""
from __future__ import annotations

from collections import OrderedDict
import fcntl
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from world_model.paddle.types import ObservationLatent, PlanningState
from world_model.paddle.training import (CoordinateVariance, images, seed_all,
    optimizer_for, frozen_encode, latent_loss)
from .checkpoints import (SCHEMA_VERSION, TENSOR_SCHEMA, ACTION_SCHEMA_VERSION,
    atomic_checkpoint, atomic_checkpoint_copy, json_atomic, code_fingerprint,
    fingerprint_modules, load_observer, read_checkpoint, resolve_device,
    restore_rng, rng_state, versions)
from .models import initial_previous_action, rollout


class Samples:
    def __init__(self, data, split, *, episode_limit=None, prefix=None):
        from .data import EpisodeDataset
        self.dataset = EpisodeDataset(data, split)
        self.cache = OrderedDict()
        self.lengths = self.dataset.lengths[:episode_limit]
        if prefix: self.lengths = [min(n, prefix) for n in self.lengths]
        self.frames = [(e,t) for e,n in enumerate(self.lengths) for t in range(n)]

    def episode(self, index):
        index = int(index)
        if index not in self.cache:
            episode = self.dataset[index]
            n = self.lengths[index]
            episode = {k: (v[:n-1] if k in ('actions','actions_world') else v[:n])
                       if isinstance(v,np.ndarray) and v.ndim > 0 else v for k,v in episode.items()}
            self.cache[index] = episode
            if len(self.cache) > 32: self.cache.popitem(last=False)
        self.cache.move_to_end(index)
        return self.cache[index]

    def frame_batch(self, indices, device):
        pairs = [self.frames[int(i)] for i in indices]
        selected = [(self.episode(e),t) for e,t in pairs]
        return (images(np.stack([e['frames'][t] for e,t in selected]),device),
                torch.as_tensor(np.stack([e['pose_targets'][t] for e,t in selected]),device=device))

    def windows(self, horizon):
        return [(e,t) for e,n in enumerate(self.lengths) for t in range(2,n-horizon)]


def supervised_memory_loss(estimate, target, mask):
    if estimate.shape != target.shape or mask.shape != estimate.shape:
        raise ValueError('Memory scalar supervision shapes must match')
    return (estimate-target).square().masked_select(mask).mean()


def pose_metrics(estimate, target):
    estimate, target = estimate.detach(),target.detach()
    delta = torch.atan2(estimate[...,4],estimate[...,5])-torch.atan2(target[...,4],target[...,5])
    angle = torch.atan2(delta.sin(),delta.cos())
    return dict(position_mae=((estimate[...,:4]-target[...,:4]).abs().mean(0)*512).tolist(),
                angle_mae_deg=float(angle.abs().mean()*180/np.pi),
                angle_norm_mean=float(estimate[...,4:6].norm(dim=-1).mean()),
                pusher_position_mse=float(((estimate[...,:2]-target[...,:2])*512).square().mean()),
                block_position_mse=float(((estimate[...,2:4]-target[...,2:4])*512).square().mean()),
                angle_mse=float(angle.square().mean()))


def perception_batch(models, samples, indices, device):
    x,y = samples.frame_batch(indices,device); s = models['E'](x)
    reconstruction, pose = models['D'](s),models['H'](s)
    pixel,position = (reconstruction-x).square().mean(),(pose-y).square().mean()
    return pixel+position, dict(image_mse=float(pixel.detach()),pose_mse=float(position.detach()),**pose_metrics(pose,y))


def memory_batch(models, samples, indices, device, encoder_batch_size=128):
    episodes = [samples.episode(i) for i in indices]
    lengths = [len(e['frames']) for e in episodes]; length,batch=max(lengths),len(episodes)
    encoded = [frozen_encode(models['E'],e['frames'],device,encoder_batch_size) for e in episodes]
    memory = torch.zeros(batch,128,device=device); estimates=[]
    target = torch.zeros(batch,length,11,device=device)
    valid = torch.zeros(batch,length,dtype=torch.bool,device=device)
    mask = torch.zeros_like(target,dtype=torch.bool)
    previous = torch.zeros(batch,length,2,device=device); previous[:,0] = -1
    for b,e in enumerate(episodes):
        n=lengths[b]; valid[b,:n]=True
        target[b,:n] = torch.as_tensor(np.concatenate([e['pose_targets'],e['motion_targets']],-1),device=device)
        mask[b,:n,:6]=True
        mask[b,:n,6:]=torch.as_tensor(e['motion_mask'],device=device)
        previous[b,1:n] = torch.as_tensor(e['actions'],device=device)
    for t in range(length):
        s=ObservationLatent.cat([v[min(t,lengths[b]-1):min(t,lengths[b]-1)+1] for b,v in enumerate(encoded)])
        candidate=models['U'](memory,s,previous[:,t])
        memory=torch.where(valid[:,t,None],candidate,memory)
        estimates.append(models['R'](memory))
    estimate=torch.stack(estimates,1)
    loss=supervised_memory_loss(estimate,target,mask)
    metrics=pose_metrics(estimate[valid][:,:6],target[valid][:,:6])
    after=mask[...,6]
    scales=torch.tensor(samples.dataset.manifest['normalization']['motion_scales'],device=device)
    metrics.update(motion_mae=((estimate.detach()[...,6:]-target[...,6:])[after].abs().mean(0)*scales).tolist(),
                   observations=sum(lengths),post_warmup_observations=int(after.sum()),
                   supervised_scalar_count=int(mask.sum()))
    return loss,metrics


def _mean_metrics(rows):
    result={}
    counts=('windows','observations','post_warmup_observations','supervised_scalar_count')
    for key in rows[0][1]:
        values=[np.asarray(r[1][key],dtype=float) for r in rows]
        if key in counts: value=sum(values)
        else:
            weight_key='post_warmup_observations' if key=='motion_mae' else 'observations'
            weights=[m.get(weight_key,n) for n,m in rows]
            value=sum(w*v for w,v in zip(weights,values))/sum(weights)
        result[key]=value.tolist() if value.ndim else float(value)
    return result


def predictive_gate(metrics, smoke=False):
    improved={key:bool(np.all(np.asarray(metrics[key])<np.asarray(metrics['copy_'+key])))
              for key in ('pusher_position_mse','block_position_mse','angle_mse')}
    latent=bool(metrics['loss']<metrics['copy_loss'])
    return dict(passed=latent and all(improved.values()),latent_improved=latent,
                diagnostics_improved=improved,smoke_bypass=bool(smoke))


def compute_statistics(encoder, samples, run, fingerprint, batch_size=128, device='cpu'):
    if samples.dataset.split != 'train':
        raise ValueError('Latent statistics require training samples; held-out splits are forbidden')
    identity=dict(encoder_fingerprint=fingerprint,dataset_fingerprint=samples.dataset.fingerprint,
                  tensor_schema=TENSOR_SCHEMA,precision='float32',lengths=samples.lengths,split='train')
    path=Path(run)/'statistics.json'
    if path.exists():
        result=json.loads(path.read_text())
        if all(result.get(k)==v for k,v in identity.items()): return result
    fine,coarse=CoordinateVariance(),CoordinateVariance()
    for e in range(len(samples.lengths)):
        frames=samples.episode(e)['frames']
        for start in range(0,len(frames),batch_size):
            with torch.no_grad(): s=encoder(images(frames[start:start+batch_size],device))
            fine.update(s.fine); coarse.update(s.coarse)
    result=dict(**identity,v_fine=fine.variance(),v_coarse=coarse.variance(),count=fine.count)
    json_atomic(path,result); return result


def observer_cache(models,samples,root,dependencies,device,encoder_batch_size=128):
    root=Path(root); root.mkdir(parents=True,exist_ok=True)
    identity=dict(dependencies={k:v for k,v in dependencies.items() if k in ('perception','memory')},
        dataset=samples.dataset.fingerprint,split=samples.dataset.split,lengths=samples.lengths,schema=TENSOR_SCHEMA)
    fingerprint=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest(); cache=[]
    for e,n in enumerate(samples.lengths):
        path=root/f'{e:06d}.npz'; memories=None
        if path.exists():
            with np.load(path,allow_pickle=False) as saved:
                if str(saved['fingerprint'])==fingerprint:
                    candidate=saved['memories']
                    if candidate.shape==(n,128) and np.isfinite(candidate).all(): memories=candidate
        if memories is None:
            episode=samples.episode(e); s=frozen_encode(models['E'],episode['frames'],device,encoder_batch_size)
            memory=torch.zeros(1,128,device=device); result=[]
            with torch.no_grad():
                for t in range(n):
                    action=initial_previous_action(1,device) if t==0 else torch.as_tensor(episode['actions'][t-1:t],device=device)
                    memory=models['U'](memory,s[t:t+1],action); result.append(memory.cpu().numpy()[0])
            memories=np.stack(result)
            temporary=path.with_suffix('.tmp')
            with temporary.open('wb') as f: np.savez_compressed(f,memories=memories,fingerprint=np.array(fingerprint))
            temporary.replace(path)
        cache.append(memories)
    json_atomic(root/'manifest.json',dict(**identity,fingerprint=fingerprint,frames=sum(samples.lengths),bytes=sum(x.nbytes for x in cache)))
    return cache


def predictor_batch(models,samples,memories,windows,indices,horizon,statistics,device):
    pairs=[windows[int(i)] for i in indices]; batch=len(pairs)
    episodes=[samples.episode(e) for e,t in pairs]
    frames=np.stack([ep['frames'][t:t+horizon+1] for ep,(_,t) in zip(episodes,pairs)])
    actions=torch.as_tensor(np.stack([ep['actions'][t:t+horizon] for ep,(_,t) in zip(episodes,pairs)]),device=device)
    targets=torch.as_tensor(np.stack([ep['pose_targets'][t+1:t+horizon+1] for ep,(_,t) in zip(episodes,pairs)]),device=device)
    encoded=frozen_encode(models['E'],frames.reshape(-1,64,64,3),device)
    fine=encoded.fine.reshape(batch,horizon+1,256,64); coarse=encoded.coarse.reshape(batch,horizon+1,64,64)
    source=ObservationLatent(fine[:,0],coarse[:,0])
    initial=PlanningState(source,torch.as_tensor(np.stack([memories[e][t] for e,t in pairs]),device=device))
    states=rollout(initial,actions,models['P'],models['U'])
    loss=torch.zeros((),device=device); copy_loss=torch.zeros_like(loss); rows=[]; copies=[]
    for j,state in enumerate(states):
        target=ObservationLatent(fine[:,j+1],coarse[:,j+1])
        loss=loss+latent_loss(state.observation,target,statistics)/horizon
        with torch.no_grad():
            copy_loss+=latent_loss(source,target,statistics)/horizon
            rows.append(pose_metrics(models['H'](state.observation),targets[:,j]))
            copies.append(pose_metrics(models['H'](source),targets[:,j]))
    metrics={k:[r[k] for r in rows] for k in rows[0]}
    metrics.update({'copy_'+k:[r[k] for r in copies] for k in copies[0]})
    return loss,dict(**metrics,copy_loss=float(copy_loss),windows=batch)


def perception_debug(models,samples,indices,run,device,step):
    from PIL import Image
    x,y=samples.frame_batch(indices[:8],device)
    with torch.no_grad():
        s=models['E'](x); reconstructed=models['D'](s); pose=models['H'](s)
    grid=torch.cat([torch.cat(list(x),dim=2),torch.cat(list(reconstructed),dim=2)],dim=1)
    Image.fromarray((grid.clamp(0,1).permute(1,2,0).cpu().numpy()*255).round().astype(np.uint8)).save(Path(run)/'reconstruction.png')
    json_atomic(Path(run)/'reconstruction.json',dict(selected_update=step,readout=pose.tolist(),target=y.tolist(),indices=indices[:8].tolist()))


def _stage_impl(config, data, run, stage, perception=None, memory=None, horizon=1, initialize_from=None, resume=False):
    from .models import Encoder, Decoder, PoseReadout, MemoryUpdater, StateReadout, Predictor
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
    limits=config.get('development_subset',{})
    if limits and not config.get('smoke',False):
        raise ValueError('Development subsets require an explicit smoke config')
    train=Samples(data,'train',episode_limit=limits.get('train_episodes'),prefix=limits.get('prefix'))
    validation=Samples(data,'validation',episode_limit=limits.get('validation_episodes'),prefix=limits.get('prefix'))
    source_frame_count = len(train.frames)
    population = {'source_dataset_fingerprint':train.dataset.fingerprint,
                  'source_frames':source_frame_count,'supplement_frames':0,
                  'presentation_unit': {'perception':'frame','memory':'episode','predictor':'window'}[stage]}
    mixed_perception = stage == 'perception' and bool(config.get('perception_supplement'))
    if mixed_perception:
        from .perception_data import SupplementFrames, MixedPerceptionSamples
        supplement_config = config['perception_supplement']
        supplement = SupplementFrames(supplement_config['directory'],supplement_config['fingerprint'])
        train = MixedPerceptionSamples(train,supplement)
        population.update(supplement_frames=supplement.manifest['count'],
                          supplement_fingerprint=supplement.manifest['fingerprint'],
                          supplement_directory=supplement_config['directory'],
                          sampling='uniform concatenated equal-sized populations; expected 50/50, one sampler')
    presentations = {'source':0,'supplement':0}
    setting_key = stage if stage != 'predictor' else f'predictor_{horizon}'
    settings = config['training'][setting_key]; common = config['training']
    deps = {}; statistics = None; models = {}; cache_train = cache_val = None
    if stage == 'perception':
        models = {'E': Encoder(), 'D': Decoder(), 'H': PoseReadout()}
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
        if (mixed_perception or 'training_population' in saved) and saved.get('training_population') != population:
            raise ValueError('Resume requires the identical training population and supplement fingerprint')
        presentations = saved.get('population_presentations',{'source':saved['examples_processed'],'supplement':0})
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
        result_path = run/'pusht_result.json'
        if result_path.exists():
            prior = json.loads(result_path.read_text())
            if prior.get('global_update') == start and prior.get('status') in ('completed','failed_quality_gate'):
                selected_gate = (selected_snapshot if best_checkpoint else saved).get('quality_gate',{})
                failed_gate = (stage == 'predictor' and horizon == 1 and not config.get('smoke',False)
                               and not selected_gate.get('passed',False))
                if prior['status'] == 'failed_quality_gate' or failed_gate:
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
    json_atomic(run/'pusht_manifest.json', {'stage': stage,'horizon': horizon,'config':config,
         'dependencies':deps,'dataset_fingerprint':train.dataset.fingerprint,'versions':versions(),
         'validation_indices':val_indices.tolist(),'tensor_schema':TENSOR_SCHEMA,
         **({'continuation':continuation} if continuation else {}),
         'normalization':train.dataset.manifest['normalization'],
         'training_population':population,
         'input_storage':{'kind':'read-only RGB64 uint8 memory map'},
         'training_lengths':train.lengths,'validation_lengths':validation.lengths})
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
            return perception_batch(models,samples,indices,device)
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
        for m in trainable.values(): m.train()
        return metrics

    def save(update, metrics, improved, stop_reason=None):
        nonlocal gate, best_checkpoint
        if stage == 'predictor':
            gate = predictive_gate(metrics,smoke=config.get('smoke',False))
        model_states = {k:m.state_dict() for k,m in trainable.items()}
        if improved:
            best_checkpoint = f'checkpoints/best_{update:08d}.pt'
        value = {'schema_version':SCHEMA_VERSION,'tensor_schema':TENSOR_SCHEMA,'action_schema':ACTION_SCHEMA_VERSION,
                 'normalization':train.dataset.manifest['normalization'],'stage':stage,
                 'horizon':horizon,'global_update':update,'models':model_states,
                 'model_fingerprint':fingerprint_modules(trainable),'optimizer':opt.state_dict(),
                 'rng':rng_state(rng),'config':config,'versions':versions(), 'code_fingerprint':source_hash,
                 'dataset_fingerprint':train.dataset.fingerprint,'dependencies':deps,
                 'statistics':statistics,'metrics':metrics,'best_validation':best,
                 'training_population':population,'population_presentations':dict(presentations),
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
        if start == 0 and not resume:
            initial_metrics = validate()
            with (run/'validation.jsonl').open('a') as f:
                f.write(json.dumps({'step':0,**initial_metrics})+'\n')
            print(f'{setting_key} initial validation: {initial_metrics}',flush=True)
            best = initial_metrics['loss']
            save(0,initial_metrics,True)
        completed_update = start
        for update in range(start+1,(start if training_complete else int(settings['updates']))+1):
            tick = time.monotonic()
            indices = rng.integers(0,train_count,size=int(settings['batch_size']))
            opt.zero_grad(set_to_none=True); loss,metrics = batch_loss(train,indices)
            if not torch.isfinite(loss): raise RuntimeError(f'Nonfinite {stage} loss at update {update}')
            loss.backward(); norm = nn.utils.clip_grad_norm_(
                [p for m in trainable.values() for p in m.parameters()],common['grad_clip'])
            opt.step(); examples += int(settings['batch_size']); completed_update = update
            supplemental = int(np.count_nonzero(indices >= source_frame_count)) if mixed_perception else 0
            presentations['supplement'] += supplemental
            presentations['source'] += len(indices)-supplemental
            row = {'step':update,'loss':float(loss.detach()),'grad_norm':float(norm),
                   'examples_processed':examples,'seconds':time.monotonic()-tick,
                   'population_presentations':dict(presentations),**metrics}
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
        failed_gate = (stage == 'predictor' and horizon == 1 and not config.get('smoke',False)
                       and not selected.get('quality_gate',{}).get('passed',False))
        result = {'stage':stage,'horizon':horizon,'status':'failed_quality_gate' if failed_gate else 'completed',
                  'smoke':bool(config.get('smoke',False)),'global_update':completed_update,
                  'selected_update':selected['global_update'],'metrics':selected['metrics'],
                  'quality_gate':selected.get('quality_gate',{}),'checkpoint':str(run/'best.pt'),
                  'elapsed_seconds':elapsed_previous+time.monotonic()-begin,
                  'examples_processed':examples,'dataset_fingerprint':train.dataset.fingerprint,
                  'normalization':train.dataset.manifest['normalization'],
                  'training_population':population,'population_presentations':dict(presentations),
                  'peak_cuda_bytes':torch.cuda.max_memory_allocated(device) if device.type=='cuda' else 0}
        result.update(continuation_work(completed_update,examples,result['elapsed_seconds']))
        json_atomic(run/'pusht_result.json',result); json_atomic(run/'status.json',result)
        if failed_gate:
            raise RuntimeError('One-step prediction did not beat copying on latent loss, pusher pose, block pose, and circular angle. Diagnostics retained; five-step stage blocked.')
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
