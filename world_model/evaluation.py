"""Held-out predictive checks; controls are reported alongside loss and variance."""
import torch
from world_model.data import preprocess_pixels, normalize_actions
from world_model.training import autocast_context


@torch.no_grad()
def evaluate_prediction(model, loader, stats, device, image_size=224, batches=8, precision='float32'):
    was_training=model.training
    model.eval()
    sums={}; count=0
    for index,batch in enumerate(loader):
        if index>=batches: break
        pixels=preprocess_pixels(batch['pixels'].to(device),image_size)
        actions=normalize_actions(batch['action'].to(device),stats)
        with autocast_context(device,precision):
            z=model.encode(pixels)
            pred=model.predict(z[:,:-1],actions[:,:-1])
            zero=model.predict(z[:,:-1],torch.zeros_like(actions[:,:-1]))
            shuffled=model.predict(z[:,:-1],actions.roll(1,0)[:,:-1])
            rollout=model.rollout(z[:,:1],actions[:,:-1])
        z,pred,zero,shuffled,rollout=[x.float() for x in [z,pred,zero,shuffled,rollout]]
        target=z[:,1:]
        metrics=dict(pred_mse=(pred-target).square().mean(),
            identity_mse=(z[:,:-1]-target).square().mean(),
            zero_action_mse=(zero-target).square().mean(),
            shuffled_action_mse=(shuffled-target).square().mean(),
            rollout_mse=(rollout-target).square().mean(),
            embedding_std=z.std(dim=0).mean(),
            action_effect=(pred-shuffled).square().mean())
        b=len(z); count+=b
        for k,v in metrics.items(): sums[k]=sums.get(k,0)+float(v)*b
    model.train(was_training)
    if not count: raise ValueError('Empty evaluation loader')
    return {**{k:v/count for k,v in sums.items()},'examples':count}


def sample_source_cases(lengths, offsets, count, seed=42, goal_offset=25, population=None):
    """Pinned LeWM eval.py window sampling, including its final-row exclusion.

    A restricted population is for held-out diagnostics only. Full reference
    evaluation supplies every episode. Source IDs are contiguous episode indices.
    """
    import numpy as np
    lengths,offsets=np.asarray(lengths),np.asarray(offsets)
    population=list(range(len(lengths))) if population is None else list(population)
    if count<1 or goal_offset<1 or not population:
        raise ValueError('Positive case count, offset and nonempty population required')
    if len(set(population))!=len(population) or min(population)<0 or max(population)>=len(lengths):
        raise ValueError('Invalid episode population')
    ranges=[np.arange(int(offsets[e]),int(offsets[e]+lengths[e]-goal_offset))
            for e in sorted(population) if lengths[e]>goal_offset]
    valid=np.concatenate(ranges) if ranges else np.empty(0,dtype=np.int64)
    if count>len(valid)-1:raise ValueError('Too few eligible reference windows')
    rows=np.sort(valid[np.random.default_rng(seed).choice(len(valid)-1,size=count,replace=False)])
    cases=[]
    for row in rows:
        ep=int(np.searchsorted(offsets,row,side='right')-1)
        cases.append(dict(row=int(row),episode=ep,start=int(row-offsets[ep])))
    return cases
