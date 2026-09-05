"""Explicit baseline data populations shared by preparation and training."""
import hashlib
import h5py
import numpy as np
import torch
from torch.utils.data import Subset
from world_model.data import TrajectoryDataset, split_episodes, explicit_episode_split, action_statistics


def prepare_training_data(cfg, ds_cfg):
    with h5py.File(ds_cfg['path'],'r') as source:
        count=len(source['ep_len'])
    args=dict(path=ds_cfg['path'],frameskip=ds_cfg['frameskip'],num_steps=ds_cfg['history']+1,
              cache_bytes=cfg.get('cache_bytes',0))
    protocol=cfg.get('split_protocol','episodes')
    if protocol=='random_windows':
        if cfg.get('episode_split') or cfg.get('train_episodes'):
            raise ValueError('Random-window protocol cannot truncate or split episodes')
        if cfg.get('normalization_population')!='full_source':
            raise ValueError('Random-window reproduction requires full_source normalization')
        tr=va=list(range(count))
        base=TrajectoryDataset(episodes=tr,**args)
        fraction=cfg['train_fraction']
        if not 0<fraction<1: raise ValueError('Invalid window training fraction')
        order=torch.randperm(len(base),generator=torch.Generator().manual_seed(cfg['seed'])).tolist()
        cut=int(fraction*len(base))
        if not 0<cut<len(base): raise ValueError('Empty training or validation windows')
        train,val=Subset(base,order[:cut]),Subset(base,order[cut:])
        digest=lambda values:hashlib.sha256(np.asarray(values,dtype='<i8').tobytes()).hexdigest()
        receipt=dict(split_protocol=protocol,normalization_population='full_source',
                     population='Random windows; training and validation share source episodes/configurations',
                     window_count=len(base),train_windows=cut,val_windows=len(base)-cut,
                     split_seed=cfg['seed'],rounding='floor(train_fraction * window_count)',
                     train_indices_sha256=digest(train.indices),val_indices_sha256=digest(val.indices))
        return train,val,tr,va,action_statistics(ds_cfg['path'],tr),receipt
    if protocol!='episodes': raise ValueError(f'Unknown split protocol: {protocol}')
    if cfg.get('normalization_population','training_episodes')!='training_episodes':
        raise ValueError('Episode-disjoint protocol requires training_episodes normalization')
    receipt=None
    if cfg.get('episode_split'):
        tr,va,receipt=explicit_episode_split(cfg['episode_split'],count)
        if cfg.get('train_episodes'):raise ValueError('Cannot truncate an explicit split')
    else:
        tr,va=split_episodes(count,cfg['seed'],cfg['train_fraction'])
        if cfg.get('train_episodes'):tr=tr[:cfg['train_episodes']]
    return (TrajectoryDataset(episodes=tr,**args),TrajectoryDataset(episodes=va,**args),
            tr,va,action_statistics(ds_cfg['path'],tr),receipt)
