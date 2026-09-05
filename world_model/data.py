"""Generic HDF5 trajectories with explicit episodes and action block alignment.

The schema is the authors' pixels/action/ep_len/ep_offset layout. Raw pixels stay
uint8 until batched preprocessing; state/proprio labels are never training inputs.
"""
import os
import h5py
import hdf5plugin  # registers filters used by the source datasets
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2


def preprocess_pixels(pixels, image_size=224):
    x = pixels.float() / 255
    mean = x.new_tensor([.485, .456, .406]).view(-1,1,1)
    std = x.new_tensor([.229, .224, .225]).view(-1,1,1)
    shape = x.shape[:-3]
    x = v2.functional.resize(((x-mean)/std).flatten(0, len(shape)-1),
                             [image_size], antialias=True)
    return x.reshape(*shape, *x.shape[-3:])


def split_episodes(count, seed=3072, train_fraction=0.9):
    if count < 2 or not 0 < train_fraction < 1:
        raise ValueError('An episode-disjoint split requires >=2 episodes')
    order = torch.randperm(count, generator=torch.Generator().manual_seed(seed)).tolist()
    cut = max(1, min(count-1, int(count*train_fraction)))
    return order[:cut], order[cut:]


class TrajectoryDataset(Dataset):
    def __init__(self, path, episodes, frameskip=5, num_steps=4, cache_bytes=0):
        self.path = str(path)
        self.episodes = np.asarray(episodes, dtype=np.int64)
        self.frameskip, self.num_steps = frameskip, num_steps
        if frameskip < 1 or num_steps < 2:
            raise ValueError('Invalid sequence length or frameskip')
        with h5py.File(path, 'r') as f:
            self.lengths, self.offsets = f['ep_len'][:], f['ep_offset'][:]
            self.action_dim = f['action'].shape[-1]
            self._cache = None
            if cache_bytes:
                required = sum(f[k].size*f[k].dtype.itemsize for k in ('pixels','action'))
                if required > cache_bytes:
                    raise ValueError(f'Dataset cache needs {required} bytes, exceeds configured cap')
                self._cache = {k:f[k][:] for k in ('pixels','action')}
        if len(set(self.episodes.tolist())) != len(self.episodes):
            raise ValueError('Duplicate episodes')
        if np.any(self.episodes < 0) or np.any(self.episodes >= len(self.lengths)):
            raise ValueError('Invalid episode ids')
        self.counts = np.maximum(0, self.lengths[self.episodes] - frameskip*num_steps + 1)
        self.cumulative = np.cumsum(self.counts)
        self._file, self._pid = None, None

    def __len__(self):
        return int(self.cumulative[-1]) if len(self.cumulative) else 0

    def locate(self, idx):
        if idx < 0: idx += len(self)
        if idx < 0 or idx >= len(self): raise IndexError(idx)
        row = int(np.searchsorted(self.cumulative, idx, side='right'))
        start = idx - (int(self.cumulative[row-1]) if row else 0)
        return int(self.episodes[row]), start

    def _open(self):
        if self._file is None or self._pid != os.getpid():
            self._file = h5py.File(self.path, 'r', rdcc_nbytes=32*1024**2)
            self._pid = os.getpid()
        return self._file

    def __getstate__(self):
        return {**self.__dict__, '_file': None, '_pid': None}

    def __getitem__(self, idx):
        ep, local = self.locate(idx)
        start = int(self.offsets[ep]) + local
        stop = start + self.frameskip*self.num_steps
        f = self._cache if self._cache is not None else self._open()
        pixels = torch.from_numpy(f['pixels'][start:stop:self.frameskip])
        if pixels.shape[-1] in (1,3): pixels = pixels.permute(0,3,1,2)
        actions = torch.from_numpy(f['action'][start:stop]).float().reshape(self.num_steps,-1)
        # The last action block is unused by the training target. Earlier NaNs
        # indicate a broken episode/action alignment, not padding to learn from.
        if not torch.isfinite(actions[:-1]).all():
            raise ValueError(f'Non-finite transition actions in episode {ep}, start {local}')
        return dict(pixels=pixels, action=actions, episode=ep, start=local)


def action_statistics(path, episodes):
    """Unbiased source-action std, fitted only on declared episodes."""
    with h5py.File(path, 'r') as f:
        offsets, lengths = f['ep_offset'][:], f['ep_len'][:]
        # Actions are small; keep the reference float32 reduction semantics.
        x = torch.from_numpy(np.concatenate([f['action'][int(offsets[e]):int(offsets[e]+lengths[e])]
                                            for e in episodes])).float()
    x = x[torch.isfinite(x).all(dim=-1)]
    mean, std = x.mean(0), x.std(0)
    if len(x)<2 or not torch.isfinite(std).all() or (std<=0).any():
        raise ValueError('Insufficient or degenerate action statistics')
    return dict(mean=mean.tolist(), std=std.tolist(), count=len(x))


def normalize_actions(actions, stats):
    d = len(stats['mean'])
    mean, std = actions.new_tensor(stats['mean']), actions.new_tensor(stats['std'])
    x = (actions.reshape(*actions.shape[:-1], -1, d)-mean)/std
    return torch.nan_to_num(x.reshape_as(actions), nan=0.0)
