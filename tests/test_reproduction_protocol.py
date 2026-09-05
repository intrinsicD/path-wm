"""Verify source-scale data contracts without allocating or training a model."""
import h5py
import numpy as np
import pytest
import torch
from world_model.protocol import prepare_training_data


def test_author_window_split_and_all_source_normalization(tmp_path):
    source=tmp_path/'source.h5'
    with h5py.File(source,'w') as f:
        f['ep_len']=[12,12,12,12]; f['ep_offset']=[0,12,24,36]
        f['pixels']=np.zeros((48,8,8,3),np.uint8)
        f['action']=np.arange(96,dtype=np.float32).reshape(48,2)
    ds_cfg=dict(path=str(source),frameskip=1,history=3,action_dim=2)
    cfg=dict(seed=3072,train_fraction=.9,split_protocol='random_windows',normalization_population='full_source')
    train,val,tr,va,stats,receipt=prepare_training_data(cfg,ds_cfg)
    assert len(train)==32 and len(val)==4  # Explicit floor(0.9 * 36).
    assert set(train.indices).isdisjoint(val.indices)
    assert sorted(train.indices+val.indices)==list(range(36))
    assert tr==va==[0,1,2,3]  # Window validation is not episode-held-out.
    x=torch.arange(96,dtype=torch.float32).reshape(48,2)
    np.testing.assert_allclose(stats['mean'],x.mean(0).numpy())
    np.testing.assert_allclose(stats['std'],x.std(0).numpy())
    assert stats['count']==48 and receipt['split_protocol']=='random_windows'
    again=prepare_training_data(cfg,ds_cfg)
    assert train.indices==again[0].indices and receipt==again[-1]
    with pytest.raises(ValueError,match='normalization'):
        prepare_training_data({**cfg,'normalization_population':'training_episodes'},ds_cfg)
