import h5py
import numpy as np
import torch
from world_model.data import action_statistics, normalize_actions, preprocess_pixels


def test_normalization_excludes_held_out_episodes_and_preserves_action_blocks(tmp_path):
    path=tmp_path/'data.h5'
    with h5py.File(path,'w') as f:
        f['ep_len']=[3,3];f['ep_offset']=[0,3]
        f['action']=np.array([[0,10],[2,12],[4,14],[100,100],[200,200],[300,300]],np.float32)
    stats=action_statistics(path,[0])
    assert stats['mean']==[2,12]
    assert stats['std']==[2,2]
    x=normalize_actions(torch.tensor([[[0.,10.,2.,12.]]]),stats)
    torch.testing.assert_close(x,torch.tensor([[[-1.,-1.,0.,0.]]]))


def test_pixel_preprocessing_preserves_batch_time_channel_layout():
    pixels=torch.full((2,4,3,8,8),255,dtype=torch.uint8)
    x=preprocess_pixels(pixels,28)
    assert x.shape==(2,4,3,28,28)
    expected=(torch.ones(3)-torch.tensor([.485,.456,.406]))/torch.tensor([.229,.224,.225])
    torch.testing.assert_close(x[1,3,:,4,5],expected)
