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


def test_pixel_window_reader_matches_slice_across_chunks_and_episode_edges(tmp_path):
    from world_model.data import read_pixel_window, TrajectoryDataset
    rng = np.random.default_rng(16)
    pixels = rng.integers(0, 256, (240, 12, 12, 3), dtype=np.uint8)
    path = tmp_path/'chunked.h5'
    with h5py.File(path, 'w') as f:
        f.create_dataset('pixels', data=pixels, chunks=(100,12,12,3), compression='gzip')
        f['ep_len'] = [120,120]; f['ep_offset'] = [0,120]
        f['action'] = rng.normal(size=(240,2)).astype(np.float32)
    with h5py.File(path, 'r') as f:
        for start, stop, skip in [(95,115,5),(100,120,5),(215,235,5),(20,28,1)]:
            expected = pixels[start:stop:skip]
            assert np.array_equal(read_pixel_window(f['pixels'],start,stop,skip), expected)
            assert np.array_equal(read_pixel_window(pixels,start,stop,skip), expected)
        chw = pixels.transpose(0,3,1,2)
        assert np.array_equal(read_pixel_window(chw,95,115,5), chw[95:115:5])
    streamed = TrajectoryDataset(path,[0,1],frameskip=5,num_steps=4)
    cached = TrajectoryDataset(path,[0,1],frameskip=5,num_steps=4,cache_bytes=10_000_000)
    for index in [0,95,100,101,len(streamed)-1]:
        left,right=streamed[index],cached[index]
        assert left['episode']==right['episode'] and left['start']==right['start']
        torch.testing.assert_close(left['pixels'],right['pixels'],rtol=0,atol=0)
        torch.testing.assert_close(left['action'],right['action'],rtol=0,atol=0)
