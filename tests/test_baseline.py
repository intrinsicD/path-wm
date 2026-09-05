import copy
import h5py
import numpy as np
import torch
from torch import nn
from world_model.model import WorldModel, build_model
from world_model.data import TrajectoryDataset, split_episodes
from world_model.objective import loss_from_embeddings


def test_episode_windows_and_action_timing(tmp_path):
    path = tmp_path / 'two_episodes.h5'
    with h5py.File(path, 'w') as f:
        f['ep_len'] = [12, 12]
        f['ep_offset'] = [0, 12]
        f['pixels'] = np.arange(24, dtype=np.uint8)[:, None, None, None].repeat(3, 3)
        f['action'] = np.arange(24, dtype=np.float32)[:, None]
    ds = TrajectoryDataset(path, [1], frameskip=2, num_steps=4)
    assert len(ds) == 5
    clip = ds[0]
    assert clip['pixels'][:, 0, 0, 0].tolist() == [12, 14, 16, 18]
    assert clip['action'].tolist() == [[12,13],[14,15],[16,17],[18,19]]
    assert ds[-1]['action'][-1, -1] == 23
    cached = TrajectoryDataset(path, [1], frameskip=2, num_steps=4, cache_bytes=10000)
    torch.testing.assert_close(cached[-1]['pixels'], ds[-1]['pixels'])
    torch.testing.assert_close(cached[-1]['action'], ds[-1]['action'])
    train, val = split_episodes(20)
    assert not set(train) & set(val)
    assert sorted(train + val) == list(range(20))


def test_target_encoder_receives_gradients():
    class Predictor(nn.Module):
        def forward(self, x, a):
            return x * 0 + a * 0
    model = WorldModel(nn.Identity(), Predictor(), nn.Identity(), nn.Identity(), nn.Identity())
    z = torch.randn(4, 4, 8, requires_grad=True)
    terms = loss_from_embeddings(model, z, torch.zeros_like(z), lambda x: x.sum()*0)
    terms['loss'].backward()
    assert z.grad[:, -1].abs().sum() > 0


def test_predictor_causality_and_component_replacement():
    model = build_model({'width': 12, 'encoder_depth': 1, 'encoder_heads': 3,
                         'predictor_depth': 1, 'predictor_heads': 2, 'head_dim': 4,
                         'mlp_dim': 24, 'projector_dim': 24, 'image_size': 28})
    model.eval()
    z, a = torch.randn(2,3,12), torch.randn(2,3,10)
    before = model.predict(z, a)
    z[:, -1] += 100
    a[:, -1] -= 100
    torch.testing.assert_close(before[:, :2], model.predict(z,a)[:, :2])
    model.projector = nn.Identity()
    assert model.encode(torch.randn(2,4,3,28,28)).shape == (2,4,12)
