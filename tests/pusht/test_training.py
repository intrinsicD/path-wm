"""Training failures that could silently change the PushT experiment."""
import copy

import numpy as np
import pytest
import torch

from world_model.pusht import training, checkpoints
from world_model.pusht.models import Encoder


def test_memory_objective_masks_only_declared_scalars():
    estimate = torch.ones(2, 4, 11, requires_grad=True)
    target = torch.zeros_like(estimate)
    mask = torch.ones_like(estimate, dtype=torch.bool)
    mask[:, :2, 6:] = False
    mask[1, 3] = False
    estimate.data[~mask] = 1000
    loss = training.supervised_memory_loss(estimate, target, mask)
    assert loss.item() == 1
    loss.backward()
    assert torch.equal(estimate.grad[~mask], torch.zeros_like(estimate.grad[~mask]))
    assert estimate.grad[mask].sum().item() == pytest.approx(2.)


def test_gate_cannot_hide_orientation_failure_or_claim_smoke_pass():
    metrics = dict(loss=.5, copy_loss=1., pusher_position_mse=[1.], copy_pusher_position_mse=[2.],
                   block_position_mse=[1.], copy_block_position_mse=[2.], angle_mse=[.3], copy_angle_mse=[.2])
    assert not training.predictive_gate(metrics, smoke=True)['passed']
    metrics['angle_mse'] = [.1]
    assert training.predictive_gate(metrics)['passed']
    metrics['loss'] = 1.
    assert not training.predictive_gate(metrics)['passed']


def test_angle_metrics_wrap_and_memory_aggregation_uses_observations():
    target = torch.tensor([[0., 0., 0., 0., np.sin(.01), np.cos(.01)]], dtype=torch.float32)
    pred = target.clone(); pred[:, 4] = np.sin(-.01)
    metrics = training.pose_metrics(pred, target)
    assert metrics['angle_mae_deg'] == pytest.approx(np.degrees(.02), rel=1e-5)
    rows = [(1, dict(position_mae=[1.]*4, motion_mae=[2.]*5, angle_mae_deg=1., angle_norm_mean=1.,
                     observations=3, post_warmup_observations=1, supervised_scalar_count=23)),
            (1, dict(position_mae=[3.]*4, motion_mae=[4.]*5, angle_mae_deg=3., angle_norm_mean=1.,
                     observations=5, post_warmup_observations=3, supervised_scalar_count=45))]
    got = training._mean_metrics(rows)
    assert got['position_mae'] == [2.25]*4
    assert got['motion_mae'] == [3.5]*5
    assert got['supervised_scalar_count'] == 68


def test_checkpoint_rejects_paddle_and_corrupted_weights(tmp_path):
    model = Encoder()
    path = tmp_path/'checkpoint.pt'
    value = dict(schema_version=checkpoints.SCHEMA_VERSION, tensor_schema=checkpoints.TENSOR_SCHEMA,
                 action_schema=checkpoints.ACTION_SCHEMA_VERSION, stage='perception', models={'E':model.state_dict()},
                 model_fingerprint=checkpoints.fingerprint_modules({'E':model}), dependencies={}, dataset_fingerprint='data')
    checkpoints.atomic_checkpoint(path, value)
    assert checkpoints.read_checkpoint(path, dataset_fingerprint='data')['stage'] == 'perception'
    wrong = copy.deepcopy(value); wrong['tensor_schema'] = 'paddle'
    checkpoints.atomic_checkpoint(path, wrong)
    with pytest.raises(ValueError, match='schema|ordering'): checkpoints.read_checkpoint(path)
    wrong = copy.deepcopy(value); next(iter(wrong['models']['E'].values())).add_(1)
    checkpoints.atomic_checkpoint(path, wrong)
    with pytest.raises(ValueError, match='fingerprint'): checkpoints.read_checkpoint(path)


def test_stage_resume_is_exact_and_completed_resume_is_idempotent(tmp_path, monkeypatch):
    class TinySamples:
        def __init__(self, data, split, **kwargs):
            from types import SimpleNamespace
            self.dataset = SimpleNamespace(fingerprint='tiny', split=split, manifest={'normalization':{'motion_scales':[1.]*5}})
            self.frames = [(0,t) for t in range(4)]; self.lengths=[4]
        def frame_batch(self, indices, device):
            x = torch.tensor(indices, device=device).float()[:,None,None,None].expand(-1,3,64,64)/4
            y = torch.zeros(len(indices), 6, device=device); y[:,5] = 1
            return x, y
    monkeypatch.setattr(training, 'Samples', TinySamples)
    config = dict(seed=3, device='cpu', cpu_threads=1, smoke=True, training=dict(
        learning_rate=3e-4, weight_decay=1e-4, grad_clip=1., validate_every=1, early_stopping=False,
        perception=dict(updates=3, batch_size=2, validation_examples=4)))
    full, interrupted = tmp_path/'full', tmp_path/'interrupted'
    training.train_perception(config, 'unused', full)
    original = training.atomic_checkpoint
    def stop_after_committed_step(path, value):
        original(path, value)
        if str(path).endswith('last.pt') and value['global_update'] == 1:
            raise KeyboardInterrupt('planned interruption after checkpoint commit')
    monkeypatch.setattr(training, 'atomic_checkpoint', stop_after_committed_step)
    with pytest.raises(KeyboardInterrupt): training.train_perception(config, 'unused', interrupted)
    monkeypatch.setattr(training, 'atomic_checkpoint', original)
    training.train_perception(config, 'unused', interrupted, resume=True)
    a,b = [checkpoints.read_checkpoint(p/'last.pt') for p in (full,interrupted)]
    assert a['global_update'] == b['global_update'] == 3
    for name in a['models']:
        for key in a['models'][name]: assert torch.equal(a['models'][name][key],b['models'][name][key])
    assert a['rng']['sampler'] == b['rng']['sampler']
    before = (interrupted/'training.jsonl').read_bytes()
    training.train_perception(config, 'unused', interrupted, resume=True)
    assert (interrupted/'training.jsonl').read_bytes() == before
