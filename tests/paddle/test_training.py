"""Scientific checks for statistics, recurrent masks, and checkpoint identity."""
import numpy as np
import pytest
import torch

from world_model.paddle.training import CoordinateVariance, supervised_memory_loss
from world_model.paddle.checkpoints import fingerprint_modules, atomic_checkpoint, read_checkpoint, TENSOR_SCHEMA


def test_variance_excludes_static_spatial_offsets_and_is_streaming():
    offsets = torch.arange(12, dtype=torch.float64).reshape(1, 3, 4) * 100
    frames = offsets + torch.tensor([0., 2., 4.]).reshape(3, 1, 1)
    stats = CoordinateVariance()
    stats.update(frames[:1]); stats.update(frames[1:])
    assert stats.count == 3
    assert stats.variance() == pytest.approx(8 / 3)


def test_memory_loss_masks_unknown_velocity_and_padding():
    estimate = torch.zeros(1, 4, 5, requires_grad=True)
    target = torch.ones_like(estimate)
    valid = torch.tensor([[True, True, True, False]])
    loss = supervised_memory_loss(estimate, target, valid)
    loss.backward()
    assert loss.item() == 1
    assert estimate.grad[0, :2, 2:4].abs().sum() == 0
    assert estimate.grad[0, 3].abs().sum() == 0
    assert estimate.grad[0, 2, 2:4].abs().sum() > 0


def test_checkpoint_rejects_dependency_change(tmp_path):
    model = torch.nn.Linear(3, 2)
    identity = fingerprint_modules({'E': model})
    checkpoint = {'stage': 'memory', 'schema_version': 1, 'tensor_schema': TENSOR_SCHEMA,
                  'dependencies': {'perception': identity},
                  'dataset_fingerprint': 'dataset-a'}
    path = tmp_path / 'last.pt'
    atomic_checkpoint(path, checkpoint)
    assert read_checkpoint(path, dependencies={'perception': identity})['stage'] == 'memory'
    with torch.no_grad():
        model.weight[0, 0] += 1
    with pytest.raises(ValueError, match='dependenc'):
        read_checkpoint(path, dependencies={'perception': fingerprint_modules({'E': model})})
    with pytest.raises(ValueError, match='dataset'):
        read_checkpoint(path, dataset_fingerprint='dataset-b')
