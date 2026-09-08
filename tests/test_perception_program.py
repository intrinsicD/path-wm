"""Checks for scientific invariants in the overnight frozen-package comparison."""
import numpy as np
import pytest
import torch

from world_model.curriculum.perception_heads import make_heads, spatial_expectation
from world_model.curriculum.perception_cache import validate_alignment


def test_heads_are_independent_and_common_initialization_is_paired():
    cnn = make_heads(64, 9107)
    vit = make_heads(384, 9107)
    for name in cnn:
        c, v = cnn[name].state_dict(), vit[name].state_dict()
        for key in c:
            if not key.startswith('input.'):
                assert torch.equal(c[key], v[key]), key
    fine = torch.randn(2, 256, 64)
    coarse = torch.randn(2, 64, 64)
    cnn['pose'](fine, coarse).square().mean().backward()
    assert any(p.grad is not None for p in cnn['pose'].parameters())
    assert all(p.grad is None for name in ('rgb', 'mask') for p in cnn[name].parameters())
    assert set(map(id, cnn['pose'].parameters())).isdisjoint(map(id, cnn['rgb'].parameters()))


def test_spatial_coordinates_have_xy_order_and_full_boundary_support():
    logits = torch.full((1, 2, 3, 3), -100.)
    logits[0, 0, 0, 2] = 100.
    logits[0, 1, 2, 0] = 100.
    assert torch.allclose(spatial_expectation(logits), torch.tensor([[1., 0., 0., 1.]]))
    assert torch.allclose(spatial_expectation(torch.zeros(1, 2, 3, 3)), torch.full((1, 4), .5))


def test_cache_alignment_rejects_reordered_rows_and_corrupted_targets():
    rows = np.array([13, 6, 82])
    target = np.arange(18, dtype='float32').reshape(3, 6)
    validate_alignment(rows, target, rows.copy(), target.copy())
    with pytest.raises(ValueError, match='row'):
        validate_alignment(rows, target, rows[::-1], target)
    bad = target.copy(); bad[1, 4] += 1
    with pytest.raises(ValueError, match='target'):
        validate_alignment(rows, target, rows, bad)
