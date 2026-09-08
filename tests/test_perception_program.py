"""Checks for scientific invariants in the overnight frozen-package comparison."""
import numpy as np
import pytest
import torch

from world_model.curriculum.perception_heads import make_heads, spatial_expectation
from world_model.curriculum.perception_cache import validate_alignment
from world_model.curriculum.perception_training import checkpoint_state, restore_checkpoint, selected_better


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


def test_selection_does_not_use_rgb_to_break_exact_pose_ties():
    first = {'q': 1.2, 'image_mse': .1, 'step': 100}
    assert not selected_better({'q': 1.2, 'image_mse': .01, 'step': 200}, first)
    assert selected_better({'q': 1.1, 'image_mse': .2, 'step': 200}, first)


def test_checkpoint_resumes_sampler_and_optimizer_exactly(tmp_path):
    def construct():
        torch.manual_seed(83)
        heads = {'pose': torch.nn.Linear(3, 2)}
        opt = {'pose': torch.optim.AdamW(heads['pose'].parameters(), lr=.01)}
        return heads, opt, np.random.default_rng(12)
    def step(heads, opt, rng):
        x = torch.tensor(rng.normal(size=(4, 3)), dtype=torch.float32)
        opt['pose'].zero_grad()
        (heads['pose'](x) - torch.randn(4, 2)).square().mean().backward()
        opt['pose'].step()
    heads, opt, rng = construct()
    for _ in range(2): step(heads, opt, rng)
    path = tmp_path / 'checkpoint.pt'
    torch.save(checkpoint_state(heads, opt, rng), path)
    for _ in range(3): step(heads, opt, rng)
    resumed, resumed_opt, resumed_rng = construct()
    restore_checkpoint(torch.load(path, weights_only=False), resumed, resumed_opt, resumed_rng)
    for _ in range(3): step(resumed, resumed_opt, resumed_rng)
    for key, value in heads['pose'].state_dict().items():
        assert torch.equal(value, resumed['pose'].state_dict()[key])
