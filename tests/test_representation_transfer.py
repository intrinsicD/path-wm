"""Numeric and evidence integrity checks, not tests of a research claim."""
import pytest
import torch
from experiments.representation_transfer import fit_affine, apply_affine, transform, memory_check


def test_affine_matches_independent_augmented_normal_equations():
    g = torch.Generator().manual_seed(73)
    x = torch.randn(30, 4, generator=g, dtype=torch.float64)
    y = x @ torch.randn(4, 3, generator=g, dtype=torch.float64) + 2
    code = fit_affine(x, y, ridge=.001)
    design = torch.cat([x, torch.ones(30, 1)], 1)
    penalty = torch.diag(torch.tensor([.001] * 4 + [0.], dtype=torch.float64))
    expected = torch.linalg.solve(design.T @ design + penalty, design.T @ y)
    assert torch.allclose(code, expected, atol=1e-11, rtol=1e-10)
    assert (apply_affine(x, code) - y).square().mean() < 1e-6
    assert torch.allclose(fit_affine(x.flip(0), y.flip(0)), code, atol=1e-11)
    assert not torch.allclose(fit_affine(x, y.flip(0)), code)


def test_rank_deficient_and_invalid_inputs():
    x = torch.ones(5, 3)
    y = torch.arange(10.).reshape(5, 2)
    assert torch.allclose(apply_affine(x, fit_affine(x, y)), y.mean(0).expand_as(y).double())
    for a, b, penalty in [(x[:0], y[:0], .001), (x, y[:2], .001),
                          (x * float('nan'), y, .001), (x, y, 0), (x, y, float('nan'))]:
        with pytest.raises(ValueError):
            fit_affine(a, b, ridge=penalty)
    with pytest.raises(ValueError):
        apply_affine(torch.ones(2, 7), fit_affine(x, y))


def test_operator_pixels_are_independent_of_fitting_and_no_clipping():
    x = torch.rand(7, 3, 16, 16)
    assert torch.equal(transform(x, 'reverse'), x[:, [2, 1, 0]])
    assert torch.equal(transform(x, 'mix_reverse'), .7 * x + .3 * x[:, [2, 1, 0]])
    assert torch.equal(transform(x, 'threshold'), (x > .4).float())
    with pytest.raises(ValueError):
        transform(x, 'unknown')


def test_source_correction_and_context_restart(tmp_path):
    g = torch.Generator().manual_seed(17)
    x = torch.randn(12, 3, generator=g)
    y = x * 2 + 1
    checks = memory_check(x, y, x[:4], 'test-codec', tmp_path)
    assert all(checks.values()), checks
