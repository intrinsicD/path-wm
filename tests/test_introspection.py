"""Model-internals diagnostics must be scale-aware, causal and side-effect free."""
import numpy as np
import pytest
import torch

from world_model.introspection import (attention_entropy, covariance_spectrum, encoder_maps, gaussianity,
                                       gradient_norms, linear_probe, parameter_norms, predictor_internals,
                                       rollout_horizon, scalar_summary, sensitivity, state_digest)
from world_model.model import build_model
from world_model.objective import SIGReg

SMALL = dict(width=12, image_size=28, encoder_depth=2, encoder_heads=3, predictor_depth=2, predictor_heads=2,
             head_dim=4, mlp_dim=24, projector_dim=24, dropout=0.1, action_dim=2, frameskip=5, history=3)


def small_model(seed=0):
    torch.manual_seed(seed)
    return build_model(SMALL).eval()


def test_effective_rank_separates_isotropic_from_collapsed_embeddings():
    g = torch.Generator().manual_seed(0)
    isotropic = covariance_spectrum(torch.randn(4000, 16, generator=g))
    collapsed = covariance_spectrum(torch.randn(4000, 1, generator=g) @ torch.randn(1, 16, generator=g))
    assert isotropic['dimension'] == 16 and len(isotropic['eigenvalues']) == 16
    assert isotropic['eigenvalues'] == sorted(isotropic['eigenvalues'], reverse=True)
    assert 15.0 < isotropic['effective_rank'] <= 16.0 and 15.0 < isotropic['rankme'] <= 16.0
    assert collapsed['effective_rank'] < 1.05 and collapsed['participation_ratio'] < 1.05
    assert collapsed['top_eigenvalue_fraction'] > 0.99 and isotropic['top_eigenvalue_fraction'] < 0.1


def test_attention_entropy_is_normalized_between_one_hot_and_uniform():
    uniform = torch.full((1, 2, 4, 4), 0.25)
    one_hot = torch.eye(4).expand(1, 2, 4, 4)
    assert torch.allclose(attention_entropy(uniform), torch.ones(2))
    assert torch.allclose(attention_entropy(one_hot), torch.zeros(2))


def test_gaussianity_flags_non_gaussian_dimensions():
    rng = np.random.default_rng(0)
    normal = gaussianity(rng.normal(size=(600, 6)))
    skewed = gaussianity(rng.exponential(size=(600, 6)))
    assert normal['shapiro_fraction_below_0_95'] == 0.0 and skewed['shapiro_fraction_below_0_95'] == 1.0
    assert abs(normal['excess_kurtosis_mean']) < 0.5 and skewed['excess_kurtosis_mean'] > 3
    assert len(normal['qq']['theoretical']) == len(normal['qq']['sample']) > 10


def test_linear_probe_recovers_an_exact_linear_readout():
    rng = np.random.default_rng(1)
    z, w, b = rng.normal(size=(400, 8)), rng.normal(size=(8, 3)), rng.normal(size=3)
    result = linear_probe(z[:300], z[:300] @ w + b, z[300:], z[300:] @ w + b, targets=['a', 'b', 'c'])
    assert min(result['r2'].values()) > 0.999 and set(result['r2']) == {'a', 'b', 'c'}
    noise = linear_probe(z[:300], rng.normal(size=(300, 1)), z[300:], rng.normal(size=(100, 1)), targets=['n'])
    assert noise['r2']['n'] < 0.2


def test_predictor_gates_start_at_zero_and_attention_is_causal():
    model = small_model()
    states, actions = torch.randn(4, 3, 12), torch.randn(4, 3, 10)
    result = predictor_internals(model, states, actions)
    assert torch.all(result['gate_msa'] == 0) and torch.all(result['gate_mlp'] == 0)  # AdaLN-zero init
    attention = result['attention']
    assert attention.shape == (2, 2, 3, 3)
    assert torch.allclose(attention.sum(-1), torch.ones(2, 2, 3), atol=1e-5)
    assert torch.all(attention.triu(1) == 0)
    assert result['attention_entropy'].shape == (2, 2) and result['action_embedding_norm'] > 0


def test_sensitivity_vanishes_when_actions_cannot_reach_the_predictor():
    model = small_model()
    states, actions = torch.randn(4, 3, 12), torch.randn(4, 3, 10)
    with torch.no_grad():
        model.action_encoder.embed[-1].weight.zero_()
        model.action_encoder.embed[-1].bias.zero_()
    result = sensitivity(model, states, actions, directions=4, seed=0)
    assert result['action'] == pytest.approx(0.0, abs=1e-6) and result['state'] > 0
    assert result['action_over_state'] == pytest.approx(0.0, abs=1e-6)


def test_gradient_norms_leave_weights_buffers_and_mode_unchanged():
    model = small_model()
    before = state_digest(model)
    pixels, actions = torch.randn(4, 4, 3, 28, 28), torch.randn(4, 4, 10)
    result = gradient_norms(model, pixels, actions, SIGReg(knots=5, num_proj=8), weight=0.09)
    assert set(result) >= {'encoder', 'projector', 'predictor', 'action_encoder', 'pred_proj', 'total'}
    assert result['total'] > 0 and all(v >= 0 for v in result.values())
    assert state_digest(model) == before and not model.training
    assert all(p.grad is None for p in model.parameters())
    names = parameter_norms(model)
    assert set(names) >= {'encoder', 'projector', 'predictor', 'action_encoder', 'pred_proj'}


def test_rollout_horizon_reports_matching_copy_baseline():
    model = small_model()
    pixels, actions = torch.randn(2, 5, 3, 28, 28), torch.randn(2, 5, 10)
    result = rollout_horizon(model, pixels, actions)
    assert result['horizon'] == [1, 2, 3, 4]
    with torch.no_grad():
        z = model.encode(pixels)
    expected = [(z[:, 0] - z[:, h]).square().mean().item() for h in range(1, 5)]
    assert result['copy_mse'] == pytest.approx(expected, rel=1e-4)
    assert len(result['prediction_mse']) == 4 and all(v > 0 for v in result['prediction_mse'])


def test_encoder_maps_align_with_patch_grid_and_restore_attention_mode():
    model = small_model()
    implementation = model.encoder.config._attn_implementation
    before = state_digest(model)
    result = encoder_maps(model, torch.randn(3, 3, 28, 28))
    assert result['cls_attention'].shape == (3, 3, 2, 2)
    assert torch.allclose(result['cls_attention'].flatten(2).sum(-1), torch.ones(3, 3), atol=1e-5)
    assert result['patch_pca'].shape == (3, 2, 2, 3)
    assert result['patch_pca'].min() >= 0 and result['patch_pca'].max() <= 1
    assert result['attention_entropy'].shape == (2,)
    assert model.encoder.config._attn_implementation == implementation and state_digest(model) == before


def test_state_digest_tracks_batchnorm_buffers():
    model = small_model()
    before = state_digest(model)
    with torch.no_grad():
        model.projector.net[1].running_mean += 1.0
    assert state_digest(model) != before


def test_scalar_summary_is_finite_and_side_effect_free():
    model = small_model()
    before = state_digest(model)
    summary = scalar_summary(model, torch.randn(6, 4, 3, 28, 28), torch.randn(6, 4, 10), directions=2)
    assert {'effective_rank', 'gate_msa_mean', 'sensitivity_action_over_state', 'param_norm_encoder', 'examples'} <= set(summary)
    assert all(np.isfinite(v) for v in summary.values()) and summary['examples'] == 6
    assert state_digest(model) == before and not model.training
