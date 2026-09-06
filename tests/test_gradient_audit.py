"""Essential scientific invariants for disposable training-gradient probes."""
import pytest
import torch

from scripts.gradient_audit_math import gradient_geometry, module_geometry, adam_delta, preserved_state


def test_gradient_noise_signed_signal_and_zero_vectors():
    r = gradient_geometry([torch.tensor([1., 1.]), torch.tensor([1., -1.])])
    assert r['mean_norm_squared'] == pytest.approx(1)
    assert r['sample_covariance_trace'] == pytest.approx(2)
    assert r['signal_estimate'] == pytest.approx(0)
    assert r['noise_to_signal'] is None
    r = gradient_geometry([torch.ones(2), torch.ones(2)])
    assert r['pairwise_cosine_mean'] == pytest.approx(1)
    assert r['noise_to_signal'] == pytest.approx(0)
    r = gradient_geometry([torch.zeros(2), torch.zeros(2)])
    assert r['pairwise_cosine_mean'] is None


def test_gradient_decomposition_preserves_cancellation():
    r = module_geometry(torch.tensor([3., 0.]), torch.tensor([-2., 0.]))
    assert r['prediction_norm'] == pytest.approx(3)
    assert r['weighted_sigreg_norm'] == pytest.approx(2)
    assert r['cosine'] == pytest.approx(-1)
    assert r['total_norm'] == pytest.approx(1)


def test_analytical_adam_matches_real_discarded_optimizer_step():
    p = torch.nn.Parameter(torch.tensor([.5, -.7], dtype=torch.float64))
    opt = torch.optim.AdamW([p], lr=.003, weight_decay=.02)
    p.grad = torch.tensor([.2, -.1], dtype=p.dtype)
    opt.step()
    g = torch.tensor([.8, .3], dtype=p.dtype)
    expected = adam_delta(p.detach(), g, opt.state[p], opt.param_groups[0])
    before = p.detach().clone()
    p.grad = g
    opt.step()
    torch.testing.assert_close(p.detach()-before, expected, atol=1e-14, rtol=1e-10)


def test_probe_restores_batchnorm_rng_modes_and_parameters_on_error():
    model = torch.nn.Sequential(torch.nn.BatchNorm1d(2), torch.nn.Dropout(.2), torch.nn.Linear(2, 2))
    model.eval()
    before = {k:v.clone() for k,v in model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    with pytest.raises(RuntimeError, match='probe failure'):
        with preserved_state(model):
            model.train()
            model(torch.randn(8, 2)).sum().backward()
            raise RuntimeError('probe failure')
    assert not model.training
    assert all(torch.equal(v, model.state_dict()[k]) for k,v in before.items())
    assert torch.equal(rng, torch.get_rng_state())
    assert all(p.grad is None for p in model.parameters())
