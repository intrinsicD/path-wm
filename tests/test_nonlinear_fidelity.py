"""Essential numeric/oracle-isolation checks for the frozen output diagnostic."""
import pytest
import torch
from experiments.nonlinear_fidelity import project_range, fit_threshold, apply_threshold, infer


def test_projection_matches_lstsq_and_is_not_encoder_reconstruction():
    g = torch.Generator().manual_seed(27)
    w = torch.randn(9, 3, generator=g, dtype=torch.float64)
    b = torch.randn(9, generator=g, dtype=torch.float64)
    target = torch.randn(7, 9, generator=g, dtype=torch.float64)
    p = project_range(w, b, target)
    z = torch.linalg.lstsq(w, (target-b).T).solution.T
    assert torch.allclose(p['prediction'], z @ w.T+b, atol=1e-11)
    assert ((target-p['prediction']) @ w).abs().max() < 1e-11
    assert torch.allclose(p['codes'] @ w.T+b, p['prediction'], atol=1e-11)
    # Rank deficiency and a zero map must remain well-defined.
    for a in (torch.cat([w, w[:, :1]], 1), torch.zeros_like(w)):
        r = project_range(a, b, target)
        assert torch.isfinite(r['prediction']).all()
        assert ((target-r['prediction']) @ a).abs().max() < 1e-10
    assert torch.equal(project_range(torch.zeros_like(w), b, target)['prediction'], b.expand_as(target))
    with pytest.raises(ValueError): project_range(w, b, target*float('nan'))


def test_support_threshold_bounds_and_rejects_unidentified_or_inconsistent_data():
    x = torch.tensor([[.1,.2,.3],[.4,.5,.6],[.7,.8,.9]])
    y = (x > torch.tensor([.3,.6,.4])).float()
    state = fit_threshold(x, y)
    assert torch.equal(apply_threshold(x, state), y)
    assert torch.equal(fit_threshold(x.flip(0), y.flip(0)), state)
    for labels in (torch.ones_like(y), 1-y, y+.1):
        with pytest.raises(ValueError): fit_threshold(x, labels)
    with pytest.raises(ValueError): fit_threshold(x*float('nan'), y)


def test_inference_has_no_target_input_and_preserves_codec():
    from pathwm.models.detail_memory import DetailCodec
    from pathwm.data.detail_views import sample_tiles
    from pathwm.io import state_hash
    model = DetailCodec(hidden=32, variant='linear').eval().requires_grad_(False)
    g = torch.Generator().manual_seed(18)
    support, query = sample_tiles(g, 4), sample_tiles(g, 3)
    labels = (support > .4).float()
    before = state_hash(model)
    a = infer(model, support, labels, query)
    poisoned_target = torch.full_like(query, float('nan'))
    b = infer(model, support, labels, query)
    assert torch.isnan(poisoned_target).all()
    assert a.keys() == b.keys()
    for name in a['predictions']:
        assert torch.equal(a['predictions'][name], b['predictions'][name])
    assert state_hash(model) == before
    assert all(p.grad is None for p in model.parameters())
