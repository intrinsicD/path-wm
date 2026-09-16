import pytest
import torch
from torch.nn import functional as F


def test_wide_correlation_matches_reference_sign_bounds_and_gradients():
    from pathwm.models.video_vae import local_correlation

    torch.manual_seed(101)
    previous = torch.randn(2, 8, 12, 14, requires_grad=True)
    current = torch.zeros_like(previous).detach()
    current[..., 3:] = previous.detach()[..., :-3]
    current.requires_grad_()
    scores = local_correlation(previous, current, radius=3)
    p = F.normalize(previous, dim=1, eps=1e-6)
    c = F.normalize(current, dim=1, eps=1e-6)
    reference = torch.stack(
        [
            torch.stack(
                [
                    (p[:, :, y, 3 + s : 11 + s] * c[:, :, y, 3:11]).sum(1)
                    for y in range(3, 9)
                ],
                1,
            )
            for s in range(-3, 4)
        ],
        1,
    )
    torch.testing.assert_close(scores, reference)
    assert scores.shape == (2, 7, 6, 8)
    assert (scores.mean((-1, -2)).argmax(1) == 0).all()  # -3 offset: rightward
    g = torch.autograd.grad(scores.square().mean(), (previous, current))
    assert all(torch.isfinite(v).all() and v.abs().sum() > 0 for v in g)
    dirty = previous.detach().clone()
    dirty[..., -1] = 1000
    torch.testing.assert_close(
        local_correlation(dirty, current, radius=3)[:, 0], scores[:, 0]
    )


def test_readout_matching_arms_share_shape_initialization_and_common_interior():
    from experiments.video_order import OrderReadout
    from pathwm.models.video_vae import local_correlation

    x = torch.randn(2, 3, 4, 12, 12)
    original = x.clone()
    models = []
    inputs = []
    for mode, active in [("train", 3), ("correlation", 2), ("correlation", 3)]:
        torch.manual_seed(103)
        model = OrderReadout(4, mode, correlation_radius=3, active_radius=active)
        model.head[0].register_forward_pre_hook(
            lambda _, a: inputs.append(a[0].detach().clone())
        )
        model(x).square().mean().backward()
        models.append(model)
    for m in models[1:]:
        assert all(
            torch.equal(v, m.state_dict()[k]) for k, v in models[0].state_dict().items()
        )
    assert inputs[0].shape == (2, 15, 6, 6)
    assert all(torch.equal(v[:, :8], inputs[0][:, :8]) for v in inputs)
    full = local_correlation(x[:, -2], x[:, -1], radius=3)
    assert inputs[0][:, 8:].count_nonzero() == 0
    assert inputs[1][:, [8, 14]].count_nonzero() == 0
    torch.testing.assert_close(inputs[1][:, 9:14], full[:, 1:6])
    torch.testing.assert_close(inputs[2][:, 8:], full)
    assert models[0].head[0].weight.grad[:, 8:].count_nonzero() == 0
    assert models[1].head[0].weight.grad[:, [8, 14]].count_nonzero() == 0
    assert models[2].head[0].weight.grad[:, [8, 14]].abs().sum() > 0
    assert torch.equal(x, original)


def test_wide_readout_single_clip_batch_independence():
    from experiments.video_order import OrderReadout

    torch.manual_seed(107)
    model = OrderReadout(4, "correlation", correlation_radius=3, active_radius=3).eval()
    x = torch.randn(3, 3, 4, 12, 12)
    expected = model(x[:1])
    x[1:] *= 100
    torch.testing.assert_close(model(x)[:1], expected, atol=1e-6, rtol=1e-5)
    assert model(x).shape == (3, 2)


def test_matching_radius_validation_and_legacy_defaults():
    from experiments.video_order import OrderReadout

    for radius, active in [
        (0, None),
        (True, None),
        (2.5, None),
        (2, 3),
        (3, 0),
        (3, True),
    ]:
        with pytest.raises(ValueError):
            OrderReadout(4, "train", correlation_radius=radius, active_radius=active)
    torch.manual_seed(109)
    a = OrderReadout(4, "correlation")
    torch.manual_seed(109)
    b = OrderReadout(4, "correlation", correlation_radius=2, active_radius=2)
    x = torch.randn(2, 3, 4, 10, 12)
    assert torch.equal(a(x), b(x))
