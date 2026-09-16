"""Shared weights, causal video posterior and image-path retention contracts."""

import copy
import torch
from pathwm.models.modalities import Observation
from pathwm.models.spatial_vae_v2 import HierarchicalVAE
from pathwm.models.video_vae import VideoVAE


def test_configurable_mixer_spatial_reach_preserves_temporal_horizon():
    from pathwm.models.video_vae import CausalLatentMixer

    torch.manual_seed(97)
    for kernel, loops, radius in [(3, 0, 1), (5, 0, 2), (3, 2, 3)]:
        mixer = CausalLatentMixer(2, spatial_kernel=kernel, spatial_iterations=loops)
        torch.nn.init.normal_(mixer.output.weight, std=0.2)
        x = torch.randn(1, 5, 2, 11, 11, requires_grad=True)
        times = torch.arange(5, dtype=torch.float64)[None]
        valid = torch.ones(1, 5, dtype=torch.bool)
        out = mixer(x, times, valid)
        grad = torch.autograd.grad(out[0, 3, :, 5, 5].sum(), x)[0]
        assert torch.count_nonzero(grad[:, 0]) == 0
        assert torch.count_nonzero(grad[:, 4]) == 0
        support = grad[:, 1:4].abs().sum((0, 1, 2))
        assert support[5, 5 + radius] > 0
        assert support[5, 6 + radius] == 0
        invalid = valid.clone()
        invalid[:, 2] = False
        dirty = x.detach().clone()
        dirty[:, 2] = float('nan')
        torch.testing.assert_close(
            mixer(dirty, times, invalid), mixer(x.detach(), times, invalid),
            rtol=0, atol=0,
        )


def test_injected_shared_spatial_loop_keeps_image_identity_and_checkpoint():
    from pathwm.models.video_vae import CausalLatentMixer

    first = CausalLatentMixer(2, spatial_iterations=1)
    second = CausalLatentMixer(2, spatial_iterations=2)
    assert sum(p.numel() for p in first.parameters()) == sum(
        p.numel() for p in second.parameters()
    )
    second.load_state_dict(first.state_dict(), strict=True)
    image = codec()
    model = VideoVAE(image, temporal=second)
    assert model.temporal is second
    x = observation(torch.rand(1, 3, 3, 11, 9))
    y, _ = model(x, sample=False)
    baseline, _ = image(x.values.flatten(0, 1), sample=False)
    torch.testing.assert_close(y.flatten(0, 1), baseline, rtol=0, atol=0)
    torch.nn.init.normal_(first.output.weight, std=0.2)
    second.load_state_dict(first.state_dict())
    values = torch.rand(1, 3, 2, 5, 7)
    mask = torch.ones(1, 3, dtype=torch.bool)
    assert not torch.equal(first(values, x.times, mask), second(values, x.times, mask))


def codec():
    return HierarchicalVAE(stem_channels=4, channels=(8,), latent_channels=2)


def observation(x, valid=None):
    return Observation(
        x,
        torch.arange(x.shape[1], dtype=torch.float64)[None].expand(len(x), -1) / 4,
        valid,
    )


def test_actual_sharing_initial_equivalence_and_unique_checkpoint():
    torch.manual_seed(81)
    image = codec()
    model = VideoVAE(image, temporal=True)
    assert model.image is image
    x = torch.rand(2, 3, 3, 9, 13)
    baseline, p = image(x.flatten(0, 1), sample=False)
    actual, vp = model(observation(x), sample=False)
    torch.testing.assert_close(actual.flatten(0, 1), baseline, rtol=0, atol=0)
    torch.testing.assert_close(vp.mu, p.mu, rtol=0, atol=0)
    saved = model.state_dict()
    assert sum(k.startswith("image.encoder.") for k in saved) == len(
        image.encoder.state_dict()
    )
    assert len({p.data_ptr() for p in saved.values()}) == len(saved)
    restored = VideoVAE(codec(), temporal=True)
    restored.load_state_dict(saved, strict=True)
    torch.testing.assert_close(
        restored(observation(x), sample=False)[0], actual, rtol=0, atol=0
    )
    disabled = VideoVAE(image, temporal=False)
    assert not list(disabled.temporal.parameters())
    torch.testing.assert_close(
        disabled(observation(x), sample=False)[0], actual, rtol=0, atol=0
    )


def test_active_temporal_path_is_causal_including_equal_times():
    torch.manual_seed(82)
    model = VideoVAE(codec(), temporal=True)
    torch.nn.init.normal_(model.temporal.output.weight, std=0.2)
    x = torch.rand(1, 4, 3, 10, 12)
    times = torch.zeros(1, 4, dtype=torch.float64)
    before = model(Observation(x, times), sample=False)[0]
    future = x.clone()
    future[:, 2:] += 4
    after = model(Observation(future, times), sample=False)[0]
    torch.testing.assert_close(before[:, :2], after[:, :2], rtol=0, atol=0)
    assert not torch.equal(before[:, 2:], after[:, 2:])
    past = x.clone()
    past[:, 0] += 4
    changed = model(Observation(past, times), sample=False)[0]
    assert not torch.equal(before[:, 1], changed[:, 1])
    prefix = model(Observation(x[:, :2], times[:, :2]), sample=False)[0]
    torch.testing.assert_close(prefix, before[:, :2], rtol=1e-5, atol=1e-6)


def test_invalid_nan_frames_are_masked_before_encoding_and_have_zero_gradient():
    torch.manual_seed(83)
    model = VideoVAE(codec(), temporal=True)
    torch.nn.init.normal_(model.temporal.output.weight, std=0.2)
    valid = torch.tensor([[True, False, True]])
    x = torch.rand(1, 3, 3, 11, 9)
    x[:, 1] = float("nan")
    x.requires_grad_()
    trace = {}
    y, p = model(observation(x, valid), sample=False, trace=trace)
    assert torch.isfinite(y).all() and torch.isfinite(p.mu).all()
    assert torch.count_nonzero(y[:, 1]) == 0
    clean = x.detach().clone()
    clean[:, 1] = 300
    torch.testing.assert_close(
        y, model(observation(clean, valid), sample=False)[0], rtol=0, atol=0
    )
    y.square().sum().backward()
    assert torch.isfinite(x.grad).all() and torch.count_nonzero(x.grad[:, 1]) == 0
    assert all(not t.requires_grad for t in trace.values())


def test_image_and_video_losses_update_same_encoder_and_keep_resolution_flexible():
    torch.manual_seed(84)
    model = VideoVAE(codec(), temporal=True)
    for h, w in [(9, 11), (12, 8)]:
        x = torch.rand(2, 3, 3, h, w)
        image, _ = model.image(x[:, 0], sample=False)
        y, p = model(observation(x), sample=True)
        assert y.shape == x.shape and p.mu.shape[:2] == (6, 2)
        anchor = model.image.encoder.stem[0].weight
        gi = torch.autograd.grad(image.square().mean(), anchor, retain_graph=True)[0]
        gv = torch.autograd.grad(y.square().mean(), anchor, retain_graph=True)[0]
        assert gi.abs().sum() > 0 and gv.abs().sum() > 0
        (
            y.square().mean() + image.square().mean() + 0.001 * p.kl_per_image().mean()
        ).backward()
        assert model.temporal.output.weight.grad.abs().sum() > 0
        assert all(
            p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()
        )
        model.zero_grad(set_to_none=True)


def test_trace_does_not_change_sampling_or_gradients():
    torch.manual_seed(85)
    model = VideoVAE(codec(), temporal=True)
    other = copy.deepcopy(model)
    x = observation(torch.rand(1, 3, 3, 9, 11))
    torch.manual_seed(86)
    a = model(x)[0]
    state_a = torch.get_rng_state()
    torch.manual_seed(86)
    b = other(x, trace={})[0]
    state_b = torch.get_rng_state()
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert torch.equal(state_a, state_b)
    a.sum().backward()
    b.sum().backward()
    for p, q in zip(model.parameters(), other.parameters()):
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)


def test_future_time_validity_and_gradients_do_not_leak():
    torch.manual_seed(91)
    model = VideoVAE(codec(), temporal=True)
    torch.nn.init.normal_(model.temporal.output.weight, std=0.2)
    x = torch.rand(1, 4, 3, 8, 10, requires_grad=True)
    original = observation(x)
    y, _ = model(original, sample=False)
    gradient = torch.autograd.grad(y[:, :2].sum(), x)[0]
    assert torch.count_nonzero(gradient[:, 2:]) == 0
    times = original.times.clone()
    times[:, 2:] += 100
    changed = model(
        Observation(x, times, torch.tensor([[True, True, False, True]])), sample=False
    )[0]
    torch.testing.assert_close(y[:, :2], changed[:, :2], rtol=0, atol=0)
    # Origin of the clock carries no content; only elapsed time is used.
    shifted = model(Observation(x, original.times + 1000), sample=False)[0]
    torch.testing.assert_close(y, shifted, rtol=0, atol=0)


def test_legacy_spatial_vae_and_invalid_inputs():
    import pytest
    from pathwm.models.spatial_vae import SpatialVAE

    model = VideoVAE(SpatialVAE(channels=(8,), latent_channels=2), temporal=True)
    x = torch.rand(1, 2, 3, 7, 11)
    assert model(observation(x), sample=False)[0].shape == x.shape
    with pytest.raises(ValueError, match="nondecreasing"):
        model(Observation(x, torch.tensor([[1.0, 0.0]])))
    with pytest.raises(ValueError, match="finite"):
        model(Observation(x * float("nan"), torch.tensor([[0.0, 1.0]])))
    valid = torch.zeros(1, 2, dtype=torch.bool)
    y, p = model(
        Observation(x * float("nan"), torch.full((1, 2), float("nan")), valid),
        sample=False,
    )
    assert not torch.count_nonzero(y) and not torch.count_nonzero(p.kl_per_image())


def test_single_frame_and_repeated_still_have_different_temporal_gradient_support():
    torch.manual_seed(92)
    model = VideoVAE(codec(), temporal=True)
    # Activate the residual to inspect its input kernel, beyond neutral startup.
    torch.nn.init.normal_(model.temporal.output.weight, std=0.2)
    still = torch.rand(2, 1, 3, 9, 11)
    for frames in [1, 4]:
        x = still.expand(-1, frames, -1, -1, -1).clone()
        y, p = model(observation(x), sample=False)
        assert y.shape == x.shape
        p.mu.square().mean().backward()
        grad = model.temporal.input.weight.grad
        # One-frame images train current-frame processing, not past-frame taps.
        assert grad[:, :, 2].abs().sum() > 0
        if frames == 1:
            assert torch.count_nonzero(grad[:, :, :2]) == 0
        else:
            assert grad[:, :, :2].abs().sum() > 0
        model.zero_grad(set_to_none=True)
