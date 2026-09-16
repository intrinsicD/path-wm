import torch


def test_pan_pairs_have_identical_current_and_multiset_but_opposite_motion():
    from pathwm.data.video_order import pan_pairs

    image = torch.zeros(1, 3, 64, 80)
    image[..., 32, 40] = 1
    data = pan_pairs(image, shifts=(2, 4))
    x, y = data["frames"], data["labels"]
    assert torch.equal(x[:, 0, -1], x[:, 1, -1])
    assert torch.equal(x[:, 0, :2], x[:, 1, :2].flip(1))
    assert torch.equal(x[:, 0].sum(1), x[:, 1].sum(1))
    assert torch.equal(y[:, 0], 1 - y[:, 1])
    columns = x.sum((-3, -2)).argmax(-1)
    expected = (columns[..., -1] > columns[..., -2]).long()
    assert torch.equal(y, expected)
    assert torch.equal(image[..., 32, 40], torch.ones(1, 3))


def test_separate_temporal_features_preserve_forward_and_causality():
    from pathwm.models.video_vae import CausalLatentMixer

    mixer = CausalLatentMixer(3)
    torch.nn.init.normal_(mixer.output.weight, std=0.2)
    x = torch.randn(2, 4, 3, 7, 9, requires_grad=True)
    before = x.detach().clone()
    times = torch.arange(4, dtype=torch.float64)[None].expand(2, -1)
    valid = torch.ones(2, 4, dtype=torch.bool)
    features = mixer.features(x, times, valid)
    torch.testing.assert_close(mixer(x, times, valid), x + features, rtol=0, atol=0)
    gradient = torch.autograd.grad(features[:, :2].sum(), x)[0]
    assert gradient[:, 2:].count_nonzero() == 0
    changed = x.detach().clone()
    changed[:, 2:] += 100
    torch.testing.assert_close(
        mixer.features(changed, times, valid)[:, :2], features[:, :2], rtol=0, atol=0
    )
    assert torch.equal(x.detach(), before)
    changed_now = x.detach().clone()
    changed_now[:, 1] += 5
    assert not torch.equal(
        mixer.features(changed_now, times, valid)[:, 1], features[:, 1]
    )


def test_local_correlation_recovers_shift_without_wraparound():
    from pathwm.models.video_vae import local_correlation

    torch.manual_seed(7501)
    previous = torch.randn(2, 8, 10, 12)
    current = torch.zeros_like(previous)
    current[..., 1:] = previous[..., :-1]
    scores = local_correlation(previous, current, radius=2)
    assert scores.shape == (2, 5, 6, 8)
    # Current content moved right: match at previous-coordinate offset-1.
    assert torch.equal(scores.mean((-1, -2)).argmax(1), torch.ones(2, dtype=torch.long))
    dirty = previous.clone()
    dirty[..., -1] = 1000
    # Common valid interior never reads the far-right column for offset-1.
    torch.testing.assert_close(
        local_correlation(dirty, current, radius=2)[:, 1], scores[:, 1]
    )


def test_readout_controls_and_frozen_feature_contracts():
    from experiments.video_order import OrderReadout, input_control, metrics

    torch.manual_seed(7510)
    a = torch.randn(3, 3, 4, 8, 10)
    x = torch.stack((a, a[:, [1, 0, 2]]), 1).flatten(0, 1)
    original = x.clone()
    labels = torch.tensor([1, 0]).expand(3, -1)
    for control in ("current", "unordered"):
        model = OrderReadout(4, "train")
        logits = model(input_control(x, control)).reshape(3, 2, 2)
        assert torch.equal(logits[:, 0], logits[:, 1])
        score = metrics(logits.detach(), labels)
        assert score["accuracy"] == 0.5 and score["pair_accuracy"] == 0
    for mode in ("frozen", "train", "correlation_only"):
        model = OrderReadout(4, mode)
        torch.nn.init.normal_(model.temporal.output.weight, std=0.1)
        model(x).square().mean().backward()
        assert all(p.grad is not None for p in model.head.parameters())
        assert all(
            (p.grad is None) == (mode != "train") for p in model.temporal.parameters()
        )
    assert torch.equal(x, original) and not x.requires_grad


def test_pair_swapping_and_raw_alignment_have_correct_direction():
    from experiments.video_order import pixel_oracle, OrderReadout, input_control
    from pathwm.data.video_order import pan_pairs

    torch.manual_seed(7511)
    data = pan_pairs(torch.rand(1, 3, 64, 80), shifts=(2, 4, 6, 8))
    predicted, ambiguous = pixel_oracle(data["frames"])
    assert torch.equal(predicted, data["labels"]) and not ambiguous.any()
    # Uniform images are genuinely ambiguous; retain rather than drop those rows.
    flat = pan_pairs(torch.zeros(1, 3, 64, 80))
    assert pixel_oracle(flat["frames"])[1].all()
    x = torch.randn(2, 3, 4, 8, 8)
    paired = torch.stack((x, x[:, [1, 0, 2]]), 1).flatten(0, 1)
    model = OrderReadout(4, "correlation")
    logits = model(paired).reshape(2, 2, 2)
    swapped = model(input_control(paired, "swap")).reshape(2, 2, 2)
    torch.testing.assert_close(swapped, logits.flip(1), rtol=0, atol=0)
