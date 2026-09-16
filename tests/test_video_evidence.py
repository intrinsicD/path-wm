import pytest
import torch
from torch.nn import functional as F


def test_evidence_head_initial_cosine_reference_and_shared_reversal():
    from pathwm.models.video_vae import CorrespondenceDirectionHead

    torch.manual_seed(201)
    model = CorrespondenceDirectionHead(radius=3)
    x = torch.rand(5, 7, 4, 6)
    means = x.mean((-1, -2))
    d = means[:, :3].amax(1) - means[:, 4:].amax(1)
    expected = torch.stack((-d / 2, d / 2), -1)
    torch.testing.assert_close(model(x), expected, rtol=0, atol=0)
    assert sum(p.numel() for p in model.parameters()) == 81
    with torch.no_grad():
        model.residual[-1].weight.normal_(0, 0.2)
        model.log_scale.fill_(0.7)
    torch.testing.assert_close(model(x.flip(1)), model(x).flip(1), rtol=0, atol=1e-7)
    tied = x.clone()
    tied[:, 4:] = tied[:, :3].flip(1)
    assert model(tied).count_nonzero() == 0
    assert model(x).sum(1).count_nonzero() == 0


def test_evidence_residual_can_change_decisions_and_learns_both_layers():
    from pathwm.models.video_vae import CorrespondenceDirectionHead

    m = CorrespondenceDirectionHead(radius=3)
    # Negative best at distance1; positive best at distance3. An absolute-distance
    # preference can change the decision without a side-specific parameter.
    x = torch.tensor([0.0, 0.1, 0.9, 1.0, 0.1, 0.2, 0.8])[None, :, None, None]
    assert m(x).argmax(1).item() == 1
    with torch.no_grad():
        for p in m.residual.parameters():
            p.zero_()
        m.residual[0].weight[0, 1] = 1
        m.residual[-1].weight[0, 0] = 2
    assert m(x).argmax(1).item() == 0
    torch.manual_seed(203)
    m = CorrespondenceDirectionHead(radius=3)
    x = torch.rand(12, 7, 3, 5, requires_grad=True)
    y = torch.arange(12) % 2
    opt = torch.optim.SGD(m.parameters(), lr=0.1)
    for step in range(2):
        opt.zero_grad()
        F.cross_entropy(m(x), y).backward()
        assert m.residual[-1].weight.grad.abs().sum() > 0
        if step:
            assert m.residual[0].weight.grad.abs().sum() > 0
            assert m.log_scale.grad.abs() > 0
        opt.step()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0


def test_evidence_readout_uses_only_two_frames_and_independent_clips():
    from experiments.video_order import OrderReadout
    from pathwm.models.video_vae import local_correlation

    m = OrderReadout(4, "correlation", correlation_radius=3, readout="evidence")
    assert m.temporal is None
    assert sum(p.numel() for p in m.parameters()) == 81
    x = torch.randn(3, 3, 4, 12, 12)
    before = x.clone()
    trace = {}
    expected = m(x, trace=trace)
    torch.testing.assert_close(
        expected, m.head(local_correlation(x[:, -2], x[:, -1], radius=3))
    )
    assert torch.equal(x, before)
    x[:, 0] *= 100
    assert torch.equal(m(x), expected)
    x[1:] *= 100
    torch.testing.assert_close(m(x[:1]), expected[:1], atol=1e-7, rtol=1e-6)
    assert trace["candidate_scores"].shape == (3, 7)
    assert trace["residual"].count_nonzero() == 0


def test_evidence_validation_and_legacy_pooled_state():
    from experiments.video_order import OrderReadout
    from pathwm.models.video_vae import CorrespondenceDirectionHead

    for bad in [0, -1, True, 2.5]:
        with pytest.raises(ValueError):
            CorrespondenceDirectionHead(radius=bad)
    m = CorrespondenceDirectionHead(radius=2)
    with pytest.raises(ValueError):
        m(torch.rand(2, 7, 3, 3))
    for kwargs in [
        dict(mode="train", readout="evidence"),
        dict(mode="correlation", readout="bad"),
        dict(mode="correlation", readout="evidence", active_radius=1),
    ]:
        with pytest.raises(ValueError):
            OrderReadout(4, **kwargs)
    torch.manual_seed(207)
    a = OrderReadout(4, "correlation", correlation_radius=3)
    torch.manual_seed(207)
    b = OrderReadout(4, "correlation", correlation_radius=3, readout="pooled")
    x = torch.randn(2, 3, 4, 12, 12)
    assert torch.equal(a(x), b(x))
    assert all(torch.equal(v, b.state_dict()[k]) for k, v in a.state_dict().items())
