import pytest
import torch
from torch.nn import functional as F


def test_pair_center_loss_numeric_reference_and_gradient():
    from experiments.video_order import direction_loss

    logits = torch.tensor([[4.0, 0.0], [2.0, 0.0]], requires_grad=True)
    labels = torch.tensor([0, 1])
    loss, parts = direction_loss(logits, labels, pair_center_weight=0.1)
    # Class scores4,2: common offset3, SmoothL1(3)=2.5.
    ce = F.cross_entropy(logits, labels)
    torch.testing.assert_close(loss, ce + 0.25)
    assert float(parts["pair_center_penalty"]) == 2.5
    penalty_gradient = torch.autograd.grad(loss - ce, logits)[0]
    torch.testing.assert_close(
        penalty_gradient, torch.tensor([[0.05, -0.05], [0.05, -0.05]])
    )
    # Opposite offsets in different pairs must not cancel before penalization.
    both = torch.cat((logits.detach(), logits.detach().flip(-1)))
    _, parts = direction_loss(both, torch.tensor([0, 1, 1, 0]), pair_center_weight=0.1)
    assert float(parts["pair_center_penalty"]) == 2.5


def test_zero_weight_exact_ce_and_pair_invariances():
    from experiments.video_order import direction_loss

    x = torch.tensor(
        [[2.0, -1.0], [0.0, 3.0], [-2.0, 1.0], [4.0, -3.0]], requires_grad=True
    )
    y = torch.tensor([0, 1, 1, 0])
    loss, _ = direction_loss(x, y, pair_center_weight=0)
    reference = F.cross_entropy(x, y)
    assert torch.equal(loss, reference)
    assert torch.equal(
        torch.autograd.grad(loss, x, retain_graph=True)[0],
        torch.autograd.grad(reference, x)[0],
    )
    base, _ = direction_loss(x, y, pair_center_weight=0.1)
    for a, b in [
        (x + torch.tensor([[8.0], [4.0], [2.0], [16.0]]), y),
        (x.flip(-1), 1 - y),
        (x.reshape(2, 2, 2).flip(1).flatten(0, 1), y.reshape(2, 2).flip(1).flatten()),
    ]:
        other, _ = direction_loss(a, b, pair_center_weight=0.1)
        torch.testing.assert_close(base, other)
    for bad in (-0.1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            direction_loss(x, y, pair_center_weight=bad)
    with pytest.raises(ValueError):
        direction_loss(x, torch.tensor([0, 0, 1, 0]), pair_center_weight=0.1)
    with pytest.raises(ValueError):
        direction_loss(x[:3], y[:3], pair_center_weight=0.1)


def test_pair_scores_detect_common_preference_without_counting_it_as_accuracy():
    from experiments.video_order import metrics

    logits = torch.tensor([[[4.0, 0.0], [2.0, 0.0]], [[2.0, 0.0], [0.0, 2.0]]])
    labels = torch.tensor([[0, 1], [0, 1]])
    m = metrics(logits, labels)
    assert m["accuracy"] == 0.75 and m["pair_accuracy"] == 0.5
    assert m["pair_ordering_accuracy"] == 1
    assert m["pair_offset_mean_abs"] == 1.5
    assert m["pair_signal_mean_abs"] == 1.5
    assert m["pair_bias_fraction"] == 0.375
    shifted = metrics(logits + 11, labels)
    assert shifted == m


def test_readout_single_sequence_has_no_dependency_on_other_batch_members():
    from experiments.video_order import OrderReadout

    torch.manual_seed(61)
    model = OrderReadout(2, "train").eval()
    x = torch.randn(3, 3, 2, 8, 8)
    together = model(x)
    individual = torch.cat([model(v[None]) for v in x])
    torch.testing.assert_close(together, individual, atol=1e-6, rtol=1e-5)
    changed = x.clone()
    changed[1:] = torch.randn_like(changed[1:]) * 100
    torch.testing.assert_close(model(changed)[0], together[0])
