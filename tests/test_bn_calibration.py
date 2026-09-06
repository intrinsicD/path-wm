# Claude-authored via a context-free MCP task; integrated and verified by Codex.
"""Essential tests for bn_calibration.recalibrate_bn_statistics."""
from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from scripts import bn_calibration


class TinyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 6)
        self.bn1 = nn.BatchNorm1d(6)
        self.drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(6, 5)
        self.bn2 = nn.BatchNorm1d(5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn2(self.fc2(self.drop(torch.relu(self.bn1(self.fc1(x))))))


def _make_case():
    torch.manual_seed(0)
    model = TinyNet()
    with torch.no_grad():  # "pretrained", clearly stale statistics
        model.bn1.running_mean.fill_(3.0)
        model.bn1.running_var.fill_(7.0)
        model.bn2.running_mean.fill_(-2.0)
        model.bn2.running_var.fill_(0.25)
    g = torch.Generator().manual_seed(123)
    batches = [torch.randn(8, 4, generator=g) * (i + 1) + i for i in range(3)]
    return model, batches, (lambda: iter(batches)), (lambda m, b: m(b))


def test_original_untouched_params_identical_and_report():
    model, batches, factory, fwd = _make_case()
    model.train()
    flags = {n: m.training for n, m in model.named_modules()}
    state = {k: v.clone() for k, v in model.state_dict().items()}
    rng = torch.get_rng_state()

    out, report = bn_calibration.recalibrate_bn_statistics(model, factory, fwd)

    for k, v in model.state_dict().items():  # original buffers/params untouched
        assert torch.equal(v, state[k]), k
    assert {n: m.training for n, m in model.named_modules()} == flags
    assert torch.equal(torch.get_rng_state(), rng)
    assert out is not model
    for (n, p), (_, q) in zip(model.named_parameters(), out.named_parameters()):
        assert torch.equal(p, q), n
    assert {n: m.training for n, m in out.named_modules()} == flags
    assert not torch.equal(out.bn1.running_mean, model.bn1.running_mean)

    assert [l.name for l in report.layers] == ["bn1", "bn2"]
    assert [l.order for l in report.layers] == [0, 1]
    for layer in report.layers:
        assert layer.batches == len(batches)
        assert layer.samples == sum(b.shape[0] for b in batches)
        assert layer.original_digest["running_mean"] != layer.new_digest["running_mean"]
    assert report.batches_per_pass == len(batches)
    assert report.samples_per_pass == sum(b.shape[0] for b in batches)

    # dropout disabled -> fully deterministic and repeatable
    again, _ = bn_calibration.recalibrate_bn_statistics(model, factory, fwd)
    assert torch.equal(again.bn2.running_mean, out.bn2.running_mean)


def test_downstream_layer_sees_eval_normalized_upstream():
    model, batches, factory, fwd = _make_case()
    out, _ = bn_calibration.recalibrate_bn_statistics(model, factory, fwd)

    out.eval()
    seen: list[torch.Tensor] = []
    handle = out.bn2.register_forward_pre_hook(lambda m, i: seen.append(i[0].detach().clone()))
    with torch.no_grad():
        for b in batches:
            out(b)
    handle.remove()

    expected_mean = torch.stack([c.mean(0) for c in seen]).mean(0)
    expected_var = torch.stack([c.var(0, unbiased=True) for c in seen]).mean(0)
    assert torch.allclose(out.bn2.running_mean, expected_mean, atol=1e-5)
    assert torch.allclose(out.bn2.running_var, expected_var, atol=1e-5)

    naive = copy.deepcopy(model)  # all BN layers trained simultaneously
    naive.eval()
    for bn in (naive.bn1, naive.bn2):
        bn.reset_running_stats()
        bn.momentum = None
        bn.train(True)
    with torch.no_grad():
        for b in batches:
            naive(b)
    assert (naive.bn2.running_mean - out.bn2.running_mean).abs().max() > 1e-3


def test_empty_and_too_small_batches_are_rejected():
    model, _, _, fwd = _make_case()
    empty = [torch.zeros(0, 4)]
    with pytest.raises(ValueError, match="empty input batch"):
        bn_calibration.recalibrate_bn_statistics(model, lambda: iter(empty), fwd)

    single = [torch.randn(1, 4)]
    with pytest.raises(ValueError, match="per channel"):
        bn_calibration.recalibrate_bn_statistics(model, lambda: iter(single), fwd)

    with pytest.raises(ValueError, match="no batches"):
        bn_calibration.recalibrate_bn_statistics(model, lambda: iter([]), fwd)
