"""Resource counts and exact event scheduling, independent of learned quality."""

from dataclasses import replace

import pytest
import torch
from torch import nn

from experiments.multimodal import build_model
from pathwm.models.belief_state import Packet
from pathwm.models.modalities import Attend, Observation
from pathwm.models.blocks import Attention


def packet_reference(model, state, observations):
    ordinal = state.ordinal + 1
    pending = model.begin_event(state, event_id=f"event-{ordinal}", ordinal=ordinal, time=1.)
    for name, obs in sorted(observations.items()):
        pending = model.add_packet(pending, Packet(f"{ordinal}/{name}", name, obs))
    return model.commit_event(pending)


def test_counts_are_rectangular_and_repeats_are_not_unique_layers():
    from pathwm.evaluation.workload import Workload

    model = nn.ModuleDict({"cross": Attend(8), "self": Attention(8, heads=2)})
    q, kv = torch.randn(2, 3, 8), torch.randn(2, 7, 8)
    valid = torch.tensor([[True] * 5 + [False] * 2] * 2)
    with Workload(model) as work:
        for _ in range(4):
            model["cross"](q, kv, valid=valid)
        model["self"](q, q, q)
    result = work.summary()
    assert len(result["attention"]) == 5  # no double-counted nested MHA
    assert result["query_token_evaluations"] == 15
    assert result["self_attention_pairs"] == 9
    assert result["cross_attention_pairs"] == 84  # padded positions still cost
    assert result["per_layer"]["cross"]["calls"] == 4
    assert result["per_layer"]["cross"]["N_key"] == [7] * 4


def test_instrumentation_preserves_outputs_gradients_rng_and_removes_hooks():
    from pathwm.evaluation.workload import Workload

    model = Attend(8)
    q, kv = torch.randn(2, 3, 8), torch.randn(2, 7, 8)
    expected = model(q, kv)
    expected.square().sum().backward()
    grads = {n: p.grad.clone() for n, p in model.named_parameters()}
    model.zero_grad(set_to_none=True)
    before = torch.get_rng_state().clone()
    with Workload(model) as work:
        actual = model(q, kv)
        actual.square().sum().backward()
    assert torch.equal(expected, actual)
    assert torch.equal(before, torch.get_rng_state())
    for name, p in model.named_parameters():
        assert torch.equal(grads[name], p.grad), name
    assert len(work.attention) == 1
    with pytest.raises(RuntimeError, match="stop"):
        with Workload(model):
            raise RuntimeError("stop")
    assert all(not m._forward_pre_hooks and not m._forward_hooks for m in model.modules())


def test_complete_event_encodes_once_with_same_posterior_and_gradients():
    torch.manual_seed(71)
    model = build_model(width=16, state_model="belief").eval()
    start = model.initial_state(2)
    observations = {
        "image": Observation(torch.randn(2, 1, 3, 16, 16), torch.ones(2, 1)),
        "audio": Observation(torch.randn(2, 1, 32), torch.ones(2, 1)),
    }
    # Include an entirely absent member and masked NaNs.
    observations = {name: replace(obs, valid=torch.tensor([[True], [False]]),
                    values=obs.values.masked_fill(torch.arange(2).reshape(2, *([1] * (obs.values.ndim - 1))) == 1, float("nan")))
                    for name, obs in observations.items()}
    counts = {name: 0 for name in observations}
    def count(name):
        def hook(*args):
            counts[name] += 1
        return hook
    hooks = [model.encoders[name].register_forward_hook(count(name)) for name in counts]
    rng = torch.get_rng_state()
    expected = packet_reference(model, start, observations)
    expected.logits.square().sum().backward(retain_graph=True)
    grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    torch.set_rng_state(rng)
    counts.update({name: 0 for name in counts})
    actual = model.observe(start, observations, time=1.)
    actual.logits.square().sum().backward()
    for hook in hooks:
        hook.remove()
    assert counts == {"image": 1, "audio": 1}
    for field in ("tokens", "h", "logits", "z", "evidence", "source_valid", "source_start", "source_end"):
        torch.testing.assert_close(getattr(actual, field), getattr(expected, field), atol=1e-6, rtol=1e-5)
    assert actual.sources == expected.sources and actual.memory.sources == expected.memory.sources
    assert actual.observation_count == expected.observation_count == 1
    for name, p in model.named_parameters():
        if name in grads:
            torch.testing.assert_close(p.grad, grads[name], atol=2e-6, rtol=2e-5)


def test_resampler_has_fixed_budget_and_does_not_read_invalid_detail():
    from pathwm.models.resampler import LatentResampler

    layer = LatentResampler(16, tokens=64)
    values = torch.randn(2, 19, 16, requires_grad=True)
    valid = torch.zeros(2, 19, dtype=torch.bool)
    valid[0, :7] = True
    changed = values.detach().clone().masked_fill(~valid[..., None], float("nan"))
    out = layer(values, valid)
    torch.testing.assert_close(out, layer(changed, valid))
    assert out.shape == (2, 64, 16) and torch.isfinite(out).all()
    assert torch.count_nonzero(out[1]) == 0
    out.square().sum().backward()
    assert values.grad[valid].abs().sum() > 0
    assert torch.count_nonzero(values.grad[~valid]) == 0
    assert layer.queries.grad.abs().sum() > 0
    assert not torch.equal(layer(values.detach(), valid), layer(torch.zeros_like(values), valid))
    with pytest.raises(ValueError):
        LatentResampler(16, tokens=0)
