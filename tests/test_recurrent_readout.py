import copy

import pytest
import torch

from pathwm.models.readout import RecurrentOutputAdapter, TemporalImageDecoder


def test_adapter_initial_identity_and_shared_iteration_parameters():
    torch.manual_seed(12)
    adapter = RecurrentOutputAdapter(16, iterations=1)
    x = torch.randn(2, 5, 16)
    original = x.clone()
    assert torch.equal(adapter(x), x)
    identities = [id(p) for p in adapter.parameters()]
    with torch.no_grad():
        adapter.gain.fill_(0.5)
    one = adapter(x)
    calls = []
    hook = adapter.cross.register_forward_hook(lambda *args: calls.append(1))
    adapter.iterations = 4
    four = adapter(x)
    hook.remove()
    assert len(calls) == 4
    assert identities == [id(p) for p in adapter.parameters()]
    assert not torch.allclose(one, four)
    assert torch.equal(x, original)


def test_adapter_mask_excludes_nan_payload_and_gradient():
    adapter = RecurrentOutputAdapter(16, iterations=2)
    with torch.no_grad():
        adapter.gain.fill_(0.5)
    x = torch.randn(2, 5, 16, requires_grad=True)
    valid = torch.tensor([[True, True, False, False, False]]).expand(2, -1)
    output = adapter(x.masked_fill(~valid[..., None], float('nan')), valid=valid)
    expected = adapter(x[:, :2])
    torch.testing.assert_close(output[:, :2], expected)
    assert output[:, 2:].count_nonzero() == 0
    output.square().sum().backward()
    assert x.grad[:, :2].abs().sum() > 0
    assert x.grad[:, 2:].count_nonzero() == 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in adapter.parameters())
    with pytest.raises(ValueError, match='valid'):
        adapter(x, valid=torch.zeros_like(valid))


def test_adapter_trace_preserves_output_gradients_rng():
    torch.manual_seed(10)
    adapter = RecurrentOutputAdapter(16, iterations=2)
    with torch.no_grad():
        adapter.gain.fill_(0.5)
    twin = copy.deepcopy(adapter)
    x = torch.randn(2, 4, 16)
    state = torch.get_rng_state().clone()
    a = adapter(x)
    a.square().sum().backward()
    trace = {}
    b = twin(x, trace=trace)
    b.square().sum().backward()
    assert torch.equal(a, b)
    assert torch.equal(torch.get_rng_state(), state)
    assert 'readout.loop.1.tokens' in trace
    for p, q in zip(adapter.parameters(), twin.parameters()):
        assert torch.equal(p.grad, q.grad)


def test_temporal_decoder_reads_only_context_and_requested_times():
    decoder = TemporalImageDecoder(16, image_size=16)
    context = torch.randn(2, 5, 16, requires_grad=True)
    before = context.detach().clone()
    times = torch.tensor([0., 1., 2., 3.])
    result = decoder(context, times)
    assert result.shape == (2, 4, 3, 16, 16)
    assert not torch.equal(result[:, 0], result[:, -1])
    result.square().mean().backward()
    assert context.grad.abs().sum() > 0
    assert torch.equal(context.detach(), before)
    assert all(p.grad is not None for p in decoder.parameters())
    with pytest.raises(ValueError, match='increasing'):
        decoder(context, times.flip(0))


@pytest.mark.parametrize('iterations', [-1, 1.5, True])
def test_bad_iterations_rejected(iterations):
    with pytest.raises(ValueError, match='iterations'):
        RecurrentOutputAdapter(16, iterations=iterations)
