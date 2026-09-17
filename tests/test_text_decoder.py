"""Independent references for the decoder's training and last-position paths."""

import copy

import pytest
import torch

from pathwm.models.modalities import TextDecoder, position


def original_forward(module, tokens, prefix, valid=None, trace=None):
    x = module.embedding(prefix) + position(
        torch.arange(prefix.shape[1], device=tokens.device, dtype=tokens.dtype),
        module.width,
    )
    x = module.self_attention(
        x,
        x,
        valid=prefix != 0,
        causal=True,
        trace=trace,
        name="decode.text.causal_attention",
    )
    return module.output(
        module.read(
            x,
            tokens,
            valid=valid,
            trace=trace,
            name="decode.text.state_attention",
        )
    )


@pytest.mark.parametrize("length", [1, 7, 16])
def test_last_logits_match_full_prefix_with_masked_nan_context(length):
    torch.manual_seed(214)
    decoder = TextDecoder(16).eval()
    tokens = torch.randn(3, 5, 16)
    valid = torch.tensor([[True, False, True, False, False]]).expand(3, -1)
    tokens = tokens.masked_fill(~valid[..., None], float("nan"))
    prefix = torch.randint(3, 259, (3, length))
    prefix[:, 0] = 1
    before = {k: v.clone() for k, v in decoder.state_dict().items()}
    rng = torch.get_rng_state().clone()
    with torch.no_grad():
        expected = original_forward(decoder, tokens, prefix, valid)[:, -1:]
        actual = decoder(tokens, prefix, valid=valid, last_only=True)
    assert actual.shape == (3, 1, 259)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=1e-5)
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(before[k], v) for k, v in decoder.state_dict().items())


def test_full_logits_gradients_and_traces_keep_original_path():
    torch.manual_seed(55)
    decoder = TextDecoder(16)
    original = copy.deepcopy(decoder)
    tokens = torch.randn(2, 4, 16, requires_grad=True)
    source = tokens.detach().clone().requires_grad_()
    prefix = torch.tensor([[1, 16, 31, 9], [1, 20, 0, 0]])
    trace, reference_trace = {}, {}
    actual = decoder(tokens, prefix, trace=trace)
    expected = original_forward(original, source, prefix, trace=reference_trace)
    assert torch.equal(actual, expected)
    actual.square().sum().backward()
    expected.square().sum().backward()
    assert torch.equal(tokens.grad, source.grad)
    for p, q in zip(decoder.parameters(), original.parameters()):
        assert torch.equal(p.grad, q.grad)
    assert trace.keys() == reference_trace.keys()
    assert all(torch.equal(trace[k], reference_trace[k]) for k in trace)
    # Teacher forcing must remain causal at every earlier output position.
    changed = prefix.clone()
    changed[0, -1] = 45
    assert torch.equal(decoder(tokens, changed)[:, :-1], actual[:, :-1])


def test_generation_keeps_mixed_eos_and_budget_contract(monkeypatch):
    decoder = TextDecoder(16)
    seen = []

    def forward(tokens, prefix, trace=None, *, valid=None, last_only=False):
        seen.append(last_only)
        out = tokens.new_zeros(len(prefix), 1 if last_only else prefix.shape[1], 259)
        out[..., :2] = 100  # PAD/BOS must stay prohibited even when highest.
        out[0, -1, 2] = 10
        out[1, -1, 2 if prefix.shape[1] >= 3 else 100] = 10
        return out

    monkeypatch.setattr(decoder, "forward", forward)
    tokens = torch.randn(2, 4, 16)
    assert decoder.generate(tokens, 5, last_only=True).tolist() == [
        [1, 2, 2, 2],
        [1, 100, 100, 2],
    ]
    assert all(seen)
    seen.clear()
    assert decoder.generate(tokens, 2).tolist() == [[1, 2, 2], [1, 100, 100]]
    assert not any(seen)
    with pytest.raises(ValueError, match="positive"):
        decoder.generate(tokens, 0)


@pytest.mark.parametrize(
    "prefix",
    [torch.tensor([[0, 5]]), torch.tensor([[1, 0, 8]]), torch.tensor([[1, 259]])],
)
def test_last_logits_keep_prefix_validation(prefix):
    with pytest.raises(ValueError):
        TextDecoder(16)(torch.randn(1, 4, 16), prefix, last_only=True)
