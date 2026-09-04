"""Structural conformance of the predictor P_phi: (W_t, a_{t:t+k}, delta_t) -> W_hat_{t+delta_t} (§5.6, §16.1).

What: with random weights, predict returns a batch of ABI states (docs/abi/abi_v1.yaml via world_state/abi.py)
and leaves its inputs (W and the action chunk) alone; the register count is one the ABI declares; every chunk
length and delta_t the contract allows is accepted and every one it forbids raises; the output does not depend
on earlier calls and is not rewritten by them; actions and delta_t reach the output; every output token is
LayerNormed without affine parameters; rows of a batch do not interact. Random weights, CPU, tiny tensors, no
marker: the structural layer of CLAUDE.md §4. The threshold layer (s(w) and transition error on the fixed probe
set) is not here.
How: W and W_next are synthetic LayerNormed ABI states (tests/conformance/conftest.py); actions are drawn in the
ABI action range in the chunk form (B, k, action_dims) float32. The predictor comes only from build("predictor"),
so until CLAUDE.md §2 step 3 writes predictors/ every test here fails naming the missing builder. Register counts
are parametrized from the ABI at collection time (load_abi() at module level); everything else resolves through
the `abi` fixture.
Why: the predictor is the frozen consumer of H1 (§4): E2 stitches a foreign adapter against it, so what it
accepts and emits must be exactly the ABI, and it must be Markov in W (DDR §5; Invariant 4: registers are created
per call and discarded) so that a reservoir snapshot or a planner fork (§7.1) sees the same function every time.
Serves E1_reference, the first experiment that runs this module, and the E2 gate.
"""
from __future__ import annotations

import copy

import pytest
import torch

import contracts
from world_state.abi import load_abi

REGISTER_OPTIONS = load_abi().register_options  # parametrization needs the ABI at collection time; the tests use `abi`


def _actions(gen, k, B, abi):
    """(B, k, action_dims) float32 in the ABI action range: the chunk form Predictor.predict takes."""
    return torch.rand(B, k, abi.action_dims, generator=gen) * 2 - 1


def _with_registers(cfg, n):
    """The dev spec with predictor.registers = n; deepcopied because `cfg` is session-scoped."""
    c = copy.deepcopy(cfg)
    c["predictor"]["registers"] = n
    return c


def test_predict_returns_abi_state_and_leaves_input_alone(build, abi, gen, W):
    """W_hat = predict(W, a, 1) is (B, n_tokens, dim) in the activation dtype, finite, one row per input row;
    W and the action chunk are untouched.

    Both inputs are reservoir objects (§7.1): an in-place delta update on W, or an in-place rescale of the
    chunk (actions.float() is a no-op view for float32), would corrupt the snapshots the planner keeps.
    """
    P = build("predictor")
    # runtime_checkable: predict and n_registers exist, not their shapes. On Python 3.12+ isinstance looks the
    # members up with inspect.getattr_static, so n_registers must be a plain attribute or property: an int kept
    # as an nn.Module buffer is reached through __getattr__ and makes this assertion fail, not the ABI checks.
    assert isinstance(P, contracts.Predictor)
    a = _actions(gen, 1, W.shape[0], abi)
    Wc, ac = W.clone(), a.clone()
    y = P.predict(W, a, 1)
    abi.check_state(y)
    assert torch.isfinite(y).all(), "predictor output has non-finite entries"
    assert y.shape[0] == W.shape[0], f"batch {y.shape[0]} != {W.shape[0]} input rows"  # check_state leaves B free
    # an in-place delta update on W, or an in-place op on the chunk, would corrupt the caller's reservoir copies (§7.1)
    assert torch.equal(W, Wc), "predict modified its input W"
    assert torch.equal(a, ac), "predict modified its actions"


@pytest.mark.parametrize("n", REGISTER_OPTIONS)
def test_registers_are_declared_in_the_abi(build, abi, cfg, n):
    """A predictor built with predictor.registers = n reports n_registers == n, and n is an ABI register option."""
    P = build("predictor", cfg=_with_registers(cfg, n))
    assert isinstance(P, contracts.Predictor)
    assert P.n_registers == n, f"n_registers {P.n_registers} != configured {n}"
    abi.check_registers(P.n_registers)  # abi_v1.yaml registers.count_options


@pytest.mark.parametrize("k,delta_t", [(1, 1), (4, 4), (4, 1), ("max", "max")])
def test_chunk_lengths_within_the_contract(build, abi, gen, W, k, delta_t):
    """Every (k, delta_t) with 1 <= delta_t <= k <= max_chunk yields an ABI state (contracts.py Predictor)."""
    P = build("predictor")
    assert isinstance(P, contracts.Predictor)
    k, delta_t = (abi.max_chunk if v == "max" else v for v in (k, delta_t))
    abi.check_state(P.predict(W, _actions(gen, k, W.shape[0], abi), delta_t))


@pytest.mark.parametrize(
    "bad",
    [
        lambda a: (2, 0, a.action_dims),                # delta_t < 1
        lambda a: (2, 3, a.action_dims),                # delta_t > k: steps without actions
        lambda a: (a.max_chunk + 1, 1, a.action_dims),  # chunk longer than the ABI allows
        lambda a: (2, 1, a.action_dims + 1),            # wrong action width
    ],
    ids=["delta_t_zero", "delta_t_beyond_chunk", "chunk_beyond_max", "wrong_action_width"],
)
def test_invalid_chunks_raise(build, abi, gen, W, bad):
    """A chunk outside 1 <= delta_t <= k <= max_chunk, or actions of the wrong width, raise ValueError.

    Why: silent acceptance of delta_t > k means steps without actions. ABIError subclasses ValueError;
    torch's own shape errors are RuntimeError, so this is not vacuous.
    """
    P = build("predictor")
    assert isinstance(P, contracts.Predictor)
    k, delta_t, dims = bad(abi)
    actions = torch.rand(W.shape[0], k, dims, generator=gen) * 2 - 1
    with pytest.raises(ValueError):
        P.predict(W, actions, delta_t)


@pytest.mark.parametrize("n", REGISTER_OPTIONS)
def test_stateless_across_calls(build, abi, cfg, gen, W, W_next, n):
    """The same (W, a, delta_t) gives the same W_hat whatever was predicted in between, and an earlier
    W_hat is not rewritten by a later call.

    The Markov / stateless contract (contracts.py Predictor; DDR §5: no context window, no hidden state).
    For n > 0 this is Invariant 4: registers are created per call and discarded (abi_v1.yaml
    registers.reset). n = 0 (the dev config) cannot violate Invariant 4, so the ABI's other counts are here.
    The clone check is the output-side twin of "leaves input alone": the reservoir (§7.1) stores predictor
    outputs as persistent objects, and an output that aliases a reused internal buffer is overwritten under
    it by the next call, which the final equality alone cannot see (it would compare the buffer with itself).
    """
    P = build("predictor", cfg=_with_registers(cfg, n))
    assert isinstance(P, contracts.Predictor)
    B = W.shape[0]
    a1, a2, a4 = (_actions(gen, k, B, abi) for k in (1, 2, 4))
    y1 = P.predict(W, a1, 1)
    y1c = y1.clone()  # kept so that a later call rewriting y1's storage shows as a change of y1
    P.predict(W_next, a4, 4)
    P.predict(W_next, a2, 2)
    assert torch.equal(y1, y1c), "a later call changed an earlier output"
    assert torch.equal(P.predict(W, a1, 1), y1), "predict depends on earlier calls"


def test_actions_and_delta_t_are_wired(build, abi, cfg, gen, W):
    """Negating the actions changes W_hat; so does changing delta_t when the predictor is delta_t-conditioned.

    An unwired action path gives s(w) = 0 exactly (§4 H1); an ignored delta_t fits a delta_t-averaged
    target (§5.6, §6.2); both would be misread from the §14 table as model failures. Requirement on the
    builder: the readout is NOT zero-initialized (§5.6 "identity by default" is the residual form, not a
    zero init). Measured on a random 2-layer transformer: actions change 38% of the output elements,
    delta_t 79%.
    """
    P = build("predictor")
    assert isinstance(P, contracts.Predictor)
    a4 = _actions(gen, 4, W.shape[0], abi)
    y = P.predict(W, a4, 4)
    assert not torch.equal(y, P.predict(W, -a4, 4)), "actions do not reach the output"
    if cfg["predictor"]["delta_t_conditioned"]:
        assert not torch.equal(y, P.predict(W, a4, 1)), "delta_t does not reach the output"


def test_readout_init_scale_reduces_the_initial_residual_without_disconnecting_actions(
    build, abi, cfg, gen, W
):
    default_cfg = copy.deepcopy(cfg)
    scaled_cfg = copy.deepcopy(cfg)
    default_cfg["predictor"]["readout_init_scale"] = 1.0
    scaled_cfg["predictor"]["readout_init_scale"] = 0.03
    default = build("predictor", cfg=default_cfg)
    scaled = build("predictor", cfg=scaled_cfg)
    actions = _actions(gen, 1, W.shape[0], abi)

    default_delta = (default.predict(W, actions, 1).float() - W.float()).square().mean()
    scaled_output = scaled.predict(W, actions, 1)
    scaled_delta = (scaled_output.float() - W.float()).square().mean()

    assert scaled_delta < default_delta * 0.1
    assert not torch.equal(scaled_output, scaled.predict(W, -actions, 1))


def test_output_tokens_are_layernormed(build, abi, gen, W, layernormed_at_all_scales):
    """Every output token has mean ~0 and variance ~1 whatever the input scale.

    Same check as the adapter's (tests/conformance/conftest.py): §5.6 readout LN(W + dW), non-affine at
    the ABI boundary (abi_v1.yaml state.normalization, DDR §13 step-2 additions). The x10 case is what
    catches a missing output LN, since at x1 a small dW on a LayerNormed W leaves var ~ 1 anyway.
    """
    P = build("predictor")
    assert isinstance(P, contracts.Predictor)
    a1 = _actions(gen, 1, W.shape[0], abi)
    layernormed_at_all_scales(lambda scale: P.predict((W.float() * scale).to(abi.dtype), a1, 1))


def test_rows_are_independent(build, abi, gen, W, per_sample):
    """predict(W, a, 1)[i] equals predict(W[i:i+1], a[i:i+1], 1)[0]: the batch is whatever the caller stacks."""
    P = build("predictor")
    assert isinstance(P, contracts.Predictor)
    per_sample(lambda W_, a_: P.predict(W_, a_, 1), W, _actions(gen, 1, W.shape[0], abi))
