"""Structural conformance of the adapter A_m: z -> W in the ABI v1 layout (§5.4, §16.1).

What: the adapter's output is a batch of ABI states (docs/abi/abi_v1.yaml via world_state/abi.py):
layout and dtype, finite, every token LayerNormed without affine parameters, rows independent. Random
weights, CPU, tiny tensors, no marker: the structural layer of CLAUDE.md §4. The threshold layer (the
§6.5 adapter losses on held-out data) is not here.
How: the adapter is built for the encoder's z, build("adapter", tuple(z.shape[1:])), because z is "any
shape" (§16.1) and cfg cannot supply it. An encoder failure therefore shows here too, and the session
report must say which module failed: the encoder (build("encoder") or encode) or the adapter.
Why: H1 (§4) is a claim about a frozen consumer (P, I_omega) on a foreign producer, and the ABI is what
makes "foreign" well-defined (§5.2); the adapter is the producer, the object under training in E2. A
producer that silently leaves the layout, the dtype or the per-token normalization would invalidate the
stitching comparison. Serves E1_reference, the first experiment that runs this module, and the E2 gate.
"""
from __future__ import annotations

import torch

import contracts


def _adapter_and_z(build, obs):
    """Encode obs with the slice's encoder, then build the adapter for that z's per-sample shape."""
    enc = build("encoder")
    assert isinstance(enc, contracts.Encoder)
    z = enc.encode(obs)
    adapter = build("adapter", tuple(z.shape[1:]))
    assert isinstance(adapter, contracts.Adapter)
    return adapter, z


def test_adapt_emits_abi_state(build, abi, obs):
    """W = adapt(z) is (B, n_tokens, dim) in the activation dtype, finite, one row per observation."""
    adapter, z = _adapter_and_z(build, obs)
    W = adapter.adapt(z)
    abi.check_state(W)
    assert torch.isfinite(W).all(), "adapter output has non-finite entries"
    assert W.shape[0] == obs.shape[0], f"batch {W.shape[0]} != {obs.shape[0]} observations"


def test_tokens_are_layernormed(build, obs, layernormed_at_all_scales):
    """Every output token has mean ~0 and variance ~1 whatever the input scale.

    abi_v1.yaml state.normalization: per_token layernorm with per_token_affine: false at the ABI
    boundary (adapter output; DDR §13 step-2 additions), so no per-producer scale or shift reaches
    the consumer; tolerances and the scale sweep are in tests/conformance/conftest.py. Random weights
    only (§4): an affine LN at init would be the identity anyway.
    Requirement on the builder: no token is constant at init (LN of a constant token is the zero
    vector, var 0), so a linear adapter's global token (W[:, 0]) must be a function of z, e.g. a
    projection of the token mean, not a zero-initialized learned constant. The sweep assumes O(1)
    per-token z (the dev ViT-S/8 ends in a LayerNorm): at x0.1 a bias-free adapter over std <= 0.1
    tokens reaches LayerNorm's eps and reads var < 1.
    """
    adapter, z = _adapter_and_z(build, obs)
    layernormed_at_all_scales(lambda scale: adapter.adapt(z * scale))


def test_rows_are_independent(build, obs, per_sample):
    """adapt(z)[i] equals adapt(z[i:i+1])[0]: the batch is whatever the caller stacks (contracts.py)."""
    adapter, z = _adapter_and_z(build, obs)
    per_sample(adapter.adapt, z)
