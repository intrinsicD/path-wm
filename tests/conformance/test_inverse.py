"""Structural conformance of the inverse-dynamics head I_omega (§5.7) against contracts.py InverseDynamics.

What: with random weights, infer_action(W, W_next) returns one ABI action per row whatever the input
scale, leaves its inputs alone, hands out an output the next call does not overwrite, and the rows of
a batch do not interact.
How: W and W_next are synthetic LayerNormed ABI states (tests/conformance/conftest.py); the head comes
only from build("inverse"), so until CLAUDE.md §2 step 3 writes world_state/inverse.py every test here
fails naming the missing builder. Random weights, CPU, tiny tensors, no marker (CLAUDE.md §4).
Why: the head is the anti-collapse term of E1 (§6.1) and the interface anchor of E2 (§5.7); its output
must be an action in the ABI action space because it feeds Environment.step (DDR §18) and the planner's
inverse proposals (§7.3). The threshold layer (§6.5 losses on a checkpoint) is not here.
"""
from __future__ import annotations

import torch

import contracts

# Input scale sweep for the range check. At x1 a random-init head emits |a| ~ 0.1 whether or not it is
# bounded, so check_actions there is vacuous: measured on an unbounded 2-layer MLP over the mean-pooled
# pair (50 inits), max |a| <= 0.21 at x1 (0 of 50 outside [-1, 1]), 2 of 50 outside at x10, 40 of 50 at
# x30, 50 of 50 at x100 (median max |a| 5.4). A bounded head (tanh, clamp) stays inside and finite at every scale.
SCALES = (100.0,)


def test_infer_action_returns_abi_action(build, abi, W, W_next):
    """One action per row: (B, action_dims) float32, finite, in the ABI action range at x1 and at x100 input
    scale; W and W_next untouched; the returned tensor survives the next call.

    The ABI action space is the contract because the output feeds Environment.step (DDR §18) and planner
    proposals (§7.3). float32 because bf16 actions have 2^-8 resolution (contracts.py InverseDynamics);
    within range because the head is bounded, the E1 loss taken before the squash (DDR §13). The x100
    input (SCALES) is what makes the range check bite. Limit: a head that LayerNorms its own input is in
    range at every input scale with or without a squash, so no input scale catches it; the only black-box
    alternative, scaling the module's parameters, would depart from "used as built" (step-2 brief).
    Inputs stay untouched because the head runs on dataset pairs and on reservoir states for inverse
    proposals (§7.3): an in-place upcast or normalization would corrupt them. The output must not be a
    view of a buffer inside the head that the next call overwrites.
    """
    inv = build("inverse")
    assert isinstance(inv, contracts.InverseDynamics)  # runtime_checkable: the method exists, not its shapes
    Wc, Wnc = W.clone(), W_next.clone()
    a = inv.infer_action(W, W_next)
    assert a.shape == (W.shape[0], abi.action_dims)
    assert a.dtype == torch.float32
    assert torch.isfinite(a).all()
    abi.check_actions(a)  # the 2-D form Environment.step takes, values within action_range
    assert torch.equal(W, Wc) and torch.equal(W_next, Wnc), "infer_action modified an input"
    a_c = a.clone()
    inv.infer_action(W_next, W)  # a different pair, so an output aliased to a buffer in the head would change
    assert torch.equal(a, a_c), "the returned action aliases state inside the head"
    for scale in SCALES:
        a = inv.infer_action((W.float() * scale).to(abi.dtype), (W_next.float() * scale).to(abi.dtype))
        assert torch.isfinite(a).all(), f"scale x{scale}: non-finite action"
        abi.check_actions(a)  # an unbounded head leaves the action range here (ABIError)


def test_rows_are_independent(build, W, W_next, per_sample):
    """Row i of a batched call equals the single-row call: the batch is whatever the caller stacks.

    atol is sized to the action output, not to per_sample's default for unit-variance tokens: a bounded
    head at random init emits |a| ~ 0.1, so a batch coupling (x - x.mean(0) on the pooled features,
    a - a.mean(0) on the output, a batch max-pool broadcast to every row) moves a row by only 0.01-0.03
    and would pass at atol 5e-2; only a train-mode BatchNorm would be caught. Measured batch-vs-single
    deviation: 7e-9 with fp32 compute, 0.0 for a bf16-compute MLP head; atol 1e-3 is ~2 bf16 ulps at
    |a| ~ 0.1 and rtol 5e-2 leaves >= 2 bf16 ulps at |a| ~ 1.
    """
    inv = build("inverse")
    assert isinstance(inv, contracts.InverseDynamics)
    per_sample(inv.infer_action, W, W_next, atol=1e-3)
