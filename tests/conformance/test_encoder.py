"""Structural conformance of the encoder E_m: o -> z (§16.1, §5.3).

What: the two checks every encoder passes with random weights before it may feed an adapter: encode
takes the raw uint8 observation exactly as the environment renders it and returns a row-batched,
floating, finite z without touching obs; and the rows of a batch do not interact. z's shape beyond
the batch dim is not checked: z is "any shape" (§16.1), mapping it to the ABI layout is the
adapter's job (§5.4, tests/conformance/test_adapter.py).
How: `obs` (B, 3, res, res) uint8 from tests/conformance/conftest.py; the encoder comes only from
`build("encoder")` (tests/conftest.py) and is asserted to be a contracts.Encoder right after build.
Structural layer (CLAUDE.md §4): random weights, CPU, tiny tensors, no marker.
Why: H1 (§4) puts a foreign encoder, through its own trained adapter, in front of the frozen P and
I_omega (E2, §5.4); swapping the encoder is one YAML line (CLAUDE.md §3) only if the data pipeline
hands every encoder the same raw obs and the encoder owns its preprocessing (contracts.py Encoder).
Preprocessing done in place would corrupt the collector's buffers (DDR §18: what the engine hands
out belongs to the caller) and read off the panel as an engine or data fault (§14); a batch-coupled
encoder would make z for an observation depend on the batch it arrived in (worlds x time in the
collector, a single goal observation in the planner, §5.11; the probe set in evaluation), so the
dataset, the goal encoding and the panel's s(w) and transition error would change with batching.
First experiment to run this module: E1_reference.
"""
import torch

import contracts


def test_encode_takes_raw_uint8_obs(build, obs):
    """z = enc.encode(obs) has B rows, is floating and finite; obs is unchanged after the call."""
    enc = build("encoder")
    assert isinstance(enc, contracts.Encoder)
    obs_before = obs.clone()
    z = enc.encode(obs)
    assert z.shape[0] == obs.shape[0]
    assert torch.is_floating_point(z)
    assert torch.isfinite(z).all()
    # Scaling and normalization are the encoder's own preprocessing (contracts.py Encoder) and must
    # not happen in place: the uint8 obs is the caller's, and a collector stores it (DDR §18).
    assert torch.equal(obs, obs_before)


def test_rows_are_independent(build, obs, per_sample):
    """encode(obs)[i] == encode(obs[i:i+1])[0] for every row: the batch is whatever the caller stacks."""
    enc = build("encoder")
    assert isinstance(enc, contracts.Encoder)
    # Rows at three brightness levels (x/8, x/2, x): on i.i.d. random rows per-row and pooled statistics
    # nearly coincide, so normalization pooled over the batch (a whole-batch LayerNorm, "(x - x.mean())
    # / x.std()" preprocessing) deviates <= 0.17 from per-row and stays within per_sample's tolerance
    # (atol + rtol * |z|); here it deviates 0.5-4.8 (measured at step 2 on ViT-like and conv fakes; one
    # dim row alone leaves a conv with a pooled output LN just inside the tolerance). A per-row encoder
    # is unaffected.
    obs = obs.clone()
    obs[0] //= 8
    obs[1] //= 2
    per_sample(enc.encode, obs)
