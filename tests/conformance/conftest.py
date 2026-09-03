"""Shared inputs and checks for the structural conformance layer (CLAUDE.md §4).

What: synthetic inputs in the contract's shapes (`obs`, `W`, `W_next`, `gen`); `per_sample`, the check
that rows of a batch do not interact; `layernormed_at_all_scales`, the check that a producer's tokens are
LayerNormed at the ABI boundary. Implementations come from tests/conftest.py's `build`.
How: inputs are drawn from private torch.Generators with fixed seeds, so they do not shift when a builder
consumes a different amount of the global RNG. per_sample compares a batched call with single-row calls;
the LayerNorm check sweeps the input scale, since LN output is scale-invariant and a merely calibrated
output is not.
Why: a module's test must not depend on other modules existing (an encoder test runs without an
environment); the batch is whatever the caller stacks (contracts.py: worlds x time), so rows must be
independent, and a predictor call is priced per row (DDR §13), which only means something if they are;
the boundary normalization is an ABI field (abi_v1.yaml state.normalization, DDR §13 step-2 additions)
that both producers, adapter and predictor, must honour.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

B = 3  # rows per synthetic batch: small, odd, more than one

# Per-token normalization at the ABI boundary. The same tokens are checked at three input scales: a
# producer that is merely calibrated at x1 (no LN; or a small residual on an already-normalized input,
# the predictor's case) fails at x0.1 / x10. Measured: LN output in bf16 has |mean| <= 4e-4 and
# |var - 1| <= 7e-3; a Linear without LN gives var 0.4-1.3 at x1 and 40-130 at x10.
SCALES = (0.1, 1.0, 10.0)
MEAN_TOL = 0.02
VAR_TOL = 0.05


def _gen(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


@pytest.fixture
def gen() -> torch.Generator:
    """For actions and any other per-test draw."""
    return _gen(1000)


@pytest.fixture
def obs(cfg) -> torch.Tensor:
    """(B, 3, res, res) uint8: exactly what the environment renders (contracts.py Encoder)."""
    res = cfg["env"]["resolution"]
    return torch.randint(0, 256, (B, 3, res, res), dtype=torch.uint8, generator=_gen(1))


def _state(abi, seed: int) -> torch.Tensor:
    """A batch of ABI states: LayerNormed tokens in the activation dtype (abi_v1.yaml state.normalization)."""
    x = torch.randn(B, abi.n_tokens, abi.dim, generator=_gen(seed))
    return F.layer_norm(x, (abi.dim,)).to(abi.dtype)


@pytest.fixture
def W(abi) -> torch.Tensor:
    return _state(abi, 2)


@pytest.fixture
def W_next(abi) -> torch.Tensor:
    return _state(abi, 3)


@pytest.fixture
def per_sample():
    """assert f(*xs)[i] == f(*(x[i:i+1]))[0] for every row i, within bf16 tolerance.

    Measured on a 2-layer 192-dim 4-head transformer with an LN readout on CPU: the worst row deviation
    between a batch of 3 and a batch of 1 is 0.0078 (fp32 compute, bf16 I/O) and 0.031 (bf16 compute);
    batch coupling (a mean over the batch, train-mode BatchNorm) is O(1). Hence 5e-2.
    """

    def _check(f, *xs: torch.Tensor, rtol: float = 5e-2, atol: float = 5e-2) -> None:
        full = f(*xs)
        for i in range(xs[0].shape[0]):
            single = f(*(x[i : i + 1] for x in xs))
            assert torch.allclose(full[i : i + 1].float(), single.float(), rtol=rtol, atol=atol), (
                f"row {i} depends on the rest of the batch"
            )

    return _check


@pytest.fixture
def layernormed_at_all_scales():
    """produce(scale) -> W for the producer's input scaled by `scale`; every token of every W must have
    |mean| < MEAN_TOL and |var - 1| < VAR_TOL. A failure names the scale and the worst token."""

    def _check(produce) -> None:
        for scale in SCALES:
            Wf = produce(scale).float()  # bf16 quantizes a variance of 1.005 to exactly 1.0078
            mean = Wf.mean(dim=-1).abs()
            var = (Wf.var(dim=-1, unbiased=True) - 1).abs()
            for name, dev, tol in (("mean", mean, MEAN_TOL), ("variance", var, VAR_TOL)):
                idx = int(dev.argmax())  # over the flattened (B, n_tokens) map, so idx % n_tokens is the token
                worst = dev.flatten()[idx].item()
                assert worst < tol, f"scale x{scale}: token {idx % dev.shape[1]} {name} off by {worst:.3g}"

    return _check
