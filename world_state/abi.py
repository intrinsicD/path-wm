"""ABI v1 layout as code: the one place that reads docs/abi/abi_v1.yaml.

What: an immutable dataclass with the fields every producer (adapter) and
consumer (predictor, inverse dynamics, updater) must agree on, plus the checks
the structural conformance tests run.
How: load_abi() parses the YAML; nothing here hardcodes a number, so code and
spec cannot drift. check_state / check_actions / check_registers raise ABIError
naming the offending field.
Why: H1 is a claim about a frozen consumer on a foreign producer, and the ABI
is what makes "foreign" well-defined (§5.2). A breaking change is a new major
version, never an in-place edit (abi_v1.yaml header, CLAUDE.md §3).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPEC = ROOT / "docs" / "abi" / "abi_v1.yaml"

_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


class ABIError(ValueError):
    """A tensor does not match the ABI layout."""


@dataclass(frozen=True)
class ABI:
    version: int
    grid_tokens: int
    global_tokens: int
    dim: int
    dtype: torch.dtype              # activations
    stats_dtype: torch.dtype        # regularizer (SIGReg) statistics
    positions: str                  # rope2d | sinusoidal; fixed and identical for every adapter
    register_options: tuple[int, ...]
    action_dims: int
    action_range: tuple[float, float]
    max_chunk: int

    @property
    def n_tokens(self) -> int:
        return self.grid_tokens + self.global_tokens

    def check_state(self, W: torch.Tensor) -> None:
        """W must be (B, n_tokens, dim) in the activation dtype."""
        if W.ndim != 3 or tuple(W.shape[1:]) != (self.n_tokens, self.dim):
            raise ABIError(f"state shape {tuple(W.shape)} != (B, {self.n_tokens}, {self.dim})")
        if W.dtype != self.dtype:
            raise ABIError(f"state dtype {W.dtype} != {self.dtype}")

    def check_actions(self, actions: torch.Tensor) -> None:
        """actions must be (B, action_dims) or a chunk (B, k, action_dims) with 1 <= k <= max_chunk,
        values within action_range. The 2-D form is what Environment.step takes and
        InverseDynamics.infer_action returns; the 3-D form is what Predictor.predict takes."""
        if actions.ndim not in (2, 3) or actions.shape[-1] != self.action_dims:
            raise ABIError(f"actions shape {tuple(actions.shape)} != (B, [k,] {self.action_dims})")
        if actions.ndim == 3 and not 1 <= actions.shape[1] <= self.max_chunk:
            raise ABIError(f"action chunk length {actions.shape[1]} not in [1, {self.max_chunk}]")
        lo, hi = self.action_range
        if actions.numel() and (actions.min() < lo or actions.max() > hi):
            raise ABIError(f"action values outside [{lo}, {hi}]")

    def check_registers(self, n_registers: int) -> None:
        if n_registers not in self.register_options:
            raise ABIError(f"register count {n_registers} not in {self.register_options}")


def load_abi(path: Path | str = DEFAULT_SPEC) -> ABI:
    """Parse the ABI spec. The YAML is the source of truth; this is its typed view."""
    spec = yaml.safe_load(Path(path).read_text())
    s, r, a, dt = spec["state"], spec["registers"], spec["actions"], spec["delta_t"]
    return ABI(
        version=int(spec["abi_version"]),
        grid_tokens=int(s["grid_tokens"]),
        global_tokens=int(s["global_tokens"]),
        dim=int(s["dim"]),
        dtype=_DTYPES[s["dtype_activations"]],
        stats_dtype=_DTYPES[s["dtype_regularizer_stats"]],
        positions=str(s["positions"]["kind"]),
        register_options=tuple(int(n) for n in r["count_options"]),
        action_dims=int(a["dims"]),
        action_range=(float(a["range"][0]), float(a["range"][1])),
        max_chunk=int(dt["max_chunk"]),
    )
