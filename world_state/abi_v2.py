"""Typed view of the modality-neutral world-state ABI v2.

What: the persistent state is a fixed set of latent belief slots; sensory evidence and action
conditions are variable-length, timestamped token streams with their own boundary checks.
How: load_abi_v2 reads docs/abi/abi_v2.yaml.  This file contains layout and validation only; model
implementations live behind the contracts in contracts.py.
Why: video grids and audio time-frequency patches do not share physical token coordinates.  Forcing
both directly into ABI v1's 8x8 visual grid would make the predictor depend on a sensor topology.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPEC = ROOT / "docs" / "abi" / "abi_v2.yaml"

_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


class ABIv2Error(ValueError):
    """A state, evidence, action, or time tensor violates ABI v2."""


@dataclass(frozen=True)
class ABIv2:
    version: int
    slot_tokens: int
    global_tokens: int
    token_order: tuple[str, ...]
    dim: int
    dtype: torch.dtype
    stats_dtype: torch.dtype
    state_positions: str
    per_token_affine: bool
    evidence_dim: int
    evidence_dtype: torch.dtype
    time_unit: str
    timestamp_reference: str
    action_dim: int
    max_action_tokens: int
    register_options: tuple[int, ...]

    @property
    def n_tokens(self) -> int:
        return self.global_tokens + self.slot_tokens

    def check_state(self, W: torch.Tensor) -> None:
        if W.ndim != 3 or tuple(W.shape[1:]) != (self.n_tokens, self.dim):
            raise ABIv2Error(f"state shape {tuple(W.shape)} != (B, {self.n_tokens}, {self.dim})")
        if W.dtype != self.dtype:
            raise ABIv2Error(f"state dtype {W.dtype} != {self.dtype}")
        if not torch.isfinite(W).all():
            raise ABIv2Error("state contains non-finite values")

    def check_evidence(
        self,
        tokens: torch.Tensor,
        timestamps: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> None:
        self._check_token_stream(
            "evidence", tokens, timestamps, valid_mask, self.evidence_dim, self.evidence_dtype
        )

    def check_action_tokens(
        self,
        tokens: torch.Tensor,
        timestamps: torch.Tensor,
        valid_mask: torch.Tensor,
        observed_mask: torch.Tensor,
    ) -> None:
        self._check_token_stream("actions", tokens, timestamps, valid_mask, self.action_dim, self.dtype)
        if observed_mask.shape != valid_mask.shape or observed_mask.dtype != torch.bool:
            raise ABIv2Error(
                f"action observed_mask must be bool {tuple(valid_mask.shape)}, got "
                f"{tuple(observed_mask.shape)} {observed_mask.dtype}"
            )
        if tokens.shape[1] > self.max_action_tokens:
            raise ABIv2Error(
                f"action token count {tokens.shape[1]} exceeds max {self.max_action_tokens}"
            )

    @staticmethod
    def _check_token_stream(
        name: str,
        tokens: torch.Tensor,
        timestamps: torch.Tensor,
        valid_mask: torch.Tensor,
        width: int,
        dtype: torch.dtype,
    ) -> None:
        if tokens.ndim != 3 or tokens.shape[-1] != width:
            raise ABIv2Error(f"{name} shape {tuple(tokens.shape)} != (B, N, {width})")
        expected = tuple(tokens.shape[:2])
        if tuple(timestamps.shape) != expected or not torch.is_floating_point(timestamps):
            raise ABIv2Error(
                f"{name} timestamps must be floating {expected}, got {tuple(timestamps.shape)} {timestamps.dtype}"
            )
        if tuple(valid_mask.shape) != expected or valid_mask.dtype != torch.bool:
            raise ABIv2Error(
                f"{name} valid_mask must be bool {expected}, got {tuple(valid_mask.shape)} {valid_mask.dtype}"
            )
        if tokens.dtype != dtype:
            raise ABIv2Error(f"{name} dtype {tokens.dtype} != {dtype}")
        if not torch.isfinite(tokens).all() or not torch.isfinite(timestamps).all():
            raise ABIv2Error(f"{name} contains non-finite values")


def load_abi_v2(path: Path | str = DEFAULT_SPEC) -> ABIv2:
    spec = yaml.safe_load(Path(path).read_text())
    if int(spec["abi_version"]) != 2:
        raise ABIv2Error(f"load_abi_v2 requires abi_version 2, got {spec['abi_version']}")
    state = spec["state"]
    evidence = spec["evidence"]
    actions = spec["action_condition"]
    registers = spec["registers"]
    return ABIv2(
        version=2,
        slot_tokens=int(state["slot_tokens"]),
        global_tokens=int(state["global_tokens"]),
        token_order=tuple(str(item) for item in state["token_order"]),
        dim=int(state["dim"]),
        dtype=_DTYPES[state["dtype_activations"]],
        stats_dtype=_DTYPES[state["dtype_regularizer_stats"]],
        state_positions=str(state["positions"]["kind"]),
        per_token_affine=bool(state["normalization"]["per_token_affine"]),
        evidence_dim=int(evidence["dim"]),
        evidence_dtype=_DTYPES[evidence["dtype_activations"]],
        time_unit=str(evidence["timestamps"]["unit"]),
        timestamp_reference=str(evidence["timestamps"]["reference"]),
        action_dim=int(actions["token_dim"]),
        max_action_tokens=int(actions["max_tokens"]),
        register_options=tuple(int(value) for value in registers["count_options"]),
    )
