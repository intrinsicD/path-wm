"""Adapter family A_m: build_adapter(cfg, z_shape) selects the kind by one YAML line (DDR §13.6).

What: a plain dict over the adapter kinds that exist today; cfg is the whole parsed spec, z_shape the
encoder's per-sample output shape (z is "any shape", §16.1, so cfg cannot supply it), the ABI comes
from cfg["abi"]. Builds on CPU; the runner moves modules.
Why: in E2 the adapter is the object under training against the frozen P and I_omega (H1, §5.4).
"""
from __future__ import annotations

import torch.nn as nn

from encoders.adapters.evidence import EvidenceProjection
from encoders.adapters.linear import LinearAdapter
from world_state.abi import ROOT, load_abi
from world_state.abi_v2 import load_abi_v2


def _linear(cfg: dict, z_shape: tuple[int, ...]) -> nn.Module:
    abi = load_abi(ROOT / cfg["abi"])
    return LinearAdapter(tuple(int(s) for s in z_shape), abi.n_tokens, abi.dim, abi.dtype)


ADAPTERS = {"linear": _linear}


def build_adapter(cfg: dict, z_shape: tuple[int, ...]) -> nn.Module:
    kind = cfg["adapter"]["kind"]
    if kind not in ADAPTERS:
        raise ValueError(f"unknown adapter.kind {kind!r}; known: {sorted(ADAPTERS)}")
    return ADAPTERS[kind](cfg, z_shape)


def build_evidence_adapter(cfg: dict, modality: str, input_dim: int) -> nn.Module:
    """Build the ABI-v2 projection belonging to one enabled evidence encoder."""
    modalities = cfg.get("modalities", {})
    if modality not in modalities or not modalities[modality].get("enabled", False):
        raise ValueError(f"modality {modality!r} is absent or disabled")
    kind = modalities[modality]["adapter"]["kind"]
    if kind != "evidence_projection":
        raise ValueError("the common-base slice implements adapter.kind='evidence_projection'")
    abi = load_abi_v2(ROOT / cfg["abi"])
    return EvidenceProjection(modality, input_dim, abi)
