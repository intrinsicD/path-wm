"""Adapter family A_m: build_adapter(cfg, z_shape) selects the kind by one YAML line (DDR §13.6).

What: a plain dict over the adapter kinds that exist today; cfg is the whole parsed spec, z_shape the
encoder's per-sample output shape (z is "any shape", §16.1, so cfg cannot supply it), the ABI comes
from cfg["abi"]. Builds on CPU; the runner moves modules.
Why: in E2 the adapter is the object under training against the frozen P and I_omega (H1, §5.4).
"""
from __future__ import annotations

import torch.nn as nn

from encoders.adapters.linear import LinearAdapter
from world_state.abi import ROOT, load_abi


def _linear(cfg: dict, z_shape: tuple[int, ...]) -> nn.Module:
    abi = load_abi(ROOT / cfg["abi"])
    return LinearAdapter(tuple(int(s) for s in z_shape), abi.n_tokens, abi.dim, abi.dtype)


ADAPTERS = {"linear": _linear}


def build_adapter(cfg: dict, z_shape: tuple[int, ...]) -> nn.Module:
    kind = cfg["adapter"]["kind"]
    if kind not in ADAPTERS:
        raise ValueError(f"unknown adapter.kind {kind!r}; known: {sorted(ADAPTERS)}")
    return ADAPTERS[kind](cfg, z_shape)
