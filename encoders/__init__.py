"""Encoder family E_m: build_encoder(cfg) selects the architecture by one YAML line (DDR §13.6, CLAUDE.md §3).

What: a plain dict over the encoder architectures that exist today; cfg is the whole parsed spec and the
builder reads cfg["encoder"] and cfg["env"]["resolution"]. Builds on CPU; the runner moves modules.
Why: H1 (§4) swaps the encoder against a frozen predictor (E2); the swap must be `encoder.arch` only.
"""
from __future__ import annotations

import torch.nn as nn

from encoders.vit import ViT


def _vit_s8(cfg: dict) -> nn.Module:
    e = cfg["encoder"]
    return ViT(
        dim=int(e["dim"]),
        layers=int(e["layers"]),
        heads=int(e["heads"]),
        resolution=int(cfg["env"]["resolution"]),
        mlp_ratio=int(e.get("mlp_ratio", 4)),  # 4 is the ViT-S constant; a spec may override it
    )


ENCODERS = {"vit_s8": _vit_s8}


def build_encoder(cfg: dict) -> nn.Module:
    arch = cfg["encoder"]["arch"]
    if arch not in ENCODERS:
        raise ValueError(f"unknown encoder.arch {arch!r}; known: {sorted(ENCODERS)}")
    return ENCODERS[arch](cfg)
