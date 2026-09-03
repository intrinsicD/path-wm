"""Predictor family P_phi: build_predictor(cfg) selects the implementation by one YAML line.

What: a plain dispatch table over predictor architectures that exist today. The builder receives the
whole parsed experiment spec and builds on CPU; run.py owns device placement.
Why: E1 trains the reference predictor and E2 freezes it while swapping encoders and adapters (H1,
§5.6), so selecting the consumer must remain a single `predictor.arch` field (CLAUDE.md §3).
"""
from __future__ import annotations

import torch.nn as nn

from predictors.transformer import TransformerPredictor
from world_state.abi import ROOT, load_abi


def _transformer(cfg: dict) -> nn.Module:
    section = cfg["predictor"]
    abi = load_abi(ROOT / cfg["abi"])
    return TransformerPredictor(
        abi=abi,
        dim=int(section["dim"]),
        layers=int(section["layers"]),
        heads=int(section["heads"]),
        mlp_ratio=int(section.get("mlp_ratio", 4)),
        n_registers=int(section["registers"]),
        delta_t_conditioned=bool(section.get("delta_t_conditioned", True)),
    )


PREDICTORS = {"transformer": _transformer}


def build_predictor(cfg: dict) -> nn.Module:
    section = cfg["predictor"]
    arch = section["arch"]
    if arch not in PREDICTORS:
        raise ValueError(f"unknown predictor.arch {arch!r}; known: {sorted(PREDICTORS)}")
    if not section.get("markov", True):
        raise ValueError("the E1 reference predictor is Markov in W; predictor.markov must be true")
    if section.get("output", "delta_layernorm") != "delta_layernorm":
        raise ValueError("the first predictor implements only predictor.output='delta_layernorm'")
    return PREDICTORS[arch](cfg)
