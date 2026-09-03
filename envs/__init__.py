"""Environment family: build_env(cfg) selects the engine by one YAML line (DDR §13.6)."""
from __future__ import annotations

from envs.causal_world import CausalWorld
from world_state.abi import ROOT, load_abi


def _causal_world(cfg: dict) -> CausalWorld:
    return CausalWorld(cfg["env"], load_abi(ROOT / cfg["abi"]))


ENVIRONMENTS = {"causal_world": _causal_world}


def build_env(cfg: dict) -> CausalWorld:
    name = cfg["env"]["name"]
    if name not in ENVIRONMENTS:
        raise ValueError(f"unknown env.name {name!r}; known: {sorted(ENVIRONMENTS)}")
    return ENVIRONMENTS[name](cfg)
