"""Loss builders for the experiment that first needs them."""

from losses.e1 import E1Objective
from world_state.abi import ABI


def build_objective(cfg: dict, abi: ABI) -> E1Objective:
    return E1Objective(cfg, abi)


__all__ = ["E1Objective", "build_objective"]
