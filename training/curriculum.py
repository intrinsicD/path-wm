"""Held-out gate controller and EMA target utility for the common-base curriculum.

What: the ordered R0 -> R1 -> B0 -> D0 -> D1 -> P0 stages, their trainable module groups, generic
metric comparisons, and a no-gradient exponential-moving-average teacher.
How: transitions fail closed when a metric is absent or violates its configured comparison. No step
counter can bypass a gate; the caller records the returned failures in the run ledger.
Why: the E1-a audit showed that merely delaying a loss can hide a broken bootstrap. Capability stages
must start only when the prerequisite representation/belief/dynamics behaviour is measured.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn as nn


_TRANSITION_GATES = {
    "representation_unimodal": "unimodal_representation_ready",
    "representation_av": "audiovisual_representation_ready",
    "belief_bootstrap": "belief_ready",
    "dynamics_one_step": "one_step_dynamics_ready",
    "dynamics_rollout": "rollout_dynamics_ready",
}

_TRAINABLE_GROUPS = {
    "representation_unimodal": frozenset({"encoders", "evidence_adapters", "representation_heads"}),
    "representation_av": frozenset({"encoders", "evidence_adapters", "representation_heads"}),
    "belief_bootstrap": frozenset({"evidence_adapters", "updater"}),
    "dynamics_one_step": frozenset({"action_adapter", "updater", "predictor"}),
    "dynamics_rollout": frozenset({"action_adapter", "updater", "predictor"}),
    "planning": frozenset({"planner", "critic"}),
}


@dataclass(frozen=True)
class GateResult:
    passed: bool
    gate: str
    failures: tuple[str, ...]


def trainable_groups(stage: str) -> frozenset[str]:
    if stage not in _TRAINABLE_GROUPS:
        raise ValueError(f"unknown curriculum stage {stage!r}")
    return _TRAINABLE_GROUPS[stage]


def _compare(actual: float, operator: str, target: float) -> bool:
    if operator == "greater_equal":
        return actual >= target
    if operator == "greater":
        return actual > target
    if operator == "less_equal":
        return actual <= target
    if operator == "less":
        return actual < target
    if operator == "equal":
        return actual == target
    raise ValueError(f"unknown gate comparison {operator!r}")


class CommonBaseCurriculum:
    def __init__(self, stages: tuple[str, ...], gates: Mapping[str, Mapping[str, Mapping[str, float | str]]]):
        if len(stages) < 2 or len(set(stages)) != len(stages):
            raise ValueError("curriculum needs at least two unique ordered stages")
        unknown = set(stages) - set(_TRAINABLE_GROUPS)
        if unknown:
            raise ValueError(f"unknown curriculum stages {sorted(unknown)}")
        if stages[-1] != "planning":
            raise ValueError("the common-base curriculum must end with planning")
        self.stages = stages
        self.transition_gates = {
            stage: _TRANSITION_GATES[stage]
            for stage in stages[:-1]
        }
        missing = set(self.transition_gates.values()) - set(gates)
        if missing:
            raise ValueError(f"curriculum is missing transition gates {sorted(missing)}")
        self.gates = {name: dict(conditions) for name, conditions in gates.items()}

    @classmethod
    def from_config(cls, cfg: Mapping) -> "CommonBaseCurriculum":
        return cls(tuple(str(stage) for stage in cfg["stages"]), cfg["gates"])

    def evaluate(self, stage: str, metrics: Mapping[str, float | int]) -> GateResult:
        if stage not in self.transition_gates:
            raise ValueError(f"stage {stage!r} has no outgoing transition")
        gate_name = self.transition_gates[stage]
        failures: list[str] = []
        for metric, condition in self.gates[gate_name].items():
            if metric not in metrics:
                failures.append(f"missing metric {metric}")
                continue
            actual = float(metrics[metric])
            target = float(condition["value"])
            operator = str(condition["op"])
            if not math.isfinite(actual):
                failures.append(f"{metric} is non-finite")
            elif not _compare(actual, operator, target):
                failures.append(f"{metric}={actual:g} does not satisfy {operator} {target:g}")
        return GateResult(not failures, gate_name, tuple(failures))


class EMATeacher(nn.Module):
    """Frozen copy of one online representation module, updated only by exponential averaging."""

    def __init__(self, online: nn.Module, decay: float) -> None:
        super().__init__()
        if not 0.0 <= decay < 1.0 or not math.isfinite(decay):
            raise ValueError("EMA decay must be finite in [0,1)")
        self.decay = float(decay)
        self.module = copy.deepcopy(online)
        self.module.requires_grad_(False)
        self.module.eval()

    @torch.no_grad()
    def update(self, online: nn.Module) -> None:
        online_state = online.state_dict()
        teacher_state = self.module.state_dict()
        if online_state.keys() != teacher_state.keys():
            raise ValueError("online and EMA teacher state layouts differ")
        for name, target in teacher_state.items():
            source = online_state[name].detach().to(device=target.device)
            if torch.is_floating_point(target):
                target.mul_(self.decay).add_(source, alpha=1.0 - self.decay)
            else:
                target.copy_(source)

    def train(self, mode: bool = True) -> "EMATeacher":
        super().train(False)
        self.module.eval()
        return self

    @torch.no_grad()
    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
