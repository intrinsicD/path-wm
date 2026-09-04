"""The staged E1 objective and horizon curriculum (§6.1-§6.4, §6.9).

What: SIGReg plus inverse dynamics are always active; by default the action term starts at k=1, then
free-running and one uniformly sampled variable-length chunk join after stage 0 as the trained horizon
grows; paired counterfactual InfoNCE and its optional paired positive MSE join at stage 2. Development
ablations can move dynamics or counterfactual onset among the same three explicit stage boundaries.
Why: this is the smallest objective that prevents collapse, anchors actions, and trains both iterative
and chunked prediction without allowing labels or environment ground truth into W (Invariant 11).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from losses.counterfactual import counterfactual_loss
from losses.inverse import inverse_loss
from losses.rollout import rollout_losses
from losses.sigreg import SIGReg
from world_state.abi import ABI


def curriculum_horizon(step: int, total_steps: int, max_horizon: int, stage0: float, growth: float) -> int:
    if total_steps < 1 or max_horizon < 1:
        raise ValueError("total_steps and max_horizon must be positive")
    stage0_steps = max(1, int(total_steps * stage0))
    growth_steps = max(1, int(total_steps * growth))
    if step < stage0_steps or max_horizon == 1:
        return 1
    progress = min(1.0, (step - stage0_steps + 1) / growth_steps)
    return min(max_horizon, 1 + math.floor(progress * (max_horizon - 1)))


def curriculum_stage_start(stage: int, total_steps: int, stage0: float, growth: float) -> int:
    """Return the first optimizer-step index for curriculum stages 0, 1, or 2."""
    if total_steps < 1:
        raise ValueError("total_steps must be positive")
    if stage == 0:
        return 0
    if stage == 1:
        return max(1, int(total_steps * stage0))
    if stage == 2:
        return max(1, int(total_steps * (stage0 + growth)))
    raise ValueError(f"curriculum stage must be 0, 1, or 2, got {stage}")


class E1Objective(nn.Module):
    def __init__(self, cfg: dict, abi: ABI) -> None:
        super().__init__()
        losses = cfg["losses"]
        reg = losses["reg"]
        if reg["kind"] != "sigreg":
            raise ValueError("the first E1 objective implements only losses.reg.kind='sigreg'")
        self.sigreg = SIGReg(
            dim=abi.dim,
            projections=int(reg["projections"]),
            knots=int(reg["knots"]),
            per_token=bool(reg["per_token"]),
        )
        self.reg_weight = float(reg["weight"])
        self.action_weight = float(losses["action"]["weight"])
        self.inverse_weight = float(losses["inverse"]["weight"])
        rollout = losses["rollout"]
        self.gamma = float(rollout["gamma"])
        self.max_horizon = int(rollout["h_train"])
        self.delta_t_max = int(rollout["delta_t_max"])
        self.free_running = bool(rollout["free_running"])
        self.stage0_fraction = float(cfg["curriculum"]["stage0_fraction"])
        self.growth_fraction = float(cfg["curriculum"]["horizon_growth_fraction"])
        self.dynamics_from_stage = int(cfg["curriculum"].get("dynamics_from_stage", 0))
        counterfactual = losses["counterfactual"]
        self.counterfactual_weight = float(counterfactual["weight"])
        self.counterfactual_positive_weight = float(counterfactual.get("positive_weight", 0.0))
        self.counterfactual_context_gradient = bool(counterfactual.get("context_gradient", True))
        self.counterfactual_k = int(counterfactual["k"])
        self.counterfactual_kappa = float(counterfactual["kappa"])
        self.counterfactual_from_stage = int(cfg["curriculum"]["counterfactual_from_stage"])
        # Validate configurable development ablations while retaining stage 0 / stage 2 as defaults.
        curriculum_stage_start(
            self.dynamics_from_stage,
            1_000,
            self.stage0_fraction,
            self.growth_fraction,
        )
        curriculum_stage_start(
            self.counterfactual_from_stage,
            1_000,
            self.stage0_fraction,
            self.growth_fraction,
        )

    def dynamics_active(self, step: int, total_steps: int) -> bool:
        start = curriculum_stage_start(
            self.dynamics_from_stage,
            total_steps,
            self.stage0_fraction,
            self.growth_fraction,
        )
        return step >= start

    def counterfactual_active(self, step: int, total_steps: int) -> bool:
        start = curriculum_stage_start(
            self.counterfactual_from_stage,
            total_steps,
            self.stage0_fraction,
            self.growth_fraction,
        )
        enabled = self.counterfactual_weight != 0.0 or self.counterfactual_positive_weight != 0.0
        return enabled and step >= start

    def forward(
        self,
        predictor,
        inverse,
        states: torch.Tensor,
        actions: torch.Tensor,
        *,
        step: int,
        total_steps: int,
        generator: torch.Generator | None = None,
        counterfactual_states: torch.Tensor | None = None,
        counterfactual_actions: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | int | None]:
        horizon = curriculum_horizon(
            step, total_steps, self.max_horizon, self.stage0_fraction, self.growth_fraction
        )
        if actions.shape[1] < horizon:
            raise ValueError(f"objective needs {horizon} action steps at training step {step}, got {actions.shape[1]}")
        stage0_steps = max(1, int(total_steps * self.stage0_fraction))
        if not self.free_running:
            horizon = 1
        chunk_delta_t = None
        dynamics_active = self.dynamics_active(step, total_steps)
        if dynamics_active and step >= stage0_steps:
            largest_chunk = min(self.delta_t_max, horizon)
            chunk_delta_t = int(torch.randint(1, largest_chunk + 1, (), generator=generator).item())

        zero = torch.zeros((), device=states.device, dtype=torch.float32)
        if dynamics_active:
            dynamics = rollout_losses(
                predictor,
                states,
                actions,
                gamma=self.gamma,
                horizon=horizon,
                action_weight=self.action_weight,
                chunk_delta_t=chunk_delta_t,
            )
        else:
            dynamics = {"total": zero, "action": zero, "rollout": zero, "chunk": zero}
        regularizer = self.sigreg(states[:, : horizon + 1].flatten(0, 1)) if self.reg_weight else zero
        inverse_value = inverse_loss(inverse, states, actions, horizon) if self.inverse_weight else zero
        counterfactual_value = zero
        counterfactual_positive = zero
        counterfactual_accuracy = None
        if self.counterfactual_active(step, total_steps):
            if counterfactual_states is None or counterfactual_actions is None:
                raise ValueError("stage-2 counterfactual loss requires paired states and actions")
            if counterfactual_states.shape[1] != self.counterfactual_k + 1:
                raise ValueError(
                    f"counterfactual states have {counterfactual_states.shape[1] - 1} branches, "
                    f"expected {self.counterfactual_k}"
                )
            counterfactual = counterfactual_loss(
                predictor,
                counterfactual_states[:, 0],
                counterfactual_states[:, 1:],
                counterfactual_actions,
                kappa=self.counterfactual_kappa,
            )
            counterfactual_value = counterfactual["loss"]
            counterfactual_positive = counterfactual["positive_mse"]
            counterfactual_accuracy = counterfactual["accuracy"]
        total = (
            dynamics["total"]
            + self.reg_weight * regularizer
            + self.inverse_weight * inverse_value
            + self.counterfactual_weight * counterfactual_value
            + self.counterfactual_positive_weight * counterfactual_positive
        )
        return {
            "total": total,
            "reg": regularizer,
            "inverse": inverse_value,
            "action": dynamics["action"],
            "rollout": dynamics["rollout"],
            "chunk": dynamics["chunk"],
            "counterfactual": counterfactual_value,
            "counterfactual_positive": counterfactual_positive,
            "counterfactual_accuracy": counterfactual_accuracy,
            "horizon": horizon,
            "delta_t": chunk_delta_t,
        }
