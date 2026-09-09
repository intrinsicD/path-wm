"""Finite-horizon planning for explicit latent states and bounded actions."""

from dataclasses import dataclass
import math

import torch

from pathwm.io import evaluation_mode


@dataclass(frozen=True)
class Plan:
    indices: torch.Tensor
    actions: torch.Tensor  # [B,H,A], best full sequences
    scores: torch.Tensor  # [B,C]

    @property
    def first_action(self):
        return self.actions[:, 0]


@torch.no_grad()
def plan(
    model,
    state,
    candidates,
    cost,
    *,
    lower,
    upper,
    dt=1.0,
    discount=1.0,
    uncertainty_weight=0.0,
    samples=None,
    trace=None,
):
    """Score cost(next_state)->[B] at each step; lower scores are better.

    Mean rollouts by default; samples>1 averages stochastic rollouts. A conditional
    latent-variance penalty is optional and is not claimed to measure epistemic
    uncertainty. Candidate generation, goals and action units belong to the recipe.
    """
    if (
        candidates.ndim != 4
        or candidates.shape[0] != len(state.tokens)
        or min(candidates.shape[1:3]) < 1
    ):
        raise ValueError("Candidates must be nonempty [B,C,H,A]")
    if (
        candidates.shape[-1] != model.action_width
        or not torch.isfinite(candidates).all()
    ):
        raise ValueError("Candidate actions must be finite with the declared width")
    lo = torch.as_tensor(lower, device=candidates.device, dtype=candidates.dtype)
    hi = torch.as_tensor(upper, device=candidates.device, dtype=candidates.dtype)
    if (
        lo.ndim > 1
        or hi.ndim > 1
        or lo.numel() not in (1, model.action_width)
        or hi.numel() not in (1, model.action_width)
        or not torch.isfinite(lo).all()
        or not torch.isfinite(hi).all()
        or (lo >= hi).any()
        or (candidates < lo).any()
        or (candidates > hi).any()
    ):
        raise ValueError("Candidate actions violate declared bounds")
    categorical = hasattr(model, "sample_beliefs")
    samples = (4 if categorical else 1) if samples is None else samples
    if categorical and uncertainty_weight:
        raise ValueError(
            "Gaussian variance penalty is undefined for categorical beliefs"
        )
    if (
        not 0 < discount <= 1
        or not math.isfinite(uncertainty_weight)
        or uncertainty_weight < 0
        or not isinstance(samples, int)
        or samples < 1
    ):
        raise ValueError(
            "Invalid planning discount, uncertainty weight or sample count"
        )
    scores = []
    with evaluation_mode(model):
        starts = (
            model.sample_beliefs(state, samples) if categorical else [state] * samples
        )
        # Common future noise makes identical action sequences receive identical scores.
        cpu_rng = torch.get_rng_state()
        cuda_rng = (
            torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
        )
        for c in range(candidates.shape[1]):
            if categorical:
                torch.set_rng_state(cpu_rng)
                if cuda_rng is not None:
                    torch.cuda.set_rng_state_all(cuda_rng)
            total = candidates.new_zeros(len(candidates))
            for initial in starts:
                branch = initial
                for h in range(candidates.shape[2]):
                    branch = model.imagine(
                        branch,
                        candidates[:, c, h],
                        dt=dt,
                        sample=categorical or samples > 1,
                    )
                    value = cost(branch)
                    if (
                        value.shape != (len(candidates),)
                        or not torch.isfinite(value).all()
                    ):
                        raise ValueError("Planner cost must return finite [B] values")
                    if (
                        not torch.isfinite(branch.tokens).all()
                        or not torch.isfinite(branch.log_scale).all()
                    ):
                        raise ValueError("Planner produced a nonfinite latent state")
                    penalty = (
                        0.0
                        if categorical
                        else branch.log_scale.mul(2).exp().mean((1, 2))
                    )
                    total += (
                        discount**h * (value + uncertainty_weight * penalty) / samples
                    )
            scores.append(total)
    scores = torch.stack(scores, 1)
    if not torch.isfinite(scores).all():
        raise ValueError("Planner accumulated nonfinite scores")
    best = scores.argmin(1)
    actions = candidates[
        torch.arange(len(candidates), device=candidates.device), best
    ].clone()
    if trace is not None:
        trace["plan.scores"] = scores.detach().cpu().clone()
        trace["plan.selected"] = best.detach().cpu().clone()
        trace["plan.candidates"] = candidates.detach().cpu().clone()
    return Plan(best, actions, scores)
