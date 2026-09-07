"""Image-goal CEM with primitive absolute targets and independent imagined memory."""

from dataclasses import dataclass
import math

import torch
from torch import Tensor

from world_model.paddle.types import ObservationLatent, PlanningState
from world_model.paddle.rollout import rollout_step
from .models import validate_actions


PLANNER_SCHEMA_VERSION = 'pusht-cem-k5-absolute-block-xy20-anglepi9-v1'


class PlanningFailure(RuntimeError):
    def __init__(self, message, stats=None):
        super().__init__(message)
        self.stats = dict(stats or {})


@dataclass(frozen=True)
class GoalPose:
    """H output prepared from an encoded desired image, separate from rollout."""
    values: Tensor


@dataclass(frozen=True)
class PlanResult:
    action: Tensor
    sequence: Tensor
    cost: float
    stats: dict
    states: list[PlanningState]

    @property
    def invalid_candidates(self):
        return self.stats['invalid_candidates']


def pose_cost(predicted: Tensor, goal: Tensor) -> Tensor:
    """Terminal block-only cost from H's normalized XY and raw sin/cos outputs."""
    if predicted.ndim != 2 or predicted.shape[1] != 6 or goal.shape != (1, 6):
        raise ValueError('predicted pose must be [B,6] and learned goal pose [1,6]')
    goal_norm = torch.linalg.vector_norm(goal[:, 4:6], dim=1)
    if not bool((torch.isfinite(goal).all(1) & torch.isfinite(goal_norm) & (goal_norm >= 1e-6)).all()):
        raise PlanningFailure('invalid learned goal pose or orientation norm')
    norm = torch.linalg.vector_norm(predicted[:, 4:6], dim=1)
    valid = torch.isfinite(predicted).all(1) & torch.isfinite(norm) & (norm >= 1e-6)
    unit = predicted[:, 4:6] / norm.clamp_min(1e-6)[:, None]
    goal_unit = goal[:, 4:6] / goal_norm[:, None]
    theta = torch.atan2(unit[:, 0], unit[:, 1])
    goal_theta = torch.atan2(goal_unit[:, 0], goal_unit[:, 1])
    difference = theta-goal_theta
    angle = torch.atan2(difference.sin(), difference.cos())
    block_xy = (predicted[:, 2:4]-goal[:, 2:4])*512
    costs = block_xy.square().sum(1)/20**2 + angle.square()/(math.pi/9)**2
    return torch.where(valid & torch.isfinite(costs), costs, torch.full_like(costs, float('inf')))


class CEMPlanner:
    def __init__(self, predictor, updater, readout, *, candidates=64, iterations=4,
                 elites=8, horizon=5, seed=3107, initial_std=.2):
        if horizon != 5:
            raise ValueError('PushT baseline horizon is five primitive actions')
        if not isinstance(candidates, int) or not 1 <= elites <= candidates or iterations < 1:
            raise ValueError('CEM requires positive iterations and 1 <= elites <= candidates')
        self.predictor, self.updater, self.readout = predictor, updater, readout
        self.candidates, self.iterations, self.elites = candidates, iterations, elites
        self.horizon, self.seed = horizon, int(seed)
        scale = torch.as_tensor(initial_std, dtype=torch.float64)
        if scale.ndim == 0:
            scale = scale.repeat(2)
        if scale.shape != (2,) or not bool((torch.isfinite(scale) & (scale > 0)).all()):
            raise ValueError('initial_std must be a positive finite scalar or XY pair')
        self.initial_std = tuple(scale.tolist())
        # CPU sampling gives each controller an explicit RNG independent of global
        # model/CUDA streams; sampled executable actions move to the model device.
        self.generator = torch.Generator(device='cpu').manual_seed(self.seed)

    @torch.no_grad()
    def prepare_goal(self, observation: ObservationLatent) -> GoalPose:
        if not isinstance(observation, ObservationLatent) or observation.fine.shape[0] != 1:
            raise ValueError('goal must be one encoded desired image')
        values = self.readout(observation).detach().clone()
        pose_cost(values, values)  # Explicit finite/unit-pair validity before CEM.
        return GoalPose(values)

    def _evaluate(self, state, actions, goal, stats, *, retain_states=False):
        validate_actions(actions)
        batch = actions.shape[0]
        imagined = PlanningState(ObservationLatent(
            state.observation.fine.expand(batch, -1, -1).clone(),
            state.observation.coarse.expand(batch, -1, -1).clone()),
            state.memory.expand(batch, -1).clone())
        valid = torch.ones(batch, dtype=torch.bool, device=state.memory.device)
        selected = []
        for index in range(self.horizon):
            imagined = rollout_step(imagined, actions[:, index], self.predictor, self.updater)
            stats['predictor_transitions'] += batch
            stats['predictor_calls'] += 1
            valid &= torch.isfinite(imagined.observation.fine).flatten(1).all(1)
            valid &= torch.isfinite(imagined.observation.coarse).flatten(1).all(1)
            valid &= torch.isfinite(imagined.memory).all(1)
            if retain_states:
                selected.append(imagined.clone())
        costs = pose_cost(self.readout(imagined.observation), goal.values)
        valid &= torch.isfinite(costs)
        costs = torch.where(valid, costs, torch.full_like(costs, float('inf')))
        stats['invalid_candidates'] += int((~valid).sum())
        stats['candidate_evaluations'] += batch
        return costs, selected

    @torch.no_grad()
    def plan(self, state: PlanningState, goal: GoalPose | ObservationLatent) -> PlanResult:
        if not isinstance(state, PlanningState) or state.memory.shape[0] != 1:
            raise ValueError('CEM expects one real PlanningState')
        if isinstance(goal, ObservationLatent):
            goal = self.prepare_goal(goal)
        if not isinstance(goal, GoalPose):
            raise TypeError('goal must be an encoded image or prepared learned GoalPose')
        goal = GoalPose(goal.values.to(device=state.memory.device, dtype=state.memory.dtype).clone())
        pose_cost(goal.values, goal.values)
        current = self.readout(state.observation)
        if current.shape != (1, 6) or not bool(torch.isfinite(current[:, :2]).all()):
            raise PlanningFailure('invalid current learned pusher position')
        if not all(bool(torch.isfinite(value).all()) for value in
                   (state.observation.fine, state.observation.coarse, state.memory)):
            raise PlanningFailure('invalid current planning state')
        mean = current[0, :2].clamp(0, 1).repeat(self.horizon, 1)
        std = mean.new_tensor(self.initial_std).expand_as(mean).clone()
        stats = {'schema': PLANNER_SCHEMA_VERSION, 'seed': self.seed, 'horizon': self.horizon,
                 'candidates': self.candidates, 'iterations': self.iterations, 'elites': self.elites,
                 'initial_std': list(self.initial_std),
                 'candidate_evaluations': 0, 'predictor_transitions': 0, 'predictor_calls': 0,
                 'invalid_candidates': 0, 'final_mean_evaluated': False, 'iteration_records': []}
        for iteration in range(self.iterations):
            noise = torch.randn((self.candidates, self.horizon, 2), generator=self.generator,
                                dtype=mean.dtype, device='cpu').to(mean.device)
            actions = (mean[None] + std[None]*noise).clamp(0, 1)
            actions[0] = mean
            costs, _ = self._evaluate(state, actions, goal, stats)
            finite = int(torch.isfinite(costs).sum())
            if finite == 0:
                raise PlanningFailure('all CEM candidates are invalid', stats)
            count = min(self.elites, finite)
            indices = torch.argsort(costs, stable=True)[:count]
            elite_actions = actions[indices]
            mean = elite_actions.mean(0)
            std = elite_actions.std(0, correction=0)
            stats['iteration_records'].append({'iteration': iteration, 'valid_candidates': finite,
                                               'elite_count': count, 'best_cost': float(costs[indices[0]])})
        mean = mean.clamp(0, 1)
        stats['final_mean_evaluated'] = True
        final_cost, states = self._evaluate(state, mean[None], goal, stats, retain_states=True)
        if not bool(torch.isfinite(final_cost).all()):
            raise PlanningFailure('final mean candidate is invalid', stats)
        sequence = mean.detach().clone()
        return PlanResult(sequence[0].clone(), sequence, float(final_cost[0]), stats, states)
