"""Exhaustive five-action search, terminal-prefix scores, and explicit failures."""

from dataclasses import dataclass
import itertools
import math
from typing import Sequence
import warnings

import torch
from torch import Tensor, nn

from .rollout import rollout_step
from .types import ObservationLatent, PlanningState, action_one_hot


Score = tuple[int, int, float, int, int]


class PlanningFailure(RuntimeError):
    """No candidate has finite predicted features, memory, and position readouts."""


@dataclass(frozen=True)
class PlanResult:
    action_id: int
    sequence: tuple[int, ...]
    score: Score
    states: list[PlanningState] | None = None
    invalid_candidates: int = 0


def all_action_sequences() -> tuple[tuple[int, ...], ...]:
    """Reproducible order is stay, left, right, intentionally unlike numeric IDs."""
    return tuple(itertools.product((1, 0, 2), repeat=5))


def sequence_score(positions_world, sequence: Sequence[int]) -> Score:
    """Score a five-step rollout, permitting a shorter prefix ending in a miss.

    Input positions are ball x/y and paddle x in world units. This also supports
    the separately labelled privileged simulator reference without model inputs.
    """
    if len(sequence) != 5 or any(action not in (0, 1, 2) for action in sequence):
        raise ValueError("the baseline planner requires five executable actions")
    positions = torch.as_tensor(positions_world).detach().cpu().tolist()
    if not 1 <= len(positions) <= 5 or any(len(row) != 3 for row in positions):
        raise ValueError("positions must have shape [1..5,3]")
    movement_count, squared_sum, first_miss = 0, 0., 0
    for step, ((x, y, paddle), action) in enumerate(zip(positions, sequence), 1):
        if not all(math.isfinite(value) for value in (x, y, paddle)):
            raise PlanningFailure("non-finite candidate position prediction")
        squared_sum += ((x-paddle)/64)**2
        movement_count += action != 1
        if y >= 61:
            first_miss = step
            break
    if not first_miss and len(positions) != 5:
        raise ValueError("a nonterminal candidate needs all five predicted positions")
    tie = 0
    order = {1: 0, 0: 1, 2: 2}
    for action in sequence:
        tie = 3 * tie + order[action]
    evaluated_steps = first_miss or 5
    return (int(first_miss > 0), 5-first_miss if first_miss else 0, squared_sum/evaluated_steps, movement_count, tie)


class ExhaustivePlanner:
    def __init__(self, predictor: nn.Module, updater: nn.Module, readout: nn.Module, candidate_batch_size: int = 243):
        if not 1 <= candidate_batch_size <= 243:
            raise ValueError("candidate_batch_size must be between 1 and 243")
        self.predictor = predictor
        self.updater = updater
        self.readout = readout
        self.candidate_batch_size = candidate_batch_size
        self.sequences = all_action_sequences()

    @torch.no_grad()
    def plan(self, state: PlanningState, goal_config=None) -> PlanResult:
        if state.memory.shape[0] != 1:
            raise ValueError("plan expects one real planning state")
        if goal_config not in (None, {}):
            raise ValueError("the baseline uses the fixed five-step goal score from the brief")
        best = None
        invalid_count = 0
        for start in range(0, 243, self.candidate_batch_size):
            sequences = self.sequences[start:start + self.candidate_batch_size]
            batch = len(sequences)
            # Materialize copies: neither branch updates nor caller updates share storage.
            imagined = PlanningState(ObservationLatent(
                state.observation.fine.expand(batch, -1, -1).clone(),
                state.observation.coarse.expand(batch, -1, -1).clone(),
            ), state.memory.expand(batch, -1).clone())
            device = state.memory.device
            actions = action_one_hot(torch.tensor(sequences, device=device)).to(state.memory.dtype)
            active = torch.ones(batch, dtype=torch.bool, device=device)
            invalid = torch.zeros_like(active)
            positions = torch.zeros(batch, 5, 3, dtype=state.memory.dtype, device=device)
            states = []
            for step in range(5):
                indices = active.nonzero(as_tuple=True)[0]
                if indices.numel():
                    predicted = rollout_step(imagined[indices], actions[indices, step], self.predictor, self.updater)
                    world = self.readout(predicted.observation) * 64
                    finite = torch.isfinite(world).all(dim=1)
                    finite &= torch.isfinite(predicted.observation.fine).flatten(1).all(1)
                    finite &= torch.isfinite(predicted.observation.coarse).flatten(1).all(1)
                    finite &= torch.isfinite(predicted.memory).all(1)
                    invalid[indices] |= ~finite
                    positions[indices, step] = world
                    # No post-terminal computation can change a candidate's score or state.
                    good_indices = indices[finite]
                    updated = imagined.clone()
                    updated.observation.fine[good_indices] = predicted.observation.fine[finite]
                    updated.observation.coarse[good_indices] = predicted.observation.coarse[finite]
                    updated.memory[good_indices] = predicted.memory[finite]
                    imagined = updated
                    active[indices] = finite & (world[:, 1] < 61)
                states.append(imagined.clone())
            positions_cpu = positions.cpu()
            invalid_cpu = invalid.cpu().tolist()
            invalid_count += sum(invalid_cpu)
            for index, sequence in enumerate(sequences):
                if invalid_cpu[index]:
                    continue
                score = sequence_score(positions_cpu[index], sequence)
                if best is None or score < best[0]:
                    best = (score, sequence, [value[index].clone() for value in states])
        if best is None:
            raise PlanningFailure(f"all 243 candidates have non-finite predictions ({invalid_count} rejected)")
        if invalid_count:
            warnings.warn(f"planner rejected {invalid_count}/243 candidates with non-finite predictions", RuntimeWarning, stacklevel=2)
        score, sequence, states = best
        return PlanResult(sequence[0], sequence, score, states, invalid_count)
