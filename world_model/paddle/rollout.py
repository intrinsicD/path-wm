"""Free-running P-then-U recurrence with no access to actual future targets."""

from torch import Tensor, nn

from .types import PlanningState


def rollout_step(state: PlanningState, action_onehot: Tensor, predictor: nn.Module, updater: nn.Module) -> PlanningState:
    observation = predictor(state.observation, state.memory, action_onehot)
    # Do not use no_grad or detach here: frozen U must transmit later P losses.
    memory = updater(state.memory, observation, action_onehot)
    return PlanningState(observation, memory)


def rollout(state: PlanningState, actions: Tensor, predictor: nn.Module, updater: nn.Module) -> list[PlanningState]:
    if actions.ndim != 3 or actions.shape[0] != state.memory.shape[0] or actions.shape[2] != 3:
        raise ValueError("rollout actions must have shape [B,K,3]")
    imagined = state.clone()
    states = []
    for step in range(actions.shape[1]):
        imagined = rollout_step(imagined, actions[:, step], predictor, updater)
        states.append(imagined)
    return states
