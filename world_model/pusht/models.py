"""PushT E/U/P adaptation: absolute XY actions and pose/observable-motion heads."""

import torch
from torch import Tensor, nn

from world_model.paddle.models import Encoder, Decoder, Attention, TransformerBlock
from world_model.paddle.types import ObservationLatent, PlanningState
from world_model.paddle.rollout import rollout_step


MODEL_SCHEMA_VERSION = 'pusht-eup-v1-rgb64-fine256-coarse64-width64-memory128-action2-pose6-motion5'
ACTION_SCHEMA_VERSION = 'pusht-absolute-xy512-start-minus1-v1'


def initial_previous_action(batch: int, device=None, dtype=torch.float32) -> Tensor:
    """Only the first observer input receives this non-executable marker."""
    return torch.full((batch, 2), -1., device=device, dtype=dtype)


def validate_actions(actions: Tensor, *, previous: bool = False) -> None:
    """Validate normalized absolute targets; only U accepts complete start rows."""
    kind = 'previous action' if previous else 'executable action'
    if not isinstance(actions, Tensor) or actions.ndim < 2 or actions.shape[-1] != 2:
        raise ValueError(f'{kind} must have final width two')
    executable = torch.isfinite(actions).all(-1) & ((actions >= 0) & (actions <= 1)).all(-1)
    allowed = executable | (actions == -1).all(-1) if previous else executable
    if not bool(allowed.all()):
        raise ValueError(f'{kind} must be finite absolute XY in [0,1]' + (' or initial (-1,-1)' if previous else ''))


class MemoryUpdater(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = Attention(query_width=128)
        self.gru = nn.GRUCell(input_size=66, hidden_size=128)

    def forward(self, memory_before: Tensor, observation: ObservationLatent, previous_action: Tensor) -> Tensor:
        validate_actions(previous_action, previous=True)
        if previous_action.shape != (memory_before.shape[0], 2):
            raise ValueError('previous action must have shape [B,2]')
        tokens = observation.tokens()
        read = self.attention(memory_before.unsqueeze(1), tokens, tokens).squeeze(1)
        return self.gru(torch.cat((read, previous_action), -1), memory_before)


class Predictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.action_projection = nn.Linear(2, 64)
        self.role_embeddings = nn.Parameter(torch.empty(3, 64))
        nn.init.normal_(self.role_embeddings, std=.02)
        self.blocks = nn.ModuleList([TransformerBlock(), TransformerBlock()])
        self.output_projection = nn.Linear(64, 64)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, observation: ObservationLatent, memory_after: Tensor, candidate_action: Tensor) -> ObservationLatent:
        validate_actions(candidate_action)
        if candidate_action.shape != (memory_after.shape[0], 2):
            raise ValueError('executable action must have shape [B,2]')
        original = observation.tokens()
        memory_tokens = memory_after.reshape(memory_after.shape[0], 2, 64) + self.role_embeddings[:2]
        action_token = self.action_projection(candidate_action).unsqueeze(1) + self.role_embeddings[2]
        tokens = torch.cat((original, memory_tokens, action_token), 1)
        for block in self.blocks:
            tokens = block(tokens)
        return ObservationLatent.from_tokens(original + self.output_projection(tokens[:, :320]))


class PoseReadout(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(20480, 6)

    def forward(self, observation: ObservationLatent) -> Tensor:
        return self.linear(observation.tokens().flatten(1))


class StateReadout(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(128, 11)

    def forward(self, memory_after: Tensor) -> Tensor:
        return self.linear(memory_after)


def rollout(state: PlanningState, actions: Tensor, predictor: nn.Module, updater: nn.Module) -> list[PlanningState]:
    """Causal imagined P→U recurrence; frozen U retains activation derivatives."""
    validate_actions(actions)
    if actions.ndim != 3 or actions.shape[0] != state.memory.shape[0]:
        raise ValueError('rollout actions must have shape [B,K,2]')
    imagined = state.clone()
    states = []
    for index in range(actions.shape[1]):
        imagined = rollout_step(imagined, actions[:, index], predictor, updater)
        states.append(imagined)
    return states
