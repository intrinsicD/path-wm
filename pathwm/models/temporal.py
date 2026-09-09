"""Causal state update and imagined transitions; persistent tensors belong to callers."""

import torch
from torch import nn
from .blocks import Attention, TransformerBlock
from .features import tokens, grids


def shared_width(spec):
    widths = {s.channels for s in spec.values()}
    if len(widths) != 1:
        raise ValueError(
            "Temporal inputs need a shared width; use ProjectFeatures explicitly"
        )
    return next(iter(widths))


class MemoryUpdater(nn.Module):
    def __init__(self, spec, action_width=2, memory_width=128):
        super().__init__()
        self.spec, self.action_width, self.memory_width = (
            dict(spec),
            action_width,
            memory_width,
        )
        width = shared_width(spec)
        self.attention = Attention(width, query_width=memory_width)
        self.gru = nn.GRUCell(width + action_width, memory_width)

    def forward(self, memory_before, features, previous_action):
        if memory_before.ndim != 2 or memory_before.shape[1] != self.memory_width:
            raise ValueError("Memory shape differs from MemoryUpdater.memory_width")
        if previous_action.shape != (len(memory_before), self.action_width):
            raise ValueError("Previous action shape differs from declared action_width")
        read = self.attention(
            memory_before.unsqueeze(1),
            tokens(features, self.spec),
            tokens(features, self.spec),
        ).squeeze(1)
        return self.gru(torch.cat((read, previous_action), -1), memory_before)


class Predictor(nn.Module):
    def __init__(self, spec, action_width=2, memory_width=128, depth=2):
        super().__init__()
        self.spec, self.action_width, self.memory_width = (
            dict(spec),
            action_width,
            memory_width,
        )
        self.width = shared_width(spec)
        self.memory_projection = (
            nn.Identity()
            if memory_width == 2 * self.width
            else nn.Linear(memory_width, 2 * self.width)
        )
        self.action_projection = nn.Linear(action_width, self.width)
        self.role_embeddings = nn.Parameter(torch.empty(3, self.width))
        nn.init.normal_(self.role_embeddings, std=0.02)
        self.blocks = nn.ModuleList(
            [TransformerBlock(self.width) for _ in range(depth)]
        )
        self.output_projection = nn.Linear(self.width, self.width)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, features, memory_after, candidate_action):
        original = tokens(features, self.spec)
        if candidate_action.shape != (
            len(original),
            self.action_width,
        ) or memory_after.shape != (len(original), self.memory_width):
            raise ValueError(
                "Candidate action or memory shape differs from predictor contract"
            )
        memory = (
            self.memory_projection(memory_after).reshape(-1, 2, self.width)
            + self.role_embeddings[:2]
        )
        action = (
            self.action_projection(candidate_action).unsqueeze(1)
            + self.role_embeddings[2]
        )
        x = torch.cat((original, memory, action), 1)
        for block in self.blocks:
            x = block(x)
        return grids(
            original + self.output_projection(x[:, : original.shape[1]]), self.spec
        )


def observe(encoder, updater, history, history_actions, initial_previous_action):
    """Observe images[0:H] using actions[0:H-1]. Final candidate action is absent."""
    if history.ndim != 5 or history.shape[1] < 1:
        raise ValueError("History must be [B,H,C,Y,X] with H >= 1")
    if history_actions.shape != (
        len(history),
        history.shape[1] - 1,
        updater.action_width,
    ):
        raise ValueError(
            "History actions must connect the H observed frames, exactly H-1"
        )
    memory = history.new_zeros(len(history), updater.memory_width)
    for t in range(history.shape[1]):
        features = encoder(history[:, t])
        previous = initial_previous_action if t == 0 else history_actions[:, t - 1]
        memory = updater(memory, features, previous)
    return features, memory


def imagine(features, memory, actions, predictor, updater):
    """Free-running P then U. No future observations; frozen U keeps input derivatives."""
    if (
        actions.ndim != 3
        or actions.shape[0] != len(memory)
        or actions.shape[2] != predictor.action_width
    ):
        raise ValueError("Actions must be [B,horizon,action_width]")
    states = []
    for t in range(actions.shape[1]):
        features = predictor(features, memory, actions[:, t])
        memory = updater(memory, features, actions[:, t])
        states.append((features, memory))
    return states


class WorldModel(nn.Module):
    """Explicit composition used by the dynamics recipe. Targets never enter forward."""

    def __init__(self, encoder, updater, predictor):
        super().__init__()
        self.encoder, self.updater, self.predictor = encoder, updater, predictor

    def forward(self, history, history_actions, actions, initial_previous_action):
        features, memory = observe(
            self.encoder,
            self.updater,
            history,
            history_actions,
            initial_previous_action,
        )
        return imagine(features, memory, actions, self.predictor, self.updater)
