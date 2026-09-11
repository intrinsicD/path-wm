"""Direct and single-event text-fact controls with label-free forward boundaries."""

import torch
from torch import nn

from .modalities import Attend
from .multiscale import MultiScaleTextEncoder
from .tasks import Actor, TaskRequest, TaskSession


class FactReader(nn.Module):
    def __init__(self, width=32):
        super().__init__()
        self.encoder = MultiScaleTextEncoder(width)
        self.queries = nn.Parameter(torch.randn(2, width) * 0.02)
        self.reader = Attend(width)
        self.entity_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 32))
        self.location_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 4))

    def forward(self, observation):
        tokens = self.encoder(observation).as_tokens()
        read = self.reader(
            self.queries.expand(len(tokens.values), -1, -1),
            tokens.values,
            valid=tokens.valid,
        )
        return self.entity_head(read[:, 0]), self.location_head(read[:, 1])


class EventFactReader(nn.Module):
    """Single-event control using the ordinary task interpreter and working tokens.

    The supplied direct reader contributes the matched encoder/readout initialization.
    No state persists between calls. Labels and per-fact task metadata are not inputs.
    Categorical draws retain the agent's normal behavior, including in eval mode.
    """

    def __init__(self, agent, direct):
        super().__init__()
        self.agent = agent
        self.agent.encoders["text"] = direct.encoder
        self.queries = direct.queries
        self.reader = direct.reader
        self.entity_head = direct.entity_head
        self.location_head = direct.location_head

    def forward(self, observation):
        state = self.agent.initial_state(
            len(observation.values), time=0, session_id="fact-control"
        )
        state = self.agent.observe(
            state, {"text": observation}, time=0, replay=self.training
        )
        session = TaskSession(
            TaskRequest(
                "fact-control",
                "Identify the observed entity and location.",
                Actor("user", "fact-control"),
            )
        )
        goal = self.agent.task_tokens(state, [session] * len(state.tokens))
        working = self.agent.think(state, steps=2, goal=goal)
        tokens = working.tokens[:, self.agent.layout["working"]]
        read = self.reader(self.queries.expand(len(tokens), -1, -1), tokens)
        return self.entity_head(read[:, 0]), self.location_head(read[:, 1])


def bind_facts(entity_logits, location_logits, query_entities):
    """Return log location probabilities from a fixed discriminative mixture.

    Normalize each record's predicted entity-q probability across the two records,
    then mix their predicted location distributions. This is a diagnostic selector,
    not a calibrated joint posterior. The exact query entity is a legitimate input.
    """
    if (
        entity_logits.ndim != 3
        or entity_logits.shape[1:] != (2, 32)
        or location_logits.shape != (len(entity_logits), 2, 4)
        or query_entities.shape != (len(entity_logits),)
        or query_entities.dtype != torch.long
        or ((query_entities < 0) | (query_entities >= 32)).any()
        or not torch.isfinite(entity_logits).all()
        or not torch.isfinite(location_logits).all()
    ):
        raise ValueError(
            "Binding requires finite two-record logits and valid entity queries"
        )
    entity = entity_logits.double().log_softmax(-1)
    score = entity.gather(-1, query_entities[:, None, None].expand(-1, 2, 1)).squeeze(
        -1
    )
    weights = score.log_softmax(-1)
    return torch.logsumexp(
        weights[..., None] + location_logits.double().log_softmax(-1), dim=1
    )
