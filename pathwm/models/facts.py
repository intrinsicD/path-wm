"""Direct text-fact reference; no session state or evaluator inputs."""

import torch
from torch import nn

from .modalities import Attend
from .multiscale import MultiScaleTextEncoder


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
