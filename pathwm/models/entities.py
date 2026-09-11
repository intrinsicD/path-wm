"""A small recurrent baseline on controlled candidate features, not an entity graph."""

import torch
from torch import nn
from pathwm.data.entities import FEATURES


def observed_association(inputs):
    """Synthetic input-only association to initial candidate order; no state oracle."""
    result = inputs.clone()
    reference = inputs[:, 0, :, :8]
    match = (inputs[..., :8].unsqueeze(-2) == reference[:, None, None]).all(-1)
    visible = inputs[..., 8:9].bool()
    if not (match.sum(-1)[visible.squeeze(-1)] == 1).all():
        raise ValueError("Visible descriptors require one exact initial match")
    result[..., :8] = 0
    result[..., :2] = torch.where(visible, match.to(inputs.dtype), 0.5)
    return result


class EntityReader(nn.Module):
    def __init__(self, width=64, association="raw"):
        if association not in ("raw", "observed"):
            raise ValueError("Unknown entity association mode")
        super().__init__()
        self.association = association
        self.encode = nn.Sequential(nn.Linear(FEATURES, width), nn.GELU())
        self.recurrent = nn.GRU(2 * width, 2 * width, batch_first=True)
        self.heads = nn.ModuleList(
            nn.Sequential(nn.LayerNorm(2 * width), nn.Linear(2 * width, n))
            for n in (2, 4, 4)
        )

    def forward(self, inputs):
        if inputs.ndim != 4 or inputs.shape[1:] != (3, 2, FEATURES):
            raise ValueError(
                "Entity reader requires three observations of two candidates"
            )
        if self.association == "observed":
            inputs = observed_association(inputs)
        tokens = self.encode(inputs).flatten(2)
        states, _ = self.recurrent(tokens)
        return tuple(head(states[:, -1]) for head in self.heads)
