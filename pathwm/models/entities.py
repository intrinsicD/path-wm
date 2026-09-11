"""A small recurrent baseline on controlled candidate features, not an entity graph."""

from torch import nn
from pathwm.data.entities import FEATURES


class EntityReader(nn.Module):
    def __init__(self, width=64):
        super().__init__()
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
        tokens = self.encode(inputs).flatten(2)
        states, _ = self.recurrent(tokens)
        return tuple(head(states[:, -1]) for head in self.heads)
