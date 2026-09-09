"""A visible composition of one encoder and independently chosen output heads."""

from torch import nn


class Perception(nn.Module):
    def __init__(self, encoder, heads, detached_heads=()):
        super().__init__()
        if not set(detached_heads) <= heads.keys():
            raise ValueError("Detached heads must name existing outputs")
        self.encoder, self.heads = encoder, nn.ModuleDict(heads)
        self.detached_heads = tuple(detached_heads)

    def forward(self, rgb):
        features = self.encoder(rgb)
        detached = {k: v.detach() for k, v in features.items()}
        return {
            k: head(detached if k in self.detached_heads else features)
            for k, head in self.heads.items()
        }
