"""Supplied symbolic perception for the shared core's first Rule World stage.

Diagnostic stage only (docs/shared-core-plan.md, S2): tokens come from hidden scene
symbols, not pixels, so it localizes core failures and never supplies a pixel gate.
Entity order: background, left machine, right machine, four objects.

`tokens` carry everything observable (role, position, attributes, lamps).
`identity` drops the lamps; actions refer to entities through it, so an action never
reveals a lamp state the core should predict.
"""

import torch
from torch import nn
from torch.nn import functional as F

ATTRIBUTES, VALUES, ENTITIES = 4, 4, 7
ROLE = torch.tensor([0, 1, 1, 2, 2, 2, 2])  # background, machine, object


class SymbolicEncoder(nn.Module):
    def __init__(self, width=128):
        super().__init__()
        self.width = width
        self.role = nn.Embedding(3, width)
        self.position = nn.Linear(2, width)
        self.attribute = nn.Embedding(ATTRIBUTES * VALUES, width)
        self.lamp = nn.Embedding(2, width)

    def identity(self, scenes):
        """[B,7,D] lamp-free entity tokens."""
        device = self.role.weight.device
        attrs = scenes.attrs.to(device)
        b = len(attrs)
        xy = torch.cat((torch.zeros(b, 1, 2), scenes.machine_xy, scenes.object_xy), 1).to(device) / 64
        tokens = self.role(ROLE.to(device)).expand(b, -1, -1) + self.position(xy)
        offsets = torch.arange(ATTRIBUTES, device=device) * VALUES
        objects = self.attribute(attrs + offsets).sum(2)
        return tokens + torch.cat((torch.zeros(b, 3, self.width, device=device), objects), 1)

    def forward(self, scenes, lamps):
        """Observed tokens [B,7,D] for lamps [B,2]."""
        tokens = self.identity(scenes)
        machines = self.lamp(lamps.to(tokens.device))
        return tokens + torch.cat((torch.zeros_like(tokens[:, :1]), machines, torch.zeros_like(tokens[:, 3:])), 1)


class SymbolicDecoder(nn.Module):
    """Per-slot readout of role, lamp and attributes (the slim symbolic decoder)."""

    def __init__(self, width=128):
        super().__init__()
        self.role = nn.Linear(width, 3)
        self.lamp = nn.Linear(width, 2)
        self.attributes = nn.Linear(width, ATTRIBUTES * VALUES)

    def forward(self, tokens):
        return dict(role=self.role(tokens), lamp=self.lamp(tokens),
                    attributes=self.attributes(tokens).unflatten(-1, (ATTRIBUTES, VALUES)))


def targets(scenes, lamps):
    """Per-entity truth: role [B,7], lamp [B,7] (-1 where undefined), attributes [B,7,4] (-1)."""
    b = len(lamps)
    role = ROLE.expand(b, -1)
    lamp = torch.full((b, ENTITIES), -1, dtype=torch.long)
    lamp[:, 1:3] = lamps
    attrs = torch.full((b, ENTITIES, ATTRIBUTES), -1, dtype=torch.long)
    attrs[:, 3:] = scenes.attrs
    return dict(role=role, lamp=lamp, attributes=attrs)


def readout_loss(out, truth):
    """Mean cross-entropy over defined targets; out[k] [...,C], truth[k] long with -1 = ignore."""
    total = 0.0
    for key, logits in out.items():
        t = truth[key].to(logits.device)
        total = total + F.cross_entropy(logits.reshape(-1, logits.shape[-1]), t.reshape(-1), ignore_index=-1)
    return total
