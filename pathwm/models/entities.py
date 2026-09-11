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


class SharedEntityReader(nn.Module):
    """Fixed supplied object streams; learned updates, coherent assignment mixture."""

    def __init__(self, width=64):
        super().__init__()
        self.encode = nn.Sequential(nn.Linear(7, width), nn.GELU())
        self.recurrent = nn.GRU(width, width, batch_first=True)
        self.heads = nn.ModuleList(
            [
                nn.Linear(width, 1),
                nn.Linear(width, 2),
                nn.Sequential(
                    nn.Linear(width + 1, width), nn.GELU(), nn.Linear(width, 2)
                ),
            ]
        )

    def forward(self, inputs):
        if inputs.ndim != 4 or inputs.shape[1:] != (3, 2, FEATURES):
            raise ValueError(
                "Shared entity reader requires three two-candidate observations"
            )
        associated = observed_association(inputs)
        if not inputs[:, :2, :, 8].bool().all():
            raise ValueError(
                "Shared diagnostic requires visible initial/action descriptors"
            )
        # Route each observed candidate to its initial object stream. Only features
        # from the first two observations enter the recurrent update.
        routed = torch.einsum(
            "btce,btcf->btef", associated[:, :2, :, :2], inputs[:, :2, :, 8:]
        )
        batch = len(inputs)
        encoded = self.encode(routed).transpose(1, 2).reshape(batch * 2, 2, -1)
        history, _ = self.recurrent(encoded)
        states = history[:, -1].reshape(batch, 2, -1)
        # Evaluate both current-order assignments; select the visible one, or mix
        # them when final recognition is unavailable. Never average object states.
        outputs = [[], [], []]
        for order in ([0, 1], [1, 0]):
            current = states[:, order]
            identity = self.heads[0](current).squeeze(-1).softmax(-1)
            state = self.heads[1](current).softmax(-1)
            effect = self.heads[2](
                torch.cat([current, inputs[:, -1, :, 14:15]], -1)
            ).softmax(-1)
            outputs[0].append(identity)
            for dest, probability in zip(outputs[1:], (state, effect)):
                dest.append(
                    (probability[:, 0, :, None] * probability[:, 1, None, :]).flatten(1)
                )
        weights = associated[:, -1, 0, :2]
        return tuple(
            (torch.stack(values, 1) * weights[..., None]).sum(1).clamp_min(1e-30).log()
            for values in outputs
        )
