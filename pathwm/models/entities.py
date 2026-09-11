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

    def __init__(self, width=64, association="observed"):
        if association not in ("observed", "learned"):
            raise ValueError("Shared reader requires observed or learned association")
        super().__init__()
        self.association = association
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

        # Construct after the reference modules to preserve their initialization.
        if association == "learned":
            self.matcher = nn.Sequential(nn.Linear(8, 32), nn.GELU(), nn.Linear(32, 1))

    def assignment_weights(self, inputs):
        """Task-trained probabilities over both complete candidate bijections."""
        difference = inputs[..., :8].unsqueeze(-2) - inputs[:, None, None, 0, :, :8]
        scores = self.matcher(difference.square()).squeeze(-1)
        assignments = torch.stack(
            [
                scores[..., 0, 0] + scores[..., 1, 1],
                scores[..., 0, 1] + scores[..., 1, 0],
            ],
            -1,
        )
        visible = inputs[..., 8].bool().all(-1, keepdim=True)
        return torch.where(visible, assignments.softmax(-1), 0.5)

    def forward(self, inputs):
        if inputs.ndim != 4 or inputs.shape[1:] != (3, 2, FEATURES):
            raise ValueError(
                "Shared entity reader requires three two-candidate observations"
            )
        if not inputs[:, :2, :, 8].bool().all():
            raise ValueError(
                "Shared diagnostic requires visible initial/action descriptors"
            )
        if self.association == "observed":
            return self._assigned(inputs, observed_association(inputs))
        weights = self.assignment_weights(inputs)
        paths = []
        for order in ([0, 1], [1, 0]):
            associated = inputs.new_zeros(inputs.shape)
            associated[:, 0, :, :2] = torch.eye(
                2, device=inputs.device, dtype=inputs.dtype
            )
            associated[:, 1, :, :2] = torch.eye(
                2, device=inputs.device, dtype=inputs.dtype
            )[order]
            associated[:, -1, 0, :2] = weights[:, -1]
            paths.append(self._assigned(inputs, associated))
        return tuple(
            (torch.stack([path[j].exp() for path in paths], 1) * weights[:, 1, :, None])
            .sum(1)
            .clamp_min(1e-30)
            .log()
            for j in range(3)
        )

    def _assigned(self, inputs, associated):
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


class EntityMatchReader(nn.Module):
    """Learned compatibility scores compete with a trainable new-entity logit."""

    def __init__(self, width=64):
        super().__init__()
        self.matcher = nn.Sequential(
            nn.Linear(8, width), nn.GELU(), nn.Linear(width, 1)
        )
        self.null = nn.Parameter(torch.zeros(()))

    def forward(self, inputs):
        if inputs.ndim != 4 or inputs.shape[1:] != (3, 2, FEATURES):
            raise ValueError("Matching reader requires the entity input envelope")
        difference = inputs[:, 0, :, :8] - inputs[:, -1, 0, None, :8]
        scores = self.matcher(difference.square()).squeeze(-1)
        return (torch.cat([scores, self.null.expand(len(inputs), 1)], -1),)
