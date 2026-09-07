"""Versioned tensor and executable-action contracts for the paddle model."""

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor
from torch.nn import functional as F


TENSOR_SCHEMA_VERSION = "paddle-latents-v1-fine256-coarse64-width64-memory128"
ACTION_SCHEMA_VERSION = "paddle-actions-v1-left0-stay1-right2-start000"


@dataclass(frozen=True)
class ObservationLatent:
    fine: Tensor
    coarse: Tensor

    def __post_init__(self):
        if self.fine.ndim != 3 or self.fine.shape[1:] != (256, 64):
            raise ValueError(f"fine must have shape [B,256,64], got {tuple(self.fine.shape)}")
        if self.coarse.shape != (self.fine.shape[0], 64, 64):
            raise ValueError(f"coarse must have shape [B,64,64], got {tuple(self.coarse.shape)}")

    def tokens(self) -> Tensor:
        return torch.cat((self.fine, self.coarse), dim=1)

    @classmethod
    def from_tokens(cls, tokens: Tensor) -> "ObservationLatent":
        if tokens.ndim != 3 or tokens.shape[1:] != (320, 64):
            raise ValueError(f"tokens must have shape [B,320,64], got {tuple(tokens.shape)}")
        return cls(tokens[:, :256], tokens[:, 256:])

    @classmethod
    def cat(cls, values: Sequence["ObservationLatent"], dim: int = 0) -> "ObservationLatent":
        return cls(torch.cat([v.fine for v in values], dim=dim), torch.cat([v.coarse for v in values], dim=dim))

    def clone(self) -> "ObservationLatent":
        return ObservationLatent(self.fine.clone(), self.coarse.clone())

    def detach(self) -> "ObservationLatent":
        return ObservationLatent(self.fine.detach(), self.coarse.detach())

    def to(self, *args, **kwargs) -> "ObservationLatent":
        return ObservationLatent(self.fine.to(*args, **kwargs), self.coarse.to(*args, **kwargs))

    def __getitem__(self, index) -> "ObservationLatent":
        fine, coarse = self.fine[index], self.coarse[index]
        if fine.ndim == 2:
            fine, coarse = fine.unsqueeze(0), coarse.unsqueeze(0)
        return ObservationLatent(fine, coarse)


@dataclass(frozen=True)
class PlanningState:
    observation: ObservationLatent
    memory: Tensor

    def __post_init__(self):
        if self.memory.shape != (self.observation.fine.shape[0], 128):
            raise ValueError(f"memory must have shape [B,128], got {tuple(self.memory.shape)}")

    def clone(self) -> "PlanningState":
        return PlanningState(self.observation.clone(), self.memory.clone())

    def detach(self) -> "PlanningState":
        return PlanningState(self.observation.detach(), self.memory.detach())

    def to(self, *args, **kwargs) -> "PlanningState":
        return PlanningState(self.observation.to(*args, **kwargs), self.memory.to(*args, **kwargs))

    def __getitem__(self, index) -> "PlanningState":
        memory = self.memory[index]
        if memory.ndim == 1:
            memory = memory.unsqueeze(0)
        return PlanningState(self.observation[index], memory)


def action_one_hot(ids: Tensor) -> Tensor:
    """Convert executable IDs to float32; only callers create the zero start marker."""
    ids = torch.as_tensor(ids)
    if ids.is_floating_point() and not torch.equal(ids, ids.round()):
        raise ValueError("action IDs must be integers in {0,1,2}")
    if torch.any((ids < 0) | (ids > 2)):
        raise ValueError("action IDs must be in {0,1,2}; the initial marker is not an ID")
    return F.one_hot(ids.long(), num_classes=3).float()
