"""Learned per-entity state with transactional identity binding."""

import copy
import torch
from torch import nn
from pathwm.io import state_hash
from pathwm.models.entity_memory import EntityMemory


class EntityStateCell(nn.Module):
    def __init__(self, width=16, preserve_no_information=False):
        super().__init__()
        self.width = width
        self.cell = nn.GRUCell(4, width)
        self.head = nn.Linear(width, 2)
        # Absence preserves legacy checkpoint keys and semantics. Presence is hashed.
        if preserve_no_information:
            self.register_buffer("_preserve_no_information", torch.tensor(True))

    def update(self, observations, previous):
        updated = self.cell(observations, previous)
        if hasattr(self, "_preserve_no_information"):
            idle = (observations == observations.new_tensor([0, 0, 0, 1])).all(-1)
            updated = torch.where(
                (idle & self._preserve_no_information)[..., None], previous, updated
            )
        return updated

    def forward(self, observations, slots):
        hidden = observations.new_zeros((len(observations), 2, self.width))
        for t in range(observations.shape[1]):
            safe = slots[:, t].clamp_min(0)
            previous = hidden[torch.arange(len(hidden)), safe]
            updated = self.update(observations[:, t], previous)
            mask = torch.nn.functional.one_hot(safe, 2).bool() & (
                slots[:, t, None] >= 0
            )
            hidden = torch.where(mask[:, :, None], updated[:, None], hidden)
        return self.head(hidden), hidden


class EntityStateMemory:
    """Single-writer state updates, atomically staged with identity receipts."""

    def __init__(self, matcher, cell, capacity=2):
        self.memory = EntityMemory(matcher, capacity=capacity)
        self.cell = copy.deepcopy(cell).cpu().eval().requires_grad_(False)
        self.latents = []

    def observe(self, event_id, descriptor, timestamp, observation):
        observation = torch.as_tensor(observation, dtype=torch.float32).detach().cpu()
        if observation.shape != (4,) or not torch.isfinite(observation).all():
            raise ValueError("Expected four finite state features")
        before = self.memory.snapshot()
        staged = EntityMemory.restore(self.memory._model, before)
        receipt = staged.observe(
            event_id, descriptor, timestamp, content=observation.tolist()
        )
        if staged.snapshot() == before:
            return receipt
        latents = copy.deepcopy(self.latents)
        identity = receipt["entity_id"]
        if identity is not None:
            previous = (
                torch.zeros(self.cell.width)
                if identity == len(latents)
                else torch.tensor(latents[identity])
            )
            with torch.inference_mode():
                updated = self.cell.update(observation[None], previous[None])[0]
            if updated.shape != (self.cell.width,) or not torch.isfinite(updated).all():
                raise ValueError("Invalid state update")
            if identity == len(latents):
                latents.append(updated.tolist())
            else:
                latents[identity] = updated.tolist()
        self.memory, self.latents = staged, latents
        return receipt

    def read(self, identity):
        if type(identity) is not int or not 0 <= identity < len(self.latents):
            raise ValueError("Unknown entity ID")
        with torch.inference_mode():
            return self.cell.head(torch.tensor(self.latents[identity]))

    def snapshot(self):
        return dict(
            schema=1,
            memory=self.memory.snapshot(),
            cell_sha256=state_hash(self.cell),
            latents=copy.deepcopy(self.latents),
        )

    @classmethod
    def restore(cls, matcher, cell, snapshot):
        store = cls(matcher, cell, capacity=snapshot["memory"]["capacity"])
        if snapshot["schema"] != 1 or snapshot["cell_sha256"] != state_hash(store.cell):
            raise ValueError("Incompatible state model")
        store.memory = EntityMemory.restore(matcher, snapshot["memory"])
        latents = copy.deepcopy(snapshot["latents"])
        if len(latents) != len(snapshot["memory"]["records"]):
            raise ValueError("State and record counts disagree")
        if latents:
            values = torch.tensor(latents)
            if (
                values.shape != (len(latents), cell.width)
                or not torch.isfinite(values).all()
            ):
                raise ValueError("Invalid persisted state")
        store.latents = latents
        return store
