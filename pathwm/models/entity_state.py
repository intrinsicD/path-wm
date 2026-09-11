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

    def forward(self, observations, slots, sources=None, entity_count=2):
        hidden = observations.new_zeros((len(observations), entity_count, self.width))
        for t in range(observations.shape[1]):
            safe = slots[:, t].clamp_min(0)
            previous = hidden[torch.arange(len(hidden)), safe]
            updated = self.update(observations[:, t], previous)
            if sources is not None:
                source = hidden[torch.arange(len(hidden)), sources[:, t].clamp_min(0)]
                interaction = self.interact(previous, source)
                updated = torch.where(
                    (sources[:, t] >= 0)[:, None], interaction, updated
                )
            mask = torch.nn.functional.one_hot(safe, entity_count).bool() & (
                slots[:, t, None] >= 0
            )
            hidden = torch.where(mask[:, :, None], updated[:, None], hidden)
        return self.head(hidden), hidden


class EntityInteractionCell(EntityStateCell):
    """A supplied directed relation; only its latent update is learned."""

    def __init__(self, width=16, blind=False):
        super().__init__(width, preserve_no_information=True)
        self.cell.requires_grad_(False)
        self.head.requires_grad_(False)
        self.interaction = nn.Sequential(
            nn.Linear(2 * width, 2 * width),
            nn.Tanh(),
            nn.Linear(2 * width, width),
            nn.Tanh(),
        )
        self.register_buffer("_interaction_blind", torch.tensor(blind))

    def interact(self, destination, source):
        source = torch.where(self._interaction_blind, torch.zeros_like(source), source)
        return self.interaction(torch.cat([destination, source], -1))


class EntityStateMemory:
    """Single-writer state updates, atomically staged with identity receipts."""

    def __init__(self, matcher, cell, capacity=2):
        self.memory = EntityMemory(matcher, capacity=capacity)
        self.cell = copy.deepcopy(cell).cpu().eval().requires_grad_(False)
        self.latents = []

    def observe(
        self,
        event_id,
        descriptor,
        timestamp,
        observation,
        *,
        source_id=None,
        source_query=None,
    ):
        observation = torch.as_tensor(observation, dtype=torch.float32).detach().cpu()
        if observation.shape != (4,) or not torch.isfinite(observation).all():
            raise ValueError("Expected four finite state features")
        content = observation.tolist()
        if source_id is not None:
            if (
                type(source_id) is not int
                or not 0 <= source_id < len(self.latents)
                or observation.any()
                or not hasattr(self.cell, "interact")
            ):
                raise ValueError(
                    "Interaction requires a known source and zero state features"
                )
            content = dict(observation=content, source_id=source_id)
        if source_query is not None:
            if (
                source_id is not None
                or observation.any()
                or not hasattr(self.cell, "interact")
            ):
                raise ValueError(
                    "Supply either a source ID or query with zero state features"
                )
            source_query = self.memory._descriptor(source_query)
            content = dict(observation=observation.tolist(), source_query=source_query)
        before = self.memory.snapshot()
        staged = EntityMemory.restore(self.memory._model, before)
        receipt = staged.observe(event_id, descriptor, timestamp, content=content)
        if staged.snapshot() == before:
            return receipt
        if source_query is not None:
            source_id = self.memory.lookup(source_query)["entity_id"]
            if source_id is None:
                raise LookupError("Unresolved interaction source")
        latents = copy.deepcopy(self.latents)
        identity = receipt["entity_id"]
        if identity is not None:
            previous = (
                torch.zeros(self.cell.width)
                if identity == len(latents)
                else torch.tensor(latents[identity])
            )
            with torch.inference_mode():
                if source_id is None:
                    updated = self.cell.update(observation[None], previous[None])[0]
                else:
                    if identity == source_id:
                        raise ValueError("Interaction endpoints must differ")
                    updated = self.cell.interact(
                        previous[None], torch.tensor(latents[source_id])[None]
                    )[0]
            if updated.shape != (self.cell.width,) or not torch.isfinite(updated).all():
                raise ValueError("Invalid state update")
            if identity == len(latents):
                latents.append(updated.tolist())
            else:
                latents[identity] = updated.tolist()
        if source_query is not None:
            receipt["source_id"] = source_id
            staged._state["receipts"][-1]["result"] = copy.deepcopy(receipt)
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
        for receipt in snapshot["memory"]["receipts"]:
            result = receipt["result"]
            if "source_id" in result and (
                type(result["source_id"]) is not int
                or not 0 <= result["source_id"] < len(latents)
            ):
                raise ValueError("Invalid persisted source")
        store.latents = latents
        return store
