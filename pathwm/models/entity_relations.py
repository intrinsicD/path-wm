"""Learned relation keys in an explicit, transactional per-destination slot."""

import copy
import torch
from torch import nn
from pathwm.io import state_hash
from pathwm.models.entity_state import EntityStateMemory


class RelationKey(nn.Module):
    def __init__(self, width=16):
        super().__init__()
        self.width = width
        self.encoder = nn.Linear(8, width)
        self.decoder = nn.Linear(width, 8)

    def encode(self, cue):
        return self.encoder(cue).tanh()

    def decode(self, key):
        return nn.functional.normalize(self.decoder(key), dim=-1)

    def forward(self, cue):
        return self.decode(self.encode(cue))


class RelationWriteGate(nn.Module):
    """Learned context comparison; storage IDs and source states are not inputs."""

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(4, 16), nn.Tanh(), nn.Linear(16, 1))

    def forward(self, active, cue):
        return self.network((active - cue).square()).squeeze(-1).sigmoid()


class EntityRelationMemory:
    """One supplied relation type; bindings store latents, never source IDs."""

    def __init__(self, matcher, cell, key_model, capacity=3, *, gate_model=None):
        self.state = EntityStateMemory(matcher, cell, capacity=capacity)
        self.key_model = copy.deepcopy(key_model).cpu().eval().requires_grad_(False)
        self.relations = {}
        self.gate_model = (
            copy.deepcopy(gate_model).cpu().eval().requires_grad_(False)
            if gate_model is not None
            else None
        )

    def observe(self, *args, **kwargs):
        return self.state.observe(*args, **kwargs)

    def _stage(self, event_id, destination, timestamp, content):
        before = self.state.snapshot()
        staged = EntityStateMemory.restore(
            self.state.memory._model, self.state.cell, before
        )
        receipt = staged.memory.observe(
            event_id, destination, timestamp, content=content
        )
        replay = staged.snapshot() == before
        if not replay and receipt["reason"] != "matched":
            raise LookupError("Unresolved relation destination")
        return staged, receipt, replay

    def bind(self, event_id, destination, source_query, timestamp):
        query = self.state.memory._descriptor(source_query)
        staged, receipt, replay = self._stage(
            event_id, destination, timestamp, dict(relation_write=query)
        )
        if replay:
            return receipt
        source = staged.memory.lookup(query)["entity_id"]
        if source is None:
            raise LookupError("Unresolved relation source")
        if source == receipt["entity_id"]:
            raise ValueError("Relation endpoints must differ")
        with torch.inference_mode():
            key = self.key_model.encode(torch.tensor(query))
        if key.shape != (self.key_model.width,) or not torch.isfinite(key).all():
            raise ValueError("Invalid relation key")
        relations = copy.deepcopy(self.relations)
        relations[str(receipt["entity_id"])] = key.tolist()
        self.state, self.relations = staged, relations
        return receipt

    def consider(
        self,
        event_id,
        destination,
        source_query,
        active_context,
        cue_context,
        timestamp,
    ):
        """Consider a replacement; rejected proposals do not resolve their source."""
        if self.gate_model is None:
            raise ValueError("Missing write gate")
        contexts = []
        for context in (active_context, cue_context):
            value = torch.as_tensor(context, dtype=torch.float32).detach().cpu()
            if value.shape != (4,) or not torch.isfinite(value).all():
                raise ValueError("Invalid context")
            if not torch.isclose(value.norm(), value.new_tensor(1.0), atol=1e-5):
                raise ValueError("Context must have unit norm")
            contexts.append(value)
        query = self.state.memory._descriptor(source_query)
        staged, receipt, replay = self._stage(
            event_id,
            destination,
            timestamp,
            dict(
                relation_proposal=query,
                active_context=contexts[0].tolist(),
                cue_context=contexts[1].tolist(),
            ),
        )
        if replay:
            return receipt
        if str(receipt["entity_id"]) not in self.relations:
            raise LookupError("Missing relation")
        with torch.inference_mode():
            probability = self.gate_model(*contexts)
        if probability.shape != () or not torch.isfinite(probability):
            raise ValueError("Invalid write probability")
        probability = float(probability)
        if not 0 <= probability <= 1:
            raise ValueError("Invalid write probability")
        relations = copy.deepcopy(self.relations)
        if probability > 0.5:
            source = staged.memory.lookup(query)["entity_id"]
            if source is None:
                raise LookupError("Unresolved relation source")
            if source == receipt["entity_id"]:
                raise ValueError("Relation endpoints must differ")
            with torch.inference_mode():
                key = self.key_model.encode(torch.tensor(query))
            if key.shape != (self.key_model.width,) or not torch.isfinite(key).all():
                raise ValueError("Invalid relation key")
            relations[str(receipt["entity_id"])] = key.tolist()
        receipt.update(write=probability > 0.5, write_probability=probability)
        staged.memory._state["receipts"][-1]["result"] = copy.deepcopy(receipt)
        self.state, self.relations = staged, relations
        return receipt

    def recall(self, event_id, destination, timestamp):
        staged, receipt, replay = self._stage(
            event_id, destination, timestamp, dict(relation_read=True)
        )
        if replay:
            return receipt
        identity = receipt["entity_id"]
        if str(identity) not in self.relations:
            raise LookupError("Missing relation")
        with torch.inference_mode():
            query = self.key_model.decode(torch.tensor(self.relations[str(identity)]))
        if query.shape != (8,) or not torch.isfinite(query).all():
            raise ValueError("Invalid decoded relation")
        if not torch.isclose(query.norm(), query.new_tensor(1.0), atol=1e-5):
            raise LookupError("Unresolved relation query")
        source = staged.memory.lookup(query)["entity_id"]
        if source is None:
            raise LookupError("Unresolved relation source")
        if source == identity:
            raise LookupError("Relation resolves to its destination")
        with torch.inference_mode():
            updated = staged.cell.interact(
                torch.tensor(staged.latents[identity])[None],
                torch.tensor(staged.latents[source])[None],
            )[0]
        if updated.shape != (staged.cell.width,) or not torch.isfinite(updated).all():
            raise ValueError("Invalid relation update")
        staged.latents[identity] = updated.tolist()
        receipt["source_id"] = source
        staged.memory._state["receipts"][-1]["result"] = copy.deepcopy(receipt)
        self.state = staged
        return receipt

    def snapshot(self):
        snapshot = dict(
            schema=1,
            state=self.state.snapshot(),
            key_model_sha256=state_hash(self.key_model),
            relations=copy.deepcopy(self.relations),
        )

        if self.gate_model is not None:
            snapshot["gate_model_sha256"] = state_hash(self.gate_model)
        return snapshot

    @classmethod
    def restore(cls, matcher, cell, key_model, snapshot, *, gate_model=None):
        store = cls(
            matcher,
            cell,
            key_model,
            capacity=snapshot["state"]["memory"]["capacity"],
            gate_model=gate_model,
        )
        if snapshot["schema"] != 1 or snapshot["key_model_sha256"] != state_hash(
            store.key_model
        ):
            raise ValueError("Incompatible relation model")
        expected_gate = (
            state_hash(store.gate_model) if store.gate_model is not None else None
        )
        if snapshot.get("gate_model_sha256") != expected_gate:
            raise ValueError("Incompatible write gate")
        store.state = EntityStateMemory.restore(matcher, cell, snapshot["state"])
        relations = copy.deepcopy(snapshot["relations"])
        if not isinstance(relations, dict):
            raise ValueError("Invalid relations")
        for identity, key in relations.items():
            if identity not in {str(i) for i in range(len(store.state.latents))}:
                raise ValueError("Unknown relation destination")
            value = torch.as_tensor(key)
            if value.shape != (key_model.width,) or not torch.isfinite(value).all():
                raise ValueError("Invalid persisted relation key")
        store.relations = relations
        return store
