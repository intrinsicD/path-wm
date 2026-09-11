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


class EntityRelationMemory:
    """One supplied relation type; bindings store latents, never source IDs."""

    def __init__(self, matcher, cell, key_model, capacity=3):
        self.state = EntityStateMemory(matcher, cell, capacity=capacity)
        self.key_model = copy.deepcopy(key_model).cpu().eval().requires_grad_(False)
        self.relations = {}

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
        return dict(
            schema=1,
            state=self.state.snapshot(),
            key_model_sha256=state_hash(self.key_model),
            relations=copy.deepcopy(self.relations),
        )

    @classmethod
    def restore(cls, matcher, cell, key_model, snapshot):
        store = cls(
            matcher, cell, key_model, capacity=snapshot["state"]["memory"]["capacity"]
        )
        if snapshot["schema"] != 1 or snapshot["key_model_sha256"] != state_hash(
            store.key_model
        ):
            raise ValueError("Incompatible relation model")
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
