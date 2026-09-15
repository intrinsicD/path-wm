"""Bounded single-writer store with replayable operations and revocable identity links.

Four logical views share one revision. No implicit neural inference or eviction.
The log retains all admitted records; capacity exhaustion requires an explicit new
retention policy. This is a reference implementation, not a concurrent database.
"""

import copy
from dataclasses import asdict, replace
import json
import math
from pathlib import Path

import torch
from pathwm.io import atomic_json, digest
from .records import (
    Component,
    Endpoint,
    Entity,
    Event,
    Evidence,
    Limits,
    Relation,
    identifier,
)


def primitive(value):
    """Reject NaNs/nonportable metadata and break aliases to caller-owned objects."""
    return json.loads(json.dumps(value, allow_nan=False))


def confidence(value):
    if value is not None and (
        isinstance(value, bool)
        or not isinstance(value, (float, int))
        or not math.isfinite(value)
        or not 0 <= value <= 1
    ):
        raise ValueError("Confidence must be absent or a finite score in [0,1]")


class Transaction:
    def __init__(self, store, event):
        self.base_revision = store.revision
        self.event = Event(**primitive(asdict(event)))
        self.operations = []
        self._store = store

    def _id(self, kind):
        return f"{self.event.id}/{kind}/{len(self.operations)}"

    def _append(self, kind, value):
        self.operations.append(dict(op=kind, value=primitive(value)))

    def create_entity(self, label="", *, kind="instance", entity_id=None):
        entity_id = identifier(entity_id or self._id("entity"))
        self._append(
            "entity", asdict(Entity(entity_id, label, kind, self.event.available_at))
        )
        return entity_id

    def add_evidence(
        self, source, modality, *, content_ref="", content_hash="", data=None
    ):
        if self.event.kind != "observation":
            raise ValueError("Only new observations can create source evidence")
        evidence_id = self._id("evidence")
        self._append(
            "evidence",
            asdict(
                Evidence(
                    evidence_id,
                    self.event.id,
                    identifier(source),
                    identifier(modality),
                    self.event.occurred_at,
                    self.event.available_at,
                    content_ref,
                    content_hash,
                    data or {},
                )
            ),
        )
        return evidence_id

    def update_entity(self, entity_id, changes, *, evidence=()):
        """Revise labels/type/existence claims; never change identity or last_seen."""
        if not changes or not set(changes) <= {"label", "kind", "existence_confidence"}:
            raise ValueError("Unsupported entity fields")
        self._append(
            "entity_update",
            dict(id=entity_id, changes=changes, evidence=list(evidence)),
        )

    def put_component(
        self,
        entity_id,
        name,
        value,
        *,
        space,
        model_version,
        evidence=(),
        role="observed",
        confidence=None,
        data=None,
        parents=(),
    ):
        value = torch.empty(0) if value is None else torch.as_tensor(value)
        if value.requires_grad or value.grad_fn is not None:
            raise ValueError("Explicitly detach at the persistent commit boundary")
        if not value.is_floating_point() or not torch.isfinite(value).all():
            raise ValueError("Latent must be a finite floating tensor")
        component_id = self._id("component")
        record = Component(
            component_id,
            entity_id,
            name,
            tuple(value.detach().cpu().reshape(-1).tolist()),
            tuple(value.shape),
            space,
            model_version,
            tuple(evidence),
            self.event.occurred_at,
            self.event.available_at,
            self.base_revision + 1,
            role,
            confidence,
            data or {},
            tuple(parents),
        )
        self._append("component", asdict(record))
        return component_id

    def relate(
        self, source, target, relation_type, *, evidence=(), confidence=None, data=None
    ):
        relation_id = self._id("relation")
        source = Endpoint(source) if isinstance(source, str) else source
        target = Endpoint(target) if isinstance(target, str) else target
        record = Relation(
            relation_id,
            source,
            target,
            relation_type,
            tuple(evidence),
            self.event.occurred_at,
            self.event.available_at,
            self.base_revision + 1,
            confidence,
            True,
            data or {},
        )
        self._append("relation", asdict(record))
        return relation_id

    def merge(self, a, b, *, evidence):
        """Accept an identity hypothesis; preserve both original records and values."""
        return self.relate(a, b, "same_as", evidence=evidence)

    def split(self, relation_id):
        self._append("retract_relation", dict(id=relation_id))

    def reassign(self, component_id, entity_id):
        self._append("reassign", dict(id=component_id, entity_id=entity_id))

    def preview(self):
        """Isolated tentative view for within-event binding; never publishes."""
        result = self._store.clone()
        result.commit(self)
        return result


class WorldStore:
    schema = "pathwm-world-state-v1"

    def __init__(self, limits=None):
        self.limits = limits or Limits()
        self._entities, self._components, self._relations, self._evidence = (
            {},
            {},
            {},
            {},
        )
        self._events = []

    @property
    def revision(self):
        return len(self._events)

    def clone(self):
        return copy.deepcopy(self)

    def begin(
        self, event_id, *, occurred_at, available_at, kind="observation", payload=None
    ):
        return Transaction(
            self, Event(event_id, occurred_at, available_at, kind, payload or {})
        )

    def receipt(self, event_id, payload=None):
        for entry in self._events:
            if entry["event"]["id"] == event_id:
                if payload is not None and digest(entry["event"]["payload"]) != digest(
                    payload
                ):
                    raise ValueError("Conflicting event retry")
                return copy.deepcopy(entry["receipt"])
        return None

    def commit(self, transaction):
        event = transaction.event
        ops = primitive(transaction.operations)
        fingerprint = digest([asdict(event), ops])
        for old in self._events:
            if old["event"]["id"] == event.id:
                if fingerprint != old["fingerprint"]:
                    raise ValueError("Conflicting event retry")
                return copy.deepcopy(old["receipt"])
        if transaction.base_revision != self.revision:
            raise ValueError("stale transaction base revision")
        if (
            self._events
            and event.available_at < self._events[-1]["event"]["available_at"]
        ):
            raise ValueError("Availability clock cannot move backward")
        draft = self.clone()
        for op in ops:
            draft._apply(op, event)
        for entity in draft._entities.values():
            times = [
                c.valid_from
                for c in draft._components.values()
                if c.entity_id == entity.id and c.role == "observed" and c.active
            ]
            draft._entities[entity.id] = replace(
                entity, last_seen=max(times, default=None)
            )
        receipt = dict(
            event_id=event.id,
            revision=self.revision + 1,
            fingerprint=fingerprint,
            operations=len(ops),
        )
        draft._events.append(
            dict(
                event=asdict(event),
                operations=ops,
                fingerprint=fingerprint,
                receipt=receipt,
            )
        )
        draft._validate()
        # One publication after all checks. Previously acquired records are copies.
        self.__dict__ = draft.__dict__
        return copy.deepcopy(receipt)

    def _apply(self, op, event):
        value, kind = op["value"], op["op"]
        table = None
        if kind == "entity":
            record, table = Entity(**value), self._entities
            if record.created_at != event.available_at or record.last_seen is not None:
                raise ValueError("Invalid entity creation time")
        elif kind == "entity_update":
            if (
                value["id"] not in self._entities
                or not value["changes"]
                or not set(value["changes"])
                <= {"label", "kind", "existence_confidence"}
            ):
                raise ValueError("Invalid entity update reference/fields")
            if any(
                e not in self._evidence
                or self._evidence[e].available_at > event.available_at
                for e in value["evidence"]
            ):
                raise ValueError("Invalid entity evidence reference")
            if (
                value["changes"].get("existence_confidence") is not None
                and not value["evidence"]
            ):
                raise ValueError("Existence confidence requires evidence")
            self._entities[value["id"]] = replace(
                self._entities[value["id"]], **value["changes"]
            )
            return
        elif kind == "evidence":
            record, table = Evidence(**value), self._evidence
            if (
                event.kind != "observation"
                or record.event_id != event.id
                or record.occurred_at != event.occurred_at
                or record.available_at != event.available_at
            ):
                raise ValueError("Invalid source evidence event")
        elif kind == "component":
            value = dict(
                value,
                values=tuple(value["values"]),
                shape=tuple(value["shape"]),
                evidence=tuple(value["evidence"]),
                parents=tuple(value["parents"]),
            )
            record, table = Component(**value), self._components
            if any(
                p not in self._components or not self._components[p].active
                for p in record.parents
            ):
                raise ValueError("Invalid or inactive parent component reference")
            if record.role not in {"observed", "inferred", "predicted"}:
                raise ValueError("Invalid component role")
            if (
                record.revision != self.revision + 1
                or record.available_at != event.available_at
                or record.valid_from != event.occurred_at
            ):
                raise ValueError("Invalid component revision/time")
            if record.role == "observed":
                if event.kind != "observation" or not record.evidence:
                    raise ValueError("Observed state requires new source evidence")
                sources = [self._evidence.get(e) for e in record.evidence]
                if not any(e is not None and e.event_id == event.id for e in sources):
                    raise ValueError("Observed state needs evidence from this event")
                entity = self._entities.get(record.entity_id)
                if entity is None:
                    raise ValueError("Unknown entity reference")
                self._entities[entity.id] = replace(
                    entity, last_seen=max(entity.last_seen or 0, event.occurred_at)
                )
        elif kind == "relation":
            value = dict(
                value,
                source=Endpoint(**value["source"]),
                target=Endpoint(**value["target"]),
                evidence=tuple(value["evidence"]),
            )
            record, table = Relation(**value), self._relations
            if (
                record.revision != self.revision + 1
                or record.available_at != event.available_at
                or record.valid_from != event.occurred_at
            ):
                raise ValueError("Invalid relation revision/time")
        elif kind == "retract_relation":
            if event.kind != "correction" or value["id"] not in self._relations:
                raise ValueError("Correction requires existing relation reference")
            record = self._relations[value["id"]]
            if not record.active:
                raise ValueError("Relation already retracted")
            self._relations[record.id] = replace(record, active=False)
            return
        elif kind == "reassign":
            if (
                event.kind != "correction"
                or value["id"] not in self._components
                or value["entity_id"] not in self._entities
            ):
                raise ValueError("Correction requires valid component/entity reference")
            record = self._components[value["id"]]
            self._components[record.id] = replace(
                record, entity_id=value["entity_id"], revision=self.revision + 1
            )
            affected = {record.id}
            while True:
                more = {
                    c.id for c in self._components.values() if set(c.parents) & affected
                } - affected
                if not more:
                    break
                affected.update(more)
            for cid in affected - {record.id}:
                c = self._components[cid]
                self._components[cid] = replace(
                    c, active=False, data={**c.data, "invalidated_by": event.id}
                )
            return
        else:
            raise ValueError("Unknown world-state operation")
        identifier(record.id)
        if record.id in table:
            raise ValueError("Duplicate record identifier")
        table[record.id] = record

    def _endpoint(self, endpoint):
        if endpoint.kind == "entity" and endpoint.ref in self._entities:
            return endpoint.ref
        if endpoint.kind == "component" and endpoint.ref in self._components:
            return self._components[endpoint.ref].entity_id
        raise ValueError("Invalid relation endpoint reference")

    def _validate(self):
        if len(json.dumps(self._events, allow_nan=False).encode()) > self.limits.bytes:
            raise ValueError("Store serialized payload capacity exhausted")
        for name in ("entities", "components", "relations", "evidence", "events"):
            if len(getattr(self, "_" + name)) > getattr(self.limits, name):
                raise ValueError(f"Store {name} capacity exhausted")
        if sum(len(c.values) for c in self._components.values()) > self.limits.values:
            raise ValueError("Store numeric payload capacity exhausted")
        for entity in self._entities.values():
            identifier(entity.kind)
            if not isinstance(entity.label, str):
                raise ValueError("Entity label must be text")
            confidence(entity.existence_confidence)
        for evidence in self._evidence.values():
            identifier(evidence.source)
            identifier(evidence.modality)
            if not isinstance(evidence.data, dict):
                raise ValueError("Evidence metadata must be a mapping")
        for c in self._components.values():
            identifier(c.name)
            identifier(c.space)
            identifier(c.model_version)
            confidence(c.confidence)
            if type(c.active) is not bool or not isinstance(c.data, dict):
                raise ValueError("Invalid component metadata/activity")
            if c.entity_id not in self._entities or any(
                e not in self._evidence for e in c.evidence
            ):
                raise ValueError("Invalid component/evidence reference")
            if (
                c.shape != (0,) and any(type(n) is not int or n < 1 for n in c.shape)
            ) or math.prod(c.shape) != len(c.values):
                raise ValueError("Invalid latent shape")
            if any(
                isinstance(x, bool)
                or not isinstance(x, (float, int))
                or not math.isfinite(x)
                for x in c.values
            ):
                raise ValueError("Invalid latent values")
            if any(self._evidence[e].available_at > c.available_at for e in c.evidence):
                raise ValueError("Future evidence reference")
        for r in self._relations.values():
            identifier(r.type)
            confidence(r.confidence)
            a, b = self._endpoint(r.source), self._endpoint(r.target)
            if any(
                e not in self._evidence
                or self._evidence[e].available_at > r.available_at
                for e in r.evidence
            ):
                raise ValueError("Invalid relation evidence reference")
            if r.type == "same_as" and (
                r.source.kind != "entity"
                or r.target.kind != "entity"
                or a == b
                or not r.evidence
            ):
                raise ValueError("Identity links require two entities and evidence")

    def canonical(self, entity_id):
        if entity_id not in self._entities:
            raise ValueError("Unknown entity reference")
        found, pending = {entity_id}, [entity_id]
        while pending:
            current = pending.pop()
            for r in self._relations.values():
                if r.type != "same_as" or not r.active:
                    continue
                a, b = r.source.ref, r.target.ref
                other = b if a == current else a if b == current else None
                if other is not None and other not in found:
                    found.add(other)
                    pending.append(other)
        return min(found)

    def entity(self, key):
        return copy.deepcopy(self._entities[key])

    def component(self, key):
        return copy.deepcopy(self._components[key])

    def entities(self):
        return tuple(copy.deepcopy(list(self._entities.values())))

    def components(self, entity_id=None, name=None):
        return tuple(
            copy.deepcopy(
                [
                    c
                    for c in self._components.values()
                    if (entity_id is None or c.entity_id == entity_id)
                    and (name is None or c.name == name)
                ]
            )
        )

    def latest(self, entity_id, name):
        result = max(
            self.components(entity_id, name),
            key=lambda c: (c.valid_from, c.revision, c.id),
            default=None,
        )
        return result if result is not None and result.active else None

    def relations(self):
        return tuple(copy.deepcopy(list(self._relations.values())))

    def evidence(self):
        return tuple(copy.deepcopy(list(self._evidence.values())))

    def events(self):
        return tuple(Event(**copy.deepcopy(e["event"])) for e in self._events)

    def at(self, *, revision=None, known_at=None):
        if revision is not None and (
            type(revision) is not int or not 0 <= revision <= self.revision
        ):
            raise ValueError("Invalid snapshot revision")
        entries = self._events[:revision] if revision is not None else self._events
        if known_at is not None:
            from .records import finite_time

            finite_time(known_at)
            entries = [e for e in entries if e["event"]["available_at"] <= known_at]
        return self.restore(
            dict(schema=self.schema, limits=asdict(self.limits), events=entries)
        )

    def snapshot(self):
        return copy.deepcopy(
            dict(schema=self.schema, limits=asdict(self.limits), events=self._events)
        )

    def save(self, path):
        atomic_json(path, self.snapshot())

    @classmethod
    def load(cls, path):
        return cls.restore(json.loads(Path(path).read_text()))

    @classmethod
    def restore(cls, snapshot):
        snapshot = primitive(snapshot)
        if (
            set(snapshot) != {"schema", "limits", "events"}
            or snapshot["schema"] != cls.schema
        ):
            raise ValueError("Incompatible world-state snapshot")
        result = cls(Limits(**snapshot["limits"]))
        for entry in snapshot["events"]:
            tx = Transaction(result, Event(**entry["event"]))
            tx.operations = entry["operations"]
            receipt = result.commit(tx)
            if (
                receipt != entry["receipt"]
                or receipt["fingerprint"] != entry["fingerprint"]
            ):
                raise ValueError("Snapshot event fingerprint mismatch")
        if result.snapshot() != snapshot:
            raise ValueError("Noncanonical or duplicate snapshot events")
        return result
