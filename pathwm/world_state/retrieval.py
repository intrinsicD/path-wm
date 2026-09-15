"""Exact bounded retrieval over one pinned snapshot; replace this callable for ANN."""

from dataclasses import dataclass, field, replace
import torch
from torch.nn import functional as F
from .records import Component, Entity, Event, Relation, finite_time


@dataclass(frozen=True)
class RetrievalBudget:
    entities: int = 8
    components: int = 16
    relations: int = 16
    events: int = 16
    values: int = 1024

    def __post_init__(self):
        if any(type(v) is not int or v < 0 for v in vars(self).values()):
            raise ValueError("Retrieval budgets must be nonnegative integers")


@dataclass(frozen=True)
class Query:
    key: torch.Tensor | None = None
    space: str | None = None
    model_version: str | None = None
    name: str = "recognition"
    entity_ids: tuple[str, ...] = ()
    entity_kind: str | None = None
    relation_type: str | None = None
    neighbors: bool = False
    after: float | None = None
    before: float | None = None
    known_at: float | None = None
    revision: int | None = None
    roles: tuple[str, ...] = ("observed", "inferred")


@dataclass(frozen=True)
class RetrievedContext:
    revision: int
    entities: tuple[Entity, ...] = ()
    components: tuple[Component, ...] = ()
    relations: tuple[Relation, ...] = ()
    events: tuple[Event, ...] = ()
    scores: dict[str, float] = field(default_factory=dict)
    canonical_ids: dict[str, str] = field(default_factory=dict)
    omitted: dict[str, int] = field(default_factory=dict)
    values_used: int = 0
    intervention: str | None = None


class ExactRetriever:
    """Small CPU reference. A missing or truncated hit is not evidence of absence."""

    def __call__(self, store, query, budget=None, *, trace=None):
        budget = budget or RetrievalBudget()
        for value in (query.after, query.before, query.known_at):
            if value is not None:
                finite_time(value)
        if (
            query.after is not None
            and query.before is not None
            and query.after > query.before
        ):
            raise ValueError("Invalid query interval")
        if not set(query.roles) <= {"observed", "inferred", "predicted"}:
            raise ValueError("Invalid query roles")
        # Freeze a coherent knowledge revision before event-time filtering.
        view = (
            store.at(revision=query.revision, known_at=query.known_at)
            if query.revision is not None or query.known_at is not None
            else store.clone()
        )
        all_entities = {e.id: e for e in view.entities()}
        if any(e not in all_entities for e in query.entity_ids):
            raise ValueError("Unknown query entity reference at this revision")
        canonical = {e: view.canonical(e) for e in all_entities}
        eligible = set(all_entities)
        if query.entity_kind is not None:
            eligible = {
                e for e in eligible if all_entities[e].kind == query.entity_kind
            }
        if query.entity_ids:
            groups = {canonical[e] for e in query.entity_ids}
            eligible &= {e for e in all_entities if canonical[e] in groups}

        def in_time(value):
            return (query.after is None or value >= query.after) and (
                query.before is None or value <= query.before
            )

        relations = [
            r
            for r in view.relations()
            if r.active
            and in_time(r.valid_from)
            and (query.relation_type is None or r.type == query.relation_type)
        ]
        if query.neighbors:
            # One hop, from the original seed set, not unbounded transitive traversal.
            seeds = eligible.copy()
            for r in relations:
                a, b = view._endpoint(r.source), view._endpoint(r.target)
                if a in seeds or b in seeds:
                    eligible.update((a, b))
            if query.entity_kind is not None:
                eligible = {
                    e for e in eligible if all_entities[e].kind == query.entity_kind
                }
        components = [
            c
            for c in view.components()
            if c.role in query.roles and in_time(c.valid_from)
        ]
        # Latest *eligible* value of each original-entity/name slot, never average aliases.
        latest = {}
        for c in sorted(components, key=lambda x: (x.valid_from, x.revision, x.id)):
            latest[c.entity_id, c.name] = c
        invalidated = sum(not c.active for c in latest.values())
        latest = {k: c for k, c in latest.items() if c.active}
        scores = {e: 0.0 for e in eligible}
        incompatible = 0
        if query.key is not None:
            key = torch.as_tensor(query.key).detach().float().cpu()
            if (
                key.ndim != 1
                or not torch.isfinite(key).all()
                or key.norm() == 0
                or not query.space
                or not query.model_version
            ):
                raise ValueError(
                    "Semantic query needs finite nonzero vector and representation version"
                )
            scores = {}
            for entity in eligible:
                c = latest.get((entity, query.name))
                if c is None:
                    continue
                if (
                    c.space != query.space
                    or c.model_version != query.model_version
                    or c.shape != tuple(key.shape)
                ):
                    incompatible += 1
                    continue
                memory = c.tensor()
                if memory.norm() > 0:
                    scores[entity] = float(F.cosine_similarity(key[None], memory[None]))
            eligible &= scores.keys()
        ordered = sorted(eligible, key=lambda e: (-scores[e], e))
        chosen = ordered[: budget.entities]
        chosen_set = set(chosen)
        values, picked = 0, []
        candidates = sorted(
            (c for c in latest.values() if c.entity_id in chosen_set),
            key=lambda c: (
                chosen.index(c.entity_id),
                c.name != query.name,
                c.name,
                c.id,
            ),
        )
        for c in candidates:
            if (
                len(picked) < budget.components
                and values + len(c.values) <= budget.values
            ):
                picked.append(c)
                values += len(c.values)
        rel_candidates = [
            r
            for r in relations
            if view._endpoint(r.source) in chosen_set
            and view._endpoint(r.target) in chosen_set
        ]
        picked_relations = rel_candidates[: budget.relations]
        evidence_ids = {e for c in picked for e in c.evidence} | {
            e for r in picked_relations for e in r.evidence
        }
        event_ids = {e.event_id for e in view.evidence() if e.id in evidence_ids}
        event_candidates = [
            e for e in view.events() if e.id in event_ids and in_time(e.occurred_at)
        ]
        # Do not return full event payloads (possibly containing unrelated entities/raw data).
        events = tuple(
            replace(e, payload={})
            for e in sorted(
                event_candidates, key=lambda e: (e.available_at, e.id), reverse=True
            )[: budget.events]
        )
        result = RetrievedContext(
            view.revision,
            tuple(all_entities[e] for e in chosen),
            tuple(picked),
            tuple(picked_relations),
            events,
            {e: scores[e] for e in chosen},
            {e: canonical[e] for e in chosen},
            dict(
                entities=len(ordered) - len(chosen),
                components=len(candidates) - len(picked),
                relations=len(rel_candidates) - len(picked_relations),
                events=len(event_candidates) - len(events),
                incompatible=incompatible,
                invalidated=invalidated,
            ),
            values,
        )
        if trace is not None:
            trace.record(
                "retrieval",
                dict(
                    revision=result.revision,
                    selected=chosen,
                    scores=result.scores,
                    omitted=result.omitted,
                    values=values,
                ),
            )
        return result


def intervene(context, *, remove_entities=(), replacements=None):
    """Detached-context diagnostic only; never mutate the store or fabricate evidence."""
    removed = set(remove_entities)
    replacements = replacements or {}
    if not removed <= {e.id for e in context.entities} or not set(replacements) <= {
        c.id for c in context.components
    }:
        raise ValueError("Intervention references must be in the retrieved context")
    components = []
    for c in context.components:
        if c.entity_id in removed:
            continue
        if c.id in replacements:
            value = torch.as_tensor(replacements[c.id]).detach().cpu()
            if tuple(value.shape) != c.shape or not torch.isfinite(value).all():
                raise ValueError("Intervention must preserve finite shape")
            c = replace(c, values=tuple(value.reshape(-1).tolist()))
        components.append(c)
    component_ids = {c.id for c in components}

    def retained(endpoint):
        return (
            endpoint.ref not in removed
            if endpoint.kind == "entity"
            else endpoint.ref in component_ids
        )

    return replace(
        context,
        entities=tuple(e for e in context.entities if e.id not in removed),
        components=tuple(components),
        relations=tuple(
            r for r in context.relations if retained(r.source) and retained(r.target)
        ),
        scores={k: v for k, v in context.scores.items() if k not in removed},
        canonical_ids={
            k: v for k, v in context.canonical_ids.items() if k not in removed
        },
        values_used=sum(len(c.values) for c in components),
        intervention="counterfactual context; original source attribution retained",
    )
