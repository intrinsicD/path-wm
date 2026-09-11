"""Controlled frozen-matcher allocation/revisit screen; no model fitting."""

from time import perf_counter
import torch
from pathwm.models.entity_memory import EntityMemory


def growth_inputs(seed=61, count=32):
    generator = torch.Generator().manual_seed(seed)
    families = []
    for _ in range(count):
        points = []
        for _ in range(10000):
            point = torch.randn(8, generator=generator)
            point = point / point.norm()
            if all((point - old).norm() >= 0.9 for old in points):
                points.append(point)
            if len(points) == 9:
                break
        if len(points) != 9:
            raise RuntimeError("Descriptor generation exhausted proposal budget")
        points = torch.stack(points)
        noise = torch.randn((8, 8), generator=generator)
        queries = points[:8] + 0.05 * noise / noise.norm(dim=-1, keepdim=True)
        queries /= queries.norm(dim=-1, keepdim=True)
        families.append(dict(descriptors=points.tolist(), revisits=queries.tolist()))
    return families


def evaluate_growth(model, families, max_seconds=60):
    deadline = perf_counter() + max_seconds
    episodes = []
    for family_id, family in enumerate(families):
        for capacity in (1, 2, 4, 8):
            store = EntityMemory(model, capacity=capacity)
            events = []
            for index in range(capacity):
                events.append(("create", index, family["descriptors"][index]))
            for index in reversed(range(capacity)):
                events.append(("revisit", index, family["revisits"][index]))
            events.append(("overflow", None, family["descriptors"][8]))
            receipts, retry_equal, restore_equal = [], True, True
            restored = None
            bindings = {}
            for time, (kind, identity, descriptor) in enumerate(events):
                if perf_counter() > deadline:
                    raise TimeoutError("Growing-memory screen exceeded budget")
                receipt = store.observe(str(time), descriptor, time)
                before_retry = store.snapshot()
                retry_equal &= store.observe(str(time), descriptor, time) == receipt
                retry_equal &= store.snapshot() == before_retry
                if restored is not None:
                    restore_equal &= (
                        restored.observe(str(time), descriptor, time) == receipt
                    )
                    restore_equal &= restored.snapshot() == store.snapshot()
                if time == capacity - 1:
                    restored = EntityMemory.restore(model, store.snapshot())
                expected_reason = dict(
                    create="created", revisit="matched", overflow="capacity"
                )[kind]
                if kind == "create" and receipt["reason"] == "created":
                    bindings[identity] = receipt["entity_id"]
                expected_id = bindings.get(identity)
                correct = receipt["reason"] == expected_reason and (
                    kind == "create" or receipt["entity_id"] == expected_id
                )
                receipts.append(
                    dict(
                        kind=kind,
                        expected_id=expected_id,
                        truth_entity=identity,
                        correct=correct,
                        receipt=receipt,
                    )
                )
            episodes.append(
                dict(
                    family=family_id,
                    capacity=capacity,
                    events=receipts,
                    retry_equal=retry_equal,
                    restore_equal=restore_equal,
                    final_snapshot=store.snapshot(),
                )
            )
    scores = {}
    for capacity in (1, 2, 4, 8):
        rows = [row for row in episodes if row["capacity"] == capacity]
        metrics = {}
        for kind in ("create", "revisit", "overflow"):
            events = [
                event
                for row in rows
                for event in row["events"]
                if event["kind"] == kind
            ]
            metrics[kind] = dict(
                count=len(events),
                correct=sum(e["correct"] for e in events),
                accuracy=sum(e["correct"] for e in events) / len(events),
                uncertain=sum(e["receipt"]["reason"] == "uncertain" for e in events),
            )
            metrics[kind]["wrong_id"] = sum(
                e["receipt"]["reason"] == "matched"
                and e["receipt"]["entity_id"] != e["expected_id"]
                for e in events
            )
            metrics[kind]["false_split"] = sum(
                kind == "revisit" and e["receipt"]["reason"] == "created"
                for e in events
            )
        distances = []
        if capacity > 1:
            for family in families:
                points = torch.tensor(family["descriptors"][:capacity])
                pairwise = torch.cdist(points, points)
                pairwise.fill_diagonal_(float("inf"))
                distances.extend(pairwise.min(-1).values.tolist())
        metrics["nearest_separation_min"] = min(distances) if distances else None
        metrics["nearest_separation_mean"] = (
            sum(distances) / len(distances) if distances else None
        )
        metrics["transactions"] = all(
            r["retry_equal"] and r["restore_equal"] for r in rows
        )
        metrics["passed"] = metrics["transactions"] and all(
            metrics[k]["accuracy"] >= 0.95 for k in ("create", "revisit", "overflow")
        )
        scores[str(capacity)] = metrics
    return dict(
        scores=scores,
        passed=all(s["passed"] for s in scores.values()),
        episodes=episodes,
    )
