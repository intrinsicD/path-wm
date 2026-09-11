"""Counterfactual directed copy histories with supplied, recognized endpoints."""

import itertools
import torch
from pathwm.models.entity_memory import EntityMemory


def interaction_episodes(matcher, families, seed, condition="reference"):
    rng = torch.Generator().manual_seed(seed)
    rows = []
    for family_id, family in enumerate(families):
        middle = list(
            zip(
                torch.randint(2, (4,), generator=rng).tolist(),
                torch.randint(4, (4,), generator=rng).tolist(),
            )
        )
        for a, b, destination, post in itertools.product(range(2), repeat=4):
            bits = [a, b]
            events = [(0, a, -1), (1, b, -1)]
            # Complement reset values as well as initial bits; paired source states stay balanced.
            events += [(e, op ^ bits[e] if op < 2 else op, -1) for e, op in middle]
            if condition == "idle":
                events += [(i % 2, 3, -1) for i in range(31)]
            events += [
                (destination, 4, 1 - destination),
                (destination, 2 if post else 3, -1),
            ]
            if condition == "composition":
                events += [(1 - destination, 4, destination)]
            events += [(0, 3, -1), (1, 3, -1)]
            if condition == "swapped":
                # Relabel the semantic entities while keeping allocation order 0 then 1.
                events = [(0, b, -1), (1, a, -1)] + [
                    (1 - e, op, 1 - src if src >= 0 else -1)
                    for e, op, src in events[2:]
                ]
            truth = [None, None]
            memory = EntityMemory(matcher, capacity=2)
            allocation = {}
            descriptors, slots, sources, operations = [], [], [], []
            for t, (entity, op, source) in enumerate(events):
                if op < 2:
                    truth[entity] = op
                elif op == 2:
                    truth[entity] ^= 1
                elif op == 4:
                    truth[entity] = truth[source]
                descriptor = family["descriptors" if t < 2 else "revisits"][entity]
                receipt = memory.observe(str(t), descriptor, t)
                identity = receipt["entity_id"]
                if t < 2:
                    allocation[entity] = identity
                source_id = allocation.get(source) if source >= 0 else None
                # Missing endpoints must not silently become an ordinary zero-feature event.
                if source >= 0 and source_id is None:
                    raise ValueError("Unresolved interaction source")
                descriptors.append(descriptor)
                slots.append(identity if identity is not None else -1)
                sources.append(source_id if source_id is not None else -1)
                operations.append(op)
            rows.append(
                dict(
                    family=family_id,
                    descriptors=descriptors,
                    slots=slots,
                    sources=sources,
                    operations=operations,
                    target=truth,
                    truth_entities=[e for e, _, _ in events],
                    source_entities=[src for _, _, src in events],
                )
            )
    features = torch.cat([torch.eye(4), torch.zeros(1, 4)])
    return dict(
        observations=features[torch.tensor([r["operations"] for r in rows])],
        slots=torch.tensor([r["slots"] for r in rows]),
        sources=torch.tensor([r["sources"] for r in rows]),
        targets=torch.tensor([r["target"] for r in rows]),
        manifest=rows,
    )
