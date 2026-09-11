"""Frozen-state temporal controls with independently simulated binary targets."""

import itertools
import torch
from pathwm.models.entity_memory import EntityMemory


def temporal_episodes(matcher, families, idle_lengths=()):
    cohorts = {}
    names = ("reference", "reset_order", "toggle_length", "no_information_length")
    for name in (*names, *(f"idle_{n}" for n in idle_lengths)):
        rows = []
        for family_id, family in enumerate(families):
            for a, b, u, v in itertools.product(range(2), repeat=4):
                middle = [(u, 2), (v, 2)]
                if name == "reset_order":
                    middle = [(0, u), (0, 2)] if v else [(0, 2), (0, u)]
                elif name == "toggle_length":
                    middle *= 8
                elif name == "no_information_length":
                    middle += [(i % 2, 3) for i in range(14)]
                elif name.startswith("idle_"):
                    middle += [(i % 2, 3) for i in range(int(name[5:]))]
                events = [(0, a), (1, b)] + middle + [(0, 3), (1, 3)]
                truth = [None, None]
                memory = EntityMemory(matcher, capacity=2)
                descriptors = []
                slots = []
                operations = []
                for t, (entity, operation) in enumerate(events):
                    if operation in (0, 1):
                        truth[entity] = operation
                    elif operation == 2:
                        truth[entity] ^= 1
                    descriptor = (
                        family["descriptors"][entity]
                        if t < 2
                        else family["revisits"][entity]
                    )
                    receipt = memory.observe(str(t), descriptor, t)
                    descriptors.append(descriptor)
                    operations.append(operation)
                    slots.append(
                        receipt["entity_id"] if receipt["entity_id"] is not None else -1
                    )
                rows.append(
                    dict(
                        family=family_id,
                        descriptors=descriptors,
                        operations=operations,
                        slots=slots,
                        target=truth,
                        truth_entities=[e for e, _ in events],
                    )
                )
        cohorts[name] = dict(
            observations=torch.eye(4)[torch.tensor([r["operations"] for r in rows])],
            slots=torch.tensor([r["slots"] for r in rows]),
            targets=torch.tensor([r["target"] for r in rows]),
            manifest=rows,
        )
    return cohorts
