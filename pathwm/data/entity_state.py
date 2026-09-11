"""Balanced binary histories; recognition routing never sees state labels."""

import itertools
import torch
from pathwm.models.entity_memory import EntityMemory


def state_episodes(matcher, families):
    episodes = []
    for family_id, family in enumerate(families):
        for a, b, first, second in itertools.product(range(2), repeat=4):
            truth = [a, b]
            truth[first] ^= 1
            truth[second] ^= 1
            entities = [0, 1, first, second, 0, 1]
            operations = [a, b, 2, 2, 3, 3]
            memory = EntityMemory(matcher, capacity=2)
            descriptors = [
                family["descriptors"][i] if t < 2 else family["revisits"][i]
                for t, i in enumerate(entities)
            ]
            slots = []
            for t, descriptor in enumerate(descriptors):
                receipt = memory.observe(str(t), descriptor, t)
                slots.append(
                    receipt["entity_id"] if receipt["entity_id"] is not None else -1
                )
            episodes.append(
                dict(
                    family=family_id,
                    descriptors=descriptors,
                    operations=operations,
                    slots=slots,
                    target=truth,
                )
            )
    return dict(
        observations=torch.eye(4)[torch.tensor([e["operations"] for e in episodes])],
        slots=torch.tensor([e["slots"] for e in episodes]),
        targets=torch.tensor([e["target"] for e in episodes]),
        manifest=episodes,
    )
