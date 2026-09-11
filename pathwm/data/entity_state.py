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


def mixed_state_episodes(matcher, families, seed, middle_steps=8):
    """Complemented random templates balance final pairs without target leakage."""
    rng = torch.Generator().manual_seed(seed)
    episodes = []
    for family_id, family in enumerate(families):
        for template in range(4):
            middle = list(
                zip(
                    torch.randint(2, (middle_steps,), generator=rng).tolist(),
                    torch.randint(4, (middle_steps,), generator=rng).tolist(),
                )
            )
            for flip in itertools.product(range(2), repeat=2):
                events = [(0, flip[0]), (1, flip[1])]
                events += [
                    (entity, op ^ flip[entity] if op < 2 else op)
                    for entity, op in middle
                ]
                events += [(0, 3), (1, 3)]
                truth = [None, None]
                memory = EntityMemory(matcher, capacity=2)
                descriptors = []
                slots = []
                operations = []
                for t, (entity, op) in enumerate(events):
                    if op < 2:
                        truth[entity] = op
                    elif op == 2:
                        truth[entity] ^= 1
                    descriptor = (
                        family["descriptors"][entity]
                        if t < 2
                        else family["revisits"][entity]
                    )
                    receipt = memory.observe(str(t), descriptor, t)
                    descriptors.append(descriptor)
                    operations.append(op)
                    slots.append(
                        receipt["entity_id"] if receipt["entity_id"] is not None else -1
                    )
                episodes.append(
                    dict(
                        family=family_id,
                        template=template,
                        descriptors=descriptors,
                        operations=operations,
                        slots=slots,
                        target=truth,
                        truth_entities=[e for e, _ in events],
                    )
                )
    return dict(
        observations=torch.eye(4)[torch.tensor([e["operations"] for e in episodes])],
        slots=torch.tensor([e["slots"] for e in episodes]),
        targets=torch.tensor([e["target"] for e in episodes]),
        manifest=episodes,
    )
