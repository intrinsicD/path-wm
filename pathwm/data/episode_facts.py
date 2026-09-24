"""Controlled two-slot lexical retrieval; supplied entity association is explicit.

All 24 words occur in training. Both-alias pairings occur only in the held-out
split. Values/record IDs are not selector inputs. This is not a language corpus.
"""
import torch

WORDS = (
    'shipping', 'billing', 'travel', 'package', 'code', 'date', 'place', 'status',
    'delivery', 'invoice', 'journey', 'parcel', 'number', 'day', 'site', 'stage',
    'dispatch', 'account', 'trip', 'box', 'identifier', 'when', 'location', 'condition',
)
TRAIN_VARIANTS = ((0, 0), (0, 1), (0, 2), (1, 0), (2, 0))
HELD_VARIANTS = ((1, 1), (1, 2), (2, 1), (2, 2))


def phrase(attributes, generator, split):
    variants = TRAIN_VARIANTS if split in ('train', 'iid') else HELD_VARIANTS
    if split not in ('train', 'iid', 'pairing'):
        raise ValueError('Unknown lexical split')
    offsets = torch.tensor(variants)[torch.randint(len(variants), attributes.shape, generator=generator)] * 8
    return torch.stack((attributes // 4, attributes % 4 + 4), dim=-1) + offsets


def features(attributes, entities, target_entity, query):
    """No answer payload, gold index, event position or source identity features."""
    keys = torch.stack((attributes // 4, attributes % 4 + 4), dim=-1)
    return query, keys, entities == target_entity[..., None]


def batch(generator, count=64, split='train'):
    # Each of four supplied identities has exactly eight of sixteen attributes.
    attributes = torch.rand(count, 4, 16, generator=generator).argsort(-1)[..., :8].reshape(count, 32)
    entities = torch.arange(4).repeat_interleave(8).expand(count, -1)
    order = torch.rand(count, 32, generator=generator).argsort(-1)
    attributes, entities = attributes.gather(1, order), entities.gather(1, order)
    target_entity = torch.randint(4, (count,), generator=generator)
    present = torch.zeros(count, 16, dtype=torch.bool)
    rows = torch.arange(count)[:, None].expand(-1, 32)
    matching = entities == target_entity[:, None]
    present[rows[matching], attributes[matching]] = True
    unknown = torch.rand(count, generator=generator) < .2
    eligible = torch.where(unknown[:, None], ~present, present)
    attribute = torch.rand(count, 16, generator=generator).masked_fill(~eligible, -1).argmax(-1)
    query = phrase(attribute, generator, split)
    gold = matching & (attributes == attribute[:, None])
    target = torch.where(unknown, 32, gold.long().argmax(-1))
    return dict(query=query, keys=features(attributes, entities, target_entity, query)[1],
                valid=matching, target=target, attributes=attributes, entities=entities,
                target_entity=target_entity, attribute=attribute)
