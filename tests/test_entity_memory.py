import json

import pytest
import torch

from pathwm.models.entities import EntityMatchReader
from pathwm.models.entity_memory import EntityMemory


class Scorer(EntityMatchReader):
    def match(self, query, memory):
        scores = 10 - 20 * (memory - query[:, None]).square().sum(-1)
        return torch.cat((scores, scores.new_zeros((len(query), 1))), -1)


def test_lifecycle_retries_capacity_and_snapshot():
    model = Scorer()
    store = EntityMemory(model, capacity=2, receipt_limit=2)
    a, b, c = torch.eye(8)[:3]
    first = store.observe('a', a, 1)
    assert first['entity_id'] == 0
    assert store.observe('b', b, 2)['entity_id'] == 1
    assert store.observe('a2', a, 3)['entity_id'] == 0
    before = store.snapshot()
    assert store.observe('a2', a, 3)['entity_id'] == 0
    assert store.snapshot() == before
    with pytest.raises(ValueError):
        store.observe('a2', b, 3)
    with pytest.raises(ValueError):
        store.observe('a', a, 1)
    assert store.snapshot() == before
    restored = EntityMemory.restore(model, json.loads(json.dumps(before)))
    assert restored.observe('c', c, 4) == store.observe('c', c, 4)
    assert store.snapshot() == restored.snapshot()
    assert store.observe('c', c, 4)['reason'] == 'capacity'
    assert len(store.snapshot()['records']) == 2


def test_invalid_input_uncertainty_and_ownership():
    store = EntityMemory(Scorer())
    a, b = torch.eye(8)[:2]
    store.observe('a', a, 1)
    store.observe('b', b, 2)
    before = store.snapshot()
    ambiguous = (a + b) / 2**0.5
    # Both close matches receive equal scores; neither may be selected.
    class Tie(Scorer):
        def match(self, query, memory):
            return query.new_zeros((len(query), memory.shape[1] + 1))
    uncertain = EntityMemory(Tie())
    uncertain.observe('a', a, 1)
    assert uncertain.observe('q', ambiguous, 2)['reason'] == 'uncertain'
    for bad in [torch.zeros(8), torch.full((8,), float('nan'))]:
        with pytest.raises(ValueError):
            store.observe('bad', bad, 3)
    assert store.snapshot() == before
    a.zero_()
    snapshot = store.snapshot()
    snapshot['records'][0]['descriptor'][0] = 9
    assert store.snapshot() == before
    altered = Scorer()
    with torch.no_grad():
        altered.null.add_(1)
    with pytest.raises(ValueError):
        EntityMemory.restore(altered, before)


def test_real_matcher_variable_cardinality_and_envelope():
    model = EntityMatchReader()
    inputs = torch.randn(3, 3, 2, 15)
    assert torch.equal(model(inputs)[0], model.match(inputs[:, -1, 0, :8], inputs[:, 0, :, :8]))
    assert model.match(torch.randn(3, 8), torch.randn(3, 4, 8)).shape == (3, 5)
