import copy
import pytest
import torch
from pathwm.models.entity_relations import EntityRelationMemory, RelationWriteGate
from pathwm.models.entity_state import EntityInteractionCell
from tests.test_entity_memory import Scorer
from tests.test_entity_relations import key_model


def test_gate_transactions():
    matcher, cell, key = Scorer(), EntityInteractionCell(), key_model()
    gate = RelationWriteGate()
    with torch.no_grad():
        for p in gate.parameters():
            p.zero_()
        gate.network[-1].bias.fill_(-10)
    store = EntityRelationMemory(matcher, cell, key, gate_model=gate)
    a, b, c, unknown = torch.eye(8)[:4]
    context = torch.eye(4)[0]
    for t, q in enumerate((a, b, c)):
        store.observe(str(t), q, t, [1, 0, 0, 0])
    store.bind("initial", a, b, 3)
    old = copy.deepcopy(store.relations)
    args = ("ignore", a, unknown, context, context, 4)
    receipt = store.consider(*args)
    assert not receipt["write"]
    assert store.relations == old
    restored = EntityRelationMemory.restore(
        matcher, cell, key, store.snapshot(), gate_model=gate
    )
    assert restored.consider(*args) == receipt
    with pytest.raises(ValueError):
        EntityRelationMemory.restore(matcher, cell, key, store.snapshot())
    with pytest.raises(ValueError, match="Conflicting"):
        store.consider("ignore", a, unknown, context, -context, 4)
    with torch.no_grad():
        store.gate_model.network[-1].bias.fill_(10)
    before = store.snapshot()
    with pytest.raises(LookupError):
        store.consider("bad", a, unknown, context, context, 5)
    assert store.snapshot() == before
    assert store.consider("accept", a, c, context, context, 5)["write"]
    after = store.snapshot()
    assert store.consider(*args) == receipt
    assert store.snapshot() == after
    assert store.recall("read", a, 6)["source_id"] == 2


def test_gate_gradient_and_invalid_context():
    gate = RelationWriteGate()
    p = gate(torch.eye(4), -torch.eye(4))
    p.sum().backward()
    assert any(x.grad is not None and x.grad.abs().sum() for x in gate.parameters())
    store = EntityRelationMemory(
        Scorer(), EntityInteractionCell(), key_model(), gate_model=gate
    )
    before = store.snapshot()
    with pytest.raises(ValueError):
        store.consider(
            "bad", torch.eye(8)[0], torch.eye(8)[1], [float("nan")] * 4, [1, 0, 0, 0], 0
        )
    assert store.snapshot() == before


def test_repeated_ignores_and_threshold():
    gate = RelationWriteGate()
    with torch.no_grad():
        for parameter in gate.parameters():
            parameter.zero_()
    matcher, cell, key = Scorer(), EntityInteractionCell(), key_model()
    store = EntityRelationMemory(matcher, cell, key, gate_model=gate)
    a, b, unknown = torch.eye(8)[:3]
    context = torch.eye(4)[0]
    store.observe("a", a, 0, [1, 0, 0, 0])
    store.observe("b", b, 1, [0, 1, 0, 0])
    store.bind("bind", a, b, 2)
    keys, latents = copy.deepcopy(store.relations), copy.deepcopy(store.state.latents)
    for t in range(3, 34):
        result = store.consider(str(t), a, unknown, context, context, t)
        assert result["write_probability"] == 0.5
        assert not result["write"]
        assert store.relations == keys
        assert store.state.latents == latents
    wrong = copy.deepcopy(gate)
    with torch.no_grad():
        wrong.network[-1].bias.add_(1)
    with pytest.raises(ValueError, match="write gate"):
        EntityRelationMemory.restore(
            matcher, cell, key, store.snapshot(), gate_model=wrong
        )
