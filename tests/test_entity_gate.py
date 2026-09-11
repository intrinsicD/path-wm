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


def test_gate_examples_and_frozen_credit():
    from pathwm.evaluation.entity_gate import gate_examples, gate_logits
    from pathwm.evaluation.entity_growth import growth_inputs

    data = gate_examples(growth_inputs(601, 2), 701)
    assert len(data["labels"]) == 24
    assert torch.equal(data["old"][::2], data["old"][1::2])
    assert torch.equal(data["new"][::2], data["new"][1::2])
    assert torch.all(data["labels"][::2] != data["labels"][1::2])
    matcher, key = Scorer().requires_grad_(False), key_model().requires_grad_(False)
    gate = RelationWriteGate()
    logits, _ = gate_logits(gate, key, matcher, data)
    torch.nn.functional.cross_entropy(logits, data["labels"]).backward()
    assert any(p.grad is not None and p.grad.abs().sum() for p in gate.parameters())
    assert all(p.grad is None for m in (matcher, key) for p in m.parameters())


@pytest.mark.parametrize("continuation", [False, True])
def test_gate_recipe_resume(tmp_path, monkeypatch, continuation):
    from pathwm.models.entities import EntityMatchReader
    from pathwm.evaluation import entity_growth
    from experiments.multimodal import train_entity_gate

    original = entity_growth.growth_inputs
    monkeypatch.setattr(
        entity_growth, "growth_inputs", lambda seed, count: original(seed, 1)
    )
    matcher = EntityMatchReader()
    with torch.no_grad():
        matcher.matcher[0].weight.fill_(1)
        matcher.matcher[0].bias.zero_()
        matcher.matcher[2].weight.fill_(-1)
        matcher.matcher[2].bias.fill_(10)
    donor, cell, key = [tmp_path / name for name in ("matcher.pt", "cell.pt", "key.pt")]
    torch.save(
        {"model": {"agent." + k: v for k, v in matcher.state_dict().items()}}, donor
    )
    torch.save({"model": EntityInteractionCell().state_dict()}, cell)
    from pathwm.models.entity_relations import RelationKey

    torch.save({"model": RelationKey().state_dict()}, key)
    # Use a key with dependable recognition for the transactional smoke.
    model = RelationKey()
    with torch.no_grad():
        model.encoder.weight.zero_()
        model.encoder.weight[:8].copy_(torch.eye(8))
        model.encoder.bias.zero_()
        model.decoder.weight.zero_()
        model.decoder.weight[:, :8].copy_(torch.eye(8))
        model.decoder.bias.zero_()
    torch.save({"model": model.state_dict()}, key)
    output = tmp_path / "gate"
    kwargs = {}
    if continuation:
        gate_donor = tmp_path / "gate.pt"
        torch.save({"model": RelationWriteGate().state_dict()}, gate_donor)
        kwargs = dict(gate_weights=gate_donor, augmented=True)
    train_entity_gate(donor, cell, key, output, **kwargs)
    raw = (output / "entity_gate.json").read_bytes()
    train_entity_gate(donor, cell, key, output, resume=True, **kwargs)
    assert (output / "entity_gate.json").read_bytes() == raw


def test_augmented_pairs_preserve_supervision():
    from pathwm.evaluation.entity_gate import augmented_gate_examples
    from pathwm.evaluation.entity_growth import growth_inputs

    families = growth_inputs(901, 1)
    control = augmented_gate_examples(families, 911, False)
    augmented = augmented_gate_examples(families, 911, True)
    assert len(control["labels"]) == len(augmented["labels"]) == 48
    for key in ("old", "new", "labels", "active", "accept"):
        assert torch.equal(control[key], augmented[key])
    assert torch.equal(control["cue"][:12], augmented["cue"][:12])
    assert not torch.equal(control["cue"][12:], augmented["cue"][12:])
