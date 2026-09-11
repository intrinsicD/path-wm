import copy
import pytest
import torch
from pathwm.models.entity_relations import RelationKey, EntityRelationMemory
from pathwm.models.entity_state import EntityInteractionCell
from tests.test_entity_memory import Scorer


def key_model():
    model = RelationKey(width=8)
    with torch.no_grad():
        model.encoder.weight.copy_(torch.eye(8))
        model.encoder.bias.zero_()
        model.decoder.weight.copy_(torch.eye(8))
        model.decoder.bias.zero_()
    return model


def test_relation_replacement_and_old_retry():
    matcher, cell, key = Scorer(), EntityInteractionCell(), key_model()
    store = EntityRelationMemory(matcher, cell, key)
    a, b, c = torch.eye(8)[:3]
    for t, q in enumerate((a, b, c)):
        store.observe(str(t), q, t, [1, 0, 0, 0])
    latents = copy.deepcopy(store.state.latents)
    bound = store.bind("bind", a, b, 3)
    assert store.state.latents == latents
    store.observe("toggle", b, 4, [0, 0, 1, 0])
    before = store.snapshot()
    read = store.recall("read", a, 5)
    assert read["source_id"] == 1
    assert store.state.latents[1:] == before["state"]["latents"][1:]
    store.bind("replace", a, c, 6)
    snapshot = store.snapshot()
    assert store.recall("read", a, 5) == read
    assert store.bind("bind", a, b, 3) == bound
    assert store.snapshot() == snapshot
    restored = EntityRelationMemory.restore(matcher, cell, key, snapshot)
    assert restored.recall("next", a, 7)["source_id"] == 2
    first_key = copy.deepcopy(restored.relations["0"])
    restored.bind("other", b, a, 8)
    assert restored.relations["0"] == first_key
    assert restored.recall("other_read", b, 9)["source_id"] == 0
    with pytest.raises(ValueError, match="Conflicting"):
        store.bind("bind", a, c, 3)
    assert store.snapshot() == snapshot


def test_relation_failure_and_snapshot_compatibility():
    matcher, cell, key = Scorer(), EntityInteractionCell(), key_model()
    store = EntityRelationMemory(matcher, cell, key)
    a, b, c = torch.eye(8)[:3]
    store.observe("a", a, 0, [1, 0, 0, 0])
    store.observe("b", b, 1, [0, 1, 0, 0])
    before = store.snapshot()
    with pytest.raises(LookupError):
        store.recall("missing", a, 2)
    with pytest.raises(LookupError):
        store.bind("unknown", a, c, 2)
    assert store.snapshot() == before
    store.bind("bind", a, b, 2)
    snapshot = store.snapshot()
    wrong = copy.deepcopy(key)
    with torch.no_grad():
        wrong.decoder.bias.add_(1)
    with pytest.raises(ValueError):
        EntityRelationMemory.restore(matcher, cell, wrong, snapshot)
    bad = copy.deepcopy(snapshot)
    bad["relations"]["99"] = [0.0] * 8
    with pytest.raises(ValueError):
        EntityRelationMemory.restore(matcher, cell, key, bad)

    class Broken(torch.nn.Module):
        def forward(self, x):
            raise RuntimeError("injected")

    store.state.cell.interaction = Broken()
    before = store.snapshot()
    with pytest.raises(RuntimeError, match="injected"):
        store.recall("failure", a, 3)
    assert store.snapshot() == before


def test_cue_pairs_and_frozen_gradient_path():
    from pathwm.evaluation.entity_relations import relation_examples, evaluate_relations
    from pathwm.evaluation.entity_growth import growth_inputs
    from time import perf_counter

    families = growth_inputs(501, 1)
    matcher = Scorer().requires_grad_(False)
    key = RelationKey()
    data = relation_examples(families)
    loss = torch.nn.functional.cross_entropy(
        matcher.match(key(data["cues"]), data["candidates"]), data["labels"]
    )
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in key.parameters())
    assert all(p.grad is None for p in matcher.parameters())
    result = evaluate_relations(
        matcher, EntityInteractionCell(), key, families, perf_counter() + 120
    )
    for c in result["cohorts"].values():
        assert c["integrity"]
    rows = result["cohorts"]["reference"]["episodes"]
    for left, right in zip(rows[::2], rows[1::2]):
        assert left["initial"] == right["initial"]
        assert left["destination"] == right["destination"]
        assert left["target"] != right["target"]
        assert (
            left["pre_read"]["state"]["latents"]
            == right["pre_read"]["state"]["latents"]
        )


def test_relation_recipe_resume(tmp_path, monkeypatch):
    from pathwm.models.entities import EntityMatchReader
    from pathwm.evaluation import entity_growth
    from experiments.multimodal import train_entity_relations

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
    donor, cell = tmp_path / "matcher.pt", tmp_path / "cell.pt"
    torch.save(
        {"model": {"agent." + k: v for k, v in matcher.state_dict().items()}}, donor
    )
    torch.save({"model": EntityInteractionCell().state_dict()}, cell)
    output = tmp_path / "relations"
    train_entity_relations(donor, cell, output)
    raw = (output / "entity_relations.json").read_bytes()
    train_entity_relations(donor, cell, output, resume=True)
    assert (output / "entity_relations.json").read_bytes() == raw
