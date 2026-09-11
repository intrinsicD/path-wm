import copy
import pytest
import torch
from pathwm.models.entity_state import EntityInteractionCell, EntityStateMemory
from tests.test_entity_memory import Scorer


def test_interaction_binding_retry_and_rollback():
    matcher = Scorer()
    cell = EntityInteractionCell()
    store = EntityStateMemory(matcher, cell)
    a, b = torch.eye(8)[:2]
    store.observe("a", a, 0, [1, 0, 0, 0])
    store.observe("b", b, 1, [0, 1, 0, 0])
    before = store.snapshot()
    with pytest.raises(ValueError):
        store.observe("bad", a, 2, [0, 0, 0, 0], source_id=9)
    assert store.snapshot() == before
    receipt = store.observe("copy", a, 2, [0, 0, 0, 0], source_id=1)
    assert store.latents[1] == before["latents"][1]
    snapshot = store.snapshot()
    assert store.observe("copy", a, 2, [0, 0, 0, 0], source_id=1) == receipt
    assert store.snapshot() == snapshot
    with pytest.raises(ValueError):
        store.observe("copy", a, 2, [0, 0, 0, 0], source_id=0)
    assert store.snapshot() == snapshot
    restored = EntityStateMemory.restore(matcher, cell, snapshot)
    assert restored.snapshot() == snapshot

    class Broken(torch.nn.Module):
        def forward(self, x):
            raise RuntimeError("injected")

    store.cell.interaction = Broken()
    before = store.snapshot()
    with pytest.raises(RuntimeError, match="injected"):
        store.observe("failure", a, 3, [0, 0, 0, 0], source_id=1)
    assert store.snapshot() == before


def test_source_dependency_and_frozen_base():
    cell = EntityInteractionCell()
    a, b = torch.randn(2, cell.width), torch.randn(2, cell.width)
    assert not torch.equal(cell.interact(a, b), cell.interact(a, -b))
    blind = EntityInteractionCell(blind=True)
    assert torch.equal(blind.interact(a, b), blind.interact(a, -b))
    cell.interact(a, b).sum().backward()
    assert all(p.grad is None for p in cell.cell.parameters())
    assert all(p.grad is None for p in cell.head.parameters())
    assert any(p.grad is not None for p in cell.interaction.parameters())
    incompatible = copy.deepcopy(cell)
    incompatible._interaction_blind.fill_(True)
    from pathwm.io import state_hash

    assert state_hash(incompatible) != state_hash(cell)


def test_interaction_histories_and_runtime_agree():
    from pathwm.data.entity_interaction import interaction_episodes
    from pathwm.evaluation.entity_growth import growth_inputs
    from pathwm.evaluation.entity_state import state_runtime

    matcher = Scorer()
    families = growth_inputs(301, 1)
    reference = interaction_episodes(matcher, families, 311)
    swapped = interaction_episodes(matcher, families, 311, "swapped")
    idle = interaction_episodes(matcher, families, 311, "idle")
    assert torch.equal(reference["targets"].flip(-1), swapped["targets"])
    cell = EntityInteractionCell()
    with torch.no_grad():
        expected = cell(
            reference["observations"], reference["slots"], reference["sources"]
        )[1]
        actual = cell(idle["observations"], idle["slots"], idle["sources"])[1]
    assert torch.equal(expected, actual)
    for condition in ["reference", "composition"]:
        data = interaction_episodes(matcher, families, 311, condition)
        for e in data["manifest"]:
            truth = [None, None]
            for target, op, source in zip(
                e["truth_entities"], e["operations"], e["source_entities"]
            ):
                if op < 2:
                    truth[target] = op
                elif op == 2:
                    truth[target] ^= 1
                elif op == 4:
                    truth[target] = truth[source]
            assert truth == e["target"]
        runtime = state_runtime(cell, matcher, data)
        assert runtime["transactions"] and runtime["latent_agreement"]


def test_interaction_recipe_resume(tmp_path, monkeypatch):
    from pathwm.models.entity_state import EntityStateCell
    from pathwm.models.entities import EntityMatchReader
    from pathwm.evaluation import entity_growth
    from experiments.multimodal import train_entity_interaction

    original = entity_growth.growth_inputs
    monkeypatch.setattr(
        entity_growth, "growth_inputs", lambda seed, count: original(seed, 1)
    )
    # Strong deterministic matcher weights keep both initial source records resolved.
    matcher = EntityMatchReader()
    with torch.no_grad():
        matcher.matcher[0].weight.fill_(1)
        matcher.matcher[0].bias.zero_()
        matcher.matcher[2].weight.fill_(-1)
        matcher.matcher[2].bias.fill_(10)
    donor, base = tmp_path / "matcher.pt", tmp_path / "base.pt"
    torch.save(
        {"model": {"agent." + k: v for k, v in matcher.state_dict().items()}}, donor
    )
    torch.save(
        {"model": EntityStateCell(preserve_no_information=True).state_dict()}, base
    )
    output = tmp_path / "run"
    train_entity_interaction(donor, base, output)
    raw = (output / "entity_interaction.json").read_bytes()
    train_entity_interaction(donor, base, output, resume=True)
    assert (output / "entity_interaction.json").read_bytes() == raw
