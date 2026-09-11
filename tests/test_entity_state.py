import pytest
import torch
from pathwm.models.entity_state import EntityStateCell, EntityStateMemory
from tests.test_entity_memory import Scorer


def test_state_retries_failure_and_restore():
    cell = EntityStateCell()
    matcher = Scorer()
    store = EntityStateMemory(matcher, cell)
    a, b = torch.eye(8)[:2]
    obs = torch.eye(4)
    store.observe("a", a, 0, obs[0])
    before = store.snapshot()
    store.observe("a", a, 0, obs[0])
    assert store.snapshot() == before
    with pytest.raises(ValueError):
        store.observe("a", a, 0, obs[1])
    store.observe("b", b, 1, obs[1])
    other = store.snapshot()["latents"][1]
    store.observe("toggle", a, 2, obs[2])
    assert store.snapshot()["latents"][1] == other
    restored = EntityStateMemory.restore(matcher, cell, store.snapshot())
    assert restored.snapshot() == store.snapshot()
    assert torch.equal(restored.read(0), store.read(0))


def test_sequence_and_runtime_latents_agree():
    cell = EntityStateCell()
    ops = torch.eye(4)[torch.tensor([[0, 1, 2, 3]])]
    slots = torch.tensor([[0, 1, 0, 1]])
    _, hidden = cell(ops, slots)
    store = EntityStateMemory(Scorer(), cell)
    points = torch.eye(8)[:2]
    for t in range(4):
        store.observe(str(t), points[slots[0, t]], t, ops[0, t])
    assert torch.allclose(torch.tensor(store.snapshot()["latents"]), hidden[0])


def test_failed_and_uncertain_updates_preserve_latents():
    class Broken(torch.nn.Module):
        def forward(self, *args):
            raise RuntimeError("state update failed")

    matcher = Scorer()
    cell = EntityStateCell()
    store = EntityStateMemory(matcher, cell)
    a = torch.eye(8)[0]
    obs = torch.eye(4)[0]
    store.observe("first", a, 0, obs)
    before = store.snapshot()
    store.cell.cell = Broken()
    with pytest.raises(RuntimeError):
        store.observe("second", a, 1, obs)
    assert store.memory.snapshot() == before["memory"]
    assert store.latents == before["latents"]

    class Uncertain(Scorer):
        def match(self, query, memory):
            return query.new_zeros((len(query), memory.shape[1] + 1))

    store = EntityStateMemory(Uncertain(), cell)
    store.observe("first", a, 0, obs)
    before = store.snapshot()
    assert store.observe("second", a, 1, obs)["reason"] == "uncertain"
    assert store.latents == before["latents"]
    assert store.memory.snapshot()["records"] == before["memory"]["records"]


def test_paired_histories_and_recipe_resume(tmp_path, monkeypatch):
    from pathwm.data.entity_state import state_episodes
    from pathwm.evaluation import entity_growth
    from experiments.multimodal import train_entity_state
    from pathwm.models.entities import EntityMatchReader

    original = entity_growth.growth_inputs
    families = original(seed=101, count=1)
    data = state_episodes(Scorer(), families)
    assert len({tuple(e["target"]) for e in data["manifest"]}) == 4
    assert all(
        e["descriptors"][-2:] == data["manifest"][0]["descriptors"][-2:]
        for e in data["manifest"]
    )
    monkeypatch.setattr(
        entity_growth, "growth_inputs", lambda seed, count: original(seed=seed, count=1)
    )
    donor = tmp_path / "donor.pt"
    torch.save(
        {
            "model": {
                "agent." + k: v for k, v in EntityMatchReader().state_dict().items()
            }
        },
        donor,
    )
    output = tmp_path / "state"
    train_entity_state(donor, output)
    raw = (output / "entity_state.json").read_bytes()
    train_entity_state(donor, output, resume=True)
    assert (output / "entity_state.json").read_bytes() == raw
