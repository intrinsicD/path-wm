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
    store.observe('a', a, 0, [1, 0, 0, 0])
    store.observe('b', b, 1, [0, 1, 0, 0])
    before = store.snapshot()
    with pytest.raises(ValueError):
        store.observe('bad', a, 2, [0, 0, 0, 0], source_id=9)
    assert store.snapshot() == before
    receipt = store.observe('copy', a, 2, [0, 0, 0, 0], source_id=1)
    assert store.latents[1] == before['latents'][1]
    snapshot = store.snapshot()
    assert store.observe('copy', a, 2, [0, 0, 0, 0], source_id=1) == receipt
    assert store.snapshot() == snapshot
    with pytest.raises(ValueError):
        store.observe('copy', a, 2, [0, 0, 0, 0], source_id=0)
    assert store.snapshot() == snapshot
    restored = EntityStateMemory.restore(matcher, cell, snapshot)
    assert restored.snapshot() == snapshot
    class Broken(torch.nn.Module):
        def forward(self, x):
            raise RuntimeError('injected')
    store.cell.interaction = Broken()
    before = store.snapshot()
    with pytest.raises(RuntimeError, match='injected'):
        store.observe('failure', a, 3, [0, 0, 0, 0], source_id=1)
    assert store.snapshot() == before


def test_source_dependency_and_frozen_base():
    cell = EntityInteractionCell()
    a, b = torch.randn(2, cell.width), torch.randn(2, cell.width)
    assert not torch.equal(cell.interact(a,b),cell.interact(a,-b))
    blind = EntityInteractionCell(blind=True)
    assert torch.equal(blind.interact(a,b),blind.interact(a,-b))
    cell.interact(a,b).sum().backward()
    assert all(p.grad is None for p in cell.cell.parameters())
    assert all(p.grad is None for p in cell.head.parameters())
    assert any(p.grad is not None for p in cell.interaction.parameters())
    incompatible = copy.deepcopy(cell)
    incompatible._interaction_blind.fill_(True)
    from pathwm.io import state_hash
    assert state_hash(incompatible) != state_hash(cell)
