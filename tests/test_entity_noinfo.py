import copy
import pytest
import torch
from pathwm.io import state_hash
from pathwm.models.entity_state import EntityStateCell, EntityStateMemory
from tests.test_entity_memory import Scorer


def test_exact_idle_preservation_and_gradients():
    cell = EntityStateCell(preserve_no_information=True)
    h = torch.randn(3, cell.width, requires_grad=True)
    x = torch.tensor([[0., 0., 0., 1.], [0., 0., 1., 0.], [0., 0., .1, 1.]])
    updated = cell.update(x, h)
    assert torch.equal(updated[0], h[0])
    assert torch.equal(updated[1:], cell.cell(x[1:], h[1:]))
    updated[0].sum().backward()
    assert torch.equal(h.grad[0], torch.ones_like(h[0]))
    assert all(p.grad is None or not p.grad.any() for p in cell.cell.parameters())
    previous = h.detach()
    for _ in range(101):
        previous = cell.update(torch.tensor([[0., 0., 0., 1.]]).expand(3, -1), previous)
    assert torch.equal(previous, h.detach())


def test_policy_checkpoint_and_runtime():
    legacy = EntityStateCell()
    cell = EntityStateCell(preserve_no_information=True)
    weights = copy.deepcopy(legacy.state_dict())
    cell.load_state_dict({**weights, '_preserve_no_information': torch.tensor(True)})
    assert state_hash(legacy) != state_hash(cell)
    store = EntityStateMemory(Scorer(), cell)
    descriptor = [1.] + [0.] * 7
    store.observe('first', descriptor, 0, [1., 0., 0., 0.])
    before = copy.deepcopy(store.latents)
    receipt = store.observe('idle', descriptor, 1, [0., 0., 0., 1.])
    assert store.latents == before
    snapshot = store.snapshot()
    assert store.observe('idle', descriptor, 1, [0., 0., 0., 1.]) == receipt
    assert store.snapshot() == snapshot
    assert EntityStateMemory.restore(Scorer(), cell, snapshot).snapshot() == snapshot
    with pytest.raises(ValueError, match='Incompatible'):
        EntityStateMemory.restore(Scorer(), legacy, snapshot)
    clone = EntityStateCell()
    clone.load_state_dict(weights)
    assert state_hash(clone) == state_hash(legacy)
