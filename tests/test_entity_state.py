import pytest
import torch
from pathwm.models.entity_state import EntityStateCell, EntityStateMemory
from tests.test_entity_memory import Scorer


def test_state_retries_failure_and_restore():
    cell = EntityStateCell()
    store = EntityStateMemory(Scorer(), cell)
    a,b = torch.eye(8)[:2]
    obs = torch.eye(4)
    store.observe('a',a,0,obs[0])
    before = store.snapshot()
    store.observe('a',a,0,obs[0])
    assert store.snapshot() == before
    with pytest.raises(ValueError):
        store.observe('a',a,0,obs[1])
    store.observe('b',b,1,obs[1])
    other = store.snapshot()['latents'][1]
    store.observe('toggle',a,2,obs[2])
    assert store.snapshot()['latents'][1] == other
    restored = EntityStateMemory.restore(Scorer(),cell,store.snapshot())
    assert restored.snapshot() == store.snapshot()
    assert torch.equal(restored.read(0),store.read(0))


def test_sequence_and_runtime_latents_agree():
    cell=EntityStateCell()
    ops=torch.eye(4)[torch.tensor([[0,1,2,3]])]
    slots=torch.tensor([[0,1,0,1]])
    _, hidden=cell(ops,slots)
    store=EntityStateMemory(Scorer(),cell)
    points=torch.eye(8)[:2]
    for t in range(4):
        store.observe(str(t),points[slots[0,t]],t,ops[0,t])
    assert torch.allclose(torch.tensor(store.snapshot()['latents']),hidden[0])
