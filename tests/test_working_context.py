"""Bounded derived references and supervised selection, not language competence."""
import json
import pytest
import torch
from pathwm.models.tasks import ContextSelector, TaskPolicy
from pathwm.world_state.context import WorkingContext
from pathwm.world_state.store import WorldStore


def put(store, event, entity='a', name='detail', value=' exact 01.20 µ '):
    tx = store.begin(event, occurred_at=float(store.revision), available_at=float(store.revision))
    if not any(e.id == entity for e in store.entities()):
        tx.create_entity(entity_id=entity)
    proof = tx.add_evidence('person', 'text', data={'text': value})
    cid = tx.put_component(entity, name, torch.ones(1), space='fact', model_version='v1',
                           evidence=(proof,), role='inferred', data={'text': value})
    store.commit(tx)
    return cid, proof


def test_selector_mask_null_and_gradients():
    head = ContextSelector(8)
    policy = TaskPolicy(8, context_selector=head)
    query = torch.randn(2, 8, requires_grad=True)
    keys = torch.randn(2, 3, 8, requires_grad=True)
    valid = torch.tensor([[True, False, True], [False, False, False]])
    scores = policy.rank_context(query, keys, valid)
    assert scores.shape == (2, 4) and torch.isneginf(scores[0, 1])
    assert scores[1].argmax().item() == 3
    torch.nn.functional.cross_entropy(scores, torch.tensor([0, 3])).backward()
    assert query.grad.abs().sum() > 0 and keys.grad[0, 1].abs().sum() == 0
    with pytest.raises(ValueError):
        policy.rank_context(query, keys, valid.float())
    with pytest.raises(ValueError):
        TaskPolicy(8).rank_context(query, keys, valid)


@pytest.mark.parametrize('layout', ['flat', 'local_global'])
def test_context_capacity_tasks_restart_and_exact_payload(layout):
    store = WorldStore()
    ids = [put(store, f'e{i}', entity=f'entity{i}')[0] for i in range(6)]
    context = WorkingContext(store, capacity=4, layout=layout, representations={'detail': ('fact', 'v1')})
    for i, cid in enumerate(ids):
        context.retain(cid, task='A' if i % 2 else 'B', scope='global' if i % 2 else 'local')
    assert len(context.snapshot()['entries']) <= 4
    assert context.evictions > 0
    a = context.read('A', max_components=2, max_values=2)
    assert a and all(c.entity_id in {'entity1', 'entity3', 'entity5'} for c in a)
    assert all(c.data['text'] == ' exact 01.20 µ ' for c in a)
    saved = json.loads(json.dumps(context.snapshot()))
    restored = WorkingContext.restore(WorldStore.restore(store.snapshot()), saved,
                                      representations={'detail': ('fact', 'v1')})
    assert restored.read('A', max_components=2, max_values=2) == a
    before = store.snapshot()
    context.reset('A')
    assert context.read('A', max_components=2, max_values=2) == ()
    assert store.snapshot() == before


def test_stale_retraction_replacement_version_and_tampered_restart():
    store = WorldStore()
    cid, proof = put(store, 'first')
    context = WorkingContext(store, capacity=2, representations={'detail': ('fact', 'v1')})
    context.retain(cid, task='A')
    put(store, 'replacement', value='new')
    with pytest.raises(ValueError, match='Stale'):
        context.read('A')
    context.reset('A')
    current = store.latest('a', 'detail').id
    context.retain(current, task='A')
    snapshot = context.snapshot()
    with pytest.raises(ValueError):
        WorkingContext.restore(store, snapshot, representations={'detail': ('fact', 'v2')})
    snapshot['entries'][0]['component_id'] = cid
    with pytest.raises(ValueError):
        WorkingContext.restore(store, snapshot, representations={'detail': ('fact', 'v1')})
    tx = store.begin('withdraw', occurred_at=3., available_at=3., kind='correction')
    tx.retract_evidence(store.component(current).evidence[0])
    store.commit(tx)
    with pytest.raises(ValueError, match='Stale'):
        context.read('A')


def test_budget_omission_is_explicit_and_retention_has_no_world_time():
    store = WorldStore()
    cid, _ = put(store, 'first')
    context = WorkingContext(store, capacity=2, representations={'detail': ('fact', 'v1')})
    revision = store.revision
    context.retain(cid, task='A')
    assert store.revision == revision
    with pytest.raises(ValueError, match='budget'):
        context.read('A', max_values=0)
    with pytest.raises(ValueError):
        context.retain(cid, task='A', scope='external')
    context.reset()
    assert not context.snapshot()['entries']
