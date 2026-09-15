"""World-state failures must preserve identity, provenance and the committed revision."""

import pytest
import torch

from pathwm.world_state.store import WorldStore
from pathwm.world_state.retrieval import ExactRetriever, Query, RetrievalBudget


def populate(store):
    tx = store.begin("first", occurred_at=1, available_at=1)
    a = tx.create_entity("cup A")
    b = tx.create_entity("cup B")
    proof = tx.add_evidence("camera", "image", content_ref="frame-1")
    for entity, vector in [(a, [1., 0.]), (b, [0., 1.])]:
        tx.put_component(entity, "recognition", torch.tensor(vector),
                         space="visual", model_version="v1", evidence=(proof,))
    store.commit(tx)
    return a, b, proof


def test_rejected_transaction_and_conflicting_retry_are_atomic():
    store = WorldStore()
    a, b, proof = populate(store)
    before = store.snapshot()
    tx = store.begin("invalid", occurred_at=2, available_at=2)
    tx.create_entity("never published")
    tx.relate(a, "missing", "near", evidence=(proof,))
    with pytest.raises(ValueError, match="reference"):
        store.commit(tx)
    assert store.snapshot() == before
    with pytest.raises(ValueError, match="retry"):
        store.commit(store.begin("first", occurred_at=1, available_at=1,
                                 payload={"changed": True}))
    assert store.snapshot() == before


def test_merge_write_split_preserves_originals_and_rejects_stale_write():
    store = WorldStore()
    a, b, proof = populate(store)
    stale = store.begin("stale", occurred_at=3, available_at=3)
    merge = store.begin("merge", occurred_at=2, available_at=2, kind="correction")
    link = merge.merge(a, b, evidence=(proof,))
    store.commit(merge)
    assert store.canonical(a) == store.canonical(b)
    with pytest.raises(ValueError, match="stale"):
        store.commit(stale)
    update = store.begin("update", occurred_at=3, available_at=3)
    component = update.put_component(b, "state", torch.tensor([0.7]),
                                    space="belief", model_version="v1",
                                    evidence=(proof,), role="inferred")
    store.commit(update)
    split = store.begin("split", occurred_at=4, available_at=4, kind="correction")
    split.split(link)
    store.commit(split)
    assert store.canonical(a) != store.canonical(b)
    assert store.component(component).entity_id == b
    assert store.entity(b).last_seen == 1
    assert WorldStore.restore(store.snapshot()).snapshot() == store.snapshot()


def test_historical_retrieval_does_not_see_late_correction_and_respects_budget():
    store = WorldStore()
    a, b, proof = populate(store)
    tx = store.begin("late", occurred_at=1, available_at=5, kind="correction")
    tx.merge(a, b, evidence=(proof,))
    store.commit(tx)
    retrieve = ExactRetriever()
    result = retrieve(store, Query(entity_ids=(a,), known_at=2),
                      RetrievalBudget(entities=1, components=1, values=2))
    assert result.revision == 1
    assert result.entities[0].id == a
    assert len(result.components) == 1
    assert result.components[0].entity_id == a
    assert result.values_used <= 2
    assert store.revision == 2


def test_store_rejects_attached_or_nonfinite_latents():
    store = WorldStore()
    tx = store.begin("bad", occurred_at=1, available_at=1)
    a = tx.create_entity()
    proof = tx.add_evidence("camera", "image")
    for tensor in (torch.ones(2, requires_grad=True), torch.tensor([float("nan")])):
        with pytest.raises(ValueError):
            tx.put_component(a, "state", tensor, space="s", model_version="1",
                             evidence=(proof,))
    assert store.revision == 0
