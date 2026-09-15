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
    for entity, vector in [(a, [1.0, 0.0]), (b, [0.0, 1.0])]:
        tx.put_component(
            entity,
            "recognition",
            torch.tensor(vector),
            space="visual",
            model_version="v1",
            evidence=(proof,),
        )
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
        store.commit(
            store.begin(
                "first", occurred_at=1, available_at=1, payload={"changed": True}
            )
        )
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
    component = update.put_component(
        b,
        "state",
        torch.tensor([0.7]),
        space="belief",
        model_version="v1",
        evidence=(proof,),
        role="inferred",
    )
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
    result = retrieve(
        store,
        Query(entity_ids=(a,), known_at=2),
        RetrievalBudget(entities=1, components=1, values=2),
    )
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
            tx.put_component(
                a, "state", tensor, space="s", model_version="1", evidence=(proof,)
            )
    assert store.revision == 0


def test_identical_retry_and_disk_restore_are_exact(tmp_path):
    store = WorldStore()
    tx = store.begin("event", occurred_at=1, available_at=2)
    a = tx.create_entity("a")
    proof = tx.add_evidence("camera", "image")
    tx.put_component(
        a,
        "state",
        torch.ones(2, 3),
        space="grid",
        model_version="v1",
        evidence=(proof,),
    )
    first = store.commit(tx)
    assert store.commit(tx) == first
    path = tmp_path / "snapshot.json"
    store.save(path)
    restored = WorldStore.load(path)
    assert restored.snapshot() == store.snapshot()
    assert restored.component("event/component/2").tensor().shape == (2, 3)
    snapshot = restored.snapshot()
    snapshot["events"][0]["operations"][2]["value"]["values"][0] = 99
    with pytest.raises(ValueError, match="fingerprint"):
        WorldStore.restore(snapshot)


def test_component_endpoints_follow_corrected_attribution_only_in_new_revision():
    from pathwm.world_state.records import Endpoint

    store = WorldStore()
    a, b, proof = populate(store)
    c = store.latest(a, "recognition")
    tx = store.begin("rel", occurred_at=2, available_at=2, kind="internal")
    tx.relate(Endpoint(c.id, "component"), b, "describes", evidence=(proof,))
    store.commit(tx)
    tx = store.begin("correction", occurred_at=1, available_at=3, kind="correction")
    tx.reassign(c.id, b)
    store.commit(tx)
    assert store.component(c.id).entity_id == b
    assert store.at(known_at=2).component(c.id).entity_id == a


def test_capacity_and_predicted_evidence_cannot_change_last_seen():
    from pathwm.world_state.records import Limits

    store = WorldStore(Limits(entities=1))
    tx = store.begin("first", occurred_at=1, available_at=1)
    a = tx.create_entity()
    proof = tx.add_evidence("cam", "image")
    tx.put_component(
        a, "state", torch.ones(2), space="x", model_version="v1", evidence=(proof,)
    )
    store.commit(tx)
    before = store.snapshot()
    tx = store.begin("over", occurred_at=2, available_at=2)
    tx.create_entity()
    with pytest.raises(ValueError, match="capacity"):
        store.commit(tx)
    assert store.snapshot() == before
    tx = store.begin("prediction", occurred_at=2, available_at=2, kind="prediction")
    with pytest.raises(ValueError, match="observations"):
        tx.add_evidence("model", "latent")
    tx.put_component(
        a,
        "forecast",
        torch.zeros(2),
        space="x",
        model_version="v1",
        evidence=(proof,),
        role="predicted",
    )
    store.commit(tx)
    assert store.entity(a).last_seen == 1
    result = ExactRetriever()(store, Query(entity_ids=(a,)))
    assert all(c.role != "predicted" for c in result.components)


def test_retrieval_version_mismatch_and_counterfactual_do_not_mutate_store():
    from pathwm.world_state.retrieval import intervene

    store = WorldStore()
    a, b, proof = populate(store)
    retrieve = ExactRetriever()
    missing = retrieve(
        store, Query(key=torch.ones(2), space="visual", model_version="v2")
    )
    assert not missing.entities and missing.omitted["incompatible"] == 2
    context = retrieve(store, Query(entity_ids=(a,)))
    before = store.snapshot()
    changed = intervene(
        context, replacements={context.components[0].id: torch.zeros(2)}
    )
    assert changed.intervention and changed.components[0].values == (0.0, 0.0)
    assert store.snapshot() == before and context.components[0].values == (1.0, 0.0)
    assert not intervene(context, remove_entities=(a,)).entities


def test_correction_invalidates_transitive_inferences_and_retry_cannot_resurrect():
    store = WorldStore()
    a, b, proof = populate(store)
    parent = store.latest(a, "recognition")
    tx = store.begin("derived", occurred_at=2, available_at=2, kind="internal")
    child = tx.put_component(
        a,
        "state",
        torch.ones(2),
        space="belief",
        model_version="v1",
        role="inferred",
        evidence=(proof,),
        parents=(parent.id,),
    )
    grandchild = tx.put_component(
        a,
        "forecast",
        torch.ones(2),
        space="belief",
        model_version="v1",
        role="predicted",
        evidence=(proof,),
        parents=(child,),
    )
    receipt = store.commit(tx)
    old = store.at(revision=2)
    correction = store.begin(
        "correction", occurred_at=1, available_at=3, kind="correction"
    )
    correction.reassign(parent.id, b)
    store.commit(correction)
    assert not store.component(child).active and not store.component(grandchild).active
    assert old.component(child).active and old.component(grandchild).active
    assert store.commit(tx) == receipt
    assert not store.component(child).active
    context = ExactRetriever()(
        store, Query(entity_ids=(a,), roles=("inferred", "predicted"))
    )
    assert not context.components and context.omitted["invalidated"] == 2
    assert store.entity(a).last_seen is None


def test_nonspatial_document_without_tensor_and_pinned_alias_view():
    store = WorldStore()
    a, b, proof = populate(store)
    tx = store.begin("doc", occurred_at=2, available_at=2)
    doc = tx.create_entity("notes", kind="document")
    evidence = tx.add_evidence("editor", "text", data={"text": "hello"})
    tx.put_component(
        doc,
        "document",
        None,
        space="literal",
        model_version="v1",
        evidence=(evidence,),
        data={"text": "hello"},
    )
    store.commit(tx)
    pinned = store.at(revision=2)
    merge = store.begin("merge", occurred_at=3, available_at=3, kind="correction")
    merge.merge(a, b, evidence=(proof,))
    store.commit(merge)
    result = ExactRetriever()(pinned, Query(entity_ids=(a,)))
    assert [e.id for e in result.entities] == [a]
    assert store.canonical(a) == store.canonical(b)
    document = ExactRetriever()(store, Query(entity_kind="document"))
    assert document.values_used == 0 and document.components[0].data["text"] == "hello"


def test_entity_labels_and_existence_scores_can_change_without_identity_decay():
    store = WorldStore()
    a, b, proof = populate(store)
    tx = store.begin("rename", occurred_at=20, available_at=20, kind="correction")
    tx.update_entity(
        a, {"label": "a new name", "existence_confidence": 0.8}, evidence=(proof,)
    )
    store.commit(tx)
    entity = store.entity(a)
    assert entity.id == a and entity.label == "a new name" and entity.last_seen == 1
    assert entity.existence_confidence == 0.8
    assert store.entity(b).existence_confidence is None
