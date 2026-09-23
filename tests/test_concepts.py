"""Source revision must invalidate every dependent state without resurrection."""

import numpy as np
import pytest
import torch

from pathwm.world_state.records import Limits
from pathwm.world_state.store import WorldStore


def observed(store, name, *, t, content_hash="h", source="camera"):
    tx = store.begin(name, occurred_at=t, available_at=t)
    proof = tx.add_evidence(source, "transition", content_ref=name, content_hash=content_hash)
    store.commit(tx)
    return proof


def derived_chain(store):
    first = observed(store, "obs-1", t=1, content_hash="same-bytes")
    second = observed(store, "obs-2", t=2)
    tx = store.begin("derive", occurred_at=2, available_at=2, kind="internal")
    concept = tx.create_entity("c", kind="concept")
    other = tx.create_entity("d", kind="concept")
    transitions = tx.put_component(concept, "transitions", torch.ones(1), space="s", model_version="v", role="inferred", evidence=(first, second))
    binding = tx.put_component(concept, "binding", torch.ones(1), space="s", model_version="v", role="inferred", parents=(transitions,))
    code = tx.put_component(concept, "code", torch.ones(2), space="s", model_version="v", role="inferred", parents=(binding,))
    key = tx.put_component(concept, "key", torch.ones(2), space="s", model_version="v", role="inferred", parents=(binding,))
    unrelated = tx.put_component(other, "code", torch.zeros(2), space="s", model_version="v", role="inferred", evidence=(second,))
    store.commit(tx)
    return first, second, dict(transitions=transitions, binding=binding, code=code, key=key, unrelated=unrelated)


def test_retraction_invalidates_direct_and_transitive_dependents_only():
    store = WorldStore()
    first, second, ids = derived_chain(store)
    revision = store.revision
    tx = store.begin("retract", occurred_at=1, available_at=3, kind="correction", payload={"source": "camera"})
    tx.retract_evidence(first)
    store.commit(tx)
    for name in ("transitions", "binding", "code", "key"):
        c = store.component(ids[name])
        assert not c.active and c.data["invalidated_by"] == "retract"
    assert store.component(ids["unrelated"]).active
    assert first in store.retractions() and second not in store.retractions()
    old = store.at(revision=revision)
    assert old.component(ids["code"]).active and not old.retractions()
    assert WorldStore.restore(store.snapshot()).retractions() == store.retractions()


def test_retraction_requires_correction_by_same_source():
    store = WorldStore()
    first, _, _ = derived_chain(store)
    before = store.snapshot()
    tx = store.begin("foreign", occurred_at=3, available_at=3, kind="correction", payload={"source": "someone-else"})
    tx.retract_evidence(first)
    with pytest.raises(ValueError, match="source"):
        store.commit(tx)
    tx = store.begin("not-correction", occurred_at=3, available_at=3)
    with pytest.raises(ValueError, match="correction"):
        tx.retract_evidence(first)
    assert store.snapshot() == before


def test_no_resurrection_but_identical_pixels_are_a_new_observation():
    store = WorldStore()
    first, _, ids = derived_chain(store)
    tx = store.begin("retract", occurred_at=1, available_at=3, kind="correction", payload={"source": "camera"})
    tx.retract_evidence(first)
    store.commit(tx)
    before = store.snapshot()
    tx = store.begin("reuse", occurred_at=4, available_at=4, kind="internal")
    tx.put_component(store.component(ids["code"]).entity_id, "code", torch.ones(2), space="s", model_version="v", role="inferred", evidence=(first,))
    with pytest.raises(ValueError, match="[Rr]etracted"):
        store.commit(tx)
    tx = store.begin("reparent", occurred_at=4, available_at=4, kind="internal")
    tx.put_component(store.component(ids["code"]).entity_id, "code", torch.ones(2), space="s", model_version="v", role="inferred", parents=(ids["binding"],))
    with pytest.raises(ValueError, match="inactive"):
        store.commit(tx)
    assert store.snapshot() == before
    # Replaying the original observation event is an idempotent retry, not a revival.
    retry = store.begin("obs-1", occurred_at=1, available_at=1)
    retry.add_evidence("camera", "transition", content_ref="obs-1", content_hash="same-bytes")
    store.commit(retry)
    assert first in store.retractions() and store.snapshot() == before
    # Identical bytes observed again later are valid new source evidence.
    again = observed(store, "obs-again", t=5, content_hash="same-bytes")
    assert again != first and again not in store.retractions()


def test_supersede_replaces_within_same_source_and_modality():
    store = WorldStore()
    first, _, ids = derived_chain(store)
    tx = store.begin("recapture", occurred_at=1, available_at=4)
    new = tx.add_evidence("camera", "transition", content_ref="fixed", content_hash="fixed")
    tx.supersede(first, new)
    store.commit(tx)
    assert store.retractions()[first]["replacement"] == new
    assert not store.component(ids["code"]).active
    tx = store.begin("bad", occurred_at=5, available_at=5)
    wrong = tx.add_evidence("other-camera", "transition", content_ref="x")
    tx.supersede(new, wrong)
    with pytest.raises(ValueError, match="source"):
        store.commit(tx)


def test_capacity_overflow_is_atomic():
    store = WorldStore(Limits(events=3))
    observed(store, "a", t=1)
    observed(store, "b", t=2)
    observed(store, "c", t=3)
    before = store.snapshot()
    with pytest.raises(ValueError, match="capacity"):
        observed(store, "d", t=4)
    assert store.snapshot() == before


def test_concept_memory_blobs_readsets_and_ownership(tmp_path):
    from pathwm.world_state.concepts import ConceptMemory

    memory = ConceptMemory(tmp_path / "memory", versions=dict(perception="p", core="c"))
    frames = np.zeros((2, 64, 64, 3), dtype=np.uint8)
    ref, sha = memory.put_blob(frames)
    assert np.array_equal(memory.get_blob(ref, sha), frames)
    (tmp_path / "memory" / ref).write_bytes(b"x" * frames.nbytes)
    with pytest.raises(ValueError, match="hash"):
        memory.get_blob(ref, sha)
    tx = memory.begin("internal")
    e = tx.create_entity("x", kind="concept")
    cid = tx.put_component(e, "code", torch.ones(2), space="rule-code", model_version="c", role="inferred")
    memory.commit(tx)
    read = memory.read_set([cid])
    assert memory.is_current(read)
    value = memory.tensor(cid)
    value += 5
    assert torch.equal(memory.tensor(cid), torch.ones(2))
    memory.save()
    restored = ConceptMemory.load(tmp_path / "memory", versions=dict(perception="p", core="c"))
    assert restored.store.snapshot() == memory.store.snapshot()
    # R1 rejects model-version changes outright; migration is a deferred contract.
    with pytest.raises(ValueError, match="migration"):
        ConceptMemory.load(tmp_path / "memory", versions=dict(perception="p", core="other"))
    with pytest.raises(TypeError):
        ConceptMemory.load(tmp_path / "memory", versions=dict(perception="p", core="other"), migrate=True)
    assert restored.is_current(read)


def test_relation_on_invalidated_component_endpoint_is_withdrawn():
    from pathwm.world_state.records import Endpoint

    store = WorldStore()
    first, _, ids = derived_chain(store)
    tx = store.begin("link", occurred_at=3, available_at=3, kind="internal")
    tx.relate(Endpoint(ids["code"], "component"), Endpoint(ids["unrelated"], "component"), "similar")
    store.commit(tx)
    relation = store.relations()[-1]
    assert relation.active
    tx = store.begin("retract", occurred_at=1, available_at=4, kind="correction", payload={"source": "camera"})
    tx.retract_evidence(first)
    store.commit(tx)
    assert not next(r for r in store.relations() if r.id == relation.id).active


def test_read_set_is_stale_when_a_newer_derivation_supersedes_an_active_one(tmp_path):
    from pathwm.world_state.concepts import ConceptMemory

    memory = ConceptMemory(tmp_path, versions=dict(perception="p", core="c"))
    tx = memory.begin("internal")
    concept = tx.create_entity("x", kind="concept")
    old = tx.put_component(concept, "code", torch.ones(2), space="rule-code", model_version="c", role="inferred")
    memory.commit(tx)
    read = memory.read_set([old])
    tx = memory.begin("internal")
    tx.put_component(concept, "code", torch.zeros(2), space="rule-code", model_version="c", role="inferred")
    memory.commit(tx)
    assert memory.store.component(old).active  # still a valid historical record
    assert not memory.is_current(read)  # but no longer the current derivation
