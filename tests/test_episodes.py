"""Evidence, protocol, revision and restart contracts; not language capability."""

import pytest
import torch
from torch import nn

from experiments.multimodal import build_model
from pathwm.world_state.episodes import EpisodeClient
from pathwm.world_state.modules import AssociationBinder, ContextEncoder, ReplaceUpdater
from pathwm.world_state.session import WorldSession


def setup():
    agent = build_model(width=16, state_model="belief", memory_recent=2,
                        memory_block=2, memory_blocks=1).eval()
    binder = AssociationBinder()
    binder.scorer.eval()
    modules = dict(agent=agent, binder=binder, updater=ReplaceUpdater(2).eval(),
                   context_encoder=ContextEncoder(16, {"detail": nn.Linear(2, 16)},
                                                  {"detail": ("detail", "v1")}).eval())
    session = WorldSession(**modules)
    return modules, session, EpisodeClient(session, representations={"detail": ("detail", "v1")})


def test_protocol_retry_interrupt_and_resume():
    modules, session, client = setup()
    client.create("create", "chat")
    read = client.load("chat", ())
    assert client.response("chat", read) == "wait"
    first = client.utterance("u1", "chat", turn_id="turn", phase="start", source="alex", time=1, text="hello")
    assert client.utterance("u1", "chat", turn_id="turn", phase="start", source="alex", time=1, text="hello") == first
    with pytest.raises(ValueError):
        client.validate(read)
    assert client.response("chat", client.load("chat", ())) == "wait"
    before = session.store.revision
    with pytest.raises(ValueError):
        client.utterance("bad", "chat", turn_id="other", phase="end", source="alex", time=2)
    assert session.store.revision == before
    client.utterance("u2", "chat", turn_id="turn", phase="chunk", source="alex", time=2, text=" world")
    client.utterance("u3", "chat", turn_id="turn", phase="end", source="alex", time=3)
    read = client.load("chat", ())
    assert client.response("chat", read) == "respond"
    assert client.response("chat", read, needs_clarification=True) == "clarify"
    restored = WorldSession.restore(session.snapshot(), **modules)
    other = EpisodeClient(restored, representations=client.representations)
    assert other.response("chat", read) == "respond"
    other.utterance("u4", "chat", turn_id="next", phase="start", source="alex", time=4)
    other.utterance("u5", "chat", turn_id="next", phase="cancel", source="alex", time=5)
    assert other.response("chat", other.load("chat", ())) == "wait"


def test_detail_correction_stales_read_and_preserves_other_part():
    _, session, client = setup()
    client.create("create", "person", kind="instance")
    a = client.observe("front", "person", source="camera", modality="image", occurred_at=1, available_at=1)
    b = client.observe("back", "person", source="camera", modality="image", occurred_at=2, available_at=2)
    front = client.publish("p1", "person", "detail", torch.tensor([1., 2.]), evidence=(a,))
    client.representations["back"] = ("detail", "v1")
    back = client.publish("p2", "person", "back", torch.tensor([3., 4.]), evidence=(b,))
    read = client.load("person", ("detail", "back"))
    assert [c.id for c in read.components] == [front, back]
    c = client.observe("correct", "person", source="camera", modality="image", occurred_at=3, available_at=3, supersedes=a)
    with pytest.raises(ValueError):
        client.validate(read)
    current = client.load("person", ("detail", "back"))
    assert [x.id for x in current.components] == [back]
    assert current.omitted == ("detail",)
    replacement = client.publish("p3", "person", "detail", torch.tensor([5., 6.]), evidence=(c,))
    assert client.publish("p3", "person", "detail", torch.tensor([5., 6.]), evidence=(c,)) == replacement
    assert client.load("person", ("detail",), max_values=1).omitted == ("detail",)
    with pytest.raises(ValueError):
        client.publish("p3", "person", "detail", torch.tensor([9., 6.]), evidence=(c,))


def test_validation_is_atomic_and_generated_cannot_be_source():
    _, session, client = setup()
    client.create("create", "chat")
    revision = session.store.revision
    for kwargs in ({"provenance": "generated"}, {"phase": "bogus"}, {"text": 23}):
        args = dict(turn_id="a", phase="start", source="alex", time=1)
        args.update(kwargs)
        with pytest.raises(ValueError):
            client.utterance("bad", "chat", **args)
    with pytest.raises(ValueError):
        client.publish("bad", "chat", "detail", torch.ones(2, requires_grad=True))
    assert session.store.revision == revision
    proof = client.observe("proof", "chat", source="alex", modality="text", occurred_at=1, available_at=1)
    client.publish("p", "chat", "detail", torch.ones(2), evidence=(proof,))
    read = client.load("chat", ("detail",))
    client.representations["detail"] = ("detail", "v2")
    with pytest.raises(ValueError):
        client.validate(read)
    assert client.load("chat", ("detail",)).omitted == ("detail",)


def test_retraction_no_fallback_parent_dependencies_and_episode_isolation():
    modules, session, client = setup()
    client.create("a", "a")
    client.create("b", "b")
    proof = client.utterance("start", "a", turn_id="turn", phase="start", source="alex", time=1,
                             audio_ref="recordings/one.wav")
    client.utterance("end", "a", turn_id="turn", phase="end", source="alex", time=2)
    old = client.publish("old", "a", "detail", torch.tensor([1., 2.]), evidence=(proof,))
    new_proof = client.observe("new-source", "a", source="alex", modality="text", occurred_at=3, available_at=3)
    latest = client.publish("latest", "a", "detail", torch.tensor([3., 4.]), evidence=(new_proof,))
    client.representations["derived"] = ("detail", "v1")
    client.publish("derived", "a", "derived", torch.tensor([5., 6.]), parents=(latest,))
    saved = client.load("a", ("detail", "derived"))
    restored = WorldSession.restore(session.snapshot(), **modules)
    replay = EpisodeClient(restored, representations=client.representations)
    assert replay.validate(saved)
    assert torch.equal(replay.load("a", ("detail",)).components[0].tensor(), torch.tensor([3., 4.]))
    unrelated = client.load("b", ())
    tx = session.store.begin("withdraw", occurred_at=3, available_at=3, kind="correction", payload={"source": "alex"})
    tx.retract_evidence(new_proof)
    session.commit(tx, advance_belief=False)
    assert client.validate(unrelated)
    with pytest.raises(ValueError):
        client.validate(saved)
    assert client.load("a", ("detail", "derived")).omitted == ("detail", "derived")
    assert session.store.component(old).active
    with pytest.raises(ValueError):
        client.response("a", unrelated)
    tx = session.store.begin("withdraw-turn", occurred_at=3, available_at=3, kind="correction", payload={"source": "alex"})
    tx.retract_evidence(proof)
    session.commit(tx, advance_belief=False)
    assert client.response("a", client.load("a", ())) == "wait"


def test_failed_source_replacement_and_future_observation_leave_store_unchanged():
    _, session, client = setup()
    client.create("create", "instance", kind="instance")
    source = client.observe("proof", "instance", source="camera", modality="image", occurred_at=1, available_at=1)
    before = session.snapshot()
    with pytest.raises(ValueError):
        client.observe("wrong", "instance", source="other", modality="image", occurred_at=2, available_at=2, supersedes=source)
    with pytest.raises(ValueError):
        client.observe("future", "instance", source="camera", modality="image", occurred_at=3, available_at=2)
    assert session.store.snapshot() == before["world"]
    assert torch.equal(session.snapshot()["rng"], before["rng"])


def test_output_chunks_revalidate_interruptions_and_complete_once():
    _, session, client = setup()
    client.create("create", "chat")
    client.utterance("s1", "chat", turn_id="t1", phase="start", source="alex", time=1)
    client.utterance("e1", "chat", turn_id="t1", phase="end", source="alex", time=2)
    read = client.load("chat", ())
    count = len(session.store.evidence())
    client.emit("chunk1", "chat", read, audio_ref="generated/chunk1", complete=False)
    assert len(session.store.evidence()) == count
    assert client.response("chat", read) == "respond"
    client.utterance("s2", "chat", turn_id="t2", phase="start", source="alex", time=3)
    revision = session.store.revision
    with pytest.raises(ValueError):
        client.emit("stale-chunk", "chat", read, audio_ref="generated/chunk2")
    assert session.store.revision == revision
    assert client.emit("chunk1", "chat", read, audio_ref="generated/chunk1", complete=False) == "chunk1"
    with pytest.raises(ValueError):
        client.emit("chunk1", "chat", read, text="conflicting retry", complete=False)
    client.utterance("e2", "chat", turn_id="t2", phase="end", source="alex", time=4)
    current = client.load("chat", ())
    client.emit("answer2", "chat", current, text="response")
    assert client.response("chat", current) == "wait"
    with pytest.raises(ValueError):
        client.emit("duplicate-answer", "chat", current, text="response")
    client.utterance("s3", "chat", turn_id="t3", phase="start", source="alex", time=5)
    client.utterance("e3", "chat", turn_id="t3", phase="end", source="alex", time=6)
    assert client.response("chat", client.load("chat", ())) == "respond"


def test_output_rejects_corrected_dependencies_and_distinguishes_omissions():
    _, session, client = setup()
    client.create("chat", "chat")
    client.create("person", "person", kind="instance")
    client.utterance("start", "chat", turn_id="turn", phase="start", source="alex", time=1)
    client.utterance("end", "chat", turn_id="turn", phase="end", source="alex", time=2)
    proof = client.observe("proof", "person", source="camera", modality="image", occurred_at=3, available_at=3)
    client.publish("detail", "person", "detail", torch.ones(2), evidence=(proof,))
    working = client.load("chat", ())
    dependency = client.load("person", ("detail",))
    capacity = client.load("person", ("detail",), max_values=1)
    assert capacity.omission_reasons == (("detail", "capacity"),)
    assert client.load("chat", ("detail",)).omission_reasons == (("detail", "never_observed"),)
    with pytest.raises(ValueError):
        client.emit("too-small", "chat", working, dependencies=(capacity,))
    client.observe("replacement", "person", source="camera", modality="image",
                   occurred_at=4, available_at=4, supersedes=proof)
    with pytest.raises(ValueError):
        client.emit("outdated", "chat", working, dependencies=(dependency,))
    invalid = client.load("person", ("detail",))
    assert invalid.omission_reasons == (("detail", "invalidated"),)
    with pytest.raises(ValueError):
        client.emit("silent-fallback", "chat", working, dependencies=(invalid,))
    assert not any(e.payload.get("operation") == "episode-emit" for e in session.store.events())


def test_read_metadata_integrity_speaker_checks_and_version_output():
    _, session, client = setup()
    client.create("chat", "chat")
    proof = client.utterance("start", "chat", turn_id="turn", phase="start", source="alex", time=1)
    revision = session.store.revision
    with pytest.raises(ValueError):
        client.utterance("wrong-speaker", "chat", turn_id="turn", phase="end", source="other", time=2)
    assert session.store.revision == revision
    client.utterance("end", "chat", turn_id="turn", phase="end", source="alex", time=2)
    client.publish("state", "chat", "detail", torch.ones(2), evidence=(proof,), data={"target": "person"})
    mutated = client.load("chat", ("detail",))
    mutated.components[0].data["target"] = "someone else"
    with pytest.raises(ValueError):
        client.emit("mutated", "chat", mutated, text="answer")
    read = client.load("chat", ("detail",))
    client.representations["detail"] = ("detail", "v2")
    with pytest.raises(ValueError):
        client.emit("version-change", "chat", read)
    incompatible = client.load("chat", ("detail",))
    assert incompatible.omission_reasons == (("detail", "representation"),)
    with pytest.raises(ValueError):
        client.emit("incompatible", "chat", incompatible)


def test_update_between_output_check_and_commit_rejects_transaction(monkeypatch):
    _, session, client = setup()
    client.create("chat", "chat")
    client.utterance("start", "chat", turn_id="turn", phase="start", source="alex", time=1)
    client.utterance("end", "chat", turn_id="turn", phase="end", source="alex", time=2)
    read = client.load("chat", ())
    original = session.commit

    def interrupt(tx, **kwargs):
        client.utterance("new-input", "chat", turn_id="next", phase="start", source="alex", time=3)
        return original(tx, **kwargs)

    monkeypatch.setattr(session, "commit", interrupt)
    with pytest.raises(ValueError):
        client.emit("race", "chat", read, text="obsolete")
    assert session.store.receipt("race") is None


def test_abort_unknown_rederivation_and_output_history_stay_live():
    _, session, client = setup()
    client.create("chat", "chat")
    client.create("person", "person", kind="instance")
    client.utterance("start", "chat", turn_id="turn", phase="start", source="alex", time=1)
    client.utterance("end", "chat", turn_id="turn", phase="end", source="alex", time=2)
    proof = client.observe("proof", "person", source="camera", modality="image", occurred_at=3, available_at=3)
    client.publish("detail", "person", "detail", torch.ones(2), evidence=(proof,))
    main = client.load("chat", ())
    part = client.load("person", ("detail",))
    client.emit("answer", "chat", main, dependencies=(part,), text="generated answer")
    assert client.emission_status("answer") == {"status": "current", "complete": True}
    tx = session.store.begin("withdraw", occurred_at=3, available_at=3, kind="correction", payload={"source": "camera"})
    tx.retract_evidence(proof)
    session.commit(tx, advance_belief=False)
    assert client.emission_status("answer")["status"] == "stale"
    client.record_abort("abort", "chat", main, reason="invalidation")
    assert client.emission_status("answer")["status"] == "aborted"
    assert client.response("chat", client.load("chat", ())) == "respond"
    with pytest.raises(ValueError):
        client.emit("old-read", "chat", main, dependencies=(part,))
    with pytest.raises(ValueError):
        client.emit("implicit-unknown", "chat", client.load("chat", ()),
                    dependencies=(client.load("person", ("detail",)),))
    count = len(session.store.evidence())
    unknown = client.publish_unknown("unknown", "person", "detail", reason="all source evidence withdrawn")
    assert len(session.store.evidence()) == count
    current = client.load("person", ("detail",))
    assert current.components[0].id == unknown and current.components[0].shape == (0,)
    assert current.components[0].data["availability"] == "unknown"
    assert current.components[0].role == "inferred" and not current.components[0].evidence
    client.emit("revised-answer", "chat", client.load("chat", ()), dependencies=(current,), text="uncertain estimate")
    assert client.emission_status("revised-answer")["status"] == "current"
    assert client.response("chat", client.load("chat", ())) == "wait"
    # Retry of the old abort must not withdraw the later revised response.
    assert client.record_abort("abort", "chat", main, reason="invalidation") == "abort"
    assert client.emission_status("revised-answer")["status"] == "current"
    with pytest.raises(ValueError):
        client.record_abort("abort", "chat", main, reason="input")


def test_abort_requires_new_reads_even_when_source_pins_are_unchanged():
    _, _, client = setup()
    client.create("chat", "chat")
    client.utterance("start", "chat", turn_id="turn", phase="start", source="alex", time=1)
    client.utterance("end", "chat", turn_id="turn", phase="end", source="alex", time=2)
    read = client.load("chat", ())
    client.emit("chunk", "chat", read, text="partial", complete=False)
    client.record_abort("abort", "chat", read)
    with pytest.raises(ValueError):
        client.emit("not-refreshed", "chat", read, text="retry")
    client.emit("refreshed", "chat", client.load("chat", ()), text="retry")
    assert client.emission_status("chunk")["status"] == "aborted"
    assert client.emission_status("refreshed")["status"] == "current"
