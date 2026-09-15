"""Neural interchangeability, diagnosis neutrality and the real agent integration."""

import json
import pytest
import torch
from torch import nn

from experiments.multimodal import build_model
from pathwm.world_state.modules import (
    Candidate,
    CandidateEncoder,
    AssociationScorer,
    AssociationBinder,
    ContextEncoder,
    RecurrentUpdater,
    ReplaceUpdater,
    TransitionPredictor,
)
from pathwm.world_state.inspection import WorldTrace, inspect_store
from pathwm.world_state.session import WorldSession
from pathwm.world_state.store import WorldStore
from pathwm.world_state.retrieval import Query, ExactRetriever


def parts(updater=None):
    agent = build_model(
        width=16, state_model="belief", memory_recent=2, memory_block=2, memory_blocks=1
    ).eval()
    binder = AssociationBinder()
    binder.scorer.eval()
    updater = (updater if updater is not None else ReplaceUpdater(2)).eval()
    context = ContextEncoder(
        16,
        {"state": nn.Linear(updater.state_width, 16)},
        {"state": ("belief", "state-v1")},
    ).eval()
    return dict(agent=agent, binder=binder, updater=updater, context_encoder=context)


def candidate(name="a", key=(1.0, 0.0), value=(0.2, 0.8), group=None):
    return Candidate(
        name,
        "camera",
        "image",
        torch.tensor(key),
        torch.tensor(value),
        "visual",
        "v1",
        exclusive_group=group,
    )


def test_differentiable_blocks_and_interchangeable_backbone():
    encoder = CandidateEncoder(
        4, 3, 2, backbone=nn.Sequential(nn.Linear(4, 4), nn.SiLU())
    )
    scorer = AssociationScorer(3, projection=nn.Linear(3, 3))
    updater = RecurrentUpdater(2, 5)
    predictor = TransitionPredictor(5, 2)
    x = torch.randn(8, 4, requires_grad=True)
    k, v = encoder(x)
    scores = scorer(k[:4], k[4:])
    state = updater(torch.zeros(8, 5), v, 1.0)
    prediction, scale = predictor(state, torch.ones(8, 2), 1.0)
    loss = (prediction - 1).square().mean() + scale.mean() + scores.square().mean()
    loss.backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    for module in (encoder, scorer, updater, predictor):
        assert all(
            p.grad is not None and torch.isfinite(p.grad).all()
            for p in module.parameters()
        )


def test_debug_hooks_preserve_outputs_rng_gradients_and_release_graphs(tmp_path):
    torch.manual_seed(12)
    m = nn.Sequential(nn.Linear(4, 5), nn.SiLU(), nn.Linear(5, 2))
    x = torch.randn(3, 4)
    baseline = m(x)
    baseline.square().sum().backward()
    grads = [p.grad.clone() for p in m.parameters()]
    m.zero_grad()
    rng = torch.get_rng_state().clone()
    trace = WorldTrace(max_records=30, tensor_values=10)
    with trace.capture(m, ["0", "2"], gradients=True):
        output = m(x)
        output.square().sum().backward()
    trace.parameters(m)
    assert torch.equal(rng, torch.get_rng_state())
    assert torch.equal(baseline, output)
    assert all(torch.equal(p.grad, g) for p, g in zip(m.parameters(), grads))
    assert all(
        not t.requires_grad and t.grad_fn is None for t in trace._tensors.values()
    )
    assert trace._values <= 10
    assert not m[0]._forward_hooks and not m[2]._forward_hooks
    assert any(k.startswith("backward.") for k in trace)
    record = trace.export(tmp_path)
    assert json.loads((tmp_path / "world_trace.json").read_text()) == record


def test_session_retry_restart_reasoner_and_failed_publication(tmp_path, monkeypatch):
    torch.manual_seed(4)
    modules = parts()
    session = WorldSession(**modules)
    trace = WorldTrace()
    rng = torch.get_rng_state().clone()
    result = session.observe(
        "one", occurred_at=1, available_at=1, candidates=(candidate(),), trace=trace
    )
    assert torch.equal(rng, torch.get_rng_state())
    owner = result["bindings"][0]["entity_id"]
    assert owner is not None
    original = session.snapshot()
    assert (
        session.observe("one", occurred_at=1, available_at=1, candidates=(candidate(),))
        == result
    )
    assert torch.equal(session.state.tokens, original["state"]["tokens"])
    with pytest.raises(ValueError, match="retry"):
        session.observe(
            "one",
            occurred_at=1,
            available_at=1,
            candidates=(candidate(value=(1.0, 0.0)),),
        )
    session.save(tmp_path / "session.pt")
    loaded = torch.load(tmp_path / "session.pt", weights_only=True)
    restored = WorldSession.restore(loaded, **modules)
    second = dict(
        event_id="two",
        occurred_at=2,
        available_at=2,
        candidates=(candidate(value=(0.9, 0.1)),),
    )
    assert session.observe(**second) == restored.observe(**second)
    assert torch.equal(session.state.logits, restored.state.logits)
    assert session.store.snapshot() == restored.store.snapshot()
    start = session.state
    context = ExactRetriever()(session.store, Query(entity_ids=(owner,)))
    assert context.components
    working, _, tokens = session.think(Query(entity_ids=(owner,)), trace=trace)
    assert tokens.values.shape == (1, 1, 16) and tokens.component_ids
    assert torch.equal(start.h, working.h) and torch.equal(start.logits, working.logits)
    assert not torch.equal(start.tokens, working.tokens)
    assert trace.records
    before = session.snapshot()

    def fail(*args, **kwargs):
        raise OSError("disk write failed")

    monkeypatch.setattr("pathwm.world_state.session.atomic_torch", fail)
    with pytest.raises(OSError, match="disk"):
        session.observe(
            "three",
            occurred_at=3,
            available_at=3,
            candidates=(candidate(),),
            save_to=tmp_path / "fail.pt",
        )
    assert before["world"] == session.snapshot()["world"]
    assert torch.equal(before["state"]["tokens"], session.state.tokens)
    assert torch.equal(before["rng"], session.snapshot()["rng"])


def test_unresolved_and_late_candidates_preserve_evidence():
    modules = parts()
    session = WorldSession(**modules)
    first = session.observe(
        "a", occurred_at=2, available_at=2, candidates=(candidate(),)
    )
    owner = first["bindings"][0]["entity_id"]
    late = session.observe(
        "late", occurred_at=1, available_at=3, candidates=(candidate(),)
    )
    assert late["bindings"][0]["status"] == "unresolved"
    assert len(session.store.evidence()) == 2
    assert session.store.latest(owner, "state").valid_from == 2
    collided = session.observe(
        "pair",
        occurred_at=4,
        available_at=4,
        candidates=(candidate("one", group="frame"), candidate("two", group="frame")),
    )
    assert [b["status"] for b in collided["bindings"]] == ["matched", "unresolved"]
    assert len(session.store.evidence()) == 4
    assert len(session.store.entities()) == 1


def test_neural_failure_does_not_publish_graph():
    class Broken(ReplaceUpdater):
        def forward(self, *args):
            raise RuntimeError("broken updater")

    session = WorldSession(**parts(Broken(2)))
    before = session.snapshot()
    with pytest.raises(RuntimeError, match="broken"):
        session.observe("bad", occurred_at=1, available_at=1, candidates=(candidate(),))
    assert session.store.snapshot() == before["world"]
    assert torch.equal(session.state.tokens, before["state"]["tokens"])


def test_model_mutation_rejected_and_context_dimensions_checked():
    modules = parts()
    session = WorldSession(**modules)
    with torch.no_grad():
        next(modules["context_encoder"].projections.parameters()).add_(1)
    with pytest.raises(ValueError, match="model changed"):
        session.snapshot()


def test_optional_concept_control_feedback_and_selection_are_explicit():
    from pathwm.world_state.extensions import (
        ControlBinding,
        Feedback,
        ActionProposal,
        select_action,
        create_prototype,
        RegulatoryModulator,
    )

    store = WorldStore()
    tx = store.begin("e", occurred_at=1, available_at=1)
    proof = tx.add_evidence("camera", "image")
    ids = [tx.create_entity(label) for label in ("a", "b")]
    for i, e in enumerate(ids):
        tx.put_component(
            e,
            "recognition",
            torch.eye(2)[i],
            space="s",
            model_version="1",
            evidence=(proof,),
        )
    store.commit(tx)
    tx = store.begin("concept", occurred_at=2, available_at=2, kind="internal")
    concept = create_prototype(
        tx, [store.latest(e, "recognition") for e in ids], label="proposed cups"
    )
    store.commit(tx)
    assert store.entity(concept).kind == "concept"
    assert store.latest(concept, "prototype").values == (0.5, 0.5)
    control = ControlBinding(ids[0], actions=("look",))
    assert (
        select_action(
            [ActionProposal("write", novelty=99), ActionProposal("look", novelty=1)],
            control,
            mode="novelty",
        ).action
        == "look"
    )
    with pytest.raises(ValueError, match="supplied"):
        select_action([ActionProposal("look")], control, mode="information_gain")
    feedback = Feedback(novelty=0.2).features()
    gain = RegulatoryModulator()(feedback)
    assert gain.shape == (4,) and (gain >= 0.5).all() and (gain <= 1.5).all()


def test_replay_after_reattribution_uses_retained_values():
    from pathwm.world_state.session import rebuild_entity_state

    model = parts()
    session = WorldSession(**model)
    result = session.observe(
        "one",
        occurred_at=1,
        available_at=1,
        candidates=(candidate(), candidate("b", key=(0.0, 1.0), value=(0.8, 0.2))),
    )
    a, b = [r["entity_id"] for r in result["bindings"]]
    source = session.store.latest(a, "recognition")
    tx = session.store.begin("fix", occurred_at=1, available_at=2, kind="correction")
    tx.reassign(source.id, b)
    session.store.commit(tx)
    assert session.store.latest(a, "state") is None
    tx = session.store.begin(
        "rebuild", occurred_at=1, available_at=3, kind="correction"
    )
    result = rebuild_entity_state(tx, session.store, b, model["updater"])
    session.store.commit(tx)
    assert session.store.component(result).active
    assert set(session.store.component(result).parents) == {
        c.id for c in session.store.components(b, "recognition")
    }
    assert len(session.store.evidence()) == 2


def test_context_projection_training_and_runtime_share_exact_values():
    modules = parts()
    session = WorldSession(**modules)
    result = session.observe(
        "one", occurred_at=1, available_at=1, candidates=(candidate(),)
    )
    owner = result["bindings"][0]["entity_id"]
    record = session.store.latest(owner, "state")
    context = ExactRetriever()(session.store, Query(entity_ids=(owner,)))
    encoder = modules["context_encoder"]
    value = record.tensor()[None].requires_grad_()
    training = encoder.project_value("state", value, age=2)
    runtime = encoder(context, now=3).values[:, 0]
    assert torch.equal(training, runtime)
    training.square().sum().backward()
    assert value.grad.abs().sum() > 0


def test_simultaneous_lateness_and_collision_is_diagnosable():
    session = WorldSession(**parts())
    session.observe("one", occurred_at=2, available_at=2, candidates=(candidate(),))
    result = session.observe(
        "late",
        occurred_at=1,
        available_at=3,
        candidates=(candidate("a", group="f"), candidate("b", group="f")),
    )
    assert all(
        r["status"] == "unresolved" and "replay" in r["reason"]
        for r in result["bindings"]
    )
    assert len(session.store.evidence()) == 3 and len(session.store.entities()) == 1


def test_relational_context_and_query_network_are_trainable():
    from dataclasses import replace
    from pathwm.world_state.modules import RelationEncoder, QueryGenerator

    store = WorldStore()
    tx = store.begin("event", occurred_at=1, available_at=1)
    a, b = tx.create_entity("a"), tx.create_entity("b")
    proof = tx.add_evidence("camera", "image")
    tx.relate(a, b, "near", evidence=(proof,))
    store.commit(tx)
    context = ExactRetriever()(
        store, Query(entity_ids=(a,), neighbors=True, relation_type="near")
    )
    encoder = ContextEncoder(16, {}, {}, relation_encoder=RelationEncoder(16, ["near"]))
    first = encoder(context, now=1)
    assert first.values.shape == (1, 1, 16) and len(first.relation_ids) == 1
    relation = context.relations[0]
    swapped = replace(
        context,
        relations=(replace(relation, source=relation.target, target=relation.source),),
    )
    assert not torch.equal(first.values, encoder(swapped, now=1).values)
    query = QueryGenerator(16, 4)
    key = query(first.values[:, 0])
    key.square().sum().backward()
    assert all(p.grad is not None for p in encoder.relation_encoder.parameters())
    assert all(p.grad is not None for p in query.parameters())


def test_actual_image_encoder_packets_and_candidate_features_share_one_event():
    from pathwm.models.modalities import Observation
    from pathwm.models.belief_state import Packet

    modules = parts()
    agent = modules["agent"]
    observation = Observation(torch.full((1, 1, 3, 16, 16), 0.4), torch.tensor([[1.0]]))
    encoded = agent.encoders["image"](observation).as_tokens().values.mean(1)
    encoder = CandidateEncoder(16, 4, 2).eval()
    key, value = encoder(encoded)
    c = Candidate("region-0", "camera", "image", key[0], value[0], "region", "v1")
    session = WorldSession(**modules)
    result = session.observe(
        "image",
        occurred_at=1,
        available_at=2,
        candidates=(c,),
        packets=(Packet("camera", "image", observation),),
    )
    assert result["bindings"][0]["entity_id"] is not None
    assert session.state.observation_count == 1 and session.state.evidence_valid.all()
    assert len(session.store.evidence()) == 1
    assert session.store.evidence()[0].occurred_at == 1
    assert session.store.evidence()[0].available_at == 2


def test_report_escapes_untrusted_labels_and_exposes_debug_sections(tmp_path):
    from pathwm.evaluation.world_state import world_state_inspection
    from pathwm.io import atomic_json

    store = WorldStore()
    tx = store.begin("e", occurred_at=1, available_at=1)
    tx.create_entity("<script>bad()</script>")
    store.commit(tx)
    atomic_json(tmp_path / "world_state.json", inspect_store(store))
    trace = WorldTrace(max_records=2, tensor_values=2)
    trace["x"] = torch.ones(3)
    trace.record("one", {"value": 1})
    trace.record("two", {})
    trace.record("dropped", {})
    trace.export(tmp_path)
    html = "".join(world_state_inspection(tmp_path))
    assert "<script>bad()" not in html and "&lt;script&gt;" in html
    assert "data-world-entity" in html and "world-filter" in html
    assert "Binding, retrieval" in html and "Dropped records: 1" in html
    assert not trace._tensors
