"""Neural interchangeability, diagnosis neutrality and the real agent integration."""

import json
import pytest
import torch
from torch import nn

from pathwm.world_state.modules import (
    Candidate,
    CandidateEncoder,
    AssociationScorer,
    AssociationBinder,
    ContextEncoder,
    RecurrentUpdater,
    ReplaceUpdater,
)
from pathwm.world_state.inspection import WorldTrace, inspect_store
from pathwm.world_state.store import WorldStore
from pathwm.world_state.retrieval import Query, ExactRetriever


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


@pytest.mark.parametrize("training", [False, True])
def test_attention_diagnostics_preserve_native_outputs_and_gradients(training):
    from pathwm.models.modalities import Attend

    torch.manual_seed(801)
    module = Attend(16).train(training)
    query = torch.randn(2, 4, 16, requires_grad=True)
    memory = torch.randn(2, 7, 16, requires_grad=True)
    native = module(query, memory)
    native.square().sum().backward()
    grads = [p.grad.clone() for p in module.parameters()]
    query_grad, memory_grad = query.grad.clone(), memory.grad.clone()
    module.zero_grad()
    query.grad = None
    memory.grad = None
    rng = torch.get_rng_state().clone()
    trace = WorldTrace()
    traced = module(query, memory, trace=trace)
    traced.square().sum().backward()
    assert torch.equal(native, traced)
    assert all(torch.equal(p.grad, g) for p, g in zip(module.parameters(), grads))
    assert torch.equal(query.grad, query_grad) and torch.equal(memory.grad, memory_grad)
    assert torch.equal(rng, torch.get_rng_state())
    assert trace["attention"]["shape"] == [2, 4, 4, 7]


def test_attention_diagnostic_probabilities_match_masked_reference():
    from pathwm.models.modalities import Attend

    module = Attend(16).eval()
    query = torch.randn(2, 4, 16)
    memory = torch.randn(2, 7, 16)
    valid = torch.ones(2, 7, dtype=torch.bool)
    valid[:, 5:] = False
    trace = WorldTrace(tensor_values=1024)
    module(query, memory, valid=valid, causal=True, trace=trace)
    mask = torch.ones(4, 7, dtype=torch.bool).triu(1)
    with torch.no_grad():
        q, k = module.query_norm(query), module.key_norm(memory)
        _, reference = module.attention(
            q,
            k,
            k,
            key_padding_mask=~valid,
            attn_mask=mask,
            need_weights=True,
            average_attn_weights=False,
        )
    assert torch.allclose(trace._tensors["attention"], reference, atol=1e-6)
    assert trace._tensors["attention"][..., 5:].count_nonzero() == 0


def test_inspection_handles_nested_multiscale_outputs_without_changing_them():
    from dataclasses import dataclass

    @dataclass
    class Pyramid:
        levels: tuple

    class Nested(nn.Module):
        def forward(self, x):
            return Pyramid(({"features": x * 2}, x + 1))

    model = Nested()
    x = torch.ones(2, 3, requires_grad=True)
    trace = WorldTrace(tensor_values=12)
    with trace.capture(model, [""], gradients=True):
        out = model(x)
        out.levels[0]["features"].sum().backward()
    assert any("levels.0.features" in k for k in trace)
    assert any(k.startswith("backward.") for k in trace)
    assert torch.equal(x.grad, torch.full_like(x, 2))
