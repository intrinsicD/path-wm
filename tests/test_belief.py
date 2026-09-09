"""Contracts for event filtering, bounded evidence and categorical gradients."""

from dataclasses import replace
import io

import pytest
import torch

from experiments.multimodal import build_model
from pathwm.models.belief_state import BeliefState, Packet
from pathwm.training.belief import categorical_kl, split_kl
from pathwm.models.modalities import Observation
from pathwm.evaluation.agent import plan


def model():
    torch.manual_seed(71)
    return build_model(width=16, state_model="belief", memory_recent=2,
                       memory_block=2, memory_blocks=1).eval()


def packet(source="camera", value=0.3, time=1.):
    return Packet(source, "image", Observation(
        torch.full((1, 1, 3, 16, 16), value), torch.tensor([[time]])))


def event(m, state, ordinal, value=0.3):
    pending = m.begin_event(state, event_id=f"event-{ordinal}", ordinal=ordinal,
                            time=float(ordinal), action=torch.zeros(1, 2))
    return m.commit_event(m.add_packet(pending, packet(time=float(ordinal), value=value)))


def test_event_union_deduplicates_without_reapplying_action():
    m = model()
    start = m.initial_state(1)
    pending = m.begin_event(start, event_id="one", ordinal=1, time=1., action=torch.zeros(1, 2))
    a, b = packet("a"), packet("b", 0.7)
    ab = m.add_packet(m.add_packet(pending, a), b)
    ba = m.add_packet(m.add_packet(pending, b), a)
    assert torch.equal(ab.state.logits, ba.state.logits)
    assert torch.equal(ab.state.z, ba.state.z)
    assert torch.equal(ab.state.h, pending.prior.h)
    duplicate = m.add_packet(ab, a)
    assert duplicate is ab
    with pytest.raises(ValueError, match="source"):
        m.add_packet(ab, packet("a", 0.9))
    sealed = m.commit_event(ab)
    assert sealed.observation_count == 1 and len(sealed.memory.recent) == 1
    with pytest.raises(ValueError, match="ordinal|event"):
        m.begin_event(sealed, event_id="one", ordinal=1, time=1.)
    assert start.memory is None


def test_live_and_imagined_share_transition_and_no_observation_creates_no_evidence():
    m = model()
    start = event(m, m.initial_state(1), 1)
    action = torch.ones(1, 2)
    torch.manual_seed(33)
    live = m.begin_event(start, event_id="two", ordinal=2, time=2., action=action).prior
    torch.manual_seed(33)
    imagined = m.imagine(start, action, dt=1., sample=True)
    assert torch.equal(live.h, imagined.h)
    assert torch.equal(live.logits, imagined.logits)
    empty = m.commit_event(m.begin_event(start, event_id="gap", ordinal=2, time=2.))
    assert empty.memory is start.memory
    assert torch.equal(empty.observed_time, start.observed_time)
    instant = m.begin_event(start, event_id="instant", ordinal=2, time=1., action=action)
    assert not torch.equal(instant.prior.h, start.h)
    with pytest.raises(ValueError, match="imagined|hypothetical"):
        m.begin_event(imagined, event_id="bad", ordinal=3, time=3.)


def test_source_features_are_independent_of_belief_and_hidden_values():
    m = model()
    start = m.initial_state(1)
    changed = replace(start, h=start.h + 9, tokens=start.tokens + 9)
    p = packet()
    e1 = m.add_packet(m.begin_event(start, event_id="a", ordinal=1, time=1.), p)
    e2 = m.add_packet(m.begin_event(changed, event_id="a", ordinal=1, time=1.), p)
    assert torch.equal(e1.state.evidence, e2.state.evidence)
    invalid = replace(p, observation=replace(p.observation,
        values=torch.full_like(p.observation.values, float("nan")),
        valid=torch.zeros(1, 1, dtype=torch.bool)))
    masked = m.add_packet(m.begin_event(start, event_id="m", ordinal=1, time=1.), invalid)
    assert torch.equal(masked.state.logits, masked.prior.logits)
    assert m.commit_event(masked).memory is None


def test_hierarchy_preserves_recent_envelope_and_bounds_old_history():
    m = model()
    state = m.initial_state(1)
    for i in range(1, 15):
        state = event(m, state, i, value=i / 20)
        bank = state.memory
        assert len(bank.recent) <= 2 and len(bank.staging) < 2 and len(bank.compressed) <= 1
    assert bank.consolidated is not None
    assert torch.equal(bank.recent[-1].h, state.h)
    assert torch.equal(bank.recent[-1].logits, state.logits)
    assert torch.equal(bank.recent[-1].z, state.z)
    assert not bank.recent[-1].h.requires_grad
    assert float(bank.consolidated.end_time.max()) < float(state.time.min())
    assert m.memory.storage_bytes(bank) > 0
    marked = m.mark(state, author="user", detail="retain the source")
    assert len(marked.memory.protected) == 1
    thought = m.think(marked, steps=2)
    assert torch.equal(thought.h, marked.h) and torch.equal(thought.logits, marked.logits)
    assert thought.memory is marked.memory and torch.equal(thought.time, marked.time)
    fresh = m.initial_state(1)
    assert fresh.memory is None and fresh.session_id != state.session_id


def test_safe_snapshot_roundtrip_rejects_other_schemas():
    m = model()
    state = event(m, m.initial_state(1), 1)
    stream = io.BytesIO()
    torch.save(state.to_dict(), stream)
    stream.seek(0)
    loaded = BeliefState.from_dict(torch.load(stream, weights_only=True))
    for name in ("h", "logits", "z", "time", "observed_time", "tokens"):
        assert torch.equal(getattr(state, name), getattr(loaded, name))
    assert loaded.memory.sources == state.memory.sources
    assert loaded.to("cpu").z.dtype == torch.long
    with pytest.raises(ValueError, match="schema"):
        BeliefState.from_dict({**state.to_dict(), "schema": "pathwm-latent-v2"})


def test_kl_floor_is_after_group_sum_and_routes_gradients():
    q = torch.tensor([[[2., -2.], [2., -2.]]], requires_grad=True)
    p = torch.zeros_like(q, requires_grad=True)
    raw = categorical_kl(q, p).sum(-1)
    dyn, rep = split_kl(q, p, free_nats=0.7)
    assert torch.allclose(dyn, raw.clamp_min(0.7).mean())
    dyn.backward(retain_graph=True)
    assert q.grad is None and p.grad.abs().sum() > 0
    p.grad = None
    rep.backward()
    assert p.grad is None and q.grad.abs().sum() > 0


def test_planner_reuses_complete_starting_draws_across_candidates(monkeypatch):
    m = model()
    state = event(m, m.initial_state(1), 1)
    original = m.imagine
    seen = []
    def spy(branch, *args, **kwargs):
        seen.append(branch.z.clone())
        assert branch.memory is state.memory
        return original(branch, *args, **kwargs)
    monkeypatch.setattr(m, "imagine", spy)
    candidates = torch.zeros(1, 2, 1, 2)
    result = plan(m, state, candidates, lambda s: s.tokens.square().mean((1, 2)),
                  lower=-1., upper=1., samples=4)
    assert len(seen) == 8
    assert all(torch.equal(a, b) for a, b in zip(seen[:4], seen[4:]))
    assert result.scores.shape == (1, 2)
