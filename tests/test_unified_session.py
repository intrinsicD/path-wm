"""R2 unified-session software contracts with random weights.

SOFTWARE fixtures: binding keys are supplied for the two slots the agent's OWN layout
calls machines, and "demonstrations" are supplied ok transitions aimed at the agent's
own perceived machine/object pixels (random perception rarely finds the real ones).
No scene identity, mask or rule reaches the session or the concept memory."""

import json
import math
import sys

import numpy as np
import pytest
import torch

from experiments import unified_session as us
from pathwm.data import rule_world as rw
from pathwm.models.latent_core import Decision
from pathwm.models.belief_state import Packet
from pathwm.models.modalities import Observation
from pathwm.models.slots import pointer
from pathwm.models.tasks import Actor, TaskRequest
from pathwm.world_state.concepts import SOURCE, AgentSettings
from pathwm.world_state.modules import AssociationBinder, Candidate
from pathwm.world_state.session import SourceItem
from pathwm.world_state.unified import GoalSpec, TypedAction, VerificationRecord

# Always found a new concept for enough evidence: concept membership becomes
# deterministic under random weights (software policy, not a calibrated setting).
CREATE = AgentSettings(lam=-1e6)


def unit(degrees, plane=0):
    key = torch.zeros(8)
    key[2 * plane] = math.cos(math.radians(degrees))
    key[2 * plane + 1] = math.sin(math.radians(degrees))
    return key


def setup(tmp_path, *, settings=CREATE, binder=None, seed=0):
    modules = us.build(seed)
    agent = us.new_agent(modules, tmp_path / "memory", settings=settings,
                         binder=binder or AssociationBinder(new_threshold=0.5))
    g = torch.Generator().manual_seed(seed)
    train = rw.split_rules()["train"]
    scene = rw.sample_scenes(g, torch.tensor([list(rw.KIND_SPLIT["train"][:2])]))
    return modules, agent, scene, (train[0], train[1]), g


def layout(agent, frame):
    return agent._layout(agent.percept(frame)[1])


def supplied(agent, frame, keys):
    """Test-chosen keys for the agent's own two machine slots (left to right)."""
    machines, objects = layout(agent, frame)
    assert len(machines) == 2 and len(objects) >= 2
    slots = [m["slot"] for m in machines]
    _, values = agent.candidate_encoder(agent.percept(frame)[1].slots[0])
    return lambda p, rgb: tuple(
        Candidate(f"slot-{k}", SOURCE, "image", key, values[k].detach(), "supplied", "fixture-v1",
                  exclusive_group="frame")
        for k, key in zip(slots, keys) if key is not None
    )


def see(agent, world, keys, **kwargs):
    frame = world.frame()
    return agent.observe(frame, candidates=supplied(agent, frame, keys), **kwargs)


def demos(agent, world, side, n, g):
    """Supplied ok transitions aimed at the agent's own perceived machine `side`."""
    machines, objects = layout(agent, world.frame())
    out = []
    for _ in range(n):
        i, j = torch.randperm(len(objects), generator=g)[:2].tolist()
        record = rw.ActionRecord(machines[side]["xy"], objects[i]["xy"], objects[j]["xy"])
        pre, post = (rw.render(world.scenes, torch.randint(2, (1, 2), generator=g))[0][0] for _ in "ab")
        out.append((pre, record, "ok", post))
    return out


def both(agent, world, n, g):
    return demos(agent, world, 0, n, g) + demos(agent, world, 1, n, g)


def decision(agent, side):
    """Session decision for the agent's own machine slot `side` (left to right)."""
    return agent.view.decisions[agent.view.machines[side]["slot"]]


def active(agent, instance, name):
    comps = [c for c in agent.memory.view()["components"].values() if c.entity_id == instance and c.name == name]
    latest = max(comps, key=lambda c: (c.valid_from, c.revision, c.id), default=None)
    return latest if latest is not None and latest.active else None


def supported(agent):
    """All transition evidence currently supporting any active code."""
    return {e for c in agent.memory.view()["components"].values() if c.name == "code" and c.active
            for e in c.evidence}


def test_one_store_identity_and_concept_owners(tmp_path):
    modules, agent, scene, rules, g = setup(tmp_path)
    world = rw.RuleWorld(scene, rules, torch.zeros(2, dtype=torch.long), rw.TaskContract())
    see(agent, world, (unit(0), unit(0, 1)), transitions=both(agent, world, 4, g))
    a, b = (decision(agent, s) for s in (0, 1))
    assert a["status"] == b["status"] == "new"
    more = both(agent, world, 1, g)
    see(agent, world, (unit(3), unit(2, 1)), transitions=more)
    again = decision(agent, 0)
    assert again["status"] == "matched" and again["entity_id"] == a["entity_id"]
    # One store: the concept client reads the session's store and has none of its own.
    assert agent.memory.store is agent.session.store and agent.memory._store is None
    events = agent.session.store.events()
    assert [e.id for e in events if e.kind == "observation"] == ["obs-000001", "obs-000002"]
    assert agent.session.time == events[-1].available_at == 2.0
    assert agent.session.state.event_id == "obs-000002"  # bookkeeping never steps the belief
    # Attribution comes from session identity: A holds both events' m1 transitions.
    transitions = active(agent, a["entity_id"], "transitions")
    first = [e.id for e in agent.memory.view()["evidence"].values() if e.modality == "transition"]
    assert len(transitions.evidence) == 5 and set(transitions.evidence) <= set(first)
    assert all(agent.memory.view()["components"][p].name == "attribution" for p in transitions.parents)
    concepts = {active(agent, x, "binding").data["concept"] for x in (a["entity_id"], b["entity_id"])}
    assert len(concepts) == 2
    # No bypass: observations must go through the session, direct store writes are caught.
    with pytest.raises(ValueError, match="WorldSession.observe"):
        agent.memory.begin("observation")
    answers = {c: agent.concept_state(c)[0] for c in concepts}
    agent.save(tmp_path / "memory")
    us.save_models(modules, tmp_path / "models.pt")
    restored = us.restore_agent(us.load_models(tmp_path / "models.pt", seed=5), tmp_path / "memory",
                                settings=CREATE, binder=AssociationBinder(new_threshold=0.5))
    assert restored.session.store.snapshot() == agent.session.store.snapshot()
    for c, z in answers.items():
        assert torch.equal(restored.concept_state(c)[0], z)
    bypass = agent.session.store.begin("bypass", occurred_at=2.0, available_at=2.0, kind="internal")
    bypass.create_entity("uncoordinated")
    agent.session.store.commit(bypass)
    with pytest.raises(ValueError, match="WorldSession.commit"):
        agent.memory.commit(agent.memory.begin("internal"))


def test_identity_split_invalidates_dependent_attribution_and_preserves_others(tmp_path):
    modules, agent, scene, rules, g = setup(tmp_path)
    world = rw.RuleWorld(scene, rules, torch.zeros(2, dtype=torch.long), rw.TaskContract())
    see(agent, world, (unit(0), unit(0, 1)), transitions=both(agent, world, 4, g))
    a, b = (decision(agent, s)["entity_id"] for s in (0, 1))
    b_binding, b_code = active(agent, b, "binding"), active(agent, active(agent, b, "binding").data["concept"], "code")
    see(agent, world, (unit(70), unit(0, 1)))  # identity failure: m1 looks new
    a2 = decision(agent, 0)
    assert a2["status"] == "new"
    image = next(e.id for e in agent.memory.view()["evidence"].values()
                 if e.event_id == "obs-000002" and e.modality == "image")
    tx = agent.session.store.begin("merge-1", occurred_at=3.0, available_at=3.0, kind="correction")
    link = tx.merge(a, a2["entity_id"], evidence=(image,))
    agent.correct(tx)
    _, result = see(agent, world, (unit(35), unit(0, 1)), transitions=demos(agent, world, 0, 4, g))
    third = decision(agent, 0)
    # The match exists only because of the merge: without it the two groups tie.
    assert third["status"] == "matched" and third["entity_id"] == a and third["identity_links"] == [link]
    dependent = set(result["evidence"][1:5])
    assert dependent <= set(active(agent, a, "transitions").evidence) and dependent <= supported(agent)
    tx = agent.session.store.begin("split-1", occurred_at=5.0, available_at=5.0, kind="correction")
    tx.split(link)
    _, repaired = agent.correct(tx)
    components = agent.memory.view()["components"]
    assert not components[third["recognition"]].active
    assert a in repaired and not dependent & set(active(agent, a, "transitions").evidence)
    assert not dependent & supported(agent)
    assert all(not c.active for c in components.values()
               if c.name == "attribution" and c.data["transition"] in dependent)
    # Raw evidence survives; the unaffected concept and binding are bit-identical.
    evidence = agent.memory.view()["evidence"]
    assert dependent <= set(evidence) and not dependent & set(agent.memory.view()["retracted"])
    assert active(agent, b, "binding") == b_binding
    assert active(agent, b_binding.data["concept"], "code") == b_code


def test_reassign_moves_attribution_and_reverifies_membership(tmp_path):
    modules, agent, scene, rules, g = setup(tmp_path)
    world = rw.RuleWorld(scene, rules, torch.zeros(2, dtype=torch.long), rw.TaskContract())
    _, first = see(agent, world, (unit(0), unit(0, 1)),
                   transitions=both(agent, world, 4, g))
    a, b = (decision(agent, s)["entity_id"] for s in (0, 1))
    b_binding = active(agent, b, "binding")
    b_code = active(agent, b_binding.data["concept"], "code")
    _, second = see(agent, world, (unit(0), unit(0, 1)),
                    transitions=demos(agent, world, 0, 4, g))
    r2 = decision(agent, 0)["recognition"]
    moved = set(second["evidence"][1:5])
    assert moved <= set(active(agent, a, "transitions").evidence)
    tx = agent.session.store.begin("reassign-1", occurred_at=3.0, available_at=3.0, kind="correction")
    d = tx.create_entity(kind="instance")
    tx.reassign(r2, d)
    _, repaired = agent.correct(tx)
    view = agent.memory.view()
    assert view["components"][r2].active and view["components"][r2].entity_id == d
    assert set(active(agent, a, "transitions").evidence) == set(first["evidence"][1:5])
    assert set(active(agent, d, "transitions").evidence) == moved
    assert active(agent, d, "appearance").parents == (r2,)  # re-derived from the retained frame
    assert repaired[d]["status"] in ("created", "bound") and active(agent, d, "binding").data["supports"]
    assert active(agent, b, "binding") == b_binding
    assert active(agent, b_binding.data["concept"], "code") == b_code


def test_planner_reads_only_the_shared_core(tmp_path, monkeypatch):
    modules, agent, scene, rules, g = setup(tmp_path)
    world = rw.RuleWorld(scene, rules, torch.zeros(2, dtype=torch.long), rw.TaskContract())
    see(agent, world, (unit(0), unit(0, 1)), transitions=both(agent, world, 4, g))
    belief = agent.session.agent
    calls = dict(dynamics=[], imagine=0, apply=0)
    dynamics = belief.dynamics.forward
    monkeypatch.setattr(belief.dynamics, "forward",
                        lambda state, action, *a, **k: calls["dynamics"].append(action) or dynamics(state, action, *a, **k))
    imagine = belief.imagine
    monkeypatch.setattr(belief, "imagine", lambda *a, **k: calls.__setitem__("imagine", calls["imagine"] + 1) or imagine(*a, **k))
    apply = agent.core.apply
    monkeypatch.setattr(agent.core, "apply", lambda *a, **k: calls.__setitem__("apply", calls["apply"] + 1) or apply(*a, **k))
    instances = [m["instance"] for m in agent.view.machines]
    assert None not in instances
    decision, read = agent.plan(agent.view, (1, 1), presses_done=0)
    assert calls["apply"] > 0 and calls["dynamics"] == [] and calls["imagine"] == 0
    action = TypedAction("press", instances[0], 0, 1, "t")
    agent.execute(rw.Actuator(world), action, read, candidates=supplied(agent, world.frame(), (unit(0), unit(0, 1))))
    assert len(calls["dynamics"]) == 1 and calls["dynamics"][0] is not None
    assert calls["imagine"] == 0
    assert agent.session.state.sources == (SOURCE,)  # only the camera packet, no predictions


def test_structured_goal_plans_and_free_text_asks(tmp_path):
    modules, agent, scene, rules, g = setup(tmp_path)
    world = rw.RuleWorld(scene, rules, torch.zeros(2, dtype=torch.long), rw.TaskContract())
    see(agent, world, (unit(0), unit(0, 1)))
    a, b = (m["instance"] for m in agent.view.machines)
    actuator = us.CountingActuator(world)
    request = TaskRequest("t1", "turn-both-lamps-on", Actor("user", "tester"))
    assert agent.run_task(actuator, None, request=request)["option"] == "ask" and actuator.presses == 0
    goal = GoalSpec("t1", "lamp_state", ((a, 1), (b, 0)))
    dispatch = agent.dispatch(request, goal)
    assert dispatch.kind == "plan" and dispatch.goal == (1, 0)
    decision, _ = agent.plan(agent.view, dispatch.goal, presses_done=0)
    assert decision.values["abstain"] == agent.contract.utility("abstain", 0)
    assert agent.contract.utility("false_stop", 2) <= decision.value <= agent.contract.utility("success", 0)
    assert agent.dispatch(request, GoalSpec("t1", "open_door", ((a, 1),))).kind == "unsupported"
    assert agent.dispatch(request, GoalSpec("t1", "lamp_state", (("nope", 1), (b, 0)))).kind == "unsupported"
    assert agent.dispatch(request, GoalSpec("t1", "lamp_state", ((a, 1),))).kind == "ask"  # incomplete
    for bad in (TypedAction("jump", a, 0, 1), TypedAction("press", "nope", 0, 1)):
        with pytest.raises(ValueError):
            agent.execute(actuator, bad, agent.memory.read_set(()))
    with pytest.raises(ValueError, match="Unknown action"):
        agent.action_encoder("jump", torch.zeros(1, 3, 64))
    assert actuator.presses == 0
    tx = agent.session.store.begin("merge-ab", occurred_at=2.0, available_at=2.0, kind="correction")
    tx.merge(a, b, evidence=(agent.memory.view()["evidence"]["obs-000001/evidence/0"].id,))
    agent.correct(tx)
    assert agent.dispatch(request, goal).reason == "ambiguous target reference"


def test_image_packet_encoded_once_for_all_consumers(tmp_path, monkeypatch):
    modules, agent, scene, rules, g = setup(tmp_path)
    world = rw.RuleWorld(scene, rules, torch.zeros(2, dtype=torch.long), rw.TaskContract())
    encoder = agent.perception.encoder
    assert agent.session.agent.encoders["image"] is encoder
    calls = []
    handle = encoder.register_forward_hook(lambda m, i, o: calls.append(o))
    seen = {}
    add = agent.session.agent.add_packet
    monkeypatch.setattr(agent.session.agent, "add_packet",
                        lambda pending, packet, **k: seen.update(k) or add(pending, packet, **k))
    view, result = see(agent, world, (unit(0), unit(0, 1)))
    handle.remove()
    assert len(calls) == 1
    shared = seen["features"].scales[0].values
    assert shared.data_ptr() == calls[0].scales[0].values.data_ptr()
    assert seen["features"].as_tokens().times.max() == 1.0  # capture time as metadata
    blob = agent.memory.view()["evidence"][result["evidence"][0]]
    assert blob.modality == "image" and agent.memory.get_blob(blob.content_ref, blob.content_hash).shape == (1, 64, 64, 3)
    # R1 compatibility: forward == from_pyramid(pyramid); supplied features are exact.
    rgb = world.frame()[None]
    assert torch.equal(agent.perception(rgb).slots, agent.perception.from_pyramid(agent.perception.pyramid(rgb)).slots)
    belief = agent.session.agent
    state = belief.initial_state(1, session_id="x")
    packet = Packet("cam", "image", Observation(rgb[:, None], torch.full((1, 1), 1.0)))
    pending = belief.begin_event(state, event_id="e", ordinal=0, time=1.0)
    plain = belief.add_packet(pending, packet).state
    fed = belief.add_packet(pending, packet, features=encoder(packet.observation, condition=None)).state
    assert torch.equal(plain.logits, fed.logits)
    with pytest.raises(ValueError, match="time span"):
        belief.add_packet(pending, packet, features=agent.perception.frame_pyramid(rgb, [0.5]))
    generated = Packet("gen", "image", Observation(rgb[:, None], torch.full((1, 1), 2.0),
                                                   provenance="generated"))
    with pytest.raises(ValueError, match="not source observation|Generated"):
        agent.session.observe("gen", occurred_at=2.0, available_at=2.0, packets=(generated,))


def test_verification_is_source_evidence_and_versions_are_rejected(tmp_path):
    modules, agent, scene, rules, g = setup(tmp_path)
    world = rw.RuleWorld(scene, rules, torch.zeros(2, dtype=torch.long), rw.TaskContract())
    keys = (unit(0), unit(0, 1))
    see(agent, world, keys)
    a, b = (m["instance"] for m in agent.view.machines)
    goal = GoalSpec("t2", "lamp_state", ((a, 1), (b, 1)))
    tx = agent.memory.begin("internal", payload=dict(declared="stop", task_id="t2", goal_hash=goal.digest()))
    agent.memory.commit(tx)
    assert agent.task_outcome(goal) == "unknown"  # a declared stop is not success
    _, read = agent.plan(agent.view, (1, 1), presses_done=0)
    record = lambda t: VerificationRecord("t2", goal.digest(), "success", "external", t)
    result = agent.execute(rw.Actuator(world), TypedAction("press", a, 0, 1, "t2"), read,
                           verification=record, candidates=supplied(agent, world.frame(), keys))
    evidence = agent.memory.view()["evidence"]
    verdict = next(e for e in evidence.values() if e.modality == "verification")
    assert verdict.event_id == evidence[result["evidence"]].event_id  # receipt and verdict: one event
    assert verdict.source == "external" and agent.task_outcome(goal) == "success"
    agent.save(tmp_path / "memory")
    saved = tmp_path / "models.pt"
    us.save_models(modules, saved)
    for name, prefix in (("core", "key_head"), ("perception", "slot_attention"), ("perception", "encoder")):
        changed = us.load_models(saved)
        parameter = next(k for k, _ in changed[name].named_parameters() if k.startswith(prefix))
        with torch.no_grad():
            changed[name].get_parameter(parameter).add_(0.01)
        with pytest.raises(ValueError, match="differ|Incompatible session model"):
            us.restore_agent(changed, tmp_path / "memory", settings=CREATE,
                             binder=AssociationBinder(new_threshold=0.5))


def test_tiny_cpu_life_writes_raw_output_and_report(tmp_path, monkeypatch):
    out = tmp_path / "life"
    monkeypatch.setattr(sys, "argv", ["unified_session", "--output", str(out), "--scenes", "3"])
    us.main()
    status = json.loads((out / "status.json").read_text())
    assert status == dict(result="completed", report="structural_verified", step=0, error=None)
    life = json.loads((out / "life.json").read_text())
    assert life["restart"]["equal"]
    free = [r for r in life["rows"] if r["kind"] == "free_text"]
    assert free and all(r["option"] == "ask" and r["actuator_presses"] == 0 for r in free)
    result = json.loads((out / "result.json").read_text())
    assert "SOFTWARE" in result["evaluation_scope"]
    for name in ("run.json", "metrics.jsonl", "report.html", "models.pt", "memory/session.pt", "memory/memory.json"):
        assert (out / name).exists()


# --- boundary regressions (independent review: runs/reviews/.../test_unified_codex_review.py) ---


def ready(tmp_path, demos_too=True):
    modules, agent, scene, rules, g = setup(tmp_path)
    world = rw.RuleWorld(scene, rules, torch.zeros(2, dtype=torch.long), rw.TaskContract())
    _, result = see(agent, world, (unit(0), unit(0, 1)), transitions=both(agent, world, 4, g) if demos_too else ())
    return modules, agent, world, g, result


def test_prepared_action_pins_live_observation_identity_and_objects(tmp_path):
    modules, agent, world, g, _ = ready(tmp_path)
    a = agent.view.machines[0]["instance"]
    _, read = agent.plan(agent.view, (1, 1), presses_done=0)
    assert dict(read.live)["view"] == agent.view.event_id
    # Unrelated derived bookkeeping does not stale the action.
    agent.memory.commit(agent.memory.begin("internal", payload=dict(note="bookkeeping")))
    assert agent.is_ready(read)
    # A new source observation of ANY modality stales it (here: a verifier-only event).
    t = agent.session.time + 1
    agent.session.observe("text-1", occurred_at=t, available_at=t, evidence=(
        SourceItem("external", "verification", data=dict(note="no frame")),))
    actuator = us.CountingActuator(world)
    assert agent.execute(actuator, TypedAction("press", a, 0, 1), read)["status"] == "stale"
    # resume_view skips the frameless latest event and rebuilds the last camera view.
    live = agent.resume_view()
    assert live.event_id == "obs-000001" and [m["instance"] for m in live.machines][0] == a
    _, read = agent.plan(agent.view, (1, 1), presses_done=0)
    # An identity correction that invalidates a pinned recognition stales it as well.
    link_tx = agent.session.store.begin("reassign-x", occurred_at=t + 1, available_at=t + 1, kind="correction")
    b = agent.view.machines[1]["instance"]
    link_tx.reassign(agent.view.machines[0]["recognition"], b)
    agent.correct(link_tx)
    assert not agent.is_ready(read) and actuator.presses == 0
    assert agent.execute(actuator, TypedAction("press", a, 0, 1), agent.memory.read_set(()))["status"] == "stale"


def test_identity_change_stales_an_action_even_without_concept_heads(tmp_path):
    modules, agent, world, g, _ = ready(tmp_path, demos_too=False)
    a, b = (m["instance"] for m in agent.view.machines)
    _, read = agent.plan(agent.view, (1, 1), presses_done=0)
    assert agent.is_ready(read)
    t = agent.session.time + 1
    tx = agent.session.store.begin("reassign-y", occurred_at=t, available_at=t, kind="correction")
    tx.reassign(agent.view.machines[0]["recognition"], b)
    agent.correct(tx)
    assert agent.memory.is_current(read) and not agent.is_ready(read)  # only the identity pin sees it


def test_membership_invalidated_outside_the_agent_is_rederived_not_blank(tmp_path):
    modules, agent, world, g, result = ready(tmp_path)
    a = agent.view.machines[0]["instance"]
    t = agent.session.time + 1
    tx = agent.session.store.begin("retract-1", occurred_at=1.0, available_at=t, kind="correction",
                                   payload=dict(source=SOURCE))
    tx.retract_evidence(result["evidence"][1])
    agent.session.commit(tx)  # no agent.correct: the next observation must repair
    see(agent, world, (unit(0), unit(0, 1)))
    binding = active(agent, a, "binding")
    assert binding.data["supports"] and len(active(agent, a, "transitions").evidence) == 3


def test_goal_budget_and_deadline_bound_planner_and_execution(tmp_path, monkeypatch):
    modules, agent, world, g, _ = ready(tmp_path)
    a, b = (m["instance"] for m in agent.view.machines)
    contract = agent.contract
    before = dict(vars(contract))
    decision, _ = agent.plan(agent.view, (1, 1), presses_done=0, budget=1)
    assert len(decision.sequence) <= 1  # the planner's horizon, not only execution
    assert agent.remaining(GoalSpec("g", "lamp_state", ((a, 1),), budget=9), 0) == contract.budget
    t = agent.session.time
    assert agent.remaining(GoalSpec("g", "lamp_state", ((a, 1),), deadline=t + 1.5), 0) == 1
    assert agent.remaining(GoalSpec("g", "lamp_state", ((a, 1),), deadline=t + 0.5), 0) == 0
    calls = []
    plan = agent.plan
    monkeypatch.setattr(agent, "plan", lambda *x, **k: calls.append(k["budget"]) or plan(*x, **k))
    actuator = us.CountingActuator(world)
    result = agent.run_task(actuator, GoalSpec("one", "lamp_state", ((a, 1), (b, 0)), budget=1))
    assert actuator.presses <= 1 and result["presses"] <= 1 and calls[0] == 1
    assert dict(vars(agent.contract)) == before and agent.contract is contract  # never mutated
    expired = GoalSpec("late", "lamp_state", ((a, 1), (b, 0)), deadline=agent.session.time - 0.5)
    assert agent.run_task(actuator, expired)["option"] == "expired" and actuator.presses <= 1
    real = agent.plan
    monkeypatch.setattr(agent, "plan", lambda view, goal, **k: (
        Decision("press", (0, 0, 1), 0.0, {}, 1), real(view, goal, **k)[1]))  # valid read, forced press
    pressed = actuator.presses
    zero = agent.run_task(actuator, GoalSpec("zero", "lamp_state", ((a, 1), (b, 0)), budget=0))
    assert zero["option"] == "refused" and actuator.presses == pressed
    for bad in (dict(budget=-1), dict(budget=True), dict(budget=1.0), dict(deadline=float("nan")),
                dict(deadline=float("inf")), dict(deadline=-1.0), dict(deadline="soon")):
        with pytest.raises(ValueError):
            GoalSpec("bad", "lamp_state", ((a, 1),), **bad)
    for targets in ((), ((a, "1"),), ((a, True),), ((a,),), [(a, 1)]):
        with pytest.raises(ValueError):
            GoalSpec("bad", "lamp_state", targets)


def test_action_indices_must_be_in_range_integers_before_any_press(tmp_path):
    modules, agent, world, g, _ = ready(tmp_path, demos_too=False)
    a = agent.view.machines[0]["instance"]
    for i, j in ((1.0, 0), (True, 0), (np.int64(1), 0)):
        with pytest.raises(ValueError):
            TypedAction("press", a, i, j)
    actuator = us.CountingActuator(world)
    _, read = agent.plan(agent.view, (1, 1), presses_done=0)
    for i, j in ((-1, 0), (0, len(agent.view.objects))):
        with pytest.raises(ValueError, match="not visible"):
            agent.execute(actuator, TypedAction("press", a, i, j), read)
    assert actuator.presses == 0


def test_replacement_is_attributed_from_its_own_receipt_and_target(tmp_path):
    modules, agent, world, g, result = ready(tmp_path)
    a, b = (m["instance"] for m in agent.view.machines)
    old = result["evidence"][1]  # an ok transition aimed at machine a
    other = demos(agent, world, 1, 1, g)[0]  # a successful press aimed at machine b
    _, replacement = see(agent, world, (unit(0), unit(0, 1)), transitions=(other,), supersedes={0: old})
    new = replacement["evidence"][1]
    agent.repair_identity()
    view = agent.memory.view()
    owners = {c.entity_id for c in view["components"].values()
              if c.active and c.name == "attribution" and c.data["transition"] == new}
    assert owners == {b}  # follows the replacement's own action, not the old identity
    assert old not in active(agent, a, "transitions").evidence and new not in active(agent, a, "transitions").evidence
    assert new in active(agent, b, "transitions").evidence
    assert old in view["evidence"] and view["retracted"][old]["replacement"] == new  # raw evidence kept


def test_verification_identity_time_and_freshness(tmp_path):
    modules, agent, world, g, _ = ready(tmp_path, demos_too=False)
    a, b = (m["instance"] for m in agent.view.machines)
    goal = GoalSpec("v", "lamp_state", ((a, 1), (b, 1)))
    keys = (unit(0), unit(0, 1))
    wrong = lambda t: VerificationRecord("v", "not-this-goal", "success", "external", t)
    with pytest.raises(ValueError, match="another task or goal"):
        agent.observe(world.frame(), verification=wrong, goal=goal, candidates=supplied(agent, world.frame(), keys))
    assert agent.view.event_id == "obs-000002"  # the observation itself was still published
    for bad in (lambda t: VerificationRecord("v", goal.digest(), "success", "external", t - 1),
                lambda t: VerificationRecord("v", goal.digest(), "success", SOURCE, t),
                lambda t: dict(status="success")):
        with pytest.raises(ValueError, match="not published"):
            agent.observe(world.frame(), verification=bad, goal=goal, candidates=supplied(agent, world.frame(), keys))
    assert not any(e.modality == "verification" for e in agent.memory.view()["evidence"].values())
    good = lambda t: VerificationRecord("v", goal.digest(), "success", "external", t)
    agent.observe(world.frame(), verification=good, goal=goal, candidates=supplied(agent, world.frame(), keys))
    assert agent.task_outcome(goal) == "success"
    _, read = agent.plan(agent.view, (1, 1), presses_done=0)
    agent.execute(rw.Actuator(world), TypedAction("press", a, 0, 1, "v"), read,
                  candidates=supplied(agent, world.frame(), keys))
    assert agent.task_outcome(goal) == "unknown"  # an action came after the verdict


def test_interrupted_derivation_is_recovered_from_source_on_restart(tmp_path, monkeypatch):
    modules, agent, scene, rules, g = setup(tmp_path)
    world = rw.RuleWorld(scene, rules, torch.zeros(2, dtype=torch.long), rw.TaskContract())
    transitions = both(agent, world, 4, g)
    monkeypatch.setattr(agent, "_attribute", lambda ids: (_ for _ in ()).throw(RuntimeError("crash")))
    with pytest.raises(RuntimeError, match="crash"):
        see(agent, world, (unit(0), unit(0, 1)), transitions=transitions)
    published = [e.id for e in agent.memory.view()["evidence"].values() if e.modality == "transition"]
    assert len(published) == 8  # source + identity + belief were published atomically
    assert not any(c.name == "attribution" for c in agent.memory.view()["components"].values())
    agent.session.save(tmp_path / "memory" / "session.pt")
    agent.memory.save()
    us.save_models(modules, tmp_path / "models.pt")
    restored = us.restore_agent(us.load_models(tmp_path / "models.pt", seed=3), tmp_path / "memory",
                                settings=CREATE, binder=AssociationBinder(new_threshold=0.5))
    view = restored.memory.view()
    attributed = {c.data["transition"] for c in view["components"].values() if c.name == "attribution" and c.active}
    assert attributed == set(published)
    restored.resume_view()
    assert all(m["concept"] is not None for m in restored.view.machines)


@pytest.mark.parametrize("name", ["perception", "core", "candidates", "actions"])
def test_every_integrated_module_change_is_rejected_before_effects(tmp_path, name):
    modules, agent, world, g, _ = ready(tmp_path)
    a = agent.view.machines[0]["instance"]
    _, read = agent.plan(agent.view, (1, 1), presses_done=0)
    parameter = next(p for k, p in modules[name].named_parameters() if not k.startswith("encoder."))
    original = parameter.detach().clone()
    with torch.no_grad():
        parameter.add_(0.01)  # e.g. the slot decoder, which the session does not fingerprint
    actuator = us.CountingActuator(world)
    for call in (lambda: agent.execute(actuator, TypedAction("press", a, 0, 1), read),
                 lambda: agent.plan(agent.view, (1, 1), presses_done=0),
                 lambda: agent.observe(world.frame()), lambda: agent.save(tmp_path / "memory")):
        with pytest.raises(ValueError, match="weights changed"):
            call()
    assert actuator.presses == 0
    with torch.no_grad():
        parameter.copy_(original)
    modules[name].train()
    with pytest.raises(ValueError, match="eval mode"):
        agent.plan(agent.view, (1, 1), presses_done=0)


def test_r1_repair_ignores_a_failed_replacement(tmp_path):
    from pathwm.evaluation.rule_world import fresh_agent

    modules = us.build(0)
    agent = fresh_agent(modules["perception"], modules["core"], tmp_path / "r1", AgentSettings())
    g = torch.Generator().manual_seed(0)
    scene = rw.sample_scenes(g, torch.tensor([list(rw.KIND_SPLIT["train"][:2])]))
    rules = rw.split_rules()["train"][:2]
    transitions = us.demonstrations(scene, rules, torch.zeros(2, dtype=torch.long), g, 4)
    out = agent.observe_session(transitions)
    old = out["evidence"][0]
    pre, record, _, post = transitions[0]
    new = agent.receive_supersede(old, (pre, record, "miss", post))
    components = agent.memory.view()["components"].values()
    assert not any(c.active and c.name == "transitions" and new in c.evidence for c in components)


def test_live_press_is_attributed_through_the_observation_it_was_chosen_in(tmp_path):
    modules, agent, world, g, _ = ready(tmp_path)
    a = agent.view.machines[0]["instance"]

    class OkActuator:  # SOFTWARE fixture: random perception rarely hits the real machines
        def frame(self):
            return world.frame()

        def press(self, record):
            return rw.Receipt("ok", 0.05, 1)

    _, read = agent.plan(agent.view, (1, 1), presses_done=0)
    # After the press the machine looks different: the session identifies a NEW entity.
    result = agent.execute(OkActuator(), TypedAction("press", a, 0, 1), read,
                           candidates=supplied(agent, world.frame(), (unit(0, 2), unit(0, 1))))
    assert agent.view.machines[0]["instance"] != a
    owners = {c.entity_id for c in agent.memory.view()["components"].values()
              if c.active and c.name == "attribution" and c.data["transition"] == result["evidence"]}
    assert owners == {a}
