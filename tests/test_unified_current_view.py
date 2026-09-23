"""Corrections keep the live view current without inventing a camera observation.

The first two tests are the independent follow-up reproductions (root review,
runs/reviews/integrated_architecture_20260923/test_unified_current_view.py), kept
permanently. SOFTWARE fixtures as in tests/test_unified_session.py.
"""

import torch

from pathwm.world_state.unified import TypedAction
from tests.test_unified_boundaries import active, ready
from tests.test_unified_session import SOURCE, unit


def test_retract_refreshes_live_membership_and_new_plan_is_current(tmp_path):
    _, agent, _, _, result = ready(tmp_path)
    instance = agent.view.machines[0]["instance"]
    old_binding = agent.view.machines[0]["binding"]
    agent.receive_retract(result["evidence"][1])
    current = active(agent, instance, "binding")
    assert current is not None and current.id != old_binding
    assert agent.view.machines[0]["binding"] == current.id
    _, read = agent.plan(agent.view, (1, 1), presses_done=0)
    assert agent.is_ready(read)


def test_correct_refreshes_live_membership_without_new_camera_frame(tmp_path):
    _, agent, _, _, result = ready(tmp_path)
    instance = agent.view.machines[0]["instance"]
    old = agent.memory.view()["evidence"][result["evidence"][1]]
    tx = agent.session.store.begin("root-withdraw", occurred_at=old.occurred_at,
                                   available_at=agent.session.time + 1, kind="correction",
                                   payload={"source": old.source})
    tx.retract_evidence(old.id)
    agent.correct(tx)
    current = active(agent, instance, "binding")
    assert current is not None
    assert agent.view.machines[0]["binding"] == current.id


def frame_state(agent):
    return (agent.view.event_id, agent.view.frame_sha, agent.view.percept.slots.clone(),
            [e.id for e in agent.session.store.events() if e.kind == "observation"])


def test_correction_invalidates_old_plan_and_refresh_keeps_raw_view_and_clock(tmp_path):
    _, agent, world, g, result = ready(tmp_path)
    a, b = (m["instance"] for m in agent.view.machines)
    b_binding = agent.view.machines[1]["binding"]
    _, old_read = agent.plan(agent.view, (1, 1), presses_done=0)
    event, sha, slots, observations = frame_state(agent)
    agent.receive_retract(result["evidence"][1])
    time_after_correction = agent.session.time
    assert not agent.is_ready(old_read)  # yesterday's plan never executes
    ev, sh, sl, obs = frame_state(agent)
    assert (ev, sh, obs) == (event, sha, observations) and torch.equal(sl, slots)  # no camera invented
    assert agent.view.machines[1]["binding"] == b_binding  # unaffected machine untouched
    # A fresh plan reads only current heads and can execute on the same frame.
    _, read = agent.plan(agent.view, (1, 1), presses_done=0)
    components = agent.memory.view()["components"]
    assert all(components[c].active for c, _ in read.components) and agent.is_ready(read)
    assert agent.session.time == time_after_correction  # planning/refresh never advances the clock
    prediction = agent.predict(agent.view, agent.record_for(agent.view, (0, 0, 1)))
    assert agent.memory.is_current(prediction["read_set"])
    assert prediction["concept"] == active(agent, a, "binding").data["concept"]


def test_repeated_repair_is_idempotent_and_does_not_rebind_again(tmp_path):
    _, agent, _, _, result = ready(tmp_path)
    agent.receive_retract(result["evidence"][1])
    revision = agent.session.store.revision
    for call in (agent.repair_identity, agent.repair, agent.recover):
        call()
        _, read = agent.plan(agent.view, (1, 1), presses_done=0)
        assert agent.is_ready(read)
    # Plans may compute a missing code once; nothing rebinds or re-derives repeatedly.
    settled = agent.session.store.revision
    agent.repair_identity()
    agent.plan(agent.view, (1, 1), presses_done=0)
    assert agent.session.store.revision == settled and settled - revision <= 2


def test_session_level_correction_then_public_repair_or_plan_uses_current_heads(tmp_path):
    """A correction published on the session directly (not through the agent)."""
    _, agent, _, _, result = ready(tmp_path)
    instance = agent.view.machines[0]["instance"]
    old = agent.memory.view()["evidence"][result["evidence"][2]]
    tx = agent.session.store.begin("session-withdraw", occurred_at=old.occurred_at,
                                   available_at=agent.session.time + 1, kind="correction",
                                   payload={"source": SOURCE})
    tx.retract_evidence(old.id)
    agent.session.commit(tx)
    # Planning refreshes the view's memberships first, so it never pins an inactive head.
    _, read = agent.plan(agent.view, (1, 1), presses_done=0)
    components = agent.memory.view()["components"]
    assert all(components[c].active for c, _ in read.components)
    assert agent.view.machines[0]["binding"] == active(agent, instance, "binding").id
    assert agent.is_ready(read)
    agent.repair_identity()
    assert agent.view.machines[0]["binding"] == active(agent, instance, "binding").id


def test_reassignment_to_a_new_entity_updates_view_identity_dispatch_and_concept(tmp_path):
    from pathwm.world_state.unified import GoalSpec

    _, agent, world, g, _ = ready(tmp_path)
    a, b = (m["instance"] for m in agent.view.machines)
    recognition = agent.view.machines[0]["recognition"]
    _, old_read = agent.plan(agent.view, (1, 1), presses_done=0)
    t = agent.session.time + 1
    tx = agent.session.store.begin("reassign-new", occurred_at=t, available_at=t, kind="correction")
    d = tx.create_entity(kind="instance")
    tx.reassign(recognition, d)
    agent.correct(tx)
    machine = agent.view.machines[0]
    assert machine["instance"] == d and machine["recognition"] == recognition
    assert machine["binding"] == active(agent, d, "binding").id
    assert machine["concept"] == active(agent, d, "binding").data["concept"]
    assert not agent.is_ready(old_read)  # the pinned identity changed
    assert agent.dispatch(None, GoalSpec("g", "lamp_state", ((a, 1), (b, 0)))).kind == "ask"  # a not visible
    assert agent.dispatch(None, GoalSpec("g", "lamp_state", ((d, 1), (b, 0)))).kind == "plan"
    _, read = agent.plan(agent.view, (1, 0), presses_done=0)
    assert agent.is_ready(read)
    # The refreshed plan can be executed on the SAME camera frame.
    before = agent.view.event_id
    result = agent.execute(_OkActuator(world), TypedAction("press", d, 0, 1), read,
                           candidates=_keys(agent, world))
    assert result["status"] == "ok" and agent.view.event_id != before


class _OkActuator:
    """SOFTWARE fixture: random perception rarely hits the real machines."""

    def __init__(self, world):
        self.world = world

    def frame(self):
        return self.world.frame()

    def press(self, record):
        from pathwm.data import rule_world as rw

        return rw.Receipt("ok", 0.05, 1)


def _keys(agent, world):
    from tests.test_unified_session import supplied

    return supplied(agent, world.frame(), (unit(0), unit(0, 1)))


def _session_withdraw(agent, evidence_id):
    old = agent.memory.view()["evidence"][evidence_id]
    tx = agent.session.store.begin(f"session-withdraw-{evidence_id}", occurred_at=old.occurred_at,
                                   available_at=agent.session.time + 1, kind="correction",
                                   payload={"source": SOURCE})
    tx.retract_evidence(evidence_id)
    agent.session.commit(tx)  # bypasses the agent: its live view is now behind the store


def _session_reassign(agent, side):
    t = agent.session.time + 1
    tx = agent.session.store.begin("session-reassign", occurred_at=t, available_at=t, kind="correction")
    new = tx.create_entity(kind="instance")
    tx.reassign(agent.view.machines[side]["recognition"], new)
    agent.session.commit(tx)
    return new


def test_each_consumer_refreshes_the_view_when_it_is_the_first_call(tmp_path):
    from pathwm.world_state.unified import GoalSpec

    # predict
    _, agent, _, _, result = ready(tmp_path / "predict")
    a = agent.view.machines[0]["instance"]
    _session_withdraw(agent, result["evidence"][2])
    prediction = agent.predict(agent.view, agent.record_for(agent.view, (0, 0, 1)))
    assert agent.memory.is_current(prediction["read_set"])
    assert prediction["concept"] == active(agent, a, "binding").data["concept"]
    # dispatch
    _, agent, _, _, _ = ready(tmp_path / "dispatch")
    b = agent.view.machines[1]["instance"]
    d = _session_reassign(agent, 0)
    assert agent.dispatch(None, GoalSpec("g", "lamp_state", ((d, 1), (b, 0)))).kind == "plan"
    # recover
    _, agent, _, _, result = ready(tmp_path / "recover")
    a = agent.view.machines[0]["instance"]
    _session_withdraw(agent, result["evidence"][2])
    agent.recover()
    assert agent.view.machines[0]["binding"] == active(agent, a, "binding").id
