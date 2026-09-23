"""A claim tag is validated, never trusted: only the exact claimed action, chosen in
the claim's camera observation, can witness it (root review reproductions first,
runs/reviews/integrated_architecture_20260923/test_unified_claim_provenance.py)."""

import numpy as np
import pytest

from pathwm.world_state.concepts import SOURCE, to_uint8
from pathwm.world_state.session import SourceItem
from pathwm.world_state.unified import TypedAction
from tests.test_unified_boundaries import ready
from tests.test_unified_session import see, unit
from tests.test_unified_testimony import _Actuator, authorize, claim_about, keys


def test_other_action_cannot_judge_claim_even_when_same_machine_and_claim_tag(tmp_path):
    _, agent, world, _, _ = ready(tmp_path)
    claim = claim_about(agent, 1)
    evidence = agent.receive_testimony(claim)
    view = agent.view
    other = agent.record_for(view, (0, 1, 0))
    assert other.as_dict() != agent.memory.view()["evidence"][evidence].data["record"]
    frame = world.frame()
    try:
        agent.observe(frame, transitions=((frame, other, "ok", frame),), attribute_with=view,
                      candidates=keys(agent, world), claim=evidence)
    except ValueError:
        pass
    agent.repair_identity()
    assert agent.claim_status(evidence)["status"] == "untested"


def test_later_scene_test_cannot_judge_old_scene_claim_from_tag_alone(tmp_path):
    _, agent, world, _, _ = ready(tmp_path)
    evidence = agent.receive_testimony(claim_about(agent, 1))
    see(agent, world, (unit(0), unit(0, 1)))
    view = agent.view
    assert view.event_id != agent.memory.view()["evidence"][evidence].data["perceived_in"]
    record = agent.record_for(view, (0, 0, 1))
    frame = world.frame()
    try:
        agent.observe(frame, transitions=((frame, record, "ok", frame),), attribute_with=view,
                      candidates=keys(agent, world), claim=evidence)
    except ValueError:
        pass
    agent.repair_identity()
    assert agent.claim_status(evidence)["status"] == "untested"


def test_invalid_claim_links_fail_in_observe_and_execute_before_any_effect(tmp_path):
    _, agent, world, _, result = ready(tmp_path)
    claim = claim_about(agent, 1)
    evidence = agent.receive_testimony(claim)
    machine = claim.action.machine
    image = result["evidence"][0]
    store, time = agent.session.store.snapshot(), agent.session.time
    frame = world.frame()
    with pytest.raises(ValueError, match="Invalid claim link"):
        agent.observe(frame, transitions=((frame, agent.record_for(agent.view, (0, 1, 0)), "ok", frame),),
                      candidates=keys(agent, world), claim=evidence)
    actuator = _Actuator(world)
    for link, action in ((evidence, TypedAction("press", machine, 1, 0, "test-c1")),  # other objects
                         (image, TypedAction("press", machine, 0, 1, "test-c1")),     # not a claim
                         ("no-such-claim", TypedAction("press", machine, 0, 1, "test-c1"))):
        _, read = agent.plan(agent.view, (1, 1), presses_done=0)
        with pytest.raises(ValueError, match="Invalid claim link"):
            agent.execute(actuator, action, read, goal=authorize(claim, task="test-c1"), claim=link,
                          candidates=keys(agent, world))
    assert actuator.presses == 0
    assert agent.session.store.snapshot()["events"][: len(store["events"])] == store["events"]
    agent.receive_retract(evidence)  # a withdrawn claim cannot be linked either
    _, read = agent.plan(agent.view, (1, 1), presses_done=0)
    with pytest.raises(ValueError, match="withdrawn"):
        agent.execute(actuator, TypedAction("press", machine, 0, 1), read, claim=evidence,
                      candidates=keys(agent, world))
    assert actuator.presses == 0 and agent.session.time > time


def test_malformed_or_foreign_retained_tags_cannot_poison_recovery(tmp_path):
    _, agent, world, _, result = ready(tmp_path)
    claim = claim_about(agent, 1)
    evidence = agent.receive_testimony(claim)
    record = agent.memory.view()["evidence"][evidence].data
    frame = world.frame()
    ref, blob = agent.memory.put_blob(np.stack((to_uint8(frame), to_uint8(frame))))
    # Foreign items that copy a real claim's record/observation/instance but are not
    # typed testimony: a non-claim note, and a "claim" spoken as the camera source.
    foreign = []
    for k, (source, modality) in enumerate((("mallory", "note"), (SOURCE, "claim"))):
        t = agent.session.time + 1
        foreign += agent.session.observe(f"foreign-{k}", occurred_at=t, available_at=t,
                                         evidence=(SourceItem(source, modality, data=dict(record)),))["evidence"]
    # Directly published (bypassing observe's validation) transitions with bad claim tags.
    for k, tag in enumerate((result["evidence"][0], ["not", "an", "id"], 7, "no-such-claim", *foreign)):
        data = dict(action=record["record"], receipt="ok", perceived_in=record["perceived_in"], claim=tag)
        t = agent.session.time + 1
        agent.session.observe(f"inject-{k}", occurred_at=t, available_at=t,
                              evidence=(SourceItem(SOURCE, "transition", ref, blob, data),))
    agent.recover()
    assert agent.claim_status(evidence)["status"] == "untested"
    assert not any(c.name == "judgment" for c in agent.memory.view()["components"].values())
    # The genuine path still works afterwards (fresh camera view, fresh claim).
    see(agent, world, (unit(0), unit(0, 1)))
    fresh = claim_about(agent, 1, claim_id="c2")
    e2 = agent.receive_testimony(fresh)
    out = agent.test_claim(e2, _Actuator(world), authorize(fresh, task="test-c2"), candidates=keys(agent, world))
    assert out["status"] == "judged"
