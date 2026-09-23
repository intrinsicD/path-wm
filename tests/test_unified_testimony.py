"""R2 typed testimony: a claim teaches nothing; one own bounded test may judge it.

SOFTWARE fixtures as in tests/test_unified_session.py (supplied keys for the agent's
own machine slots, supplied ok demonstrations). `_Actuator` returns a chosen receipt
for the real frames because random perception rarely hits the real machines. The
judgment's "observed" value is the agent's own (random) perception: wiring only.
"""

import pytest
import torch

from experiments import unified_session as us
from pathwm.data import rule_world as rw
from pathwm.world_state.modules import AssociationBinder
from pathwm.world_state.unified import Claim, GoalSpec, TypedAction
from tests.test_unified_boundaries import active, ready
from tests.test_unified_session import CREATE, SOURCE, see, supplied, unit


class _Actuator:
    def __init__(self, world, status="ok"):
        self.world, self.status, self.presses = world, status, 0

    def frame(self):
        return self.world.frame()

    def press(self, record):
        self.presses += 1
        return rw.Receipt(self.status, 0.05, self.presses)


def keys(agent, world):
    return supplied(agent, world.frame(), (unit(0), unit(0, 1)))


def claim_about(agent, outcome, claim_id="c1", source="alice", side=0):
    return Claim(claim_id, source, TypedAction("press", agent.view.machines[side]["instance"], 0, 1), outcome)


def authorize(claim, budget=1, deadline=None, task="test-c1"):
    return GoalSpec(task, "claim_test", ((claim.action.machine, claim.outcome),), budget=budget, deadline=deadline)


def perceived(agent, world):
    """What the agent's own perception will report for the claimed press (stub keeps frames)."""
    frame = world.frame()
    record = agent.record_for(agent.view, (0, 0, 1))
    return int(agent.transition_tokens(frame, record, frame)["outcome"])


def learning_state(agent):
    view = agent.memory.view()["components"].values()
    return sorted((c.id, c.active) for c in view if c.name in ("attribution", "transitions", "binding", "code", "key"))


def test_claim_alone_teaches_nothing_and_is_source_attributed(tmp_path):
    _, agent, world, g, _ = ready(tmp_path)
    before = learning_state(agent)
    claim = claim_about(agent, 1)
    e = agent.receive_testimony(claim)
    item = agent.memory.view()["evidence"][e]
    assert (item.source, item.modality) == ("alice", "claim")
    assert item.data["perceived_in"] == agent.view.event_id and item.data["outcome"] == 1
    assert learning_state(agent) == before  # no support, attribution, code or binding
    assert not any(x.modality == "transition" and x.event_id == item.event_id
                   for x in agent.memory.view()["evidence"].values())
    assert agent.claim_status(e)["status"] == "untested"
    assert agent.receive_testimony(claim) == e  # identical retry is idempotent
    with pytest.raises(ValueError, match="differ"):
        agent.receive_testimony(claim_about(agent, 0))
    for bad in (dict(source=SOURCE), dict(outcome=2), dict(outcome=True), dict(source="")):
        fields = dict(claim_id="x", source="alice", action=claim.action, outcome=1) | bad
        with pytest.raises(ValueError):
            Claim(**fields)
    with pytest.raises(NotImplementedError):  # the R1 signature is not reinterpreted
        agent.receive_claim(world.frame(), agent.record_for(agent.view, (0, 0, 1)), 1, _Actuator(world))


@pytest.mark.parametrize("truthful", [True, False])
def test_only_the_own_observed_press_teaches_and_judges_the_claim(tmp_path, truthful):
    _, agent, world, g, _ = ready(tmp_path)
    observed = perceived(agent, world)
    claim = claim_about(agent, observed if truthful else 1 - observed)
    e = agent.receive_testimony(claim)
    actuator = _Actuator(world)
    result = agent.test_claim(e, actuator, authorize(claim), candidates=keys(agent, world))
    assert actuator.presses == 1 and result["status"] == "judged"
    view = agent.memory.view()
    transition = view["evidence"][result["transition"]]
    assert transition.source == SOURCE and transition.data["claim"] == e and transition.data["receipt"] == "ok"
    assert "outcome" not in transition.data  # the claim's value is not copied into observed evidence
    machine = claim.action.machine
    assert transition.id in active(agent, machine, "transitions").evidence  # own press joins memory
    judgment = view["components"][result["judgment"]]
    assert judgment.evidence == (e, transition.id) and judgment.data["observed"] == observed
    status = "consistent" if truthful else "contradicted"
    assert judgment.data["status"] == status and agent.claim_status(e)["status"] == status
    # A second call never presses again.
    assert agent.test_claim(e, actuator, authorize(claim), candidates=keys(agent, world))["status"] == "judged"
    assert actuator.presses == 1


@pytest.mark.parametrize("status", ["miss", "same_object"])
def test_failed_test_neither_judges_nor_supports(tmp_path, status):
    _, agent, world, g, _ = ready(tmp_path)
    claim = claim_about(agent, 1)
    e = agent.receive_testimony(claim)
    support = set(active(agent, claim.action.machine, "transitions").evidence)
    actuator = _Actuator(world, status)
    result = agent.test_claim(e, actuator, authorize(claim), candidates=keys(agent, world))
    assert actuator.presses == 1 and result["status"] == status and result["judgment"] is None
    assert agent.claim_status(e)["status"] == "untested"
    assert set(active(agent, claim.action.machine, "transitions").evidence) == support
    assert agent.memory.view()["evidence"][result["transition"]].data["claim"] == e  # receipt kept raw
    assert agent._judge(e, result["transition"]) is None  # a failed receipt can never judge


def test_no_press_for_stale_ambiguous_unauthorized_or_changed_model(tmp_path):
    _, agent, world, g, _ = ready(tmp_path)
    claim = claim_about(agent, 1)
    e = agent.receive_testimony(claim)
    actuator = _Actuator(world)
    assert agent.test_claim(e, actuator, authorize(claim, budget=0), candidates=keys(agent, world))["status"] == "refused"
    late = authorize(claim, deadline=agent.session.time - 0.5)
    assert agent.test_claim(e, actuator, late, candidates=keys(agent, world))["status"] == "refused"
    for wrong in (GoalSpec("test-c1", "lamp_state", ((claim.action.machine, 1),)),
                  GoalSpec("test-c1", "claim_test", ((claim.action.machine, 0),))):
        with pytest.raises(ValueError, match="authorize"):
            agent.test_claim(e, actuator, wrong)
    with torch.no_grad():
        agent.core.outcome.bias.add_(0.1)
    with pytest.raises(ValueError, match="weights changed"):
        agent.test_claim(e, actuator, authorize(claim))
    assert actuator.presses == 0 and agent.claim_status(e)["status"] == "untested"


def test_identity_ambiguity_and_scene_change_leave_the_claim_untested(tmp_path):
    _, agent, world, g, _ = ready(tmp_path)
    claim = claim_about(agent, 1)
    e = agent.receive_testimony(claim)
    actuator = _Actuator(world)
    b = agent.view.machines[1]["instance"]
    t = agent.session.time + 1
    tx = agent.session.store.begin("reassign-claimed", occurred_at=t, available_at=t, kind="correction")
    tx.reassign(agent.view.machines[0]["recognition"], b)  # claimed machine no longer identifiable
    agent.correct(tx)
    assert agent.test_claim(e, actuator, authorize(claim), candidates=keys(agent, world))["status"] == "unresolved"
    _, agent, world, g, _ = ready(tmp_path / "scene")
    claim = claim_about(agent, 1)
    e = agent.receive_testimony(claim)
    see(agent, world, (unit(0), unit(0, 1)))  # a new camera observation: the claim's references are old
    assert agent.test_claim(e, actuator, authorize(claim), candidates=keys(agent, world))["status"] == "stale"
    assert actuator.presses == 0 and agent.claim_status(e)["status"] == "untested"


def test_retracting_testimony_keeps_direct_evidence_and_correcting_the_test_invalidates_judgment(tmp_path):
    _, agent, world, g, _ = ready(tmp_path)
    claim = claim_about(agent, 1)
    e = agent.receive_testimony(claim)
    result = agent.test_claim(e, _Actuator(world), authorize(claim), candidates=keys(agent, world))
    machine, transition = claim.action.machine, result["transition"]
    agent.receive_retract(e)
    assert agent.claim_status(e)["status"] == "retracted"
    assert not agent.memory.view()["components"][result["judgment"]].active
    assert transition in active(agent, machine, "transitions").evidence  # own observation stays
    assert transition not in agent.memory.view()["retracted"]
    # Separately: correcting the DIRECT test evidence invalidates the judgment, not the claim.
    _, agent, world, g, _ = ready(tmp_path / "direct")
    claim = claim_about(agent, 1)
    e = agent.receive_testimony(claim)
    result = agent.test_claim(e, _Actuator(world), authorize(claim), candidates=keys(agent, world))
    agent.receive_retract(result["transition"])
    assert not agent.memory.view()["components"][result["judgment"]].active
    assert agent.claim_status(e)["status"] == "untested" and e not in agent.memory.view()["retracted"]
    assert result["transition"] not in active(agent, claim.action.machine, "transitions").evidence


def test_restart_and_interrupted_judgment_preserve_knowledge_without_a_second_press(tmp_path, monkeypatch):
    modules, agent, world, g, _ = ready(tmp_path)
    claim = claim_about(agent, 1)
    e = agent.receive_testimony(claim)
    actuator = _Actuator(world)
    monkeypatch.setattr(agent, "_judge", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("crash")))
    with pytest.raises(RuntimeError, match="crash"):
        agent.test_claim(e, actuator, authorize(claim), candidates=keys(agent, world))
    assert actuator.presses == 1  # the press and its receipt were published
    agent.save(tmp_path / "memory")
    us.save_models(modules, tmp_path / "models.pt")
    restored = us.restore_agent(us.load_models(tmp_path / "models.pt", seed=4), tmp_path / "memory",
                                settings=CREATE, binder=AssociationBinder(new_threshold=0.5))
    restored.resume_view()
    status = restored.claim_status(e)["status"]
    assert status in ("consistent", "contradicted")  # recover derived it from the retained test
    again = restored.test_claim(e, actuator, authorize(claim), candidates=keys(restored, world))
    assert again["status"] == "judged" and actuator.presses == 1


def test_identity_correction_of_the_test_invalidates_its_judgment(tmp_path):
    _, agent, world, g, _ = ready(tmp_path)
    claim = claim_about(agent, 1)
    e = agent.receive_testimony(claim)
    pre_recognition = agent.view.machines[0]["recognition"]  # the decision the test was chosen on
    result = agent.test_claim(e, _Actuator(world), authorize(claim), candidates=keys(agent, world))
    assert agent.claim_status(e)["judgment"] == result["judgment"]
    b = agent.view.machines[1]["instance"]
    t = agent.session.time + 1
    tx = agent.session.store.begin("reassign-test", occurred_at=t, available_at=t, kind="correction")
    tx.reassign(pre_recognition, b)
    agent.correct(tx)
    view = agent.memory.view()
    assert not view["components"][result["judgment"]].active
    owners = {c.entity_id for c in view["components"].values()
              if c.active and c.name == "attribution" and c.data["transition"] == result["transition"]}
    assert owners == {b}  # the observation follows the corrected identity ...
    assert agent.claim_status(e)["status"] == "untested"  # ... but no longer judges a claim about another entity


def test_testimony_with_invisible_references_is_rejected_before_any_effect(tmp_path):
    _, agent, world, g, _ = ready(tmp_path)
    store, time = agent.session.store.snapshot(), agent.session.time
    machine = agent.view.machines[0]["instance"]
    for action in (TypedAction("press", "not-an-entity", 0, 1),
                   TypedAction("press", machine, 0, len(agent.view.objects))):
        with pytest.raises(ValueError, match="not visible|not one visible"):
            agent.receive_testimony(Claim("bad", "alice", action, 1))
    assert agent.session.store.snapshot() == store and agent.session.time == time
