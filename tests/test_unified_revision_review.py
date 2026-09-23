"""Independent adversarial software checks for the unified-session review.

Uses the implementation author's explicitly supplied identity fixture; no learned
capability is asserted. These tests exercise new observation/action/source boundaries.
"""
import torch
from tests.test_unified_session import setup, see, unit, both
from experiments import unified_session as us
from pathwm.data import rule_world as rw
from pathwm.models.latent_core import Decision
from pathwm.world_state.unified import GoalSpec, TypedAction


def live(tmp_path, demos=False):
    modules, agent, scene, rules, g = setup(tmp_path)
    world = rw.RuleWorld(scene, rules, torch.zeros(2, dtype=torch.long), rw.TaskContract())
    transitions = both(agent, world, 4, g) if demos else ()
    _, result = see(agent, world, (unit(0), unit(0, 1)), transitions=transitions)
    return agent, world, scene, transitions, result


def test_new_observation_invalidates_prepared_action(tmp_path):
    agent, world, scene, _, _ = live(tmp_path, demos=True)
    _, read = agent.plan(agent.view, (1, 1), presses_done=0)
    action = TypedAction('press', agent.view.machines[0]['instance'], 0, 1)
    see(agent, world, (unit(0), unit(0, 1)))
    actuator = us.CountingActuator(world)
    result = agent.execute(actuator, action, read)
    assert result['status'] == 'stale' and actuator.presses == 0


def force_press(agent, monkeypatch):
    # A planner proposal never authorizes exceeding the caller's hard task limits.
    monkeypatch.setattr(agent, 'plan', lambda *a, **kw: (
        Decision('press', (0, 0, 1), 0.0, {}, 1), agent.memory.read_set(())))


def test_zero_budget_never_executes(tmp_path, monkeypatch):
    agent, world, scene, _, _ = live(tmp_path)
    force_press(agent, monkeypatch)
    targets = tuple((m['instance'], 1) for m in agent.view.machines)
    actuator = us.CountingActuator(world)
    agent.run_task(actuator, GoalSpec('zero', 'lamp_state', targets, budget=0))
    assert actuator.presses == 0


def test_expired_deadline_never_executes(tmp_path, monkeypatch):
    agent, world, scene, _, _ = live(tmp_path)
    force_press(agent, monkeypatch)
    targets = tuple((m['instance'], 1) for m in agent.view.machines)
    actuator = us.CountingActuator(world)
    agent.run_task(actuator, GoalSpec('expired', 'lamp_state', targets, deadline=agent.session.time-0.1))
    assert actuator.presses == 0


def test_failed_replacement_is_not_concept_support(tmp_path):
    agent, world, scene, transitions, result = live(tmp_path, demos=True)
    old = result['evidence'][1]
    pre, record, _, post = transitions[0]
    _, replacement = see(agent, world, (unit(0), unit(0, 1)),
        transitions=((pre, record, 'miss', post),), supersedes={0: old})
    new = replacement['evidence'][1]
    agent.repair_identity()
    components = agent.memory.view()['components'].values()
    assert not any(c.active and c.name == 'attribution' and c.data['transition'] == new for c in components)
    assert not any(c.active and c.name == 'code' and new in c.evidence for c in components)


def test_runtime_core_change_rejects_prepared_action(tmp_path):
    import pytest
    agent, world, scene, _, _ = live(tmp_path, demos=True)
    _, read = agent.plan(agent.view, (1, 1), presses_done=0)
    action = TypedAction('press', agent.view.machines[0]['instance'], 0, 1)
    actuator = us.CountingActuator(world)
    with torch.no_grad():
        agent.core.outcome.bias.add_(0.1)
    with pytest.raises(ValueError, match='(?i)(model|version|weight|changed)'):
        agent.execute(actuator, action, read)
    assert actuator.presses == 0


def test_new_camera_state_expires_old_verification(tmp_path):
    from pathwm.world_state.unified import VerificationRecord
    agent, world, scene, _, _ = live(tmp_path)
    goal = GoalSpec('verify', 'lamp_state', tuple((m['instance'], 0) for m in agent.view.machines))
    see(agent, world, (unit(0), unit(0, 1)), goal=goal,
        verification=lambda t: VerificationRecord(goal.task_id, goal.digest(), 'success', 'external', t))
    assert agent.task_outcome(goal) == 'success'
    world.lamps = torch.ones(2, dtype=torch.long)  # environment changes independently
    see(agent, world, (unit(0), unit(0, 1)))
    assert agent.task_outcome(goal) == 'unknown'


def test_verifier_exception_cannot_erase_executed_receipt(tmp_path):
    import pytest
    agent, world, scene, _, _ = live(tmp_path)
    _, read = agent.plan(agent.view, (1, 1), presses_done=0)
    action = TypedAction('press', agent.view.machines[0]['instance'], 0, 1)
    actuator = us.CountingActuator(world)
    def broken_verifier(t):
        raise RuntimeError('verifier unavailable')
    try:
        agent.execute(actuator, action, read, verification=broken_verifier)
    except (RuntimeError, ValueError):
        pass
    assert actuator.presses == 1
    receipts = [e for e in agent.memory.view()['evidence'].values() if e.modality == 'transition']
    assert len(receipts) == 1


def test_restart_recovers_after_attribution_before_binding(tmp_path, monkeypatch):
    import pytest
    from tests.test_unified_session import CREATE, active
    from pathwm.world_state.modules import AssociationBinder
    modules, agent, scene, rules, g = setup(tmp_path)
    world = rw.RuleWorld(scene, rules, torch.zeros(2, dtype=torch.long), rw.TaskContract())
    transitions = both(agent, world, 4, g)
    monkeypatch.setattr(agent, '_rebind', lambda instance: (_ for _ in ()).throw(RuntimeError('after attribution')))
    with pytest.raises(RuntimeError, match='after attribution'):
        see(agent, world, (unit(0), unit(0, 1)), transitions=transitions)
    assert sum(c.name == 'attribution' and c.active for c in agent.memory.view()['components'].values()) == 8
    agent.save(tmp_path/'memory')
    us.save_models(modules, tmp_path/'models.pt')
    restored = us.restore_agent(us.load_models(tmp_path/'models.pt',seed=3), tmp_path/'memory',
        settings=CREATE, binder=AssociationBinder(new_threshold=.5))
    restored.resume_view()
    for m in restored.view.machines:
        transitions_component = active(restored, m['instance'], 'transitions')
        assert transitions_component is not None and len(transitions_component.evidence) == 4
        binding = active(restored, m['instance'], 'binding')
        assert binding.data['supports'] and m['concept'] is not None
