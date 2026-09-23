"""Inherited R1 write paths and the identity-key adapter on the unified runtime.

SOFTWARE fixtures (see tests/test_unified_session.py): supplied keys for the agent's
own machine slots and supplied ok transitions at its own perceived pixels. The saved
S1 run is written in the recipe's real checkpoint format; its weights are random, so
nothing here says learned keys identify machines.
"""

import json

import pytest
import torch

from experiments import unified_session as us
from pathwm.data import rule_world as rw
from pathwm.io import atomic_json, state_hash
from pathwm.models.latent_core import key_head
from pathwm.models.slots import SlotPerception
from pathwm.world_state.concepts import SOURCE
from pathwm.world_state.modules import AssociationBinder
from tests.test_unified_session import CREATE, active, both, demos, see, setup, supplied, unit


def ready(tmp_path):
    modules, agent, scene, rules, g = setup(tmp_path)
    world = rw.RuleWorld(scene, rules, torch.zeros(2, dtype=torch.long), rw.TaskContract())
    _, result = see(agent, world, (unit(0), unit(0, 1)), transitions=both(agent, world, 4, g))
    return modules, agent, world, g, result


def effects(agent, tmp_path):
    """Everything an unsupported call must leave untouched."""
    return dict(
        blobs=sorted(p.name for p in (tmp_path / "memory" / "blobs").iterdir()),
        store=agent.session.store.snapshot(),
        time=agent.session.time,
        belief=agent.session.state.tokens.clone(),
        view=(agent.view.event_id, [(m["instance"], m["recognition"]) for m in agent.view.machines]),
    )


def same(a, b):
    return {k: v for k, v in a.items() if k != "belief"} == {k: v for k, v in b.items() if k != "belief"} \
        and torch.equal(a["belief"], b["belief"])


def keys_are_well_formed(agent):
    width = agent.core.key_head[-1].out_features
    view = agent.memory.view()["components"].values()
    return all(c.shape == (width,) for c in view if c.name in ("key", "appearance") and c.active)


# --- F1: inherited R1 APIs ---------------------------------------------------------


def test_legacy_retract_name_corrects_through_the_session_and_rederives(tmp_path):
    modules, agent, world, g, result = ready(tmp_path)
    a, b = (m["instance"] for m in agent.view.machines)
    b_binding = active(agent, b, "binding")
    b_code = active(agent, b_binding.data["concept"], "code")
    a_concept = active(agent, a, "binding").data["concept"]
    withdrawn = result["evidence"][1]  # a successful transition attributed to machine a
    assert withdrawn in active(agent, a, "transitions").evidence
    agent.receive_retract(withdrawn)
    assert withdrawn in agent.memory.view()["retracted"]
    assert keys_are_well_formed(agent)
    remaining = active(agent, a, "transitions")
    assert remaining is not None and withdrawn not in remaining.evidence and len(remaining.evidence) == 3
    binding = active(agent, a, "binding")
    assert binding.data["supports"] and binding.parents[1] == remaining.id
    # Below min_check no behavioural re-check is possible: the prior membership is kept
    # (unverified), never re-proposed by appearance into another concept.
    assert binding.data["concept"] == a_concept and not binding.data["verified"]
    assert agent.memory.view()["components"][binding.parents[0]].name == "appearance"
    assert not any(withdrawn in c.evidence for c in agent.memory.view()["components"].values()
                   if c.active and c.name == "code")
    assert active(agent, b, "binding") == b_binding  # the unaffected concept is untouched
    assert active(agent, b_binding.data["concept"], "code") == b_code
    assert agent.repair() == {}  # the legacy name dispatches to the R2 repair: nothing left


def test_legacy_supersede_publishes_a_frameless_correction_with_original_attribution(tmp_path):
    modules, agent, world, g, result = ready(tmp_path)
    a, b = (m["instance"] for m in agent.view.machines)
    view_before = agent.view.event_id
    old = result["evidence"][1]
    pre, _, _, post = demos(agent, world, 1, 1, g)[0]
    record = demos(agent, world, 1, 1, g)[0][1]  # the corrected action hit machine b
    new = agent.receive_supersede(old, (pre, record, "ok", post))
    view = agent.memory.view()
    item = view["evidence"][new]
    assert view["retracted"][old]["replacement"] == new
    assert item.data["perceived_in"] == view["evidence"][old].data["perceived_in"]
    assert not any(e.event_id == item.event_id and e.modality == "image" for e in view["evidence"].values())
    assert agent.view.event_id == view_before and agent.resume_view().event_id == view_before
    owners = {c.entity_id for c in view["components"].values()
              if c.active and c.name == "attribution" and c.data["transition"] == new}
    assert owners == {b} and old not in active(agent, a, "transitions").evidence
    failed = agent.receive_supersede(new, (pre, record, "miss", post))
    assert not any(c.active and c.name == "attribution" and c.data["transition"] == failed
                   for c in agent.memory.view()["components"].values())
    assert keys_are_well_formed(agent)


def test_unsupported_r1_write_paths_fail_before_any_effect(tmp_path):
    modules, agent, world, g, result = ready(tmp_path)
    before = effects(agent, tmp_path)
    frame = world.frame()
    transition = demos(agent, world, 0, 1, g)[0]
    calls = dict(
        observe_scene=lambda: agent.observe_scene(frame),
        observe_session=lambda: agent.observe_session([transition]),
        receive_claim=lambda: agent.receive_claim(frame, transition[1], 1, rw.Actuator(world)),
        act=lambda: agent._act(rw.Actuator(world), agent.view, 0, transition[1]),
        supersede_unknown=lambda: agent.receive_supersede("no-such-evidence", transition),
        retract_unknown=lambda: agent.receive_retract("no-such-evidence"),
        retract_image=lambda: agent.receive_retract(result["evidence"][0]),
    )
    for name, call in calls.items():
        with pytest.raises((NotImplementedError, ValueError)):
            call()
        assert same(effects(agent, tmp_path), before), name
    agent.receive_retract(result["evidence"][1])
    with pytest.raises(ValueError, match="already withdrawn"):
        agent.receive_retract(result["evidence"][1])


def test_changed_model_refuses_legacy_corrections_before_effects(tmp_path):
    modules, agent, world, g, result = ready(tmp_path)
    before = effects(agent, tmp_path)
    transition = demos(agent, world, 1, 1, g)[0]
    with torch.no_grad():
        agent.core.outcome.bias.add_(0.1)
    for call in (lambda: agent.receive_retract(result["evidence"][1]),
                 lambda: agent.receive_supersede(result["evidence"][1], transition),
                 lambda: agent.repair()):
        with pytest.raises(ValueError, match="weights changed"):
            call()
        assert same(effects(agent, tmp_path), before)


# --- F3: identity-key adapter --------------------------------------------------------


def saved_identity_run(path, seed=7, mode="joint"):
    """A saved S1 run in the recipe's real checkpoint format (random weights)."""
    torch.manual_seed(seed)
    perception, key = SlotPerception(us.WIDTH, 7, 3), key_head(us.WIDTH, us.KEY)
    path.mkdir(parents=True)
    model = {f"perception.{k}": v for k, v in perception.state_dict().items()}
    model |= {f"key.{k}": v for k, v in key.state_dict().items()}
    torch.save(dict(schema="pathwm-run-v1", model=model, step=3), path / "last.pt")
    sizes = dict(width=us.WIDTH, key_width=us.KEY, slots=7, iterations=3, decoder_width=32)
    atomic_json(path / "run.json", dict(identity=dict(settings=dict(identity=mode, sizes=sizes))))
    return perception, key


def test_identity_key_adapter_uses_the_exported_key_exactly(tmp_path):
    perception, key = saved_identity_run(tmp_path / "s1")
    modules = us.build(0, identity_run=tmp_path / "s1")
    assert state_hash(modules["perception"]) == state_hash(perception)
    assert state_hash(modules["core"].key_head) == state_hash(key)
    assert modules["candidates"].key is modules["core"].key_head  # one key, no second projection
    agent = us.new_agent(modules, tmp_path / "memory", settings=CREATE)
    g = torch.Generator().manual_seed(0)
    scene = rw.sample_scenes(g, torch.tensor([list(rw.KIND_SPLIT["train"][:2])]))
    world = rw.RuleWorld(scene, rw.split_rules()["train"][:2], torch.zeros(2, dtype=torch.long), rw.TaskContract())
    _, result = agent.observe(world.frame())
    _, percept = agent.percept(world.frame())
    expected = agent.core.key(percept.slots[0])
    view = agent.memory.view()
    for d in result["bindings"]:
        slot = int(d["candidate"].split("-")[1])
        stored = torch.tensor(view["evidence"][d["evidence"]].data["key"])
        assert torch.equal(stored, expected[slot].detach().cpu())


def test_identity_key_choice_survives_save_and_restore_and_changes_are_rejected(tmp_path):
    saved_identity_run(tmp_path / "s1")
    modules = us.build(0, identity_run=tmp_path / "s1")
    us.save_models(modules, tmp_path / "models.pt")
    config = json.loads((tmp_path / "models.json").read_text())
    assert config["candidate_key"]["source"] == "identity_run"
    assert config["candidate_key"]["key_state_sha256"] == state_hash(modules["core"].key_head)
    assert config["candidate_key"]["architecture"] == f"pathwm.models.latent_core.key_head({us.WIDTH}, {us.KEY})"
    agent = us.new_agent(modules, tmp_path / "memory", settings=CREATE)
    g = torch.Generator().manual_seed(0)
    scene = rw.sample_scenes(g, torch.tensor([list(rw.KIND_SPLIT["train"][:2])]))
    world = rw.RuleWorld(scene, rw.split_rules()["train"][:2], torch.zeros(2, dtype=torch.long), rw.TaskContract())
    agent.observe(world.frame())
    agent.save(tmp_path / "memory")
    loaded = us.load_models(tmp_path / "models.pt", seed=9)
    assert loaded["candidates"].key is loaded["core"].key_head
    restored = us.restore_agent(loaded, tmp_path / "memory", settings=CREATE)
    frame = world.frame()
    k1 = [c.key for c in agent.candidates(agent.percept(frame)[1])]
    k2 = [c.key for c in restored.candidates(restored.percept(frame)[1])]
    assert all(torch.equal(x, y) for x, y in zip(k1, k2)) and restored.memory.versions == agent.memory.versions
    # Changing only the shared key is a model change for both owners' checks.
    with torch.no_grad():
        restored.core.key_head[0].weight.add_(0.01)
    with pytest.raises(ValueError, match="weights changed"):
        restored.observe(frame)
    # A default (random-projection) composition cannot silently restore this memory.
    with pytest.raises(ValueError, match="differ|Incompatible session model"):
        us.restore_agent(us.build(0), tmp_path / "memory", settings=CREATE)


def test_identity_key_adapter_validates_its_source(tmp_path):
    saved_identity_run(tmp_path / "plain", mode=None)
    with pytest.raises(ValueError, match="identity"):
        us.build(0, identity_run=tmp_path / "plain")
    from pathwm.world_state.modules import CandidateEncoder

    with pytest.raises(ValueError, match="key width"):
        CandidateEncoder(us.WIDTH, 16, us.VALUE, key=key_head(us.WIDTH, us.KEY))
    bad = key_head(us.WIDTH, us.KEY)
    with torch.no_grad():
        bad[-1].bias.fill_(float("nan"))
    encoder = CandidateEncoder(us.WIDTH, us.KEY, us.VALUE, key=bad)
    with pytest.raises(ValueError, match="finite"):
        encoder(torch.zeros(3, us.WIDTH))


def test_same_kind_twins_stay_distinct_or_unresolved(tmp_path):
    """SOFTWARE fixture: both machines of the agent's own layout get the SAME key (as
    same-kind twins would under an appearance key). They must never become one entity."""
    modules, agent, scene, rules, g = setup(tmp_path, binder=AssociationBinder())
    world = rw.RuleWorld(scene, rules, torch.zeros(2, dtype=torch.long), rw.TaskContract())
    twins = (unit(0), unit(0))
    for _ in range(3):
        _, result = see(agent, world, twins, transitions=demos(agent, world, 1, 2, g))
        decided = [d for d in result["bindings"]]
        owners = [d["entity_id"] for d in decided if d["entity_id"] is not None]
        assert len(owners) == len(set(owners))  # one event never binds both twins to one entity
        assert any(d["status"] == "unresolved" for d in decided)
    # Transitions aimed at the unresolved twin stay raw: nothing attributes them anywhere.
    unresolved = [m for m in agent.view.machines if m["instance"] is None]
    assert unresolved
    view = agent.memory.view()
    twin_side = [m["instance"] for m in agent.view.machines].index(None)
    aimed = {e.id for e in view["evidence"].values() if e.modality == "transition"
             and tuple(e.data["action"]["machine_xy"]) == tuple(agent.view.machines[twin_side]["xy"])}
    assert aimed and not any(c.active and c.name == "attribution" and c.data["transition"] in aimed
                             for c in view["components"].values())


def test_recipe_identity_run_records_the_key_and_restarts_equal(tmp_path, monkeypatch):
    import sys

    saved_identity_run(tmp_path / "s1", mode="detached")
    out = tmp_path / "life"
    monkeypatch.setattr(sys, "argv", ["unified_session", "--output", str(out), "--scenes", "3",
                                      "--identity-run", str(tmp_path / "s1")])
    us.main()
    assert json.loads((out / "status.json").read_text())["result"] == "completed"
    settings = json.loads((out / "run.json").read_text())["identity"]["settings"]
    assert settings["binding"].startswith("exported S1 identity key")
    source = json.loads((out / "models.json").read_text())["candidate_key"]
    assert source["identity_mode"] == "detached" and source["architecture"] == us.KEY_ARCHITECTURE
    assert json.loads((out / "life.json").read_text())["restart"]["equal"]
