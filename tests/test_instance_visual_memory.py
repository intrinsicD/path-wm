from pathlib import Path
import pytest
import torch
from experiments import unified_session as us
from pathwm.data import rule_world as rw
from pathwm.world_state.concepts import SOURCE
from pathwm.world_state.unified import VISUAL_SPACE

# ----------------------------------------------------------------------
# Per-instance visual memory through the actual R2 composition.
#
# Actual path: `experiments.unified_session.build(identity_run=J)` — native
# SlotPerception (width64, seven slots, three iterations, RGB64) sharing the belief's
# multiscale encoder, the provisional J perception+identity checkpoint, the recipe's
# default AssociationBinder and the learned slot-key candidates (no supplied keys or
# scene identity reach the agent). Train kinds, as in the R2 life recipe; with these
# uncalibrated binder thresholds the validation kind pair leaves one machine
# unresolved, so identity quality is not claimed here. Ground-truth renders/entity
# maps are used ONLY by assertions. Tests marked FAULT INJECTION write deliberately
# malformed records or disable one writer to exercise guard/backfill branches.


J = Path(__file__).resolve().parents[1] / "runs/latent_agent_r1/identity_joint_20260923"


@pytest.fixture(scope="module")
def modules():
    if not (J / "last.pt").exists():
        pytest.skip("J checkpoint not present")
    return us.build(0, identity_run=J)


def world(seed=0):
    g = torch.Generator().manual_seed(seed)
    kinds = torch.tensor([list(rw.KIND_SPLIT["train"][:2])])
    return g, kinds, rw.sample_scenes(g, kinds)


def frame(scene, lamps):
    rgb, entity = rw.render(scene, torch.tensor([lamps]))
    return rgb[0], entity[0]


def instances(agent):
    ids = [m["instance"] for m in agent.view.machines]
    assert all(ids) and len(set(ids)) == 2, "actual identity path must bind both machines"
    return ids


def machine_index(entity, machine):
    """Ground-truth scene machine under the agent's own slot pixel (assertions only)."""
    x, y = (int(v) for v in machine["xy"])
    return int(entity[y, x]) - 1


def visual(agent, **match):
    return [c for c in agent.memory.view()["components"].values()
            if c.name == "visual_slot" and all(getattr(c, k) == v for k, v in match.items())]


def fresh(modules, tmp_path, name="memory"):
    return us.new_agent(modules, tmp_path / name)


def test_acquisition_stores_native_slot_with_frame_and_recognition_lineage(modules, tmp_path):
    agent = fresh(modules, tmp_path)
    _, _, scene = world()
    rgb, _ = frame(scene, [0, 0])
    view, result = agent.observe(rgb)
    image = result["evidence"][0]
    live = agent.percept(rgb)[1]
    for m in view.machines:
        records = visual(agent, entity_id=m["instance"])
        assert len(records) == 1
        c = records[0]
        assert c.active and c.shape == (64,) and c.space == VISUAL_SPACE
        assert c.model_version == agent.memory.versions["perception"]
        assert c.evidence == (image,) and c.parents == (m["recognition"],)
        assert c.data["slot"] == m["slot"]
        assert torch.equal(c.tensor(), live.slots[0, m["slot"]].cpu())
        assert agent.visual_memory(m["instance"]) == c
    # Unbound slots (background/objects) receive no visual memory.
    assert len(visual(agent)) == 2


def test_render_uses_stored_slot_with_actual_decoder_and_heads_without_reencoding(modules, tmp_path, monkeypatch):
    agent = fresh(modules, tmp_path)
    _, _, scene = world()
    rgb, entity = frame(scene, [1, 0])
    view, _ = agent.observe(rgb)
    live = agent.percept(rgb)[1]
    with torch.no_grad():
        colors, alpha = agent.perception.decoder(live.slots)
    lamps = [1, 0]
    # FAULT INJECTION: any encoder or slot-attention call during recall fails the test.
    def forbidden(*args, **kwargs):
        raise AssertionError("recall re-encoded pixels")
    monkeypatch.setattr(agent.perception.encoder, "forward", forbidden)
    monkeypatch.setattr(agent.perception.slot_attention, "forward", forbidden)
    for m in view.machines:
        k, out = m["slot"], agent.render_memory(m["instance"])
        assert out["slot"].shape == (1, 1, 64) and out["rgb"].shape == (1, 1, 3, 64, 64)
        assert out["alpha"].shape == (1, 1, 64, 64)
        # Exact: the same decoder/heads on the live slot at the same [1,1,D] shape.
        one = live.slots[:, k:k + 1]
        same_rgb, same_alpha = agent.perception.decoder(one)
        assert torch.equal(out["slot"], one) and torch.equal(out["rgb"], same_rgb)
        assert torch.equal(out["alpha"], same_alpha)
        assert torch.equal(out["lamp"], agent.perception.heads["lamp"](one).squeeze(-1))
        # Declared float32 tolerance against the live 7-slot batch (batch-shape-dependent
        # CPU conv reductions: observed <=1e-5 rgb, <=5e-4 on alpha logits).
        assert torch.allclose(out["rgb"][0, 0], colors[0, k], rtol=0, atol=1e-4)
        assert torch.allclose(out["alpha"][0, 0], alpha[0, k], rtol=1e-5, atol=2e-3)
        assert torch.allclose(out["lamp"][0, 0], live.lamp[0, k], rtol=0, atol=1e-5)
        assert torch.allclose(out["kind"][0, 0], live.kind[0, k], rtol=0, atol=1e-5)
        assert torch.allclose(out["attributes"][0, 0], live.attributes[0, k], rtol=0, atol=1e-5)
        # Actual J lamp head on the recalled slot matches the rendered lamp.
        assert int(out["lamp"][0, 0] > 0) == lamps[machine_index(entity, m)]


def test_new_observation_becomes_current_and_old_record_is_immutable(modules, tmp_path):
    agent = fresh(modules, tmp_path)
    g, kinds, scene = world()
    rgb, _ = frame(scene, [0, 0])
    agent.observe(rgb)
    a, b = instances(agent)
    first = {i: agent.visual_memory(i) for i in (a, b)}
    moved = rw.sample_scenes(g, kinds)  # same machine kinds, new layout, one lamp toggled
    rgb2, entity2 = frame(moved, [1, 0])
    view, result = agent.observe(rgb2)
    assert {m["instance"] for m in view.machines} == {a, b}
    for m in view.machines:
        now = agent.visual_memory(m["instance"])
        assert now.evidence == (result["evidence"][0],) and now.parents == (m["recognition"],)
        assert int(agent.render_memory(m["instance"])["lamp"][0, 0] > 0) == [1, 0][machine_index(entity2, m)]
        old = agent.memory.view()["components"][first[m["instance"]].id]
        assert old == first[m["instance"]] and old.active  # history is immutable, not current
        assert now.id != old.id


def test_reassign_invalidates_newest_without_fallback_and_derives_for_new_owner(modules, tmp_path):
    agent = fresh(modules, tmp_path)
    g, kinds, scene = world()
    agent.observe(frame(scene, [0, 0])[0])
    a, b = instances(agent)
    b_record = agent.visual_memory(b)
    view, _ = agent.observe(frame(rw.sample_scenes(g, kinds), [0, 0])[0])
    m = next(m for m in view.machines if m["instance"] == a)
    newest = agent.visual_memory(a)
    tx = agent.session.store.begin("reassign-1", occurred_at=agent.session.time,
                                   available_at=agent.session.time, kind="correction")
    d = tx.create_entity(kind="instance")
    tx.reassign(m["recognition"], d)
    agent.correct(tx)
    components = agent.memory.view()["components"]
    assert not components[newest.id].active
    with pytest.raises(LookupError):  # older active sighting of `a` is NOT resurrected
        agent.visual_memory(a)
    moved = agent.visual_memory(d)
    assert moved.parents == (m["recognition"],) and moved.evidence == newest.evidence
    assert torch.equal(moved.tensor(), newest.tensor())
    # The unaffected instance is untouched.
    assert agent.visual_memory(b) != b_record  # b was re-observed in obs2 ...
    assert components[b_record.id] == b_record  # ... and its history is unchanged


def test_frame_retraction_invalidates_and_never_recreates(modules, tmp_path):
    agent = fresh(modules, tmp_path)
    _, _, scene = world()
    view, result = agent.observe(frame(scene, [0, 0])[0])
    a, b = instances(agent)
    image = result["evidence"][0]
    records = [agent.visual_memory(i).id for i in (a, b)]
    tx = agent.session.store.begin("retract-1", occurred_at=agent.session.time,
                                   available_at=agent.session.time, kind="correction",
                                   payload=dict(source=SOURCE))
    tx.retract_evidence(image)
    agent.correct(tx)
    assert agent.view is None
    assert agent.resume_view() is None
    for i, cid in zip((a, b), records):
        with pytest.raises(LookupError):
            agent.visual_memory(i)
        assert not agent.memory.view()["components"][cid].active
    for m in view.machines:  # camera-derived identity must withdraw with its frame
        assert not agent.memory.view()["components"][m["recognition"]].active
        assert not agent._needs_visual(m["recognition"])
    revision = agent.session.store.revision
    agent._backfill_visual()
    assert agent.session.store.revision == revision and all(not c.active for c in visual(agent))
    with pytest.raises(LookupError):
        agent.source_pyramid(a)
    agent.observe(frame(scene,[0,0])[0])
    assert not set(instances(agent)) & {a,b}  # fail-closed, no identity reconciliation


def test_candidate_retraction_invalidates_recognition_lineage(modules, tmp_path):
    agent = fresh(modules, tmp_path)
    _, _, scene = world()
    view, _ = agent.observe(frame(scene, [0, 0])[0])
    a, b = instances(agent)
    rec = agent.memory.view()["components"][view.machines[0]["recognition"]]
    b_record = agent.visual_memory(view.machines[1]["instance"])
    tx = agent.session.store.begin("retract-2", occurred_at=agent.session.time,
                                   available_at=agent.session.time, kind="correction",
                                   payload=dict(source=SOURCE))
    tx.retract_evidence(rec.evidence[0])
    agent.correct(tx)
    agent.recover()
    with pytest.raises(LookupError):
        agent.visual_memory(view.machines[0]["instance"])
    assert agent.visual_memory(view.machines[1]["instance"]) == b_record


def test_restart_reproduces_records_renders_and_source_pyramid(modules, tmp_path):
    agent = fresh(modules, tmp_path)
    _, _, scene = world()
    rgb, _ = frame(scene, [0, 1])
    agent.observe(rgb)
    ids = instances(agent)
    before = {i: (agent.visual_memory(i), agent.render_memory(i)) for i in ids}
    pyramid = agent.source_pyramid(ids[0])
    agent.save(tmp_path / "memory")
    us.save_models(modules, tmp_path / "models.pt")
    restored = us.restore_agent(us.load_models(tmp_path / "models.pt", 1), tmp_path / "memory")
    for i in ids:
        record, render = before[i]
        assert restored.visual_memory(i) == record
        out = restored.render_memory(i)
        for key in ("slot", "rgb", "alpha", "lamp", "kind", "attributes"):
            assert torch.equal(out[key], render[key]), key
    again = restored.source_pyramid(ids[0])
    assert len(again.scales) == len(pyramid.scales) == 2
    assert [s.grid for s in again.scales] == [(1, 16, 16), (1, 8, 8)]
    for s, t in zip(again.scales, pyramid.scales):
        assert torch.equal(s.values, t.values) and torch.equal(s.valid, t.valid)
    # Full native pyramid under the fixed hash, recomputed from the retained frame.
    live = modules["perception"].pyramid(rgb[None])
    for s, t in zip(again.scales, live.scales):
        assert torch.equal(s.values, t.values)


def test_backfill_for_pre_existing_store_only_from_valid_recognitions(modules, tmp_path, monkeypatch):
    agent = fresh(modules, tmp_path)
    _, _, scene = world()
    # FAULT INJECTION: emulate a store saved before visual memory existed.
    monkeypatch.setattr(type(agent), "_needs_visual", lambda self, recognition, view=None: False)
    rgb, _ = frame(scene, [0, 0])
    view, _ = agent.observe(rgb)
    ids = instances(agent)
    assert not visual(agent)
    agent.save(tmp_path / "memory")
    monkeypatch.undo()
    us.save_models(modules, tmp_path / "models.pt")
    restored = us.restore_agent(us.load_models(tmp_path / "models.pt", 1), tmp_path / "memory")
    live = restored.percept(rgb)[1]
    for m in view.machines:
        c = restored.visual_memory(m["instance"])
        assert c.parents == (m["recognition"],) and torch.equal(c.tensor(), live.slots[0, m["slot"]].cpu())
    revision = restored.session.store.revision
    restored.recover()
    restored._refresh()
    assert restored.session.store.revision == revision  # idempotent: nothing rewritten
    assert len(visual(restored)) == 2 and set(c.entity_id for c in visual(restored)) == set(ids)


def test_changed_perception_is_rejected(modules, tmp_path):
    agent = fresh(modules, tmp_path)
    _, _, scene = world()
    agent.observe(frame(scene, [0, 0])[0])
    a, _ = instances(agent)
    weight = modules["perception"].decoder.network[0].bias
    with torch.no_grad():
        weight[0] += 1e-3
    try:
        with pytest.raises(ValueError):
            agent.render_memory(a)
    finally:
        with torch.no_grad():
            weight[0] -= 1e-3
    assert agent.render_memory(a)["record"] == agent.visual_memory(a)


def _inject(agent, instance, **overrides):
    """FAULT INJECTION: publish a malformed newest visual_slot for `instance`."""
    good = agent.visual_memory(instance)
    value = overrides.pop("value", None)
    fields = dict(space=good.space, model_version=good.model_version, evidence=good.evidence,
                  parents=good.parents, data=dict(good.data))
    fields.update(overrides)
    tx = agent.memory.begin("internal")
    tx.put_component(instance, "visual_slot", good.tensor() if value is None else value,
                     role="inferred", **fields)
    agent.memory.commit(tx)


@pytest.mark.parametrize("case", ["version", "space", "shape", "slot", "source_modality",
                                  "source_event", "record_event", "parent_owner", "parent_kind"])
def test_reader_guards_reject_malformed_newest_record(modules, tmp_path, case):
    agent = fresh(modules, tmp_path)
    g, kinds, scene = world()
    _, first = agent.observe(frame(scene, [0, 0])[0])
    a, b = instances(agent)
    _, second = agent.observe(frame(rw.sample_scenes(g, kinds), [0, 0])[0])
    good, other = agent.visual_memory(a), agent.visual_memory(b)
    evidence = agent.memory.view()["evidence"]
    candidate = next(e for e in evidence.values() if e.event_id == "obs-000002" and e.modality == "image"
                     and e.id != second["evidence"][0])
    old_frame = first["evidence"][0]
    appearance = next(c for c in agent.memory.view()["components"].values()
                      if c.name == "appearance" and c.parents == good.parents and c.active)
    overrides = dict(
        version=dict(model_version="0" * 16),
        space=dict(space="machine-key"),
        shape=dict(value=torch.zeros(32)),
        slot=dict(data=dict(good.data, slot=(good.data["slot"] + 1) % 7)),
        source_modality=dict(evidence=(candidate.id,)),
        source_event=dict(evidence=(old_frame,)),
        record_event=dict(data=dict(good.data, event="another-event")),
        parent_owner=dict(parents=other.parents),
        parent_kind=dict(parents=(appearance.id,)),
    )[case]
    _inject(agent, a, **overrides)
    with pytest.raises(ValueError):
        agent.visual_memory(a)
    with pytest.raises(ValueError):
        agent.render_memory(a)
    assert agent.visual_memory(b) == other


def test_unknown_instance_is_lookup_error(modules, tmp_path):
    agent = fresh(modules, tmp_path)
    with pytest.raises(LookupError):
        agent.visual_memory("no-such-entity")


def test_rederived_older_sighting_does_not_override_newer_observation(modules, tmp_path):
    agent = fresh(modules, tmp_path)
    g, kinds, scene = world()
    rules = tuple(rw.split_rules()['train'][:2])
    demos = us.demonstrations(scene, rules, torch.zeros(2,dtype=torch.long), g, 3)
    first, _ = agent.observe(frame(scene,[0,0])[0], transitions=demos)
    instance, recognition = first.machines[0]['instance'], first.machines[0]['recognition']
    second, _ = agent.observe(frame(rw.sample_scenes(g,kinds),[1,1])[0])
    latest = agent.visual_memory(instance)
    assert latest.data['event'] == second.event_id
    for tag, target in (('away', None), ('back', instance)):
        tx = agent.session.store.begin(f'reassign-{tag}', occurred_at=agent.session.time,
                                      available_at=agent.session.time, kind='correction')
        destination = target or tx.create_entity(kind='instance')
        tx.reassign(recognition, destination)
        agent.correct(tx)
    assert agent.visual_memory(instance) == latest
    assert agent.render_memory(instance)['lamp'].item() > 0


def test_candidate_scope_is_explicit_and_survives_model_restore(tmp_path):
    if not (J/'last.pt').exists():
        pytest.skip('J checkpoint not present')
    _,_,scene = world()
    rgb,_ = frame(scene,[0,1])
    generic = us.build(0,identity_run=J)
    all_agent = fresh(generic,tmp_path,'all')
    percept = all_agent.percept(rgb)[1]
    assert len(all_agent.candidates(percept)) == 7
    scoped = us.build(0,identity_run=J,candidate_scope='predicted-machine')
    agent = fresh(scoped,tmp_path,'scoped')
    chosen = agent.candidates(percept)
    expected = [f'slot-{k}' for k in range(7) if percept.kind[0,k].argmax().item()==1]
    assert len(expected) == 2 and [c.id for c in chosen] == expected
    us.save_models(scoped,tmp_path/'models.pt')
    loaded = us.load_models(tmp_path/'models.pt')
    restored = fresh(loaded,tmp_path,'restored')
    assert [c.id for c in restored.candidates(percept)] == expected


@pytest.mark.parametrize('prewithdraw_candidate',[False,True])
def test_frame_withdrawal_retry_is_idempotent(modules,tmp_path,prewithdraw_candidate):
    agent=fresh(modules,tmp_path)
    _,_,scene=world()
    view,result=agent.observe(frame(scene,[0,0])[0])
    image=result['evidence'][0]
    if prewithdraw_candidate:
        recognition=agent.session.store.component(view.machines[0]['recognition'])
        tx=agent.session.store.begin('earlier',occurred_at=1.,available_at=1.,kind='correction',payload=dict(source=SOURCE))
        tx.retract_evidence(recognition.evidence[0]);agent.correct(tx)
    def transaction():
        tx=agent.session.store.begin('frame-withdraw',occurred_at=1.,available_at=2.,kind='correction',payload=dict(source=SOURCE))
        tx.retract_evidence(image)
        return tx
    tx=transaction()
    receipt,_=agent.correct(tx)
    snapshot=agent.session.store.snapshot()
    assert agent.correct(transaction())[0]==receipt
    assert agent.correct(tx)[0]==receipt
    assert agent.session.store.snapshot()==snapshot
