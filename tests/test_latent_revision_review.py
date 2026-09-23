"""Independent review: withdrawal must repair graph identity, not only values."""

import pytest
import torch

from pathwm.world_state.store import WorldStore


def test_withdrawn_identity_evidence_does_not_keep_entities_merged():
    store = WorldStore()
    tx = store.begin("seen", occurred_at=1, available_at=1)
    proof = tx.add_evidence("camera", "image")
    left, right = tx.create_entity("left"), tx.create_entity("right")
    relation = tx.merge(left, right, evidence=(proof,))
    store.commit(tx)
    before = store.revision
    assert store.canonical(left) == store.canonical(right)

    tx = store.begin(
        "withdraw", occurred_at=1, available_at=2, kind="correction",
        payload={"source": "camera"},
    )
    tx.retract_evidence(proof)
    store.commit(tx)
    assert store.canonical(left) != store.canonical(right)
    assert not next(r for r in store.relations() if r.id == relation).active
    historical = store.at(revision=before)
    assert historical.canonical(left) == historical.canonical(right)


def test_supersede_cannot_point_to_already_withdrawn_new_evidence():
    store = WorldStore()
    tx = store.begin("old", occurred_at=1, available_at=1)
    old = tx.add_evidence("camera", "image")
    store.commit(tx)
    before = store.snapshot()
    tx = store.begin("new", occurred_at=1, available_at=2)
    first = tx.add_evidence("camera", "image", content_hash="a")
    second = tx.add_evidence("camera", "image", content_hash="b")
    tx.supersede(first, second)
    tx.supersede(old, first)
    with pytest.raises(ValueError):
        store.commit(tx)
    assert store.snapshot() == before


def test_impossible_disjoint_queries_fail_instead_of_leaking_support(monkeypatch):
    from pathwm.data import rule_world as rw

    original = rw.sample_scenes

    def identical_attributes(generator, kinds, attrs=None):
        return original(generator, kinds, attrs=torch.zeros(1, 4, 4, dtype=torch.long))

    monkeypatch.setattr(rw, "sample_scenes", identical_attributes)
    # Only two semantic inputs exist; this support covers both. A capped sampler
    # must reject the unsatisfiable query rather than silently reuse its answer.
    with pytest.raises((ValueError, RuntimeError)):
        rw.sample_episodes(
            torch.Generator().manual_seed(17), rw.grammar(), range(4),
            episodes=1, support=(128,), queries=4, p_empty=0, chain=False,
        )


def test_nonfinite_action_does_not_partially_execute():
    from pathwm.data import rule_world as rw

    scene = rw.sample_scenes(torch.Generator().manual_seed(5), torch.tensor([[0, 1]]))
    env = rw.RuleWorld(scene, rw.grammar()[:2], (0, 0), rw.TaskContract())
    before = env.lamps.clone()
    with pytest.raises(ValueError):
        env.press(rw.ActionRecord((float("nan"), 10.0), (8.0, 46.0), (24.0, 46.0)))
    assert env.presses == 0
    assert torch.equal(env.lamps, before)


def test_read_set_pins_component_revision_even_when_id_stays_active(tmp_path):
    from pathwm.world_state.concepts import ConceptMemory

    memory = ConceptMemory(tmp_path, versions={"perception": "p", "core": "c"})
    tx = memory.begin("internal")
    first, second = tx.create_entity("first"), tx.create_entity("second")
    component = tx.put_component(
        first, "code", torch.ones(2), space="rule-code", model_version="c",
        role="inferred",
    )
    memory.commit(tx)
    read = memory.read_set((component,))
    assert memory.is_current(read)
    tx = memory.begin("correction")
    tx.reassign(component, second)
    memory.commit(tx)
    assert memory.store.component(component).active
    assert not memory.is_current(read)


def test_loading_new_models_requires_explicit_memory_migration(tmp_path):
    from pathwm.world_state.concepts import ConceptMemory

    memory = ConceptMemory(tmp_path, versions={"perception": "p", "core": "c"})
    memory.save()
    with pytest.raises(ValueError, match="[Vv]ersion|[Mm]odel|migration"):
        ConceptMemory.load(tmp_path, versions={"perception": "new", "core": "c"})


def test_rollout_object_inputs_do_not_use_intermediate_observations():
    from types import SimpleNamespace
    from torch.nn import functional as F
    from pathwm.data import rule_world as rw
    from pathwm.evaluation.rule_world import encode_episodes

    def contextual_perceiver(scenes, lamps):
        # Deliberately contextual features: a lamp change affects object tokens,
        # as it can in the real global encoder. Masks remain exact for this test.
        _, entity = rw.render(scenes, lamps)
        values = torch.arange(7).float()[None] + lamps.sum(-1, keepdim=True)
        return SimpleNamespace(
            slots=values[..., None].expand(-1, -1, 4),
            alpha=F.one_hot(entity, 7).permute(0, 3, 1, 2).float() * 40 - 20,
            lamp=torch.zeros(len(scenes), 7),
        )

    batch = rw.sample_episodes(
        torch.Generator().manual_seed(0), (rw.Rule("toggle", 0, 0, 0),),
        range(4), episodes=4, support=(4,), queries=2, p_empty=0,
    )
    encoded = encode_episodes(contextual_perceiver, batch, "cpu", chunk=32)
    first = batch.chain.step == 0
    second = batch.chain.step == 1
    initial = batch.chain.pre[first].sum(-1)
    assert (initial != batch.chain.pre[second].sum(-1)).any(), "fixture must expose change"
    expected = 3 + batch.chain.a[second] + initial
    assert torch.equal(encoded.chain.a[second, 0], expected.float())


def test_formal_gates_cannot_drop_a_missing_seed_measurement():
    import copy
    from pathwm.data.rule_world import FAMILIES
    from pathwm.evaluation.rule_world import aggregate_gates, FORMAL_PROTOCOL, PASS, INCOMPLETE

    summary = {
        "nu_full": dict.fromkeys(FAMILIES, 0.95),
        "nu_empty": dict.fromkeys(FAMILIES, 0.1),
        "nu_pre_restart_formal": dict.fromkeys(FAMILIES, 0.95),
        "swap": {"informative": 20, "follows_swapped_rule_under_swap": 1.0,
                 "follows_swapped_rule_own_code": 0.0},
        "tasks": {"agent": {"mean_utility": 0.9}, "empty_memory": {"mean_utility": 0.0}},
        "screen": dict.fromkeys(
            ("C1", "C3", "C4", "C5b", "C5c", "C2_ece", "I_runtime_checks"), PASS
        ),
        "lives_by_support": {"8": 4, "32": 4, "128": 4},
        "counter_evidence": {"gain": 0.3},
    }
    summaries = [copy.deepcopy(summary) for _ in range(3)]
    for seed, item in zip((1101, 1102, 1103), summaries):
        item["provenance"] = dict(
            core_seed=seed, perception_seed=seed,
            core_sha256=f"core-{seed}", perception_sha256=f"perception-{seed}",
            protocol=FORMAL_PROTOCOL, evaluation_seed=999,
            source_sha256="same-source", floors_sha256="same-floors",
        )
    assert aggregate_gates(summaries)["status"] == PASS
    del summaries[2]["nu_full"]["category"]
    result = aggregate_gates(summaries)
    assert result["gates"]["C2"] == INCOMPLETE
    assert result["status"] == INCOMPLETE


def test_formal_eligibility_survives_the_actual_json_transport():
    import json
    from pathwm.evaluation.rule_world import FORMAL_PROTOCOL, formal_eligibility

    summaries = [
        dict(
            lives_by_support={8: 4, 32: 4, 128: 4},
            provenance=dict(
                core_seed=seed, perception_seed=seed,
                core_sha256=f"core-{seed}", perception_sha256=f"perception-{seed}",
                protocol=FORMAL_PROTOCOL, evaluation_seed=999,
                source_sha256="same-source", floors_sha256="same-floors",
            ),
        )
        for seed in (1101, 1102, 1103)
    ]
    assert formal_eligibility(summaries) == []
    restored = json.loads(json.dumps(summaries))
    assert formal_eligibility(restored) == []


def test_read_set_rejects_replaced_concept_code_even_when_old_record_is_retained(tmp_path):
    from pathwm.world_state.concepts import ConceptMemory

    memory = ConceptMemory(tmp_path, versions={"core": "c", "perception": "p"})
    tx = memory.begin("internal")
    entity = tx.create_entity("concept", kind="concept")
    code = tx.put_component(entity, "code", torch.zeros(2), space="rule-code", model_version="c", role="inferred")
    memory.commit(tx)
    read = memory.read_set((code,))
    assert memory.is_current(read)
    tx = memory.begin("internal")
    tx.put_component(entity, "code", torch.ones(2), space="rule-code", model_version="c", role="inferred")
    memory.commit(tx)
    assert not memory.is_current(read)
