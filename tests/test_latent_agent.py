"""Latent core, slot consumer, planner and integrated runtime contracts for R1."""

import copy

import pytest
import torch

from pathwm.data import rule_world as rw
from pathwm.io import state_hash
from pathwm.models import latent_core as lc
from pathwm.models.slots import SlotPerception, perception_loss, pointer

WIDTH = 16


def tiny_core():
    torch.manual_seed(0)
    return lc.LatentCore(width=WIDTH, heads=2, loops=2, code_tokens=2, key_width=8)


def test_induce_is_finite_empty_padding_and_permutation_safe():
    core = tiny_core().eval()
    e = torch.randn(1, 5, WIDTH)
    valid = torch.ones(1, 5, dtype=torch.bool)
    empty = core.induce(e[:, :0], valid[:, :0])
    assert torch.isfinite(empty).all()
    none_valid = core.induce(e, torch.zeros_like(valid))
    assert torch.allclose(none_valid, empty, atol=1e-6)
    padded = torch.cat((e, torch.full((1, 3, WIDTH), float("nan"))), 1)
    pad_valid = torch.cat((valid, torch.zeros(1, 3, dtype=torch.bool)), 1)
    z = core.induce(e, valid)
    assert torch.allclose(core.induce(padded, pad_valid), z, atol=1e-6)
    assert torch.allclose(core.induce(e[:, torch.randperm(5)], valid), z, atol=1e-5)


def test_queries_do_not_communicate_within_a_batch():
    core = tiny_core().eval()
    m, a, b = (torch.randn(4, WIDTH) for _ in range(3))
    z = torch.randn(4, 2, WIDTH)
    logit, nxt = core.apply(m, a, b, z)
    m2 = m.clone()
    m2[1:] = torch.randn(3, WIDTH)
    logit2, nxt2 = core.apply(m2, a, b, z)
    assert torch.equal(logit[0], logit2[0]) and torch.equal(nxt[0], nxt2[0])


def random_tokens(episodes=2, support=6, queries=5):
    def part(n, chain=False):
        t = dict(
            m_pre=torch.randn(n, WIDTH), a=torch.randn(n, WIDTH), b=torch.randn(n, WIDTH),
            m_post=torch.randn(n, WIDTH), outcome=torch.randint(2, (n,)).float(),
            episode=torch.arange(n) % episodes,
        )
        if chain:
            t["step"] = torch.arange(n) // episodes
        return lc.TransitionTokens(**t)

    return lc.EpisodeTokens(
        support=part(support * episodes), query=part(queries * episodes),
        chain=part(2 * episodes, chain=True), keys=torch.randn(episodes, 2, WIDTH),
        episodes=episodes,
    )


def test_core_loss_reaches_all_trainable_core_parameters():
    core = tiny_core()
    loss, metrics = lc.episode_loss(core, random_tokens(), torch.ones(WIDTH))
    loss.backward()
    missing = [n for n, p in core.named_parameters() if p.grad is None or not p.grad.abs().sum() > 0]
    assert not missing, missing
    assert all(torch.isfinite(torch.tensor(v)) for v in metrics.values())


def test_perception_loss_trains_multiscale_encoder_and_slots():
    torch.manual_seed(1)
    model = SlotPerception(width=WIDTH, slots=7, iterations=2, decoder_width=8)
    g = torch.Generator().manual_seed(0)
    scenes = rw.sample_scenes(g, torch.tensor([[0, 1], [2, 3]]))
    lamps = torch.tensor([[0, 1], [1, 0]])
    rgb, entity = rw.render(scenes, lamps)
    percept = model(rgb)
    assert percept.slots.shape == (2, 7, WIDTH) and percept.alpha.shape == (2, 7, 64, 64)
    loss, metrics = perception_loss(percept, rgb, entity, scenes.attrs, lamps)
    loss.backward()
    for prefix in ("encoder.stem", "encoder.pyramid", "slot_attention", "decoder", "heads"):
        grads = [p.grad for n, p in model.named_parameters() if n.startswith(prefix)]
        assert grads and any(g is not None and g.abs().sum() > 0 for g in grads), prefix
    idx = pointer(percept.alpha, scenes.machine_xy[:, 0])
    assert idx.shape == (2,) and idx.dtype == torch.long


class FakeCore:
    """Outcome logit reads Z[0,0]: the planner must consume the learned forecast."""

    def apply(self, m, a, b, z, loops=None):
        return z[:, 0, 0] * (a[:, 0] - b[:, 0]), m + 1.0


def test_planner_consumes_forecast_and_shares_contract():
    contract = rw.TaskContract()
    machines = torch.zeros(2, 4)
    objects = torch.eye(4)
    lamps = torch.tensor([-8.0, -8.0])  # both lamps confidently off
    goal = (1, 0)
    codes = torch.zeros(2, 2, 4)
    codes[0, 0, 0] = 20.0  # machine 0: a with larger first feature -> success
    d = lc.search(FakeCore(), machines, objects, codes, lamps, goal, contract, budget_left=2, presses_done=0)
    assert d.option == "press" and d.action[0] == 0 and d.action[1] == 0
    codes[0, 0, 0] = -20.0
    d2 = lc.search(FakeCore(), machines, objects, codes, lamps, goal, contract, budget_left=2, presses_done=0)
    assert d2.action != d.action
    assert d.values["abstain"] == contract.utility("abstain", 0)
    p_now = torch.sigmoid(lamps[0]) * torch.sigmoid(-lamps[1])
    assert d.values["stop"] == pytest.approx(contract.expected_stop(float(p_now), 0))
    done = lc.search(FakeCore(), machines, objects, codes, torch.tensor([8.0, -8.0]), goal, contract, budget_left=2, presses_done=0)
    assert done.option == "stop" and done.action is None
    empty = lc.search(FakeCore(), machines, objects, codes, lamps, goal, contract, budget_left=0, presses_done=2)
    assert empty.option in ("stop", "abstain") and empty.sequences == 0
    assert lc.search(FakeCore(), machines, objects, codes, lamps, goal, contract, budget_left=2, presses_done=0).sequences == 24 + 24 * 24


# ---------------------------------------------------------------- integrated runtime


def tiny_models():
    torch.manual_seed(2)
    perception = SlotPerception(width=WIDTH, slots=7, iterations=2, decoder_width=8).eval()
    core = tiny_core().eval()
    return perception, core


def tiny_life(seed=4, n=8):
    from pathwm.evaluation import rule_world as ev

    return ev.sample_life(torch.Generator().manual_seed(seed), "validation", n_support=n, queries=4, goals=2, distract=8, counter=4)


def test_life_keeps_weights_frozen_restarts_and_ignores_poisoned_hidden_fields(tmp_path):
    from pathwm.evaluation import rule_world as ev

    perception, core = tiny_models()
    before = state_hash(perception), state_hash(core)
    spec = tiny_life()
    record = ev.run_life(perception, core, spec, tmp_path / "life", settings=ev.AgentSettings())
    assert (state_hash(perception), state_hash(core)) == before
    assert len(set(record["hashes"])) == 1
    assert record["checks"]["restart_reproduces"] is True
    assert record["checks"]["supersede_matches_clean"] is True
    assert record["checks"]["unaffected_bit_identical"] is True
    assert record["checks"]["stale_plan_rejected"] is True
    poisoned = ev.run_life(perception, core, spec.poisoned(), tmp_path / "poisoned", settings=ev.AgentSettings())
    assert poisoned["agent_outputs"] == record["agent_outputs"]


def test_planner_inputs_contain_no_environment(tmp_path):
    from pathwm.evaluation import rule_world as ev

    perception, core = tiny_models()
    spec = tiny_life(seed=6)
    task = spec.goals[0]
    env = rw.RuleWorld(task.scene, task.rules, task.lamps.clone(), rw.TaskContract())
    agent = ev.fresh_agent(perception, core, tmp_path / "m", ev.AgentSettings())
    view = agent.observe_scene(env.frame())
    plan, _ = agent.plan(view, task.goal, presses_done=0)
    env.rules = (rw.Rule("close", 0, 0, 0), rw.Rule("open", 1, 1, 1))  # hidden change
    again, _ = agent.plan(agent.observe_scene(env.frame()), task.goal, presses_done=0)
    assert (plan.option, plan.action) == (again.option, again.action)


def test_slot_candidates_and_concept_context_connect_to_existing_world_modules():
    from pathwm.world_state.concepts import concept_context, slot_candidates
    from pathwm.world_state.modules import CandidateEncoder, ContextEncoder
    from torch import nn

    perception, core = tiny_models()
    rgb, _ = rw.render(rw.sample_scenes(torch.Generator().manual_seed(0), torch.tensor([[0, 1]])), torch.zeros(1, 2, dtype=torch.long))
    with torch.no_grad():
        percept = perception(rgb)
    candidates = slot_candidates(percept, CandidateEncoder(WIDTH, 8, 8), source="rule_world_camera", model_version="p")
    assert len(candidates) == 7 and candidates[0].key.shape == (8,)
    context = ContextEncoder(WIDTH, {"code": nn.Linear(2 * WIDTH, WIDTH)}, {"code": ("rule-code", "c")}, max_tokens=4)
    tokens = concept_context(context, torch.randn(2, WIDTH))
    assert tokens.shape == (1, WIDTH)


def test_prepared_once_tokens_equal_live_agent_encoding(tmp_path):
    from pathwm.evaluation import rule_world as ev

    perception, core = tiny_models()
    g = torch.Generator().manual_seed(9)
    batch = rw.sample_episodes(g, rw.split_rules()["train"], rw.KIND_SPLIT["train"], episodes=1, support=(4,), queries=2, p_empty=0.0)
    with torch.no_grad():
        tokens = ev.encode_episodes(ev.pixel_perceiver(perception), batch, "cpu")
    agent = ev.fresh_agent(perception, core, tmp_path / "m", ev.AgentSettings())
    t = batch.support
    for i in range(len(t)):
        scene = batch.scenes.select([int(t.scene[i])])
        pre = rw.render(scene, t.pre[i][None])[0][0]
        post = rw.render(scene, t.post[i][None])[0][0]
        record = ev.record_for(scene, int(t.machine[i]), int(t.a[i]), int(t.b[i]))
        live = agent.transition_tokens(pre, record, post)
        for name in ("m_pre", "a", "b", "m_post"):
            assert torch.allclose(getattr(tokens.support, name)[i], live[name], atol=1e-5), name


def test_recipe_pause_resume_matches_uninterrupted(tmp_path, monkeypatch):
    import sys
    import experiments.latent_agent as recipe

    def run(*arguments):
        monkeypatch.setattr(sys, "argv", ["latent_agent", *arguments])
        recipe.main()

    common = ["--stage", "perception", "--size", "check", "--device", "cpu", "--updates", "4"]
    run(*common, "--stop-after", "2", "--output", str(tmp_path / "paused"))
    run("--stage", "perception", "--resume", str(tmp_path / "paused"))
    run(*common, "--output", str(tmp_path / "straight"))
    a = torch.load(tmp_path / "paused" / "last.pt", weights_only=True)
    b = torch.load(tmp_path / "straight" / "last.pt", weights_only=True)
    assert a["step"] == b["step"] == 4
    assert all(torch.equal(a["model"][k], b["model"][k]) for k in a["model"])
    assert (tmp_path / "paused" / "report.html").exists()


# ---------------------------------------------------------------- review repairs


def feedback_agent(tmp_path, **settings):
    """Tiny random-weight agent. Every concept is created (never merged) and any
    appearance score admits feedback support: deterministic software wiring only."""
    from pathwm.evaluation import rule_world as ev
    from pathwm.world_state.concepts import AgentSettings

    perception, core = tiny_models()
    config = AgentSettings(tau=-1.0, tau_feedback=-1.0, lam=-1e9, **settings)
    agent = ev.fresh_agent(perception, core, tmp_path / "m", config)
    g = torch.Generator().manual_seed(21)
    rules = (rw.Rule("toggle", 0, 0, 0), rw.Rule("relation", 1, 1, 1))
    for kinds in ([48, 49], [50, 51]):  # each session's pressed instance founds a concept
        session = ev.DemoSession(rw.sample_scenes(g, torch.tensor([kinds])), rules, ("A", "B"),
                                 [(torch.randint(2, (2,), generator=g).tolist(), 0, *rw.PAIRS[k]) for k in range(4)])
        transitions, _ = ev.demo_transitions(session, rw.TaskContract())
        agent.observe_session(transitions)
    concepts = sorted(e.id for e in agent.memory.view()["entities"].values() if e.kind == "concept")
    assert len(concepts) == 2
    scene = rw.sample_scenes(g, torch.tensor([[48, 49]]))
    world = rw.RuleWorld(scene, rules, torch.zeros(2, dtype=torch.long), rw.TaskContract())
    return ev, agent, world, scene, concepts


def test_successful_execution_feedback_updates_only_the_bound_concept(tmp_path):
    """Software wiring (random weights): NOT evidence that the update is useful."""
    ev, agent, world, scene, concepts = feedback_agent(tmp_path)
    view = agent.observe_scene(world.frame())
    record = ev.record_for(scene, 0, 0, 1)
    m = agent._owner(view.machines, record.machine_xy)
    target = view.machines[m]["concept"]
    other = next(c for c in concepts if c != target)
    z_before, ids_before = agent.concept_state(target)
    read = agent.memory.read_set(ids_before + (view.machines[m]["binding"],))
    z_other, ids_other = agent.concept_state(other)
    result = agent._act(rw.Actuator(world), view, m, record)
    assert result["status"] == "ok" and result["feedback"]["status"] == "appearance_support"
    assert result["evidence"] in agent.support_of(target)
    assert not agent.memory.is_current(read)  # the earlier plan/prediction is stale
    z_after, ids_after = agent.concept_state(target)
    assert ids_after != ids_before and not torch.equal(z_after, z_before)
    fresh = ev.ConceptAgent(agent.perception, agent.core, agent.memory, agent.contract, agent.settings)
    assert torch.allclose(z_after, fresh.induce(agent.support_of(target)), atol=1e-5)
    z_other2, ids_other2 = agent.concept_state(other)
    assert ids_other2 == ids_other and torch.equal(z_other2, z_other)


def test_failed_receipts_are_kept_but_never_teach(tmp_path):
    ev, agent, world, scene, concepts = feedback_agent(tmp_path)
    view = agent.observe_scene(world.frame())
    supports = {c: agent.support_of(c) for c in concepts}
    miss = rw.ActionRecord((0.0, 0.0), tuple(scene.object_xy[0, 0].tolist()), tuple(scene.object_xy[0, 1].tolist()))
    result = agent._act(rw.Actuator(world), view, 0, miss)
    assert result["status"] == "miss" and result["feedback"] is None
    stored = agent.memory.view()["evidence"][result["evidence"]]
    assert stored.data["receipt"] == "miss"
    assert {c: agent.support_of(c) for c in concepts} == supports


def test_claim_test_enters_the_same_feedback_path(tmp_path):
    ev, agent, world, scene, concepts = feedback_agent(tmp_path)
    record = ev.record_for(scene, 1, 2, 3)
    tested = agent.receive_claim(world.frame(), record, 1, rw.Actuator(world))
    assert tested["tested"] and tested["feedback"]["status"] == "appearance_support"
    assert tested["evidence"] in agent.support_of(tested["feedback"]["concept"])
    testimony = [e for e in agent.memory.view()["evidence"].values() if e.source == "testimony"]
    assert testimony and all(e.id not in agent.support_of(c) for e in testimony for c in concepts)


def test_feedback_can_be_disabled_for_the_declared_control(tmp_path):
    ev, agent, world, scene, concepts = feedback_agent(tmp_path, feedback_support=False)
    view = agent.observe_scene(world.frame())
    before = {c: agent.support_of(c) for c in concepts}
    result = agent._act(rw.Actuator(world), view, 0, ev.record_for(scene, 0, 0, 1))
    assert result["status"] == "ok" and result["feedback"]["status"] == "appearance"
    assert {c: agent.support_of(c) for c in concepts} == before


def test_slot_attention_padding_count_has_no_influence():
    from pathwm.models.slots import SlotAttention

    torch.manual_seed(3)
    module = SlotAttention(WIDTH, 4, 2)
    with torch.no_grad():
        module.input_norm.bias.normal_()  # padded keys/values become nonzero
    tokens = torch.randn(1, 6, WIDTH)
    valid = torch.ones(1, 6, dtype=torch.bool)
    base = module(tokens, valid)
    for pad in (1, 50):
        padded = torch.cat((tokens, torch.randn(1, pad, WIDTH)), 1)
        mask = torch.cat((valid, torch.zeros(1, pad, dtype=torch.bool)), 1)
        assert torch.allclose(module(padded, mask), base, atol=1e-6)
    empty = module(tokens, torch.zeros_like(valid))
    assert torch.isfinite(empty).all()


def formal_summary(seed, **changes):
    from pathwm.evaluation import rule_world as ev

    families = dict.fromkeys(ev.FAMILY_ORDER, 0.9)
    summary = dict(
        lives_by_support={"8": 4, "32": 4, "128": 4},
        provenance=dict(core_seed=seed, perception_seed=seed, core_sha256=f"c{seed}", perception_sha256=f"p{seed}",
                        protocol=ev.FORMAL_PROTOCOL, evaluation_seed=7, source_sha256="s", floors_sha256="f"),
        nu_full=dict(families), nu_empty=dict.fromkeys(ev.FAMILY_ORDER, 0.1),
        nu_pre_restart_formal=dict(families),
        swap=dict(informative=10, follows_swapped_rule_under_swap=0.9, follows_swapped_rule_own_code=0.1),
        tasks=dict(agent=dict(mean_utility=0.8), empty_memory=dict(mean_utility=-0.2)),
        counter_evidence=dict(gain=0.3),
        screen=dict.fromkeys(("C1", "C2_ece", "C3", "C4", "C5c", "I_runtime_checks"), ev.PASS),
    )
    summary.update(changes)
    return summary


def test_formal_gates_need_distinct_seeds_full_protocol_and_complete_data():
    from pathwm.evaluation import rule_world as ev

    good = [formal_summary(s) for s in (1, 2, 3)]
    assert ev.aggregate_gates(good)["status"] == ev.PASS
    duplicate = [formal_summary(1), formal_summary(1), formal_summary(3)]
    assert ev.aggregate_gates(duplicate)["status"] == ev.INELIGIBLE
    smoke = [formal_summary(s) for s in (1, 2, 3)]
    for s in smoke:
        s["provenance"] = dict(s["provenance"], protocol=dict(ev.FORMAL_PROTOCOL, model_size="check"))
    assert ev.aggregate_gates(smoke)["status"] == ev.INELIGIBLE
    assert ev.aggregate_gates(good[:2])["status"] == ev.INELIGIBLE
    missing = [formal_summary(s) for s in (1, 2, 3)]
    del missing[1]["nu_full"]["toggle"]
    result = ev.aggregate_gates(missing)
    assert result["gates"]["C2"] == ev.INCOMPLETE and result["status"] != ev.PASS
    incomplete_screen = [formal_summary(s) for s in (1, 2, 3)]
    incomplete_screen[0]["screen"] = dict(incomplete_screen[0]["screen"], C4=ev.INCOMPLETE)
    assert ev.aggregate_gates(incomplete_screen)["gates"]["C4"] == ev.INCOMPLETE


def test_counter_evidence_gate_is_a_paired_lower_bound_not_point_positivity():
    from pathwm.evaluation import rule_world as ev

    variable = [formal_summary(s, counter_evidence=dict(gain=g)) for s, g in ((1, 0.9), (2, 0.01), (3, 0.02))]
    assert all(s["counter_evidence"]["gain"] > 0 for s in variable)
    result = ev.aggregate_gates(variable)
    assert result["bounds"]["counter_gain"] <= 0 and result["gates"]["C5b"] == ev.FAIL


def test_screen_marks_unestablished_checks_and_unavailable_strata_incomplete(tmp_path):
    from pathwm.evaluation import rule_world as ev

    perception, core = tiny_models()
    life = ev.run_life(perception, core, tiny_life(seed=8), tmp_path / "life", settings=ev.AgentSettings())
    life = dict(life, n_support=ev.FORMAL_N)
    floors = rw.floors()
    assert life["memory"]["store_bytes"] > 0 and life["memory"]["blob_files"] > 0
    unestablished = dict(life, checks=dict(life["checks"], stale_plan_rejected=None))
    assert ev.summarize([unestablished], floors)["screen"]["I_runtime_checks"] == ev.INCOMPLETE
    unavailable = dict(life, unavailable={"reach2": 1})
    assert ev.summarize([unavailable], floors)["screen"]["C4"] == ev.INCOMPLETE
    assert ev.summarize([dict(life, n_support=8)], floors)["nu_full"] == {"groups_without_both_classes": 0}


def test_reserved_memory_ceiling_is_enforced_with_saved_state(monkeypatch):
    import argparse
    import experiments.latent_agent as recipe

    saved = []

    class Runner:
        def save(self):
            saved.append(True)

    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda device=None: 7 * 2**30)
    args = argparse.Namespace(device="cuda", max_reserved_gib=6.0)
    with pytest.raises(recipe.ResourceCeiling):
        recipe.enforce_ceiling(args, Runner())
    assert saved == [True]


# ---------------------------------------------------------------- R15-R18


def test_symbolic_diagnostic_has_no_key_objective_and_frozen_symbols():
    import experiments.latent_agent as recipe

    s = recipe.SIZES["check"]
    model = recipe.RuleModel(s, symbolic=True)
    loss, metrics = lc.episode_loss(model.core, random_tokens(), torch.ones(WIDTH), key_weight=0.0)
    loss.backward()
    assert "key_nce" not in metrics
    assert all(p.grad is None for p in model.core.key_head.parameters())
    pixel_loss, pixel_metrics = lc.episode_loss(tiny_core(), random_tokens(), torch.ones(WIDTH))
    assert "key_nce" in pixel_metrics  # the pixel arm keeps its declared objective
    source = open(recipe.__file__).read()
    assert "model.perception.requires_grad_(False)" in source and "key_weight = 0.0 if symbolic" in source


def test_formal_family_schedule_covers_every_family_and_is_required():
    from pathwm.evaluation import rule_world as ev

    covered = {f for pair in ev.FAMILY_SCHEDULE for f in pair}
    assert covered == set(rw.FAMILIES) and len(ev.FAMILY_SCHEDULE) == ev.FORMAL_PROTOCOL["lives_per_support"]
    assert ev.FORMAL_PROTOCOL["family_schedule"] == [list(p) for p in ev.FAMILY_SCHEDULE]
    for i, families in enumerate(ev.FAMILY_SCHEDULE):
        spec = ev.sample_life(torch.Generator().manual_seed(i), "test", n_support=8, queries=2, goals=0,
                              distract=8, counter=8, families=families)
        assert (spec.rules["A"].family, spec.rules["B"].family) == families
    unscheduled = [formal_summary(seed) for seed in (1, 2, 3)]
    for s in unscheduled:
        protocol = {k: v for k, v in ev.FORMAL_PROTOCOL.items() if k != "family_schedule"}
        s["provenance"] = dict(s["provenance"], protocol=protocol)
    assert ev.aggregate_gates(unscheduled)["status"] == ev.INELIGIBLE


def test_undefined_balanced_accuracy_groups_stay_visible_and_block_c2():
    from pathwm.evaluation import rule_world as ev

    rows = [dict(family="category", group="g", truth=1, s=0, p=0.9) for _ in range(3)]
    nu = ev.nu_from_rows(rows, rw.floors())
    assert nu["groups_without_both_classes"] == 1 and "category" not in nu
    summaries = [formal_summary(seed) for seed in (1, 2, 3)]
    summaries[0]["nu_full"]["groups_without_both_classes"] = 1
    assert ev.aggregate_gates(summaries)["gates"]["C2"] == ev.INCOMPLETE


def test_real_summaries_survive_json_transport_into_formal_gates(tmp_path):
    import json
    from pathwm.evaluation import rule_world as ev

    perception, core = tiny_models()
    life = ev.run_life(perception, core, tiny_life(seed=10), tmp_path / "life", settings=ev.AgentSettings())
    lives = [dict(life, n_support=n) for n in (8, 32, 128) for _ in range(4)]
    summaries = []
    for seed in (1, 2, 3):
        provenance = dict(core_seed=seed, perception_seed=seed, core_sha256=f"c{seed}", perception_sha256=f"p{seed}",
                          protocol=ev.FORMAL_PROTOCOL, evaluation_seed=7, source_sha256="s", floors_sha256="f")
        summary = ev.summarize(lives, rw.floors(), None, provenance)
        summaries.append(json.loads(json.dumps(summary, default=str)))  # the real gates-stage transport
    assert ev.formal_eligibility(summaries) == []
    result = ev.aggregate_gates(summaries)
    assert result["status"] in (ev.FAIL, ev.INCOMPLETE)  # eligible, but random weights cannot pass


def test_feedback_does_not_stale_reads_of_unrelated_concepts(tmp_path):
    ev, agent, world, scene, concepts = feedback_agent(tmp_path)
    view = agent.observe_scene(world.frame())
    record = ev.record_for(scene, 0, 0, 1)
    m = agent._owner(view.machines, record.machine_xy)
    other = next(c for c in concepts if c != view.machines[m]["concept"])
    read_other = agent.memory.read_set(agent.concept_state(other)[1])
    assert agent._act(rw.Actuator(world), view, m, record)["feedback"] is not None
    assert agent.memory.is_current(read_other)
