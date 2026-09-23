"""RuleWorld generator/executor failures would silently invalidate every R1 gate."""

import itertools

import pytest
import torch

from pathwm.data import rule_world as rw
from pathwm.io import digest

# SHA256 of the approved split lists (runs/reviews/shared_abstraction_formulas_20260923/
# split-proposal.json), computed once on 23 September; code no longer reads that file.
APPROVED_SPLIT_SHA256 = (
    "18894000e3882c55bc5cf7cdbbfe7c8a01e1cce0f84a49eb0a8a24a19aad4ffc"
)


def scalar_truth(rule, a, b, s):
    """Independent restatement of the R1 formulas (u=1 for every executed press)."""
    if rule.family == "category":
        return int(a[rule.j] in rule.values)
    m = int((a[rule.j] + rule.delta) % 4 == b[rule.k])
    return {
        "relation": m,
        "open": int(s or m),
        "close": int(s and not m),
        "toggle": int(s != m),
    }[rule.family]


def test_grammar_has_280_distinct_functions_matching_independent_truth():
    rules = rw.grammar()
    assert len(rules) == 280
    assert [r.family for r in rules].count("category") == 24
    assert [r.family for r in rules].count("relation") == 64
    domain = torch.cartesian_prod(*[torch.arange(4)] * 8, torch.arange(2))
    a, b, s = domain[:, :4], domain[:, 4:8], domain[:, 8]
    tables = set()
    tensors = rw.rule_tensors(rules)
    for index, rule in enumerate(rules):
        batch = {k: v[index].expand(len(domain), *v.shape[1:]) for k, v in tensors.items()}
        out = rw.outcome_batch(batch, a, b, s)
        tables.add(out.to(torch.uint8).numpy().tobytes())
        rows = torch.randint(len(domain), (64,), generator=torch.Generator().manual_seed(index))
        for r in rows.tolist():
            assert int(out[r]) == scalar_truth(rule, a[r].tolist(), b[r].tolist(), int(s[r]))
            assert rw.outcome(rule, a[r].tolist(), b[r].tolist(), int(s[r])) == int(out[r])
    assert len(tables) == 280


def test_split_is_deterministic_code_and_matches_approved_groups():
    groups = rw.split_groups()
    assert digest(groups) == APPROVED_SPLIT_SHA256
    assert [len(groups["category"][k]) for k in ("train", "validation", "test")] == [8, 2, 2]
    assert [len(groups["predicate"][k]) for k in ("train", "validation", "test")] == [44, 10, 10]
    rules = rw.split_rules()
    counts = {k: len(v) for k, v in rules.items()}
    assert counts == {"train": 16 + 44 + 132, "validation": 4 + 10 + 30, "test": 4 + 10 + 30}
    seen = {k: {r.group() for r in v} for k, v in rules.items()}
    assert not (seen["train"] & seen["test"]) and not (seen["validation"] & seen["test"])
    train = rules["train"]
    assert {r.j for r in train} == set(range(4)) and {r.delta for r in train if r.family != "category"} == set(range(4))
    assert {r.family for r in train} == {"category", "relation", "open", "close", "toggle"}


def test_render_is_quantized_static_and_only_lamps_change():
    g = torch.Generator().manual_seed(3)
    scenes = rw.sample_scenes(g, torch.tensor([[0, 5], [7, 9]]))
    frames = [rw.render(scenes, torch.tensor([[x, y]] * 2)) for x in (0, 1) for y in (0, 1)]
    for rgb, entity in frames:
        assert rgb.shape == (2, 3, 64, 64) and entity.shape == (2, 64, 64)
        assert torch.equal(rgb, (rgb * 255).round() / 255)
        assert set(entity.unique().tolist()) == set(range(7))
    (a, ea), (b, eb) = frames[0], frames[1]
    assert torch.equal(ea, eb)
    changed = (a != b).any(1)
    assert changed.any() and (eb[changed] == 2).all()  # right lamp only
    assert not torch.equal(rw.render(rw.sample_scenes(g, torch.tensor([[0, 5]])), torch.zeros(1, 2, dtype=torch.long))[0], a[:1])


def world(rules, lamps=(0, 0), attrs=None, seed=5):
    scenes = rw.sample_scenes(torch.Generator().manual_seed(seed), torch.tensor([[1, 2]]))
    if attrs is not None:
        scenes = rw.Scenes(scenes.kind, scenes.machine_xy, torch.tensor([attrs]), scenes.object_xy)
    return rw.RuleWorld(scenes, rules, torch.tensor(lamps), rw.TaskContract())


def test_executor_targets_one_machine_and_returns_receipts():
    rules = (rw.Rule("toggle", 0, 0, 0), rw.Rule("category", 0, values=(0, 1)))
    attrs = [[0, 0, 0, 0], [0, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]]
    env = world(rules, (0, 0), attrs)
    m, o = env.scenes.machine_xy[0], env.scenes.object_xy[0]
    receipt = env.press(rw.ActionRecord(tuple(m[0].tolist()), tuple(o[0].tolist()), tuple(o[1].tolist())))
    # toggle: (a0 + 0) % 4 == b0 -> 0 == 0 -> flips left lamp; right untouched.
    assert receipt.status == "ok" and env.lamps.tolist() == [1, 0]
    assert env.press(rw.ActionRecord((1.0, 1.0), tuple(o[0].tolist()), tuple(o[1].tolist()))).status == "miss"
    assert env.lamps.tolist() == [1, 0]
    assert env.presses == 2  # miss is executed and costs
    third = env.press(rw.ActionRecord(tuple(m[1].tolist()), tuple(o[2].tolist()), tuple(o[3].tolist())))
    assert third.status == "budget_exceeded" and env.presses == 2 and env.lamps.tolist() == [1, 0]
    env2 = world(rules, (0, 0), attrs)
    same = env2.press(rw.ActionRecord(tuple(m[1].tolist()), tuple(o[0].tolist()), tuple(o[0].tolist())))
    assert same.status == "same_object" and env2.lamps.tolist() == [0, 0]


def test_utility_is_one_function_and_counts_terminal_once():
    c = rw.TaskContract()
    assert c.utility("success", 0) == 1.0
    assert c.utility("success", 2) == pytest.approx(0.9)
    assert c.utility("false_stop", 1) == pytest.approx(-1.05)
    assert c.utility("abstain", 0) == -0.25
    assert c.expected_stop(0.8, 1) == pytest.approx(0.8 * c.utility("success", 1) + 0.2 * c.utility("false_stop", 1))


def brute_force(rules, attrs, lamps, goal, budget):
    best = None
    actions = [(m, i, j) for m in (0, 1) for i in range(4) for j in range(4) if i != j]
    for depth in range(budget + 1):
        for seq in itertools.product(actions, repeat=depth):
            state = list(lamps)
            for m, i, j in seq:
                state[m] = scalar_truth(rules[m], attrs[i], attrs[j], state[m])
            if tuple(state) == tuple(goal):
                return depth
    return best


def test_bfs_strata_match_brute_force_and_sampler_fails_explicitly():
    g = torch.Generator().manual_seed(11)
    rules_all = rw.grammar()
    for trial in range(40):
        rules = tuple(rules_all[int(i)] for i in torch.randint(280, (2,), generator=g))
        attrs = torch.randint(4, (4, 4), generator=g).tolist()
        lamps = torch.randint(2, (2,), generator=g).tolist()
        for goal in itertools.product((0, 1), repeat=2):
            assert rw.min_presses(rules, attrs, lamps, goal, 2) == brute_force(rules, attrs, lamps, goal, 2)
    # With these objects 'open' never matches: no press changes a lamp, so no
    # 'reach2' goal exists and the finite sampler must fail explicitly.
    rules = (rw.Rule("open", 0, 0, 1), rw.Rule("open", 0, 0, 1))
    attrs = [[0, 0, 0, 0]] * 4
    assert rw.goal_for_stratum(rules, attrs, [0, 0], "reach2", g) is None
    with pytest.raises(rw.StratumUnavailable):
        rw.sample_task(g, torch.tensor([1, 2]), rules, "reach2", cap=3, attrs=attrs)
    task = rw.sample_task(g, torch.tensor([1, 2]), (rw.Rule("open", 0, 0, 0),) * 2, "unreachable", cap=200)
    assert rw.min_presses(task.rules, task.scene.attrs[0].tolist(), task.lamps.tolist(), task.goal, 2) is None


def test_floors_enumerated_for_r1_domain():
    floors = rw.floors()
    assert set(floors) == {"category", "relation", "open", "close", "toggle"}
    for family, f in floors.items():
        assert 0.0 <= f["copy_s"] <= 1 and 0.0 <= f["operator_only"] <= 1
        assert f["floor"] == max(f["copy_s"], f["operator_only"], f["majority_label"], f["constant"])
    assert floors["category"]["copy_s"] == pytest.approx(0.5)


def test_episode_support_and_queries_are_symbolically_disjoint():
    g = torch.Generator().manual_seed(2)
    batch = rw.sample_episodes(g, rw.split_rules()["train"], rw.KIND_SPLIT["train"], episodes=3, support=(8, 16), queries=6, p_empty=0.0)
    for e in range(3):
        s = batch.support.episode == e
        q = batch.query.episode == e
        key = lambda t, mask: {
            (tuple(batch.scenes.attrs[t.scene[i], t.a[i]].tolist()), tuple(batch.scenes.attrs[t.scene[i], t.b[i]].tolist()), int(t.pre[i, t.machine[i]]))
            for i in mask.nonzero().flatten().tolist()
        }
        assert not key(batch.support, s) & key(batch.query, q)
        assert set(batch.scenes.kind[batch.support.scene[s], batch.support.machine[s]].tolist()) == {int(batch.target_kind[e])}
    assert (batch.support.post.gather(1, batch.support.machine[:, None]).squeeze(1) == batch.support.outcome).all()
    assert batch.chain.step.max() == 1


def test_action_records_reject_malformed_coordinates_at_construction():
    with pytest.raises(ValueError):
        rw.ActionRecord((float("inf"), 1.0), (1.0, 1.0), (2.0, 2.0))
    with pytest.raises(ValueError):
        rw.ActionRecord((1.0,), (1.0, 1.0), (2.0, 2.0))
    rw.ActionRecord((-5.0, 90.0), (1.0, 1.0), (2.0, 2.0))  # finite off-screen: a charged miss


def test_life_sampler_rejects_unsatisfiable_disjoint_queries(monkeypatch):
    from pathwm.evaluation import rule_world as ev

    original = rw.sample_scenes

    def identical(generator, kinds, attrs=None):
        return original(generator, kinds, attrs=torch.zeros(1, 4, 4, dtype=torch.long))

    monkeypatch.setattr(rw, "sample_scenes", identical)
    with pytest.raises(ValueError, match="disjoint"):
        ev.sample_life(torch.Generator().manual_seed(1), "validation", n_support=128, queries=4, goals=0, distract=8, counter=8)
