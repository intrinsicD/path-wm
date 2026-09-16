import copy

import pytest

from pathwm.evaluation.search import POLICIES, replay_search, run_search, select_policy


def history(scores, world="a"):
    rows = []
    for depth in range(1, 5):
        for branch in range(4):
            rows.append(dict(branch=branch, depth=depth,
                             score=scores[branch][depth-1], seconds=1.0))
    return dict(world=world, contract={"task": "unit", "branches": 4, "depth": 4}, nodes=rows)


def test_prefix_only_replay_and_unsupported_continuations():
    h = history([[-1, -.8, -.7, -.6]] * 4)
    a = replay_search("round_robin12", h)
    changed = copy.deepcopy(h)
    for node in changed["nodes"][4:]:
        node["score"] = 50
    b = replay_search("round_robin12", changed)
    assert a["decisions"][:4] == b["decisions"][:4]
    assert all(len(d["visible"]) == i for i, d in enumerate(a["decisions"]))
    assert a["steps"] == 12 and a["status"] == "complete"
    h["nodes"] = h["nodes"][:4]
    missing = replay_search("round_robin12", h)
    assert missing["status"] == "unsupported" and missing["utility"] is None
    assert missing["steps"] == 4


def test_live_and_replay_share_actions_and_never_jump_to_a_hidden_descendant():
    h = history([[-1, -.8, -.7, -.6], [-1, -.9, -.9, -.9],
                 [-1, -.99, -.99, -.99], [-1, -.95, -.94, -.94]])
    lookup = {(n["branch"], n["depth"]): n for n in h["nodes"]}
    for policy in POLICIES:
        seen = []
        def execute(branch, depth):
            assert depth == 1 or (branch, depth-1) in seen
            if depth == 1:
                assert all((b, 1) in seen for b in range(branch))
            seen.append((branch, depth))
            return lookup[branch, depth]
        live = run_search(policy, execute)
        replay = replay_search(policy, h)
        assert live == replay
        assert len(seen) <= (8 if policy == "round_robin8" else 12)


def test_flat_branches_stop_and_incumbent_is_included_without_test_metrics():
    h = history([[-1]*4]*4)
    r = replay_search("plateau1", h)
    assert r["steps"] == 8 and r["best_score"] == -1
    selection = select_policy([h, history([[-1]*4]*4, "b")])
    assert selection["selected"] == "round_robin8"
    assert selection["mean_utility"] >= selection["candidates"][0]["mean_utility"]
    assert selection["candidates"][0]["policy"] == "round_robin12"
    broken = copy.deepcopy(h); broken["contract"]["task"] = "changed"
    with pytest.raises(ValueError, match="contract"):
        select_policy([h, broken])
    with pytest.raises(ValueError, match="world"):
        select_policy([h,h])


def test_bad_executor_history_and_policy_fail_closed():
    with pytest.raises(ValueError):
        run_search("unknown", lambda b,d: None)
    for result in (dict(branch=2,depth=1,score=0,seconds=1),
                   dict(branch=0,depth=1,score=float("nan"),seconds=1),
                   dict(branch=0,depth=1,score=0,seconds=-1)):
        with pytest.raises(ValueError):
            run_search("round_robin12", lambda b,d: result)
    h = history([[-1]*4]*4)
    h["nodes"].append(h["nodes"][0])
    with pytest.raises(ValueError, match="Duplicate"):
        replay_search("greedy", h)
