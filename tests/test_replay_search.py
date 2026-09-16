import copy

import pytest

from pathwm.evaluation.search import POLICIES, replay_search, run_search, select_policy


def history(scores, world="a"):
    rows = []
    for depth in range(1, 5):
        for branch in range(4):
            rows.append(
                dict(
                    branch=branch,
                    depth=depth,
                    score=scores[branch][depth - 1],
                    seconds=1.0,
                )
            )
    return dict(
        world=world, contract={"task": "unit", "branches": 4, "depth": 4}, nodes=rows
    )


def test_prefix_only_replay_and_unsupported_continuations():
    h = history([[-1, -0.8, -0.7, -0.6]] * 4)
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
    h = history(
        [
            [-1, -0.8, -0.7, -0.6],
            [-1, -0.9, -0.9, -0.9],
            [-1, -0.99, -0.99, -0.99],
            [-1, -0.95, -0.94, -0.94],
        ]
    )
    lookup = {(n["branch"], n["depth"]): n for n in h["nodes"]}
    for policy in POLICIES:
        seen = []

        def execute(branch, depth):
            assert depth == 1 or (branch, depth - 1) in seen
            if depth == 1:
                assert all((b, 1) in seen for b in range(branch))
            seen.append((branch, depth))
            return lookup[branch, depth]

        live = run_search(policy, execute)
        replay = replay_search(policy, h)
        assert live == replay
        assert len(seen) <= (8 if policy == "round_robin8" else 12)


def test_flat_branches_stop_and_incumbent_is_included_without_test_metrics():
    h = history([[-1] * 4] * 4)
    r = replay_search("plateau1", h)
    assert r["steps"] == 8 and r["best_score"] == -1
    selection = select_policy([h, history([[-1] * 4] * 4, "b")])
    assert selection["selected"] == "round_robin8"
    assert selection["mean_utility"] >= selection["candidates"][0]["mean_utility"]
    assert selection["candidates"][0]["policy"] == "round_robin12"
    broken = copy.deepcopy(h)
    broken["world"] = "b"
    broken["contract"]["task"] = "changed"
    with pytest.raises(ValueError, match="contract"):
        select_policy([h, broken])
    with pytest.raises(ValueError, match="world"):
        select_policy([h, h])


def test_bad_executor_history_and_policy_fail_closed():
    with pytest.raises(ValueError):
        run_search("unknown", lambda b, d: None)
    for result in (
        dict(branch=2, depth=1, score=0, seconds=1),
        dict(branch=0, depth=1, score=float("nan"), seconds=1),
        dict(branch=0, depth=1, score=0, seconds=-1),
    ):
        with pytest.raises(ValueError):
            run_search("round_robin12", lambda b, d: result)
    h = history([[-1] * 4] * 4)
    h["nodes"].append(h["nodes"][0])
    with pytest.raises(ValueError, match="Duplicate"):
        replay_search("greedy", h)


def test_real_branch_restart_interleaving_and_frozen_core(tmp_path):
    import torch
    from experiments import modality_readout as recipe
    from pathwm.data.understanding import UnderstandingData, synthetic_records

    torch.set_num_threads(2)
    recipe.seed_everything(32)
    model = recipe.Model("native").requires_grad_(False).eval()
    before = recipe.state_hash(model)
    data = UnderstandingData.__new__(UnderstandingData)
    data.records, data.arrays, data.cases = synthetic_records("quick")
    rng = torch.get_rng_state().clone()
    cache = recipe.exploration_cache(model, data, "VID.order", 32, "cpu")
    assert torch.equal(rng, torch.get_rng_state())
    assert not (
        set(r["id"] for r in cache["calibration"]["rows"])
        & set(r["id"] for r in cache["test"]["rows"])
    )
    contract = dict(
        branches=4,
        depth=4,
        block_updates=2,
        learning_rates=[0.0003, 0.001, 0.003, 0.01],
        fixtures="unit",
    )

    def advance(root, b, d):
        return recipe.exploration_advance(
            model, cache, root, b, d, seed=32, contract=contract, device="cpu"
        )

    first = advance(tmp_path / "interleaved", 0, 1)
    advance(tmp_path / "interleaved", 1, 1)
    a = advance(tmp_path / "interleaved", 0, 2)
    advance(tmp_path / "sequential", 0, 1)
    b = advance(tmp_path / "sequential", 0, 2)
    assert a["score"] == b["score"] and a["model_sha256"] == b["model_sha256"]
    assert a["parent"] == first["sha256"]
    sa, sb = [torch.load(x["checkpoint"], weights_only=True) for x in (a, b)]

    def equal(x, y):
        if isinstance(x, torch.Tensor):
            assert torch.equal(x, y)
        elif isinstance(x, dict):
            assert x.keys() == y.keys()
            for k in x:
                equal(x[k], y[k])
        elif isinstance(x, (tuple, list)):
            assert len(x) == len(y)
            for xx, yy in zip(x, y):
                equal(xx, yy)
        else:
            assert x == y

    equal(sa, sb)
    assert recipe.state_hash(model) == before
    for k, v in model.state_dict().items():
        if not k.startswith(("outputs.decoders.text.", "outputs.adapters.text.")):
            assert torch.equal(sa["model"][k], v)
    with pytest.raises(ValueError, match="skipped|repeated"):
        advance(tmp_path / "interleaved", 0, 4)
    # Reusing a completed first action must never reset the branch silently.
    with pytest.raises(ValueError, match="frontier"):
        advance(tmp_path / "interleaved", 0, 1)
