"""Learnability ladder (plan §23): nested relation pools and support-use controls on the actual core."""

import dataclasses
import sys

import pytest
import torch

from pathwm.data import rule_world as rw
from pathwm.evaluation import rule_world as ev
from pathwm.models import latent_core as lc
from pathwm.models.slots import SymbolicSlots

WIDTH = 16


def test_relation_ladder_is_nested_structured_and_training_only():
    train = set(rw.split_rules()["train"])
    ladders = {n: rw.relation_ladder(n) for n in rw.LADDER_SIZES}
    assert rw.LADDER_SIZES == (1, 4, 16, 44)
    for n, rules in ladders.items():
        assert len(rules) == len(set(rules)) == n
        assert all(r.family == "relation" and r in train for r in rules)
    assert ladders[1] == (rw.Rule("relation", 0, 1, 0),)
    assert {(r.j, r.k) for r in ladders[4]} == {(0, 1)} and {r.delta for r in ladders[4]} == {0, 1, 2, 3}
    assert {(r.j, r.k) for r in ladders[16]} == {(0, 1), (1, 2), (2, 1), (3, 1)}
    assert set(ladders[1]) < set(ladders[4]) < set(ladders[16]) < set(ladders[44])
    with pytest.raises(ValueError):
        rw.relation_ladder(8)


def _batch(rules, episodes=6, support=(8,), seed=0):
    return rw.sample_episodes(torch.Generator().manual_seed(seed), rules, rw.KIND_SPLIT["train"],
                              episodes=episodes, support=support, queries=4, p_empty=0.0)


def test_pool_episodes_draw_only_pool_rules():
    pool = rw.relation_ladder(4)
    batch = _batch(pool, episodes=40)
    assert set(batch.rules) <= set(pool) and len(set(batch.rules)) > 1


def _tokens(batch):
    torch.manual_seed(0)
    symbolic = SymbolicSlots(WIDTH)
    return ev.symbolic_episode_tokens(symbolic, batch, "cpu")


def _core():
    torch.manual_seed(1)
    return lc.LatentCore(width=WIDTH, heads=2, loops=2, code_tokens=2, key_width=8).eval()


def test_empty_arm_is_induction_without_valid_evidence():
    batch = _batch(rw.relation_ladder(4))
    tokens, core = _tokens(batch), _core()
    with torch.no_grad():
        logit = ev.control_logits(core, tokens, "empty", batch.rules)
        empty = core.induce(torch.zeros(tokens.episodes, 0, WIDTH), torch.zeros(tokens.episodes, 0, dtype=torch.bool))
        q = tokens.query
        expected, _ = core.apply(q.m_pre, q.a, q.b, empty[q.episode])
    assert torch.allclose(logit, expected, atol=1e-6)


def test_permuted_arm_changes_only_support_outcomes_within_each_episode():
    batch = _batch(rw.relation_ladder(16), episodes=4, support=(16,))
    tokens = _tokens(batch)
    permuted = ev.permute_support_outcomes(tokens.support, torch.Generator().manual_seed(3))
    s = tokens.support
    for name in ("m_pre", "a", "b", "episode"):
        assert torch.equal(getattr(permuted, name), getattr(s, name)), name
    assert not torch.equal(permuted.m_post, s.m_post)
    for e in range(tokens.episodes):
        rows = s.episode == e
        before = sorted(map(tuple, s.m_post[rows].tolist()))
        after = sorted(map(tuple, permuted.m_post[rows].tolist()))
        assert before == after
        assert sorted(s.outcome[rows].tolist()) == sorted(permuted.outcome[rows].tolist())


def test_swapped_arm_uses_the_code_of_an_episode_with_another_rule():
    batch = _batch(rw.relation_ladder(4), episodes=8)
    source = ev.swap_sources(batch.rules)
    assert all(batch.rules[i] != batch.rules[j] for i, j in enumerate(source))
    tokens, core = _tokens(batch), _core()
    with torch.no_grad():
        z = lc.codes_for(core, tokens.support, tokens.episodes)
        q = tokens.query
        expected, _ = core.apply(q.m_pre, q.a, q.b, z[torch.tensor(source)][q.episode])
        assert torch.allclose(ev.control_logits(core, tokens, "swapped", batch.rules), expected, atol=1e-6)
    assert ev.swap_sources(rw.relation_ladder(1) * 3) is None


def test_control_logits_ignore_query_labels():
    batch = _batch(rw.relation_ladder(4))
    tokens, core = _tokens(batch), _core()
    q = tokens.query
    poisoned = dataclasses.replace(tokens, query=dataclasses.replace(q, outcome=1 - q.outcome, m_post=-q.m_post))
    with torch.no_grad():
        for arm in ("full", "empty", "swapped", "permuted"):
            assert torch.equal(ev.control_logits(core, tokens, arm, batch.rules, seed=5),
                               ev.control_logits(core, poisoned, arm, batch.rules, seed=5)), arm


def test_train_rules_is_limited_to_the_symbolic_induction_stage(monkeypatch, tmp_path):
    import experiments.latent_agent as recipe

    for argv in (["--stage", "core", "--perception", str(tmp_path), "--train-rules", "4"],
                 ["--stage", "symbolic", "--train-rules", "5"],
                 ["--stage", "symbolic", "--train-rules", "4", "--oracle-curriculum", "2", "--updates", "4"]):
        monkeypatch.setattr(sys, "argv", ["latent_agent", *argv, "--output", str(tmp_path / "x")])
        with pytest.raises(SystemExit):
            recipe.main()


def test_symbolic_ladder_run_records_pool_and_control_arms(monkeypatch, tmp_path):
    import json
    import experiments.latent_agent as recipe

    monkeypatch.setattr(sys, "argv", ["latent_agent", "--stage", "symbolic", "--size", "check", "--device", "cpu",
                                      "--updates", "2", "--train-rules", "4", "--output", str(tmp_path / "L4")])
    recipe.main()
    run = json.loads((tmp_path / "L4" / "run.json").read_text())["identity"]["settings"]
    assert run["train_rules"] == 4  # restored as the CLI value on resume
    assert run["train_rule_keys"] == [r.key() for r in rw.relation_ladder(4)]
    result = json.loads((tmp_path / "L4" / "result.json").read_text())
    pool = result["metrics"]["pool"]
    assert set(pool["nu"]) == {"full", "empty", "swapped", "permuted"}
    assert set(result["screen"]) >= {"passed", "criteria"}
    assert (tmp_path / "L4" / "report.html").exists()
    rows = [json.loads(line) for line in (tmp_path / "L4" / "metrics.jsonl").read_text().splitlines()]
    pool_rows = [r for r in rows if r.get("split") == "pool"]
    assert pool_rows and all("nu_full_relation" in r and "nu_empty_relation" in r for r in pool_rows)


def test_ladder_pause_resume_matches_uninterrupted(monkeypatch, tmp_path):
    import experiments.latent_agent as recipe

    def run(*arguments):
        monkeypatch.setattr(sys, "argv", ["latent_agent", *arguments])
        recipe.main()

    common = ["--stage", "symbolic", "--size", "check", "--device", "cpu", "--updates", "4", "--train-rules", "16"]
    run(*common, "--stop-after", "2", "--output", str(tmp_path / "paused"))
    run("--stage", "symbolic", "--resume", str(tmp_path / "paused"))
    run(*common, "--output", str(tmp_path / "straight"))
    a = torch.load(tmp_path / "paused" / "last.pt", weights_only=True)
    b = torch.load(tmp_path / "straight" / "last.pt", weights_only=True)
    assert a["step"] == b["step"] == 4
    assert all(torch.equal(a["model"][k], b["model"][k]) for k in a["model"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_control_arms_run_on_cuda_like_cpu():
    batch = _batch(rw.relation_ladder(4))
    tokens, core = _tokens(batch), _core()
    cuda_tokens = ev.symbolic_episode_tokens(copy_to(SymbolicSlots, "cuda"), batch, "cuda")
    cuda_core = _core().to("cuda")
    with torch.no_grad():
        for arm in ev.CONTROL_ARMS:
            cpu = ev.control_logits(core, tokens, arm, batch.rules, seed=2)
            gpu = ev.control_logits(cuda_core, cuda_tokens, arm, batch.rules, seed=2)
            assert torch.allclose(cpu, gpu.cpu(), atol=1e-4), arm


def copy_to(cls, device):
    torch.manual_seed(0)
    return cls(WIDTH).to(device)
