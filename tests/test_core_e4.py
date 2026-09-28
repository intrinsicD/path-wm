"""E4-A core form (docs/core-design.md): untied blocks per inner round, width 128, R1 form unchanged."""

import sys

import pytest
import torch

from pathwm.data import rule_world as rw
from pathwm.evaluation import rule_world as ev
from pathwm.models import latent_core as lc


def params(modules):
    return sum(p.numel() for m in modules for p in m.parameters())


def test_block_parameter_counts_match_the_e4_decision():
    core = lc.LatentCore(width=128, heads=4, loops=2, code_tokens=4, key_width=32, blocks=2)
    assert params([core.block]) == 264_832
    assert params([core.block, *core.more_blocks]) == 529_664


def test_one_block_default_is_bit_identical_and_keeps_checkpoint_keys():
    torch.manual_seed(0)
    old = lc.LatentCore(width=16, heads=2, loops=2, code_tokens=2, key_width=8)
    torch.manual_seed(0)
    new = lc.LatentCore(width=16, heads=2, loops=2, code_tokens=2, key_width=8, blocks=1)
    assert old.state_dict().keys() == new.state_dict().keys()
    assert not any(k.startswith("more_blocks") for k in new.state_dict())
    assert all(torch.equal(v, new.state_dict()[k]) for k, v in old.state_dict().items())


def test_extra_blocks_leave_the_initialisation_of_existing_parameters_unchanged():
    torch.manual_seed(0)
    one = lc.LatentCore(width=16, heads=2, loops=2, code_tokens=2, key_width=8, blocks=1)
    torch.manual_seed(0)
    two = lc.LatentCore(width=16, heads=2, loops=2, code_tokens=2, key_width=8, blocks=2)
    shared = one.state_dict()
    assert all(torch.equal(v, two.state_dict()[k]) for k, v in shared.items())


@pytest.mark.parametrize("reader", ["code", "evidence"])
def test_every_inner_round_applies_all_blocks_in_order(reader):
    torch.manual_seed(1)
    core = lc.LatentCore(width=16, heads=2, loops=2, code_tokens=2, key_width=8, blocks=2).eval()
    calls = []
    core.block.register_forward_hook(lambda *a: calls.append(0))
    core.more_blocks[0].register_forward_hook(lambda *a: calls.append(1))
    batch = rw.sample_episodes(torch.Generator().manual_seed(0), rw.relation_ladder(4), rw.KIND_SPLIT["train"],
                               episodes=2, support=(4,), queries=2, p_empty=0.0)
    torch.manual_seed(0)
    from pathwm.models.slots import SymbolicSlots
    tokens = ev.symbolic_episode_tokens(SymbolicSlots(16), batch, "cpu")
    with torch.no_grad():
        z, valid = lc.read_context(core, tokens.support, tokens.episodes, reader)
        expected_induce = [0, 1, 0, 1] if reader == "code" else []
        assert calls == expected_induce
        calls.clear()
        q = tokens.query
        core.apply(q.m_pre, q.a, q.b, z[q.episode], None, None if valid is None else valid[q.episode])
    assert calls == [0, 1, 0, 1]


def test_two_block_core_trains_all_blocks():
    torch.manual_seed(2)
    core = lc.LatentCore(width=16, heads=2, loops=2, code_tokens=2, key_width=8, blocks=2).train()
    batch = rw.sample_episodes(torch.Generator().manual_seed(0), rw.relation_ladder(4), rw.KIND_SPLIT["train"],
                               episodes=3, support=(4,), queries=4, p_empty=0.0)
    from pathwm.models.slots import SymbolicSlots
    tokens = ev.symbolic_episode_tokens(SymbolicSlots(16), batch, "cpu")
    for reader in ("code", "evidence"):
        core.zero_grad()
        loss, _ = lc.episode_loss(core, tokens, torch.ones(16), key_weight=0.0, auxiliary_weight=0.0, reader=reader)
        loss.backward()
        assert all(p.grad is not None and p.grad.abs().sum() > 0
                   for m in (core.block, core.more_blocks[0]) for p in m.parameters() if p.dim() > 1), reader


def test_recipe_profiles_full_is_e4_and_r1_keeps_the_former_form():
    import experiments.latent_agent as recipe

    full, r1 = recipe.SIZES["full"], recipe.SIZES["r1"]
    assert (full["core_width"], full["core_blocks"], full["loops"], full["heads"]) == (128, 2, 2, 4)
    assert (r1["core_width"], r1["core_blocks"], r1["width"], r1["loops"]) == (64, 1, 64, 2)
    assert full["width"] == r1["width"] == 64  # perception unchanged
    model = recipe.RuleModel(full, symbolic=True)
    assert params([model.core.block, *model.core.more_blocks]) == 529_664
    assert model.perception.width == 128 and model.variance.shape == (128,)
    old = recipe.RuleModel(r1, symbolic=True)
    assert not len(old.core.more_blocks) and old.core.width == 64


def test_pixel_core_with_a_wider_core_is_refused_until_the_projection_exists():
    import experiments.latent_agent as recipe

    with pytest.raises(ValueError, match="projection"):
        recipe.RuleModel(recipe.SIZES["full"], symbolic=False)
    recipe.RuleModel(recipe.SIZES["r1"], symbolic=False)  # R1 pixel path unchanged


def test_core_overrides_are_recorded_and_resume_exactly(monkeypatch, tmp_path):
    import json
    import experiments.latent_agent as recipe

    def run(*arguments):
        monkeypatch.setattr(sys, "argv", ["latent_agent", *arguments])
        recipe.main()

    common = ["--stage", "symbolic", "--size", "check", "--device", "cpu", "--updates", "4", "--train-rules", "4",
              "--core-width", "24", "--core-blocks", "2", "--core-loops", "3"]
    run(*common, "--stop-after", "2", "--output", str(tmp_path / "paused"))
    run("--stage", "symbolic", "--resume", str(tmp_path / "paused"))
    run(*common, "--output", str(tmp_path / "straight"))
    a = torch.load(tmp_path / "paused" / "last.pt", weights_only=True)
    b = torch.load(tmp_path / "straight" / "last.pt", weights_only=True)
    assert a["step"] == b["step"] == 4
    assert all(torch.equal(a["model"][k], b["model"][k]) for k in a["model"])
    sizes = json.loads((tmp_path / "straight" / "run.json").read_text())["identity"]["settings"]["sizes"]
    assert (sizes["core_width"], sizes["core_blocks"], sizes["loops"]) == (24, 2, 3)
    assert a["model"]["core.more_blocks.0.cross.in_proj_weight"].shape == (72, 24)


def test_episodes_override_sets_training_batch_size_and_resumes(monkeypatch, tmp_path):
    import json
    import experiments.latent_agent as recipe

    sizes = []
    original = rw.sample_episodes

    def spy(*args, **kwargs):
        sizes.append(kwargs.get("episodes"))
        return original(*args, **kwargs)

    monkeypatch.setattr(rw, "sample_episodes", spy)
    monkeypatch.setattr(sys, "argv", ["latent_agent", "--stage", "symbolic", "--size", "check", "--device", "cpu",
                                      "--updates", "2", "--train-rules", "4", "--episodes", "10",
                                      "--output", str(tmp_path / "E")])
    recipe.main()
    assert sizes.count(10) >= 2  # both training updates
    settings = json.loads((tmp_path / "E" / "run.json").read_text())["identity"]["settings"]
    assert settings["episodes"] == 10 and settings["sizes"]["episodes"] == 10


def test_holdout_ladder_rules_are_never_trained_and_form_the_transfer_pool(monkeypatch, tmp_path):
    import json
    import experiments.latent_agent as recipe

    drawn = []
    original = rw.sample_episodes

    def spy(generator, rules, *args, **kwargs):
        drawn.append(tuple(rules))
        return original(generator, rules, *args, **kwargs)

    monkeypatch.setattr(rw, "sample_episodes", spy)
    ladder = rw.relation_ladder(4)
    monkeypatch.setattr(sys, "argv", ["latent_agent", "--stage", "symbolic", "--size", "check", "--device", "cpu",
                                      "--updates", "2", "--train-rules", "4", "--rule-repeats", "7", "1", "1", "1",
                                      "--holdout-ladder-rules", "3", "--output", str(tmp_path / "H")])
    recipe.main()
    held = ladder[3]
    trained = [p for p in drawn if held not in p]
    assert (ladder[0],) * 7 + ladder[1:3] in trained  # skewed training pool without the held-out rule
    assert (held,) in drawn  # the transfer population
    settings = json.loads((tmp_path / "H" / "run.json").read_text())["identity"]["settings"]
    assert settings["holdout_rule_keys"] == [held.key()]
    assert settings["train_rule_keys"] == [r.key() for r in ladder[:3]]
    result = json.loads((tmp_path / "H" / "result.json").read_text())
    assert set(result["metrics"]["pool_heldout"]["nu_by_rule"]) <= {held.key()}
    for bad in (["--holdout-ladder-rules", "4"], ["--holdout-ladder-rules", "0", "1", "2", "3"],
                ["--train-families", "relation", "toggle", "--holdout-ladder-rules", "1"]):
        monkeypatch.setattr(sys, "argv", ["latent_agent", "--stage", "symbolic", "--train-rules", "4", *bad,
                                          "--output", str(tmp_path / "x")])
        with pytest.raises(SystemExit):
            recipe.main()
