"""Confusable same-palette twins in the existing S1 paired identity batches.

Actual native full configuration (width64, seven slots, three iterations, RGB64) and the
J perception+key checkpoint. Tiny CPU runs are software checks of the training data
option, not training results. The legacy baseline is the frozen recipe copy retained by a
completed run (byte-identical to the pre-change recipe), imported from its file.
Constructed held-out-like textures in one test are labelled FAULT INJECTION.
"""

import importlib.util
import json
import math
from pathlib import Path
import sys

import pytest
import torch
from torch import nn

from experiments import latent_agent as la
from experiments import unified_session as us
from pathwm.data import rule_world as rw
from pathwm.io import load_component, source_record, state_hash
from pathwm.models.latent_core import key_head
from pathwm.models.slots import SlotPerception

ROOT = Path(__file__).resolve().parents[1]
J = ROOT / "runs/latent_agent_r1/identity_joint_20260923"
LEGACY = ROOT / "runs/real_visual_joint_repair_3501_u3000_v1/recipe.py"
TRAIN = rw.KIND_SPLIT["train"]


@pytest.fixture(scope="module", autouse=True)
def needs_artifacts():
    if not (J / "last.pt").exists() or not LEGACY.exists():
        pytest.skip("J checkpoint or frozen legacy recipe not present")


@pytest.fixture(scope="module")
def legacy():
    spec = importlib.util.spec_from_file_location("legacy_latent_agent", LEGACY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def streams(seed=41):
    return (torch.Generator().manual_seed(seed), torch.Generator().manual_seed(seed + 1),
            torch.Generator().manual_seed(seed + 2))


def batch(fn, count=16, rate=None, randomize=1.0, seed=41):
    g, aug, pair = streams(seed)
    kwargs = {} if rate is None else dict(twins=rate)
    out = fn(g, TRAIN, count, "cpu", randomize=randomize, augmentation=aug, pairing=pair, **kwargs)
    return out, (g.get_state(), aug.get_state(), pair.get_state())


def body_pixels(rgb, xy):
    """Machine body texture pixels (panel and lamp disc excluded) in body coordinates."""
    x, y = int(xy[0]), int(xy[1])
    crop = rgb[:, y - 8:y + 9, x - 11:x + 12]
    dy, dx = torch.meshgrid(torch.arange(-8, 9), torch.arange(-11, 12), indexing="ij")
    keep = ~(((dx.abs() <= 4) & (dy >= -8) & (dy <= -1)) | (dx.square() + (dy + 4.5).square() <= 3.2 ** 2))
    return crop[:, keep]


@pytest.mark.parametrize("randomize", [1.0, 0.0])
def test_default_is_byte_identical_to_the_frozen_legacy_recipe(legacy, randomize):
    new, new_states = batch(la.paired_perception_batch, randomize=randomize)
    zero, zero_states = batch(la.paired_perception_batch, rate=0.0, randomize=randomize)
    old, old_states = batch(legacy.paired_perception_batch, randomize=randomize)
    for a, b, c in zip(new[:4], zero[:4], old[:4]):
        if isinstance(a, torch.Tensor):
            assert torch.equal(a, c) and torch.equal(b, c)
        else:  # scenes
            for x, y, z in zip(vars(a).values(), vars(b).values(), vars(c).values()):
                assert torch.equal(x, z) and torch.equal(y, z)
    for x, y, z in zip(vars(new[5][0]).values(), vars(zero[5][0]).values(), vars(old[5][0]).values()):
        assert torch.equal(x, z) and torch.equal(y, z)
    assert torch.equal(new[5][1], old[5][1]) and torch.equal(zero[5][1], old[5][1])
    assert new[4] == old[4] and zero[4] == old[4]  # stats: no twin keys at rate 0
    for s, t, u in zip(new_states, zero_states, old_states):  # identical RNG consumption
        assert torch.equal(s, u) and torch.equal(t, u)


def test_twins_share_palette_and_period_differ_in_equal_proportion_pattern_and_pixels():
    (scenes, lamps, rgb, entity, stats, (tex, source)), _ = batch(la.paired_perception_batch, rate=1.0)
    (base_scenes, base_lamps, _, _, _, (base_tex, base_source)), _ = batch(la.paired_perception_batch, rate=0.0)
    n = 16
    # Base scenes, lamps, pairing and machine-0 bodies are untouched by the option.
    for x, y in zip(vars(scenes).values(), vars(base_scenes).values()):
        assert torch.equal(x, y)
    assert torch.equal(lamps, base_lamps) and torch.equal(source, base_source)
    assert torch.equal(tex.colors[:n, 0], base_tex.colors[:n, 0])
    assert stats["twin_attempts"] == n and stats["twins"] + stats["twin_heldout_rejections"] == n
    assert stats["twin_rate"] == stats["twins"] / n and stats["twins"] > 0
    sampler = rw.TextureSampler()
    for b in range(n):
        twin = torch.equal(tex.colors[b, 1], tex.colors[b, 0]) and int(tex.period[b, 1]) == int(tex.period[b, 0])
        if not twin:  # a rejected proposal keeps its original texture
            assert torch.equal(tex.colors[b, 1], base_tex.colors[b, 1])
            continue
        p0, p1 = int(tex.pattern[b, 0]), int(tex.pattern[b, 1])
        assert p1 in (0, 1, 2, 3) and p1 != p0
        assert not sampler.heldout_like(tex.colors[b, 1], p1, int(tex.period[b, 1]))
        a = body_pixels(rgb[b], scenes.machine_xy[b, 0])
        c = body_pixels(rgb[b], scenes.machine_xy[b, 1])
        assert not torch.equal(a, c)  # same palette, visibly different pattern


def test_every_proposal_is_heldout_checked_and_rejections_are_counted():
    # FAULT INJECTION: machine-0 bodies copy a held-out kind's colours and period with a
    # different equal-proportion pattern, so a proposal of that kind's pattern must be rejected.
    held = rw.KIND_SPLIT["validation"][0]
    pattern, period = int(rw.KIND_PATTERN[held]), int(rw.KIND_PERIOD[held])
    assert pattern in (0, 1, 2, 3)
    count = 24
    colors = rw.KIND_COLORS[held].expand(count, 2, 2, 3).clone()
    other = (pattern + 1) % 4
    tex = rw.Textures(colors, torch.tensor([[other, other]] * count), torch.tensor([[period, period]] * count))
    out, stats = la.confusable_twins(tex, 1.0, torch.Generator().manual_seed(5))
    rejected = [b for b in range(count) if int(out.pattern[b, 1]) == other and torch.equal(out.colors[b, 1], colors[b, 1])]
    assert stats["twin_attempts"] == count and stats["twin_heldout_rejections"] == len(rejected) > 0
    assert stats["twins"] == count - len(rejected)
    assert all(int(out.pattern[b, 1]) != pattern for b in range(count))


def test_twins_are_negatives_across_and_within_frames_and_positives_are_unchanged():
    (scenes, _, rgb, _, _, (tex, source)), _ = batch(la.paired_perception_batch, rate=1.0)
    n = 32
    target = torch.argsort(source.flatten())
    false_negative = la.texture_matches(tex, n)
    twins = [b for b in range(16) if torch.equal(tex.colors[b, 1], tex.colors[b, 0])
             and int(tex.pattern[b, 1]) != int(tex.pattern[b, 0])]
    assert twins
    for b in twins:
        a0, a1 = 2 * b, 2 * b + 1
        assert not false_negative[a0, target[a1]] and not false_negative[a1, target[a0]]
    keys = torch.nn.functional.normalize(torch.randn(2 * n, 8), dim=-1)
    fn = false_negative.clone(); fn[torch.arange(n), target] = False
    same, different, within = la.pair_cosines(keys, tex, target, fn)
    assert len(same) == n and len(within) == 32  # twin frames count as within-frame negatives


def test_full_native_joint_identity_gradient_on_a_twin_batch():
    perception = SlotPerception(64, 7, 3, decoder_width=32)
    key = key_head(64, 32)
    load_component(perception, J / "last.pt", "perception")
    load_component(key, J / "last.pt", "key")
    (scenes, _, rgb, _, stats, (tex, source)), _ = batch(la.paired_perception_batch, count=8, rate=1.0)
    assert stats["twins"] > 0
    loss, metrics = la.identity_loss(key, perception(rgb), scenes, tex, source, detached=False,
                                     margin=(0.95, 0.5, 100.0))
    assert torch.isfinite(loss)
    loss.backward()
    for module in (key, perception):
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads) and any(g.abs().sum() > 0 for g in grads)


def run_main(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["latent_agent.py", *map(str, argv)])
    torch.set_num_threads(2)
    return la.main()


def joint_args(output, *extra):
    return ["--stage", "perception", "--output", output, "--size", "full", "--device", "cpu",
            "--updates", "2", "--seed", "3501", "--init-perception", J, "--init-key", "--identity", "joint",
            "--texture-randomization", "1.0", "--identity-margin-positive", "0.95",
            "--identity-margin-negative", "0.5", "--identity-margin-weight", "100",
            "--confusable-twins", "0.25", *extra]


def component(path, name, module):
    load_component(module, Path(path) / "last.pt", name)
    return state_hash(module)


def test_joint_run_records_twins_and_resumes_exactly(tmp_path, monkeypatch):
    whole, paused = tmp_path / "whole", tmp_path / "paused"
    run_main(monkeypatch, *joint_args(whole))
    settings = json.loads((whole / "run.json").read_text())["identity"]["settings"]
    assert settings["confusable_twins"] == 0.25
    rows = [json.loads(line) for line in (whole / "metrics.jsonl").read_text().splitlines()]
    train = [r for r in rows if r["split"] == "train"]
    assert train and all({"twin_attempts", "twins", "twin_heldout_rejections", "twin_rate"} <= set(r) for r in train)
    run_main(monkeypatch, *joint_args(paused, "--stop-after", "1"))
    run_main(monkeypatch, "--stage", "perception", "--resume", paused, "--device", "cpu")
    fresh = {"perception": lambda: SlotPerception(64, 7, 3, decoder_width=32), "key": lambda: key_head(64, 32)}
    for name, make in fresh.items():
        assert component(paused, name, make()) == component(whole, name, make())


@pytest.mark.parametrize("argv, message", [
    (["--identity", "joint", "--confusable-twins", "nan"], "finite probability"),
    (["--identity", "joint", "--confusable-twins", "inf"], "finite probability"),
    (["--identity", "joint", "--confusable-twins", "-0.1"], "finite probability"),
    (["--identity", "joint", "--confusable-twins", "1.5"], "finite probability"),
    (["--confusable-twins", "0.25"], "requires --identity"),
])
def test_invalid_twin_rates_are_rejected(tmp_path, monkeypatch, capsys, argv, message):
    with pytest.raises(SystemExit):
        run_main(monkeypatch, "--stage", "perception", "--output", tmp_path / "x", "--device", "cpu", *argv)
    assert message in capsys.readouterr().err
    assert not (tmp_path / "x").exists()


def test_twins_are_rejected_outside_the_perception_stage(tmp_path, monkeypatch, capsys):
    with pytest.raises(SystemExit):
        run_main(monkeypatch, "--stage", "core", "--output", tmp_path / "x", "--device", "cpu",
                 "--perception", J, "--confusable-twins", "0.25")
    assert "perception stage only" in capsys.readouterr().err


def test_latent_recipe_is_outside_the_unified_session_manifest_source_hash():
    files = source_record(us.__file__, nn.Module())["files"]
    assert str((ROOT / "experiments/latent_agent.py").resolve()) not in files
    assert str((ROOT / "experiments/unified_session.py").resolve()) in files
