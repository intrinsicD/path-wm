"""Frozen-perception identity-key repair mode of the existing S1 perception stage.

Actual native full configuration (width64, seven slots, three iterations, RGB64) and
the provisional J perception+key checkpoint. Runs are tiny (2 updates, CPU) software
checks of the training mode, not a training result. FAULT INJECTION is labelled.
"""

import json
from pathlib import Path
import sys

import pytest
import torch
from torch.nn import functional as F

from experiments import latent_agent as la
from experiments import unified_session as us
from pathwm.data import rule_world as rw
from pathwm.io import load_component, state_hash
from pathwm.models.latent_core import key_head
from pathwm.models.slots import SlotPerception

J = Path(__file__).resolve().parents[1] / "runs/latent_agent_r1/identity_joint_20260923"
FULL = la.SIZES["full"]
MARGIN = ["--identity-margin-positive", "0.95", "--identity-margin-negative", "0.5",
          "--identity-margin-weight", "1.0"]


@pytest.fixture(scope="module", autouse=True)
def needs_j():
    if not (J / "last.pt").exists():
        pytest.skip("J checkpoint not present")


def run_main(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["latent_agent.py", *map(str, argv)])
    torch.set_num_threads(2)
    return la.main()


def frozen_args(output, *extra):
    return ["--stage", "perception", "--output", output, "--size", "full", "--device", "cpu",
            "--updates", "2", "--seed", "3501", "--init-perception", J, "--init-key",
            "--identity", "detached", "--freeze-perception", *MARGIN, *extra]


def component(path, name, module):
    load_component(module, Path(path) / "last.pt", name)
    return state_hash(module)


def j_states():
    return (component(J, "perception", SlotPerception(FULL["width"], FULL["slots"], FULL["iterations"],
                                                      decoder_width=FULL["decoder_width"])),
            component(J, "key", key_head(FULL["width"], FULL["key_width"])))


@pytest.fixture(scope="module")
def frozen_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("frozen") / "run"
    mp = pytest.MonkeyPatch()
    try:
        run_main(mp, *frozen_args(out))
    finally:
        mp.undo()
    return out


def test_frozen_run_trains_only_the_key_and_keeps_perception_bit_identical(frozen_run):
    perception_j, key_j = j_states()
    width = FULL["width"]
    assert component(frozen_run, "perception", SlotPerception(width, 7, 3, decoder_width=32)) == perception_j
    assert component(frozen_run, "key", key_head(width, FULL["key_width"])) != key_j
    run = json.loads((frozen_run / "run.json").read_text())["identity"]
    settings = run["settings"]
    assert settings["freeze_perception"] and settings["init_key"]
    assert settings["frozen_perception_state_sha256"] == perception_j
    assert settings["init_key_source"]["state_sha256"] == key_j
    assert all(name.startswith("key.") for name in run["trainable"])
    # The parent's recorded randomization is read, not assumed.
    parent = json.loads((J / "run.json").read_text())["identity"]["settings"]["texture_randomization"]
    assert settings["texture_randomization"] == parent
    assert settings["key_monitor"]["seed"] == 3502 and settings["key_monitor"]["pairs"] == 64
    assert settings["key_monitor"]["kinds"] == "train"
    assert (settings["identity_margin_positive"], settings["identity_margin_negative"],
            settings["identity_margin_weight"]) == (0.95, 0.5, 1.0)
    assert settings["key_monitor"]["violation_margins"] == dict(positive=0.95, negative=0.5)
    # Top-level objective discloses the key-only loss (no perception RGB/label loss).
    assert "key only" in settings["objective"] and "absolute margins" in settings["objective"]
    assert "RGB" not in settings["objective"]


def test_frozen_run_reports_monitor_screen_and_no_validation_kind_rows(frozen_run):
    result = json.loads((frozen_run / "result.json").read_text())
    status = json.loads((frozen_run / "status.json").read_text())
    assert status["result"] == "completed" and status["report"] != "failed"
    assert (frozen_run / "report.html").exists()
    assert result["perception_unchanged"] is True
    screen = result["screen"]
    assert screen["thresholds"] == dict(positive_q01=0.93, negative_q99=0.60, within_q99=0.60)
    assert result["gate"] == screen["passed"]  # pass or fail, recorded either way
    for population in ("procedural", "kind_table"):
        m = result["metrics"]["monitor"][population]
        assert m["positive_count"] == 128 and m["within_count"] > 0 and m["negative_count"] > 0
        assert -1 <= m["negative_q99"] <= m["negative_max"] <= 1 and m["positive_min"] <= m["positive_q01"]
        for rate in ("positive_violation", "negative_violation", "within_violation"):
            assert 0 <= m[rate] <= 1  # configured-margin violation rates on the monitor
    rows = [json.loads(line) for line in (frozen_run / "metrics.jsonl").read_text().splitlines()]
    splits = {r["split"] for r in rows}
    assert "validation" not in splits and {"train", "monitor_initial", "monitor"} <= splits
    assert all("identity_margin" in r and "margin_positive_violation" in r for r in rows if r["split"] == "train")


def test_frozen_key_run_loads_in_the_r2_composition(frozen_run):
    perception_j, _ = j_states()
    modules = us.build(0, identity_run=frozen_run, candidate_scope="predicted-machine")
    assert state_hash(modules["perception"]) == perception_j
    assert state_hash(modules["core"].key_head) == component(frozen_run, "key", key_head(64, 32))


def test_exact_resume_matches_uninterrupted_run(frozen_run, tmp_path, monkeypatch):
    out = tmp_path / "paused"
    run_main(monkeypatch, *frozen_args(out, "--stop-after", "1"))
    assert json.loads((out / "status.json").read_text())["result"] == "paused"
    run_main(monkeypatch, "--stage", "perception", "--resume", out, "--device", "cpu")
    assert component(out, "key", key_head(64, 32)) == component(frozen_run, "key", key_head(64, 32))
    assert component(out, "perception", SlotPerception(64, 7, 3, decoder_width=32)) == j_states()[0]


def test_margin_is_optional_exact_when_satisfied_and_active_when_violated():
    torch.manual_seed(0)
    perception = SlotPerception(64, 7, 3, decoder_width=32)
    key = key_head(64, 32)
    load_component(perception, J / "last.pt", "perception")
    load_component(key, J / "last.pt", "key")
    perception.eval()
    scenes, _, rgb, _, _, (textures, source) = la.paired_perception_batch(
        torch.Generator().manual_seed(11), rw.KIND_SPLIT["train"], 8, "cpu", randomize=1.0,
        augmentation=torch.Generator().manual_seed(12), pairing=torch.Generator().manual_seed(13))
    with torch.no_grad():
        percept = perception(rgb)
    base, metrics = la.identity_loss(key, percept, scenes, textures, source, detached=True)
    assert set(metrics) == {"identity_nce", "identity_accuracy", "identity_false_negatives"}
    slack, _ = la.identity_loss(key, percept, scenes, textures, source, detached=True, margin=(-1.0, 1.0, 1.0))
    assert torch.equal(slack, base)  # hinge exactly zero when every margin holds
    tight, tight_metrics = la.identity_loss(key, percept, scenes, textures, source, detached=True,
                                            margin=(1.0, -1.0, 1.0))
    assert tight > base and tight_metrics["margin_positive_violation"] > 0
    tight.backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in key.parameters())


def test_equal_textures_are_never_negatives_within_or_across_frames():
    perception = SlotPerception(64, 7, 3, decoder_width=32)
    key = key_head(64, 32)
    load_component(perception, J / "last.pt", "perception")
    load_component(key, J / "last.pt", "key")
    perception.eval()
    single = [rw.KIND_SPLIT["train"][0]]  # every machine shows one kind-table texture
    scenes, _, rgb, _, _, (textures, source) = la.paired_perception_batch(
        torch.Generator().manual_seed(21), single, 4, "cpu", randomize=0.0,
        augmentation=torch.Generator().manual_seed(22), pairing=torch.Generator().manual_seed(23))
    with torch.no_grad():
        percept = perception(rgb)
        rows = torch.arange(len(scenes))
        slots = torch.stack([percept.slots[rows, la.pointer(percept.alpha, scenes.machine_xy[:, m])]
                             for m in (0, 1)], 1).flatten(0, 1)
        keys = F.normalize(key(slots), dim=-1)
    n = len(keys) // 2
    target = torch.argsort(source.flatten())
    false_negative = la.texture_matches(textures, n)
    false_negative[torch.arange(n), target] = False
    same, different, within = la.pair_cosines(keys, textures, target, false_negative)
    assert len(same) == n and len(different) == 0 and len(within) == 0
    # The margin loss on this legal all-equal batch stays finite (empty negative term is
    # zero, not NaN) and the positive hinge alone still trains the key (target 1.0 is
    # violated by every real pair; InfoNCE has only its target here).
    loss, metrics = la.identity_loss(key, percept, scenes, textures, source, detached=True,
                                     margin=(1.0, 0.5, 1.0))
    assert torch.isfinite(loss) and metrics["margin_negative_pairs"] == 0
    assert metrics["margin_positive_violation"] == 1.0
    assert metrics["margin_negative_violation"] == 0.0
    loss.backward()
    grads = [p.grad for p in key.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in grads) and any(g.abs().sum() > 0 for g in grads)


def test_frozen_guard_rejects_changed_perception():
    model = torch.nn.ModuleDict(dict(perception=SlotPerception(64, 7, 3, decoder_width=32), key=key_head(64, 32)))
    load_component(model["perception"], J / "last.pt", "perception")
    expected = state_hash(model["perception"])
    la.guard_frozen(model, expected)
    with torch.no_grad():  # FAULT INJECTION: a changed frozen weight must stop the run
        model["perception"].decoder.network[0].bias[0] += 1e-6
    with pytest.raises(RuntimeError, match="Frozen perception changed"):
        la.guard_frozen(model, expected)


def test_resource_ceiling_checks_frozen_hash_before_its_preserving_save(monkeypatch):
    """FAULT INJECTION: simulated CUDA reservation above the ceiling with a changed
    frozen perception; the guard must stop the run before any checkpoint is written."""
    saves = []
    runner = type("R", (), {"save": lambda self: saves.append(1)})()
    args = type("A", (), {"device": "cuda", "max_reserved_gib": 6.0})()
    monkeypatch.setattr(la.torch.cuda, "max_memory_reserved", lambda device=None: 7 * 2**30)

    def changed():
        raise RuntimeError("Frozen perception changed; refusing to save or report this run")

    with pytest.raises(RuntimeError, match="Frozen perception changed"):
        la.enforce_ceiling(args, runner, guard=changed)
    assert saves == []
    with pytest.raises(la.ResourceCeiling):
        la.enforce_ceiling(args, runner, guard=lambda: None)
    assert saves == [1]


@pytest.mark.parametrize("argv, message", [
    (["--init-key"], "requires --identity"),
    (["--identity", "detached", "--init-perception", J, "--freeze-perception"], "requires --init-perception, --init-key"),
    (["--identity", "joint", "--init-perception", J, "--init-key", "--freeze-perception"], "--identity detached"),
    (["--identity", "detached", *MARGIN[:2]], "given together"),
    (["--identity", "detached", *MARGIN[:4], "--identity-margin-weight", "-1"], "positive finite weight"),
    (["--identity", "detached", "--identity-margin-positive", "0.4", "--identity-margin-negative", "0.5",
      "--identity-margin-weight", "1"], "negative < positive"),
    ([*MARGIN], "with --identity"),
    (["--identity", "detached", "--init-perception", J, "--init-key", "--freeze-perception",
      "--texture-randomization", "0.5"], "recorded texture randomization"),
])
def test_incompatible_flags_are_rejected(tmp_path, monkeypatch, capsys, argv, message):
    with pytest.raises(SystemExit):
        run_main(monkeypatch, "--stage", "perception", "--output", tmp_path / "x", "--device", "cpu", *argv)
    assert message in capsys.readouterr().err
    assert not (tmp_path / "x").exists()


def test_freeze_is_rejected_outside_the_perception_stage(tmp_path, monkeypatch, capsys):
    with pytest.raises(SystemExit):
        run_main(monkeypatch, "--stage", "core", "--output", tmp_path / "x", "--device", "cpu",
                 "--perception", J, "--freeze-perception")
    assert "perception stage only" in capsys.readouterr().err
