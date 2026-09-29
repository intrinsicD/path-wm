"""Shared-core recipe, symbolic stage: the workflow runs end to end (declared downscale `check`)."""

import json

import pytest

from experiments import core as recipe
from pathwm.evaluation.rules import TRANSITION_FAMILIES


@pytest.mark.parametrize("family,flags", [("relation", []), ("toggle", ["--untied"]), ("relation", ["--continuous-only"]),
                                          ("relation", ["--reader", "evidence", "--swap-weight", "1", "--rule-repeats", "7", "1", "1", "1"])])
def test_symbolic_recipe_runs_and_writes_a_report(tmp_path, family, flags):
    out = tmp_path / "run"
    result = recipe.main(["--output", str(out), "--size", "check", "--device", "cpu", "--family", family,
                          "--updates", "3", "--episodes", "2", "--queries", "4", "--support", "2",
                          "--eval-episodes", "4", "--log-every", "1", *flags])
    assert result["complete"] and result["step"] == 3
    for pool in ("train_rules", "heldout_rules"):
        assert set(result[pool]) >= {"full", "empty", "swapped", "copy", "posterior_lamp_accuracy"}
    status = json.loads((out / "status.json").read_text())
    assert status == dict(result="completed", report="completed", step=3, error=None)
    assert (out / "report.html").exists() and (out / "last.pt").exists()


def test_resume_continues_the_same_run(tmp_path):
    out = tmp_path / "run"
    base = ["--size", "check", "--device", "cpu", "--episodes", "2", "--queries", "4", "--support", "2",
            "--eval-episodes", "4"]
    paused = recipe.main(["--output", str(out), "--updates", "4", "--max-minutes", "0", *base])
    assert paused["step"] == 0 and not paused["complete"]
    assert json.loads((out / "status.json").read_text())["result"] == "paused"
    result = recipe.main(["--resume", str(out)])
    assert result["step"] == 4 and result["complete"]


def test_copy_baseline_has_zero_nu_on_transition_families(tmp_path):
    # Copy predicts Δ = 0 everywhere, so its balanced accuracy on Δ is 0.5: ν at or below the floor.
    result = recipe.main(["--output", str(tmp_path / "run"), "--size", "check", "--device", "cpu",
                          "--family", "toggle", "--updates", "1", "--episodes", "2", "--queries", "8",
                          "--support", "2", "--eval-episodes", "8"])
    for family, nu in result["train_rules"]["copy"]["nu"].items():
        assert family in TRANSITION_FAMILIES and nu <= 0.0 + 1e-9


def test_perception_stage_runs_qualifies_and_saves_a_loadable_component(tmp_path):
    import torch
    from pathwm.io import load_component
    from pathwm.models.slots import SlotPerception

    out = tmp_path / "perception"
    result = recipe.main(["--stage", "perception", "--output", str(out), "--size", "check", "--device", "cpu",
                          "--updates", "2", "--batch", "2", "--qualify-scenes", "4", "--log-every", "1"])
    assert result["complete"] and set(result) >= {"train_kinds", "validation_kinds", "qualified"}
    gates = result["validation_kinds"]["qualification"]
    assert set(gates) == {"attribute", "lamp", "machine_pointer", "object_pointer", "passed"}
    assert (out / "report.html").exists()
    size = recipe.PERCEPTION_SIZES["check"]
    module = SlotPerception(size["width"], 7, size["iterations"], decoder_width=size["decoder_width"])
    load_component(module, out / "last.pt", "perception")
    assert module(torch.zeros(1, 3, 64, 64)).slots.shape == (1, 7, size["width"])
