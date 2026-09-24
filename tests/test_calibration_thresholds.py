"""Configurable match-threshold grid for the native visual-binding calibrator.

Software checks on the actual J checkpoint and actual sequential session with a tiny
number of TRAIN episodes (2 or 1). These are calibration-software checks, not
calibration or qualification results.
"""

import json
from pathlib import Path
import sys

import pytest

from experiments import unified_session as us

J = Path(__file__).resolve().parents[1] / "runs/latent_agent_r1/identity_joint_20260923"
LEGACY_RULE = ("Actual session grid: match in [.8,.85,.9,.95], new=match-.05, margin in [.05,.1]; "
               "maximize min acquisition/same-layout/relocated matching under false-match<=.005")


@pytest.fixture(scope="module")
def needs_j():
    if not (J / "last.pt").exists():
        pytest.skip("J checkpoint not present")


def read(path, name):
    return json.loads((Path(path) / name).read_text())


def test_default_grid_is_the_legacy_grid_with_unchanged_settings(tmp_path, needs_j):
    out = tmp_path / "default"
    us.calibrate_visual_binding(out, identity_run=J, seed=3405, scenes=1)
    settings = read(out, "run.json")["identity"]["settings"]
    assert settings["rule"] == LEGACY_RULE and "match_thresholds" not in settings
    policies = read(out, "policies.json")
    assert [(p["thresholds"]["match_threshold"], p["thresholds"]["margin"]) for p in policies] == [
        (m, g) for m in (.80, .85, .90, .95) for g in (.05, .10)]
    assert all(p["thresholds"]["new_threshold"] == p["thresholds"]["match_threshold"] - .05 for p in policies)
    assert us.LEGACY_MATCH_THRESHOLDS == (.80, .85, .90, .95)


def test_custom_grid_is_applied_recorded_and_bound_in_the_manifest(tmp_path, needs_j):
    out = tmp_path / "custom"
    grid = (0.90, 0.93)
    passed = us.calibrate_visual_binding(out, identity_run=J, seed=3405, scenes=2, match_thresholds=grid)
    settings = read(out, "run.json")["identity"]["settings"]
    assert settings["match_thresholds"] == list(grid) and "0.9, 0.93" in settings["rule"]
    policies = read(out, "policies.json")
    assert [(p["thresholds"]["match_threshold"], p["thresholds"]["margin"]) for p in policies] == [
        (m, g) for m in grid for g in (.05, .10)]
    assert passed, "fixture expects a passing tiny calibration to exercise the manifest path"
    binding = read(out, "binding.json")
    assert binding["calibration"]["match_thresholds"] == list(grid)
    assert binding["thresholds"]["match_threshold"] in grid
    binder, policy = us.visual_binding_policy(out / "binding.json", J)
    assert policy["thresholds"] == binding["thresholds"]
    # Tampering with the recorded grid breaks manifest integrity.
    binding["calibration"]["match_thresholds"] = [0.90]
    (out / "binding.json").write_text(json.dumps(binding))
    with pytest.raises(ValueError, match="changed"):
        us.visual_binding_policy(out / "binding.json", J)


@pytest.mark.parametrize("grid", [(), (0.9, float("nan")), (0.9, float("inf")), (0.9, 0.9), (0.93, 0.91),
                                  (0.9, 1.01), (-0.96,), (0.9, "0.93")])
def test_invalid_grids_are_rejected_before_any_output(tmp_path, grid):
    out = tmp_path / "bad"
    with pytest.raises(ValueError, match="match thresholds"):
        us.calibrate_visual_binding(out, identity_run=J, seed=3405, scenes=1, match_thresholds=grid)
    assert not out.exists()


def run_main(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["unified_session.py", *map(str, argv)])
    return us.main()


@pytest.mark.parametrize("values, message", [
    (["0.9", "nan"], "match thresholds"),
    (["0.93", "0.91"], "match thresholds"),
    (["0.9", "0.9"], "match thresholds"),
])
def test_cli_rejects_invalid_grids_before_output(tmp_path, monkeypatch, capsys, values, message):
    out = tmp_path / "cli"
    with pytest.raises(SystemExit):
        run_main(monkeypatch, "--output", out, "--identity-run", J, "--calibrate-visual-binding", "--scenes", "1",
                 "--binding-match-thresholds", *values)
    assert message in capsys.readouterr().err and not out.exists()


def test_cli_grid_requires_calibration_mode(tmp_path, monkeypatch, capsys):
    out = tmp_path / "cli"
    with pytest.raises(SystemExit):
        run_main(monkeypatch, "--output", out, "--identity-run", J, "--visual-memory",
                 "--binding-match-thresholds", "0.9")
    assert "--calibrate-visual-binding" in capsys.readouterr().err and not out.exists()
