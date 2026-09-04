"""Every experiment spec and dev config parses, and no numeric value has silently become a string.

PyYAML reads `1e6` and `1.0e6` as strings (only `1.0e+6` is a float). A spec value
like that would flow into a training loop unnoticed, so specs use plain integers.
"""
import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
SPECS = sorted(ROOT.glob("experiments/*.yaml")) + sorted(ROOT.glob("configs/**/*.yaml"))
NUMERIC_LOOKING = re.compile(r"^[-+]?[0-9][0-9_.]*(e[-+]?[0-9]+)?$", re.IGNORECASE)


def _leaves(node, path=""):
    if isinstance(node, dict):
        for k, v in node.items():
            yield from _leaves(v, f"{path}.{k}")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _leaves(v, f"{path}[{i}]")
    else:
        yield path, node


@pytest.mark.parametrize("spec", SPECS, ids=lambda p: str(p.relative_to(ROOT)))
def test_spec_has_no_numeric_strings(spec):
    data = yaml.safe_load(spec.read_text())
    assert isinstance(data, dict) and "experiment" in data, "a spec is a mapping with an 'experiment' key"
    bad = [(p, v) for p, v in _leaves(data) if isinstance(v, str) and NUMERIC_LOOKING.match(v)]
    assert not bad, f"numeric-looking strings (write them as plain numbers): {bad}"


def test_counterfactual_temperature_iteration_changes_only_kappa():
    previous_path = ROOT / "configs/dev/first_slice_counterfactual.yaml"
    calibrated_path = ROOT / "configs/dev/first_slice_counterfactual_scaled.yaml"
    previous = yaml.safe_load(previous_path.read_text())
    calibrated = yaml.safe_load(calibrated_path.read_text())

    assert previous["losses"]["counterfactual"]["kappa"] == 0.1
    previous["losses"]["counterfactual"]["kappa"] = 0.001
    assert calibrated == previous


@pytest.mark.parametrize(
    ("filename", "weight"),
    [
        ("first_slice_counterfactual_grid_k1e3_w1e3.yaml", 0.001),
        ("first_slice_counterfactual_grid_k1e3_w3e3.yaml", 0.003),
        ("first_slice_counterfactual_grid_k1e3_w1e2.yaml", 0.01),
        ("first_slice_counterfactual_grid_k1e3_w15e3.yaml", 0.015),
        ("first_slice_counterfactual_grid_k1e3_w2e2.yaml", 0.02),
        ("first_slice_counterfactual_grid_k1e3_w3e2.yaml", 0.03),
        ("first_slice_counterfactual_grid_k1e3_w1e1.yaml", 0.1),
    ],
)
def test_counterfactual_weight_grid_changes_only_weight(filename, weight):
    base = yaml.safe_load((ROOT / "configs/dev/first_slice_counterfactual_scaled.yaml").read_text())
    candidate = yaml.safe_load((ROOT / "configs/dev" / filename).read_text())

    base["losses"]["counterfactual"]["weight"] = weight
    assert candidate == base


@pytest.mark.parametrize(
    ("filename", "kappa", "weight"),
    [
        ("first_slice_counterfactual_grid_k5e4_w1e2.yaml", 0.0005, 0.01),
        ("first_slice_counterfactual_grid_k2e3_w4e2.yaml", 0.002, 0.04),
        ("first_slice_counterfactual_grid_k4e3_w8e2.yaml", 0.004, 0.08),
    ],
)
def test_counterfactual_temperature_grid_changes_only_kappa_and_weight(filename, kappa, weight):
    base = yaml.safe_load((ROOT / "configs/dev/first_slice_counterfactual_scaled.yaml").read_text())
    candidate = yaml.safe_load((ROOT / "configs/dev" / filename).read_text())

    base["losses"]["counterfactual"].update(kappa=kappa, weight=weight)
    assert candidate == base


@pytest.mark.parametrize(
    ("filename", "changes"),
    [
        ("first_slice_counterfactual_solution_predictor_only.yaml", {"weight": 1.0, "context_gradient": False}),
        ("first_slice_counterfactual_solution_absolute_w2.yaml", {"positive_weight": 2.0}),
        ("first_slice_counterfactual_solution_absolute_w6.yaml", {"positive_weight": 6.0}),
        ("first_slice_counterfactual_solution_absolute_w12.yaml", {"positive_weight": 12.0}),
        ("first_slice_counterfactual_solution_absolute_w24.yaml", {"positive_weight": 24.0}),
        ("first_slice_counterfactual_solution_absolute_w48.yaml", {"positive_weight": 48.0}),
    ],
)
def test_counterfactual_solution_configs_change_only_declared_fields(filename, changes):
    base = yaml.safe_load(
        (ROOT / "configs/dev/first_slice_counterfactual_grid_k2e3_w4e2.yaml").read_text()
    )
    candidate = yaml.safe_load((ROOT / "configs/dev" / filename).read_text())

    base["losses"]["counterfactual"].update(changes)
    assert candidate == base
