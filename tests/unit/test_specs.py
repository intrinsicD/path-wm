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
