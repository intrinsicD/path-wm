"""How every test obtains an implementation (CLAUDE.md §2 step 2, §3): the spec under test, its ABI, the builders.

What: `cfg` is configs/dev/first_slice.yaml, the slice's spec (tiny sizes, random weights); `abi` is its ABI;
`build(family, *args, cfg=None)` calls the family's build_<module>(cfg, *args) (CLAUDE.md §3) and returns
the module in inference mode.
How: the builder is looked up when the test body calls build(), so a missing implementation is a FAILED
test whose message names the builder to write, never an error at fixture setup and never a collection
error that would hide the rest of the suite; a builder that exists but is broken raises its own traceback.
torch is reseeded before each build (reproducible random weights) and the builder gets its own copy of cfg.
Why: step 2 of a slice commits its tests red against the step-1 interfaces; the table below is where step 3
puts each implementation (§16.2 layout, DDR §13 step-2 additions).
"""
from __future__ import annotations

import copy
import importlib
import importlib.util
from pathlib import Path

import pytest
import torch
import yaml

from world_state.abi import load_abi

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "configs" / "dev" / "first_slice.yaml"

# family -> (module, builder, cfg key that selects the implementation). cfg is the whole parsed spec, so every
# builder reads its own section plus `abi:` and swapping a module stays one YAML line. build_adapter also takes
# z_shape, the encoder's output shape without the batch dim: z is "any shape" (§16.1), so cfg cannot supply it.
# Builders build on CPU; the runner moves modules to the device it runs on.
BUILDERS = {
    "env": ("envs", "build_env", ("env", "name")),
    "encoder": ("encoders", "build_encoder", ("encoder", "arch")),
    "adapter": ("encoders.adapters", "build_adapter", ("adapter", "kind")),
    "predictor": ("predictors", "build_predictor", ("predictor", "arch")),
    "inverse": ("world_state.inverse", "build_inverse", ("inverse_dynamics", "kind")),
}


@pytest.fixture(scope="session")
def cfg() -> dict:
    return yaml.safe_load(SPEC.read_text())


@pytest.fixture(scope="session")
def abi(cfg):
    return load_abi(ROOT / cfg["abi"])


def _missing(module: str) -> bool:
    """True when the module file is absent; find_spec raises ModuleNotFoundError when a parent package is."""
    try:
        return importlib.util.find_spec(module) is None
    except ModuleNotFoundError:
        return True


@pytest.fixture
def build(cfg):
    base = cfg

    def _build(family: str, *args, cfg: dict | None = None):
        spec = base if cfg is None else cfg
        module, name, (section, key) = BUILDERS[family]
        what = f"{module}.{name}(cfg{', ...' if args else ''}) for {section}.{key}={spec[section][key]!r}"
        if _missing(module):
            pytest.fail(f"no implementation: {what}; CLAUDE.md §2 step 3 writes it")
        builder = getattr(importlib.import_module(module), name, None)  # a broken module raises its own error
        if builder is None:
            pytest.fail(f"no implementation: {what}; CLAUDE.md §2 step 3 writes it")
        torch.manual_seed(0)
        built = builder(copy.deepcopy(spec), *args)
        if isinstance(built, torch.nn.Module):
            built.eval()  # the Protocols have no mode: conformance tests the inference contract, not dropout
        return built

    return _build
