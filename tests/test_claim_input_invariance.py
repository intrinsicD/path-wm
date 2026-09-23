"""R1 poisoning must change evaluator annotations only, never an agent-visible input.

`sample_life` records the claim the agent will receive as an explicit observable field
(`claimed`, the R1 population's all-false claim, 1 - truth, drawn with no extra RNG).
`run_life` transmits that field; `LifeSpec.poisoned` flips only the truth annotation.
"""

import dataclasses
import hashlib
import json

import pytest
import torch

from pathwm.evaluation import rule_world as ev
from tests.test_latent_agent import tiny_life, tiny_models

# Fingerprints of the ORIGINAL (pre-fix) sampler output and RNG consumption.
ORIGINAL_SAMPLING = {4: "ac79d688655649187e4b3e8b36594b04fd878122243d8eaef79582fcb78d2954",
                     11: "4e44404a8c587ec4c176a93faf59c0ecbd053959be6fbc92447e6bb9081b01ca"}


def fingerprint(seed):
    g = torch.Generator().manual_seed(seed)
    spec = ev.sample_life(g, "validation", n_support=8, queries=4, goals=2, distract=8, counter=4)
    c = spec.claim
    parts = dict(
        rng=hashlib.sha256(g.get_state().numpy().tobytes()).hexdigest(),
        claim=[c["order"], c["lamps"], c["side"], c["i"], c["j"], c["truth"],
               c["scene"].attrs.tolist(), c["scene"].machine_xy.tolist(), c["scene"].object_xy.tolist()],
        queries=[(q.scene, list(q.lamps), q.machine, q.i, q.j, q.label, q.truth) for q in spec.queries],
        rules={k: list(v.key()) if isinstance(v.key(), tuple) else str(v.key()) for k, v in spec.rules.items()},
        life_id=spec.life_id,
    )
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest(), spec


def received_testimony(directory):
    """What the agent actually received: the testimony persisted in its own memory."""
    events = json.loads((directory / "memory" / "store.json").read_text())["events"]
    return [op["value"]["data"] for e in events for op in e["operations"]
            if op["op"] == "evidence" and op["value"]["source"] == "testimony"]


def test_poisoned_life_transmits_the_identical_claim(tmp_path):
    perception, core = tiny_models()
    spec = tiny_life()
    honest = ev.run_life(perception, core, spec, tmp_path / "honest", settings=ev.AgentSettings())
    poisoned = ev.run_life(perception, core, spec.poisoned(), tmp_path / "poisoned", settings=ev.AgentSettings())
    assert spec.poisoned().claim["truth"] != spec.claim["truth"]  # the annotation is poisoned
    assert received_testimony(tmp_path / "honest") == received_testimony(tmp_path / "poisoned")  # the input is not
    assert poisoned["agent_outputs"] == honest["agent_outputs"]
    assert honest["agent_outputs"]["claim"]["transmitted"] == spec.claim["claimed"]


@pytest.mark.parametrize("seed", sorted(ORIGINAL_SAMPLING))
def test_original_population_and_rng_are_unchanged_apart_from_the_observable_field(seed):
    digest, spec = fingerprint(seed)
    assert digest == ORIGINAL_SAMPLING[seed]
    assert spec.claim["claimed"] == 1 - spec.claim["truth"]  # R1's all-false claims, now explicit
    assert spec.poisoned().claim["claimed"] == spec.claim["claimed"]


def test_a_claim_without_the_observable_field_fails_before_the_life_starts(tmp_path):
    perception, core = tiny_models()
    spec = tiny_life()
    legacy = dataclasses.replace(spec, claim={k: v for k, v in spec.claim.items() if k != "claimed"})
    with pytest.raises(ValueError, match="claimed"):
        ev.run_life(perception, core, legacy, tmp_path / "legacy", settings=ev.AgentSettings())
    assert not (tmp_path / "legacy").exists()  # no truth fallback, nothing run
