"""R0 selection must retain the rank intervention only with both modalities and guardrails."""
from copy import deepcopy
from pathlib import Path
import yaml
from scripts.run_common_base_overnight import compare_r0


def _case():
    root = Path(__file__).resolve().parents[2]
    specs = {variant: yaml.safe_load((root / f"configs/dev/common_base_full_{variant}.yaml").read_text())
             for variant in ("balanced", "control")}
    gates = specs["balanced"]["curriculum"]["gates"]["unimodal_representation_ready"]
    rows = [{"variant": variant, "seed": seed,
             "metrics": {name: (0.4 if variant == "balanced" else 0.3) if "rank" in name else 0.5
                         for name in gates}, "gate": {"passed": True}}
            for seed in (0, 1) for variant in ("balanced", "control")]
    return specs, rows


def test_rank_intervention_needs_every_pair_and_temporal_guardrail():
    specs, rows = _case()
    assert compare_r0(rows, specs)["decision"] == "retain_covariance"
    assert compare_r0(rows[:-1], specs) == {"decision": "incomplete", "r1_ready": False}
    changed = deepcopy(rows)
    changed[0]["metrics"]["audio_future_prediction_advantage"] = -0.01
    assert compare_r0(changed, specs)["decision"] == "covariance_not_supported"
    changed = deepcopy(rows)
    changed[0]["metrics"]["audio_effective_rank_fraction"] = 0.2
    assert compare_r0(changed, specs)["decision"] == "covariance_not_supported"


def test_improved_rank_does_not_promote_a_failed_r0():
    specs, rows = _case()
    for row in rows:
        row["gate"]["passed"] = False
    comparison = compare_r0(rows, specs)
    assert comparison["decision"] == "retain_covariance"
    assert not comparison["r1_ready"]
