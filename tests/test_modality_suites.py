import copy

import numpy as np
import pytest
import torch


def test_partial_capability_gates_cannot_pass():
    from pathwm.evaluation.capabilities import case_record, score

    result = case_record(
        "partial",
        {},
        torch.zeros(1),
        {},
        {"a": score(1.0, 0.8), "b": score(None, 0.8)},
        "tiny",
        "raw.npz",
    )
    assert result["status"] == "not_measured"


def test_cartesian_intervention_factors_are_conditionally_balanced():
    from pathwm.data.modality_readout import dataset, observations

    d = dataset("intervention", 7201)
    y = d["targets"]["factors"].numpy()
    assert len(y) == 72 and len(set(d["ids"])) == 72
    for col, cardinality in enumerate((3, 3, 2)):
        rest = np.delete(y, col, axis=1)
        for key in np.unique(rest, axis=0):
            group = y[(rest == key).all(1), col]
            assert np.array_equal(
                np.bincount(group, minlength=cardinality), np.full(cardinality, 4)
            )
    assert set(observations(d, "complementary", omit="video")) == {
        "text",
        "image",
        "audio",
    }
    for split in ("train", "validation", "seen", "heldout"):
        other = dataset(split, 7201)
        assert not set(other["ids"]) & set(d["ids"])


def test_factor_diagnostics_validate_and_keep_failures_per_draw():
    from pathwm.evaluation.modality_suite import factor_diagnostics

    y = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 0]])
    pred = np.repeat(y[None], 3, axis=0)
    pred[2, 0, 0] = 1
    r = factor_diagnostics(pred, y, ["a", "b", "c"])
    assert r["min_joint_accuracy"] == pytest.approx(2 / 3)
    assert r["min_factor_accuracy"][0] == pytest.approx(2 / 3)
    assert r["failures"][0]["example_id"] == "a"
    assert r["failures"][0]["draw"] == 2
    assert len(r["by_combination"]) == 3
    for broken in (
        pred.astype(float) + 0.5,
        pred[:, :-1],
        np.full_like(pred, 4),
        np.full(pred.shape, np.nan),
    ):
        with pytest.raises(ValueError):
            factor_diagnostics(broken, y, ["a", "b", "c"])
    with pytest.raises(ValueError):
        factor_diagnostics(pred, y, ["a", "a", "c"])


def test_suite_keeps_missing_cases_and_does_not_promote_probes():
    from pathwm.evaluation.modality_suite import build_suite

    report = build_suite({}, [], source={"checkpoint": "abc"}, seed=7201)
    assert report["coverage"]["passed"] == 0
    assert report["coverage"]["not_implemented"] > 0
    assert report["coverage"]["not_run"] > 0
    ids = [c["id"] for c in report["cases"]]
    assert len(ids) == len(set(ids))
    assert {"text", "image", "audio", "video", "core", "action"} <= {
        c["modality"] for c in report["cases"]
    }
    bogus = [{"stage": "encoder", "factor_accuracy": [1, 1, 1], "all_correct": 1}]
    other = build_suite({}, bogus, source={"checkpoint": "abc"}, seed=7201)
    assert other["coverage"] == report["coverage"]
    assert all(c["assessment"] == "not_scored" for c in other["cases"])


def test_suite_requires_each_draw_and_every_factor_and_separates_omission():
    from pathwm.evaluation.modality_suite import build_suite, factor_diagnostics

    y = np.array([[a, b, c] for a in range(3) for b in range(3) for c in range(2)])
    ids = [str(i) for i in range(len(y))]
    pred = np.repeat(y[None], 3, axis=0)
    good = factor_diagnostics(pred, y, ids)
    measured = {
        f"{split}.{mode}": copy.deepcopy(good)
        for split in ("seen", "heldout")
        for mode in ("text", "image", "audio", "video", "all", "complementary")
    }
    broken = pred.copy()
    broken[1, :, 2] = 1 - broken[1, :, 2]
    measured["heldout.video"] = factor_diagnostics(broken, y, ids)
    measured["intervention.complementary"] = good
    missing = pred.copy()
    missing[:, :, 0] = 0
    measured["intervention.without_image"] = factor_diagnostics(missing, y, ids)
    report = build_suite(measured, [], source={"checkpoint": "abc"}, seed=7201)
    by_id = {c["id"]: c for c in report["cases"]}
    assert by_id["VID.symbolic.heldout"]["assessment"] == "fail"
    assert by_id["TXT.symbolic.heldout"]["assessment"] == "pass"
    assert by_id["CORE.source_image"]["assessment"] == "pass"
    assert by_id["CORE.source_audio"]["assessment"] == "not_scored"
    assert by_id["VID.natural_motion"]["implementation"] == "not_implemented"


def test_actual_evaluation_preserves_model_and_caller_rng_and_repeats():
    from experiments.modality_readout import Core
    from pathwm.data.modality_readout import dataset
    from pathwm.evaluation.modality_suite import measure_factors
    from pathwm.io import seed_everything, state_hash

    seed_everything(92)
    core = Core()
    core.train()
    populations = {}
    for split in ("seen", "heldout", "intervention"):
        d = dataset(split, 92)
        for group in ("inputs", "complementary"):
            from dataclasses import replace

            d[group] = {
                k: replace(
                    v,
                    values=v.values[:2],
                    times=v.times[:2],
                    valid=None if v.valid is None else v.valid[:2],
                )
                for k, v in d[group].items()
            }
        d["targets"] = {k: v[:2] for k, v in d["targets"].items()}
        d["ids"] = d["ids"][:2]
        populations[split] = d
    before, rng = state_hash(core), torch.get_rng_state().clone()
    a, arrays_a, seeds_a = measure_factors(core, populations, seed=92, device="cpu")
    b, arrays_b, seeds_b = measure_factors(core, populations, seed=92, device="cpu")
    assert a == b and seeds_a == seeds_b
    assert len(seeds_a) == 48
    assert core.training and state_hash(core) == before
    assert torch.equal(torch.get_rng_state(), rng)
    assert all(p.grad is None for p in core.parameters())
    for k in arrays_a:
        np.testing.assert_array_equal(arrays_a[k], arrays_b[k])


def test_hints_separate_failed_actual_reader_from_accessible_stage():
    from pathwm.evaluation.modality_suite import diagnostic_hints

    actual = {"heldout.text": {"min_factor_accuracy": [1, 1, 0.5]}}
    probes = [
        dict(
            split="heldout",
            input_mode="text",
            stage=s,
            probe="ridge",
            factor_accuracy=[1, 1, score],
        )
        for s, score in [("encoder", 1), ("posterior", 0.5), ("thought", 1)]
    ]
    hints = diagnostic_hints(actual, probes)
    assert len(hints) == 1
    assert "deployed factor readout" in hints[0]["next_check"]
    assert "no unique cause" in hints[0]["attribution"]
