import copy

import numpy as np
import pytest
import torch


def test_contrast_records_are_balanced_and_temporal_controls_are_identical():
    from pathwm.data.understanding import synthetic_records, validate_records

    records, arrays, cases = synthetic_records("quick")
    validate_records(records, arrays)
    assert len(cases) == 10
    for case in cases:
        rows = [r for r in records if r["case"] == case["id"] and r["split"] == "test"]
        assert {r["answer"] for r in rows} == {0, 1}
        for a, b in zip(rows[::2], rows[1::2]):
            assert a["pair"] == b["pair"] and a["question"] == b["question"]
            assert a["choices"] == b["choices"] and a["answer"] != b["answer"]
            if case["id"] in ("VID.order", "VID.history"):
                assert np.array_equal(
                    arrays[a["evidence"]["video"]][-1],
                    arrays[b["evidence"]["video"]][-1],
                )
    broken = copy.deepcopy(records)
    broken[1]["answer"] = broken[0]["answer"]
    with pytest.raises(ValueError, match="pair"):
        validate_records(broken, arrays)


def test_source_groups_cannot_cross_calibration_and_test():
    from pathwm.data.understanding import synthetic_records, validate_records

    records, arrays, _ = synthetic_records("quick")
    a = next(r for r in records if r["split"] == "calibration")
    b = next(r for r in records if r["split"] == "test")
    b["groups"] = a["groups"]
    with pytest.raises(ValueError, match="group"):
        validate_records(records, arrays)


def test_controlled_cohorts_do_not_repeat_identical_evidence():
    from pathwm.data.understanding import synthetic_records

    rows, arrays, _ = synthetic_records("full")
    seen = {}
    for row in rows:
        kind, value = next(iter(row["evidence"].items()))
        key = (row["case"], value if kind == "text" else arrays[value].tobytes())
        assert key not in seen or seen[key] == row["split"]
        seen[key] = row["split"]


def test_preparation_covers_every_subset_and_guards_real_pair_sources(
    tmp_path, monkeypatch
):
    import csv
    from itertools import combinations
    import pathwm.data.understanding as data_module

    root = tmp_path / "raw_source"
    (root / "metadata").mkdir(parents=True)
    (root / "raw").mkdir()
    rows = []
    for city in data_module.CITIES.values():
        for ci, scene in enumerate(data_module.SCENES):
            for i in range(4):
                group = f"{city}-{ci * 10 + i}"
                row = dict(
                    identifier=group,
                    scene_label=scene,
                    filename_audio=group + ".wav",
                    filename_video=group + ".mp4",
                )
                rows.append(row)
                for key in ("filename_audio", "filename_video"):
                    (root / "raw" / row[key]).write_bytes(group.encode())
    with (root / "metadata/meta.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    def fake_decode(root, row):
        rng = np.random.default_rng(sum(row["identifier"].encode()))
        return rng.random((3, 3, 8, 8), dtype=np.float32), rng.random(
            (16, 64), dtype=np.float32
        )

    monkeypatch.setattr(data_module, "_decode_recording", fake_decode)
    for profile in ("quick", "full"):
        path = data_module.prepare_understanding(tmp_path / profile, root, profile)
        data = data_module.UnderstandingData(path)
        real = [c for c in data.cases if c["domain"] == "real scene category"]
        assert len(data.cases) == 25
        assert {tuple(c["modalities"]) for c in real} == {
            x for n in range(1, 5) for x in combinations(data_module.KINDS, n)
        }
        by_pair = {}
        for r in data.records:
            if r["case"].startswith("MIX.") and r["answer"] == 0:
                sensor_groups = [
                    v.split("/")[1] for k, v in r["evidence"].items() if k != "text"
                ]
                assert len(sensor_groups) == len(set(sensor_groups))
            if r["pair"] is None:
                continue
            current = data.inputs(r, omit=r["required"])
            if r["pair"] in by_pair:
                previous = by_pair[r["pair"]]
                assert current.keys() == previous.keys()
                for kind in current:
                    torch.testing.assert_close(
                        current[kind].values, previous[kind].values, rtol=0, atol=0
                    )
                    torch.testing.assert_close(
                        current[kind].times, previous[kind].times, rtol=0, atol=0
                    )
            by_pair[r["pair"]] = current
        with (path / "arrays.npz").open("ab") as f:
            f.write(b"changed")
        with pytest.raises(ValueError, match="hash"):
            data_module.UnderstandingData(path)


def test_prior_guessing_cannot_pass_grounded_pairs_and_probes_cannot_promote():
    from pathwm.evaluation.understanding import task_metrics

    labels = np.array([0, 1, 0, 1])
    pairs = ["a", "a", "b", "b"]
    guessed = np.zeros((3, 4), dtype=int)
    score = task_metrics(guessed, guessed, guessed, labels, pairs, 2)
    assert score["accuracy_min"] == 0.5 and score["paired_min"] == 0
    assert not score["passed"]
    correct = np.tile(labels, (3, 1))
    good = task_metrics(correct, guessed, guessed, labels, pairs, 2)
    assert good["passed"]
    with pytest.raises(ValueError):
        task_metrics(correct[:1], guessed, guessed, labels, pairs, 2)


def test_reference_refuses_changed_fixtures_profile_or_readout_contract():
    from pathwm.evaluation.understanding import compare_understanding

    a = {
        "contract": {"fixtures": "a", "profile": "quick", "readout": "mc-v1"},
        "cases": [
            {
                "id": "T",
                "metrics": {"accuracy_min": 0.5, "paired_min": 0.0},
                "passed": False,
            }
        ],
    }
    assert compare_understanding(a, a)[0]["status"] == "unchanged"
    for field in a["contract"]:
        b = copy.deepcopy(a)
        b["contract"][field] = "changed"
        with pytest.raises(ValueError, match="contract"):
            compare_understanding(a, b)


def test_diagnostic_capture_is_observational_and_choice_scoring_uses_actual_decoder():
    from experiments.modality_readout import Model
    from pathwm.data.modality_readout import dataset, observations
    from pathwm.evaluation.understanding import capture_stages, choice_scores
    from pathwm.models.modalities import bytes_batch

    model = Model("native").eval()
    inputs = observations(dataset("train"), "audio", [0])
    rng = torch.get_rng_state().clone()
    expected = model.core(inputs)
    after = torch.get_rng_state().clone()
    torch.set_rng_state(rng)
    tokens, stages = capture_stages(model.core, inputs)
    assert torch.equal(tokens, expected) and torch.equal(torch.get_rng_state(), after)
    assert set(stages) == {"encoder", "posterior", "working"}
    assert all(not t.requires_grad for t in stages.values())
    candidates = ["ja", "nein"]
    actual = choice_scores(model.outputs, tokens, candidates)
    for i, answer in enumerate(candidates):
        target, valid = bytes_batch([answer])
        logits = model.outputs("text", tokens, target[:, :-1])
        expected_score = (
            logits.log_softmax(-1)
            .gather(-1, target[:, 1:, None])
            .squeeze(-1)[valid[:, 1:]]
            .mean()
        )
        torch.testing.assert_close(actual[i], expected_score)


def test_comparison_reports_a_lost_gate_even_when_headline_accuracy_is_unchanged():
    from pathwm.evaluation.understanding import compare_understanding

    before = {'contract': {'version': 'fixed'}, 'cases': [{
        'id': 'paired', 'passed': True,
        'metrics': {'accuracy_min': 1., 'paired_min': 1., 'source_gain_min': .5,
                    'gates': {'accuracy': True, 'no_evidence_gain': True}}}]}
    after = copy.deepcopy(before)
    after['cases'][0]['passed'] = False
    after['cases'][0]['metrics']['gates']['no_evidence_gain'] = False
    row = compare_understanding(before, after)[0]
    assert row['status'] == 'regressed'
    assert row['lost_gates'] == ['no_evidence_gain']
