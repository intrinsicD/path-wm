import pytest
from pathwm.models.source_choice import SourceChoice


def test_feedback_only_choice_and_permutation():
    policy = SourceChoice()
    assert policy.choose() is None
    policy.observe(0, -0.05)
    policy.observe(1, 0.95)
    assert policy.choose() == 1
    swapped = SourceChoice()
    swapped.observe(1, -0.05)
    swapped.observe(0, 0.95)
    assert swapped.choose() == 0
    before = policy.snapshot()
    for _ in range(5):
        assert policy.choose() == 1
    assert policy.snapshot() == before
    with pytest.raises(ValueError):
        policy.observe(0, float("nan"))
    assert policy.snapshot() == before


def test_calibration_only_feedback_and_resume(tmp_path):
    import json
    import torch
    from pathwm.models.entity_relations import RelationWriteGate
    from experiments.multimodal import evaluate_entity_source_choice

    donor = tmp_path / "gate.pt"
    model = RelationWriteGate()
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
    torch.save({"model": model.state_dict()}, donor)
    output = tmp_path / "choice"
    evaluate_entity_source_choice(donor, output)
    raw = (output / "entity_source_choice.json").read_bytes()
    result = json.loads(raw)
    for row in result["worlds"]:
        assert all(x["index"] < 512 for x in row["feedback"])
        assert row["choice"] is None
        assert (
            row["scores"]["learned"]
            == row["scores"]["no_feedback"]
            == row["scores"]["stop"]
        )
        assert row["values"]["counts"] == [256, 256]
    evaluate_entity_source_choice(donor, output, resume=True)
    assert (output / "entity_source_choice.json").read_bytes() == raw


def test_window_forgets_without_reset():
    policy = SourceChoice(window=2)
    policy.observe(0, 1)
    policy.observe(0, 1)
    assert policy.choose() == 0
    policy.observe(0, -0.1)
    policy.observe(0, -0.1)
    assert policy.choose() is None
    assert policy.snapshot()["counts"] == [2, 0]
    assert policy.snapshot()["history"] == [[-0.1, -0.1], []]
    with pytest.raises(ValueError):
        SourceChoice(window=0)


def test_drift_feedback_timing_and_resume(tmp_path):
    import json
    import torch
    from pathwm.models.entity_relations import RelationWriteGate
    from experiments.multimodal import evaluate_entity_source_drift

    donor = tmp_path / "gate.pt"
    torch.save({"model": RelationWriteGate().state_dict()}, donor)
    output = tmp_path / "drift"
    evaluate_entity_source_drift(donor, output)
    raw = (output / "entity_source_drift.json").read_bytes()
    data = json.loads(raw)
    for row in data["episodes"]:
        for event in row["actions"]:
            if row["policy"] in ("frozen", "no_feedback") or event["source"] is None:
                assert event["feedback"] is None
                assert event["before"] == event["after"]
            else:
                assert event["feedback"] is not None
    evaluate_entity_source_drift(donor, output, resume=True)
    assert (output / "entity_source_drift.json").read_bytes() == raw


def test_triggered_forgetting_is_evidence_driven_and_source_local():
    policy = SourceChoice(change_block=4)
    for _ in range(8):
        policy.observe(0, 0.5)
    assert policy.snapshot()["resets"] == [0, 0]
    policy.observe(1, 0.8)
    before_other = policy.snapshot()["counts"][1]
    for _ in range(3):
        policy.observe(0, -0.5)
    assert policy.snapshot()["resets"] == [0, 0]
    policy.observe(0, -0.5)
    state = policy.snapshot()
    assert state["resets"] == [1, 0]
    assert state["counts"] == [4, before_other]
    assert state["sums"][0] == -2
    assert policy.choose() == 1
    with pytest.raises(ValueError):
        policy.observe(0, float("nan"))
    assert policy.snapshot() == state
    with pytest.raises(ValueError):
        SourceChoice(window=4, change_block=4)


def test_variance_aware_change_distinguishes_noisy_and_sharp_shift():
    noisy = SourceChoice(change_block=4, change_z=3)
    for value in [-1, 1, -1, 1, -0.5, 1.5, -0.5, 1.5]:
        noisy.observe(0, value)
    assert noisy.snapshot()["resets"] == [0, 0]
    sharp = SourceChoice(change_block=4, change_z=3)
    for value in [0.5] * 4 + [-0.5] * 4:
        sharp.observe(0, value)
    assert sharp.snapshot()["resets"] == [1, 0]
    assert sharp.snapshot()["squares"] == [1, 0]
    for z in [0, -1, float("nan")]:
        with pytest.raises(ValueError):
            SourceChoice(change_block=4, change_z=z)
    with pytest.raises(ValueError):
        SourceChoice(change_z=3)


def test_uncertainty_selection_uses_only_stationary_development(monkeypatch):
    import pathwm.evaluation.source_choice as module

    calls = []

    def fake(gate, **kwargs):
        calls.append(kwargs)
        z = kwargs["change_z"]
        return dict(
            detector=dict(
                change_z=z, static_reset_episode_rate={2: 0.5, 3: 0.125, 4: 0}[z]
            ),
            passed=True,
        )

    monkeypatch.setattr(module, "evaluate_source_drift", fake)
    result = module.evaluate_source_uncertainty(None)
    assert result["selection"]["selected_z"] == 3
    assert all(c["static_only"] and c["seed"] == 1901 for c in calls[:3])
    assert calls[3] == dict(worlds=16, seed=2001, change_z=3)


def test_uncertainty_unqualified_fallback_cannot_pass(monkeypatch):
    import pathwm.evaluation.source_choice as module

    def fake(gate, **kwargs):
        return dict(
            detector=dict(change_z=kwargs["change_z"], static_reset_episode_rate=1),
            passed=True,
        )

    monkeypatch.setattr(module, "evaluate_source_drift", fake)
    result = module.evaluate_source_uncertainty(None)
    assert result["selection"]["selected_z"] == 4
    assert not result["selection"]["qualified"]
    assert not result["passed"]


def test_source_diagnosis_mixed_blocks_and_tampering():
    from pathwm.evaluation.source_choice import diagnose_source_changes
    import copy

    policy = SourceChoice(change_block=4, change_z=2)
    for _ in range(6):
        policy.observe(0, .5)
    initial = policy.snapshot()
    actions = []
    for i in range(6):
        before = policy.snapshot()
        policy.observe(0, .5)
        actions.append(dict(index=512+i, source=0, feedback=.5,
                            before=before, after=policy.snapshot()))
    data = dict(episodes=[dict(world=0, swapped=True, policy="uncertainty",
                               initial=initial, actions=actions)])
    rows = diagnose_source_changes(data)["sources"]
    assert rows[0]["mixed_checks"] == 1
    assert rows[0]["pure_checks"] == 1
    assert rows[0]["category"] == "eligible_below_threshold"
    assert rows[1]["category"] == "no_completed_block"
    bad = copy.deepcopy(data)
    bad["episodes"][0]["actions"][1]["after"]["resets"][0] += 1
    with pytest.raises(ValueError, match="reset"):
        diagnose_source_changes(bad)
