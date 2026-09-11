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
