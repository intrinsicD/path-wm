import torch
from pathwm.models.entity_relations import RelationWriteGate


def test_shift_pairing_and_scores():
    from pathwm.evaluation.entity_gate import gate_shift_examples, score_gate_shift

    data = gate_shift_examples(801, 8)
    assert data["labels"].tolist() == [True, False] * 8
    assert torch.equal(data["active"][::2], data["active"][1::2])
    assert torch.equal(data["noise"][::2], data["noise"][1::2])
    model = RelationWriteGate()
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
    result = score_gate_shift(model, data)
    assert not result["passed"]
    for c in result["cohorts"].values():
        assert c["brier"] == 0.25
        assert c["thresholds"]["0.5"]["positive_recall"] == 0
        assert c["thresholds"]["0.5"]["negative_recall"] == 1
        assert c["thresholds"]["0.4"]["positive_recall"] == 1
        assert c["thresholds"]["0.4"]["negative_recall"] == 0


def test_shift_recipe_resume(tmp_path):
    from experiments.multimodal import evaluate_entity_gate_shift

    donor = tmp_path / "gate.pt"
    torch.save({"model": RelationWriteGate().state_dict()}, donor)
    output = tmp_path / "shift"
    evaluate_entity_gate_shift(donor, output)
    raw = (output / "entity_gate_shift.json").read_bytes()
    evaluate_entity_gate_shift(donor, output, resume=True)
    assert (output / "entity_gate_shift.json").read_bytes() == raw
