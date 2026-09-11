import pytest
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


@pytest.mark.parametrize("reobserve", [False, True])
def test_shift_recipe_resume(tmp_path, reobserve):
    from experiments.multimodal import evaluate_entity_gate_shift

    donor = tmp_path / "gate.pt"
    torch.save({"model": RelationWriteGate().state_dict()}, donor)
    output = tmp_path / "shift"
    evaluate_entity_gate_shift(
        donor, output, reobserve=reobserve, correlation=0.5 if reobserve else None
    )
    raw = (output / "entity_gate_shift.json").read_bytes()
    evaluate_entity_gate_shift(
        donor,
        output,
        resume=True,
        reobserve=reobserve,
        correlation=0.5 if reobserve else None,
    )
    assert (output / "entity_gate_shift.json").read_bytes() == raw


def test_reobserve_duplicate_and_cost():
    from pathwm.evaluation.entity_gate import (
        gate_reobserve_examples,
        score_gate_reobserve,
    )

    data = gate_reobserve_examples(4)
    assert not torch.equal(data["noise"], data["second_noise"])
    model = RelationWriteGate()
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
    result = score_gate_reobserve(model, data)
    for c in result["cohorts"].values():
        strategies = c["strategies"]
        assert c["duplicate_exact"]
        assert strategies["selective"]["reread_rate"] == 1
        assert strategies["selective"]["utility"] == 0.48
        assert strategies["first"]["utility"] == 0.5


def test_correlated_noise_endpoints():
    from pathwm.evaluation.entity_gate import gate_reobserve_examples

    independent = gate_reobserve_examples(8, correlation=0)
    shared = gate_reobserve_examples(8, correlation=1)
    middle = gate_reobserve_examples(8, correlation=0.5)
    assert torch.equal(independent["noise"], shared["noise"])
    assert torch.equal(shared["noise"], shared["second_noise"])
    assert torch.allclose(
        middle["second_noise"],
        0.5 * middle["noise"] + (3**0.5 / 2) * independent["second_noise"],
    )
    with pytest.raises(ValueError):
        gate_reobserve_examples(8, correlation=1.1)


def test_alternate_sources_share_first_observation():
    from pathwm.evaluation.entity_gate import score_evidence_sources
    result = score_evidence_sources(RelationWriteGate(), pairs=4)
    for n in result['sources']['same']['cohorts']:
        a = result['sources']['same']['cohorts'][n]
        b = result['sources']['alternate']['cohorts'][n]
        assert a['defer'] == b['defer']
        assert a['strategies']['first'] == b['strategies']['first']
        assert a['duplicate_exact'] and b['duplicate_exact']
        for c,cost in ((a,.02),(b,.05)):
            s=c['strategies']['selective']
            assert abs(s['utility']-(s['accuracy']-cost*s['reread_rate']))<1e-7
