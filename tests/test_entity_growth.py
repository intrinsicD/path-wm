import torch
from pathwm.evaluation.entity_growth import growth_inputs, evaluate_growth
from tests.test_entity_memory import Scorer


def test_nested_populations_and_lifecycle():
    families = growth_inputs(seed=61, count=3)
    assert families == growth_inputs(seed=61, count=3)
    for family in families:
        points = torch.tensor(family["descriptors"])
        distances = torch.cdist(points, points)
        assert (distances[~torch.eye(9, dtype=torch.bool)] >= 0.9).all()
    result = evaluate_growth(Scorer(), families)
    assert result["passed"]
    assert len(result["episodes"]) == 12
    assert all(
        row["retry_equal"] and row["restore_equal"] for row in result["episodes"]
    )


def test_rejections_cannot_pass_as_success():
    class Reject(Scorer):
        def match(self, query, memory):
            scores = query.new_zeros((len(query), memory.shape[1] + 1))
            return scores

    results = evaluate_growth(Reject(), growth_inputs(count=1))
    assert not results["passed"]
    assert all(s["revisit"]["accuracy"] == 0 for s in results["scores"].values())
    assert all(s["transactions"] for s in results["scores"].values())


def test_recipe_frozen_screen_and_cached_resume(tmp_path):
    from experiments.multimodal import entity_growth
    from pathwm.models.entities import EntityMatchReader
    import json

    donor = tmp_path / "donor.pt"
    torch.save({"model": EntityMatchReader().state_dict()}, donor)
    output = tmp_path / "screen"
    entity_growth(donor, output)
    raw = (output / "entity_growth.json").read_bytes()
    entity_growth(donor, output, resume=True)
    assert (output / "entity_growth.json").read_bytes() == raw
    assert json.loads((output / "status.json").read_text())["result"] == "completed"
    assert "Frozen growing entity memory" in (output / "report.html").read_text()
