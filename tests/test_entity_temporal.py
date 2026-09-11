from pathwm.data.entity_temporal import temporal_episodes
from pathwm.evaluation.entity_growth import growth_inputs
from tests.test_entity_memory import Scorer


def test_temporal_controls_and_order_targets():
    data = temporal_episodes(Scorer(), growth_inputs(111, 1))
    assert {k: v["observations"].shape[1] for k, v in data.items()} == {
        "reference": 6,
        "reset_order": 6,
        "toggle_length": 20,
        "no_information_length": 20,
    }
    for cohort in data.values():
        assert {tuple(e["target"]) for e in cohort["manifest"]} == {
            (0, 0),
            (0, 1),
            (1, 0),
            (1, 1),
        }
        assert len({str(e["descriptors"][-2:]) for e in cohort["manifest"]}) == 1
    rows = data["reset_order"]["manifest"]
    for left, right in zip(rows[::2], rows[1::2]):
        assert sorted(left["operations"]) == sorted(right["operations"])
        assert left["target"][0] != right["target"][0]


def test_temporal_recipe_cache(tmp_path, monkeypatch):
    import torch
    from pathwm.models.entities import EntityMatchReader
    from pathwm.models.entity_state import EntityStateCell
    from pathwm.evaluation import entity_growth
    from experiments.multimodal import evaluate_entity_temporal

    original = entity_growth.growth_inputs
    monkeypatch.setattr(
        entity_growth, "growth_inputs", lambda seed, count: original(seed, 1)
    )
    matcher = tmp_path / "matcher.pt"
    cell = tmp_path / "cell.pt"
    torch.save(
        {
            "model": {
                "agent." + k: v for k, v in EntityMatchReader().state_dict().items()
            }
        },
        matcher,
    )
    torch.save({"model": EntityStateCell().state_dict()}, cell)
    output = tmp_path / "evaluation"
    evaluate_entity_temporal(matcher, cell, output)
    raw = (output / "entity_temporal.json").read_bytes()
    evaluate_entity_temporal(matcher, cell, output, resume=True)
    assert (output / "entity_temporal.json").read_bytes() == raw
