import json

import pytest
import torch


def config():
    from tests.test_multimodal_training import settings

    return dict(
        settings(),
        dataset="facts",
        width=16,
        train_windows=96,
        validation_windows=32,
        improve_every=0,
        batch_size=4,
        evaluate_every=1,
        max_seconds=450.0,
    )


def test_semantic_pair_split_and_forward_boundary():
    import experiments.multimodal as recipe
    from pathwm.models.facts import FactReader

    train, dev = [recipe.make_data(config(), s) for s in ("train", "validation")]
    a, b = [set(zip(d.entities, d.locations)) for d in (train, dev)]
    assert len(a) == 96 and len(b) == 32 and not a & b
    assert len(a | b) == 128
    assert {e for e, _ in a} == set(range(32))
    assert {loc for _, loc in a} == set(range(4))
    assert all((e + loc) % 4 == 0 for e, loc in b)
    with pytest.raises(ValueError, match="train/development"):
        recipe.make_data(config(), "test")
    batch = train.batch([0, 1, 2])
    model = FactReader(width=16)
    entity, location = model(batch["fact_observation"])
    assert entity.shape == (3, 32) and location.shape == (3, 4)
    loss = torch.nn.functional.cross_entropy(
        entity, batch["entities"]
    ) + torch.nn.functional.cross_entropy(location, batch["locations"])
    loss.backward()
    for module in (model.encoder, model.reader, model.entity_head, model.location_head):
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters()
        )
    assert all(
        p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()
    )


def test_binding_selector_numeric_and_query_location_order_controls():
    from pathwm.models.facts import bind_facts

    entity = torch.full((1, 2, 32), -30.0)
    entity[0, 0, 3] = entity[0, 1, 8] = 30
    location = torch.full((1, 2, 4), -30.0)
    location[0, 0, 1] = location[0, 1, 2] = 30
    first = bind_facts(entity, location, torch.tensor([3]))
    second = bind_facts(entity, location, torch.tensor([8]))
    assert first.argmax(-1).item() == 1 and second.argmax(-1).item() == 2
    torch.testing.assert_close(
        first, bind_facts(entity.flip(1), location.flip(1), torch.tensor([3]))
    )
    assert (
        bind_facts(entity, location.flip(1), torch.tensor([3])).argmax(-1).item() == 2
    )
    # Uninformative entity probabilities give the arithmetic location mixture.
    actual = bind_facts(torch.zeros_like(entity), location, torch.tensor([3])).exp()
    torch.testing.assert_close(actual, location.double().softmax(-1).mean(1))
    with pytest.raises(ValueError):
        bind_facts(entity, location, torch.tensor([32]))


def test_exhaustive_binding_reference_perfect_logits():
    from pathwm.evaluation.facts import binding_reference

    entities = torch.arange(32).repeat_interleave(4)
    locations = torch.arange(4).repeat(32)
    entity = torch.full((128, 32), -30.0).scatter_(1, entities[:, None], 30)
    location = torch.full((128, 4), -30.0).scatter_(1, locations[:, None], 30)
    summary, raw = binding_reference(entity, location, entities, locations)
    assert raw["cases"].shape == (5952, 4)
    assert raw["queries"].shape == (11904,)
    assert summary["gate"] and summary["coherent_swap_success"] == 1
    assert summary["order_max_log_probability_delta"] == 0
    for groups in (summary["groups"], summary["swapped_groups"]):
        assert sum(g["pairs"] for g in groups.values()) == 5952
        assert sum(g["queries"] for g in groups.values()) == 11904
        assert all(g["accuracy"] == g["paired_success"] == 1 for g in groups.values())
        assert all(
            g["constituent_baseline"]["factual_accuracy"] == 1
            and g["constituent_baseline"]["examples"] == g["queries"]
            for g in groups.values()
        )


def test_fact_resume_cache_and_no_heldout_selection(tmp_path, monkeypatch):
    import experiments.multimodal as recipe
    from tests.test_runs import equal_tree

    calls = []
    predict = recipe.fact_predictions

    def recording(model, data, settings, **kwargs):
        calls.append(data.split)
        return predict(model, data, settings, **kwargs)

    monkeypatch.setattr(recipe, "fact_predictions", recording)
    recipe.train(config(), tmp_path / "resumed", stop_after=1)
    assert set(calls) == {"train"}
    recipe.train(config(), tmp_path / "resumed", resume=True)
    assert calls.count("validation") == 1
    recipe.train(config(), tmp_path / "full")
    states = [
        torch.load(tmp_path / n / "last.pt", weights_only=True)
        for n in ("resumed", "full")
    ]
    for s in states:
        s["model"].pop("diagnostic_elapsed_seconds")
    for key in (
        "model",
        "optimizer",
        "sampler",
        "torch",
        "numpy",
        "random",
        "rows",
        "step",
    ):
        equal_tree(states[0][key], states[1][key])
    calls.clear()
    recipe.train(config(), tmp_path / "resumed", resume=True)
    assert not calls
    d = json.loads((tmp_path / "resumed/fact_results.json").read_text())
    assert d["final_step"] == 2 and not d["gates"]["extraction"]
    assert d["binding"]["status"] == "skipped_failed_extraction_gate"
    assert "Direct fact extraction" in (tmp_path / "resumed/report.html").read_text()
