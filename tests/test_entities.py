import json
import pytest
import torch


def config():
    from tests.test_multimodal_training import settings

    return dict(
        settings(),
        dataset="entities",
        width=16,
        history=3,
        horizon=1,
        train_windows=64,
        validation_windows=32,
        improve_every=0,
        batch_size=8,
        evaluate_every=1,
        max_seconds=450.0,
        seed=31,
    )


def test_entity_contract_oracle_pairs_and_split():
    from pathwm.data.entities import EntityEpisodes, entity_oracle

    sets = [EntityEpisodes(s, 64) for s in ("train", "validation", "test")]
    assert not set(sets[0].descriptor_ids) & set(sets[1].descriptor_ids)
    assert not set(sets[1].descriptor_ids) & set(sets[2].descriptor_ids)
    for data in sets:
        batch = data.batch(range(len(data)))
        assert set(batch) == {"entity_inputs", "entity_targets"}
        for i, record in enumerate(data.manifest):
            expected = entity_oracle(batch["entity_inputs"][i])
            for a, b in zip(expected, batch["entity_targets"]):
                torch.testing.assert_close(a, b[i])
        for group in range(0, len(data), 32):
            x = batch["entity_inputs"][group : group + 16]
            assert torch.equal(x[-1, -1], x[0, -1])
        assert data.final_view_bounds() == dict(identity=0.5, state=0.25, effect=0.25)
    with pytest.raises(ValueError):
        EntityEpisodes("train", 33)


def test_entity_recurrent_temporal_gradient_and_no_mutation():
    from pathwm.data.entities import EntityEpisodes
    from pathwm.models.entities import EntityReader

    model = EntityReader(16)
    x = EntityEpisodes("train", 32).batch(range(8))["entity_inputs"].requires_grad_()
    before = x.detach().clone()
    scores = model(x)
    assert [list(v.shape) for v in scores] == [[8, 2], [8, 4], [8, 4]]
    sum(v.square().mean() for v in scores).backward()
    assert x.grad[:, 0].abs().sum() > 0
    assert torch.equal(before, x.detach())
    assert all(
        p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()
    )


def test_entity_resume_cache_and_final_only(tmp_path, monkeypatch):
    import experiments.multimodal as recipe
    from tests.test_runs import equal_tree

    calls = []
    predict = recipe.entity_predictions

    def capture(model, data, settings, **kwargs):
        calls.append(data.split)
        return predict(model, data, settings, **kwargs)

    monkeypatch.setattr(recipe, "entity_predictions", capture)
    recipe.train(config(), tmp_path / "resume", stop_after=1)
    assert set(calls) == {"train"}
    recipe.train(config(), tmp_path / "resume", resume=True)
    recipe.train(config(), tmp_path / "full")
    states = [
        torch.load(tmp_path / n / "last.pt", weights_only=True)
        for n in ("resume", "full")
    ]
    for state in states:
        state["model"].pop("diagnostic_elapsed_seconds")
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
    recipe.train(config(), tmp_path / "resume", resume=True)
    assert not calls
    result = json.loads((tmp_path / "resume/entity_results.json").read_text())
    assert result["final_step"] == 2
    assert "Two-object entity memory" in (tmp_path / "resume/report.html").read_text()


def test_entity_scores_oracle_and_abstention():
    from pathwm.data.entities import EntityEpisodes, entity_oracle
    from pathwm.evaluation.entities import entity_metrics

    data = EntityEpisodes("validation", 64)
    batch = data.batch(range(len(data)))
    targets = batch["entity_targets"]
    scores = tuple(t.clamp_min(1e-12).log() for t in targets)
    result = entity_metrics(scores, targets, data.cohorts, data.groups)
    assert result["gates"]["passed"]
    assert result["views"]["ambiguous"]["coverage"] == 0
    uniform = entity_metrics(
        tuple(torch.zeros_like(t) for t in targets), targets, data.cohorts, data.groups
    )
    assert not uniform["gates"]["decisions"]
    assert not uniform["gates"]["passed"]
    for x, expected in zip(batch["entity_inputs"], zip(*targets)):
        swapped = entity_oracle(x.flip(1))
        torch.testing.assert_close(swapped[0], expected[0].flip(0))
        for actual, target in zip(swapped[1:], expected[1:]):
            torch.testing.assert_close(actual, target[[0, 2, 1, 3]])
