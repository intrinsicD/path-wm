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


@pytest.mark.parametrize(
    "association,reader",
    [
        ("raw", "recurrent"),
        ("observed", "recurrent"),
        ("observed", "shared"),
        ("learned", "shared"),
    ],
)
def test_entity_resume_cache_and_final_only(tmp_path, monkeypatch, association, reader):
    import experiments.multimodal as recipe
    from tests.test_runs import equal_tree

    configuration = dict(config(), entity_association=association, entity_reader=reader)
    calls = []
    predict = recipe.entity_predictions

    def capture(model, data, settings, **kwargs):
        calls.append(data.split)
        return predict(model, data, settings, **kwargs)

    monkeypatch.setattr(recipe, "entity_predictions", capture)
    recipe.train(configuration, tmp_path / "resume", stop_after=1)
    assert set(calls) == {"train"}
    recipe.train(configuration, tmp_path / "resume", resume=True)
    recipe.train(configuration, tmp_path / "full")
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
    recipe.train(configuration, tmp_path / "resume", resume=True)
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


def test_observed_association_input_only_and_missingness():
    from pathwm.models.entities import observed_association
    from pathwm.data.entities import EntityEpisodes

    x = EntityEpisodes("train", 32).inputs
    before = x.clone()
    y = observed_association(x)
    assert torch.equal(x, before)
    assert torch.equal(y[..., 8:], x[..., 8:])
    torch.testing.assert_close(y[:, 0, :, :2], torch.eye(2).expand(32, -1, -1))
    torch.testing.assert_close(y[16:, -1, :, :2], torch.full((16, 2, 2), 0.5))
    assert torch.count_nonzero(y[..., 2:8]) == 0
    swap = observed_association(x.flip(2))
    torch.testing.assert_close(swap[..., :2], y.flip(2)[..., [1, 0]])


@pytest.mark.parametrize("association", ["observed", "learned"])
def test_shared_entity_permutations_and_gradients(association):
    import itertools
    from pathwm.models.entities import SharedEntityReader
    from pathwm.data.entities import EntityEpisodes

    model = SharedEntityReader(16, association)
    x = EntityEpisodes("validation", 32).inputs.clone().requires_grad_()
    original = x.detach().clone()
    base = model(x)
    torch.testing.assert_close(base[0][16:].exp(), torch.full((16, 2), 0.5))
    torch.testing.assert_close(base[1][16:, 1].exp(), base[1][16:, 2].exp())
    assert torch.equal(x.detach(), original)
    for order in itertools.product([False, True], repeat=3):
        moved = torch.stack(
            [x[:, t].flip(1) if flip else x[:, t] for t, flip in enumerate(order)], 1
        )
        for j, (a, b) in enumerate(zip(base, model(moved))):
            permutation = [1, 0] if j == 0 else [0, 2, 1, 3]
            if order[-1]:
                b = b[:, permutation]
            torch.testing.assert_close(
                a.softmax(-1), b.softmax(-1), atol=1e-6, rtol=1e-5
            )
            torch.testing.assert_close(a.exp().sum(-1), torch.ones(len(x)))
    sum(v.square().mean() for v in base).backward()
    assert x.grad[:, 0, :, 9].abs().sum() > 0
    assert all(
        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
    )


def test_learned_association_task_gradients_and_no_exact_lookup(monkeypatch):
    import pathwm.models.entities as module
    from pathwm.data.entities import EntityEpisodes

    model = module.SharedEntityReader(16, association="learned")
    x = EntityEpisodes("validation", 32).inputs

    def forbidden(*args):
        raise AssertionError("Learned path called exact matching")

    monkeypatch.setattr(module, "observed_association", forbidden)
    output = model(x)
    targets = EntityEpisodes("validation", 32).targets
    loss = sum(-(t * p.log_softmax(-1)).sum(-1).mean() for p, t in zip(output, targets))
    loss.backward()
    assert sum(p.grad.abs().sum() for p in model.matcher.parameters()) > 0
    for p in output:
        torch.testing.assert_close(p.exp().sum(-1), torch.ones(len(x)))
    torch.testing.assert_close(
        model.assignment_weights(x)[16:, -1], torch.full((16, 2), 0.5)
    )
