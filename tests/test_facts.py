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


@pytest.mark.parametrize("reader", ["direct", "event"])
def test_fact_resume_cache_and_no_heldout_selection(tmp_path, monkeypatch, reader):
    import experiments.multimodal as recipe
    from tests.test_runs import equal_tree

    calls = []
    predict = recipe.fact_predictions

    def recording(model, data, settings, **kwargs):
        calls.append(data.split)
        return predict(model, data, settings, **kwargs)

    monkeypatch.setattr(recipe, "fact_predictions", recording)
    settings = dict(config(), fact_reader=reader, state_model="belief")
    recipe.train(settings, tmp_path / "resumed", stop_after=1)
    assert set(calls) == {"train"}
    recipe.train(settings, tmp_path / "resumed", resume=True)
    assert calls.count("validation") == 1
    recipe.train(settings, tmp_path / "full")
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
    recipe.train(settings, tmp_path / "resumed", resume=True)
    assert not calls
    d = json.loads((tmp_path / "resumed/fact_results.json").read_text())
    assert d["final_step"] == 2 and not d["gates"]["extraction"]
    assert d["binding"]["status"] == "skipped_failed_extraction_gate"
    assert d["reader"] == reader
    title = (
        "Direct fact extraction"
        if reader == "direct"
        else "Facts through the agent reader"
    )
    assert title in (tmp_path / "resumed/report.html").read_text()


def test_event_fact_matched_initialization_and_working_token_boundary(monkeypatch):
    from dataclasses import replace
    import experiments.multimodal as recipe

    torch.manual_seed(23)
    direct = recipe.build_model(width=16, facts=True)
    torch.manual_seed(23)
    event = recipe.build_model(
        width=16, facts=True, fact_reader="event", state_model="belief"
    )
    for key, value in direct.encoder.state_dict().items():
        torch.testing.assert_close(
            value, event.agent.encoders["text"].state_dict()[key], rtol=0, atol=0
        )
    for name in ("reader", "entity_head", "location_head"):
        for key, value in getattr(direct, name).state_dict().items():
            torch.testing.assert_close(
                value, getattr(event, name).state_dict()[key], rtol=0, atol=0
            )
    torch.testing.assert_close(direct.queries, event.queries, rtol=0, atol=0)
    data = recipe.make_data(config(), "train")
    batch = data.batch([0, 7])
    calls = []
    observe, task, think = (
        event.agent.observe,
        event.agent.task_tokens,
        event.agent.think,
    )

    def observing(state, observations, **kwargs):
        assert state.ordinal == -1 and state.observation_count == 0
        assert set(observations) == {"text"} and kwargs["time"] == 0
        calls.append("observe")
        return observe(state, observations, **kwargs)

    def interpreting(state, sessions, **kwargs):
        assert state.ordinal == 0 and state.observation_count == 1
        assert len({s.request.instruction for s in sessions}) == 1
        assert sessions[0].context_record() == sessions[1].context_record()
        assert "e00" not in sessions[0].request.instruction
        calls.append("interpret")
        return task(state, sessions, **kwargs)

    def thinking(state, **kwargs):
        assert kwargs["steps"] == 2
        calls.append("think")
        return think(state, **kwargs)

    monkeypatch.setattr(event.agent, "observe", observing)
    monkeypatch.setattr(event.agent, "task_tokens", interpreting)
    monkeypatch.setattr(event.agent, "think", thinking)
    encoded = []

    def retain_source(module, inputs, output):
        encoded.append(output)
        for scale in output.scales:
            scale.values.retain_grad()

    handle = event.agent.encoders["text"].register_forward_hook(retain_source)
    entity, location = event(batch["fact_observation"])
    handle.remove()
    assert calls == ["observe", "interpret", "think"]
    loss = torch.nn.functional.cross_entropy(
        entity, batch["entities"]
    ) + torch.nn.functional.cross_entropy(location, batch["locations"])
    loss.backward()
    assert len(encoded) == 2  # Observed fact first, constant task instruction second.
    assert any(
        scale.values.grad is not None
        and torch.isfinite(scale.values.grad).all()
        and scale.values.grad.abs().sum() > 0
        for scale in encoded[0].scales
    )
    for module in (
        event.agent.encoders["text"],
        event.agent.updater,
        event.agent.thinker,
        event.reader,
        event.entity_head,
        event.location_head,
    ):
        assert any(
            p.grad is not None
            and torch.isfinite(p.grad).all()
            and p.grad.abs().sum() > 0
            for p in module.parameters()
        )

    def erased(state, **kwargs):
        result = thinking(state, **kwargs)
        tokens = result.tokens.clone()
        tokens[:, event.agent.layout["working"]] = 0
        return replace(result, tokens=tokens)

    monkeypatch.setattr(event.agent, "think", erased)
    a, b = event(batch["fact_observation"])
    torch.testing.assert_close(a[0], a[1], rtol=0, atol=0)
    torch.testing.assert_close(b[0], b[1], rtol=0, atol=0)


def test_event_fact_evaluation_fixed_draws_preserve_training_rng():
    import experiments.multimodal as recipe

    settings = dict(config(), fact_reader="event", state_model="belief", batch_size=16)
    model = recipe.build_model(
        width=16, facts=True, fact_reader="event", state_model="belief"
    )
    data = recipe.make_data(settings, "validation")
    torch.manual_seed(8)
    before = torch.get_rng_state().clone()
    first = recipe.fact_predictions(model, data, settings)
    assert torch.equal(before, torch.get_rng_state()) and model.training
    torch.manual_seed(81)
    before = torch.get_rng_state().clone()
    second = recipe.fact_predictions(model, data, settings)
    assert torch.equal(before, torch.get_rng_state()) and model.training
    for key in first:
        torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)


def donor_checkpoint(tmp_path):
    from pathwm.models.facts import FactReader

    torch.manual_seed(101)
    donor = FactReader(width=16)
    with torch.no_grad():
        for p in donor.encoder.parameters():
            p.add_(0.03)
        for p in donor.entity_head.parameters():
            p.fill_(77)
    path = tmp_path / "donor.pt"
    torch.save(
        dict(
            schema="pathwm-run-v1",
            model={"agent." + k: v for k, v in donor.state_dict().items()},
            optimizer={"must_not_load": True},
            step=512,
        ),
        path,
    )
    return path, donor


def test_warm_encoder_transfer_boundary_rng_and_updates(tmp_path):
    import experiments.multimodal as recipe
    from pathwm.io import file_hash, state_hash, trainable_parameters

    path, donor = donor_checkpoint(tmp_path)
    torch.manual_seed(23)
    cold = recipe.build_model(
        width=16, facts=True, fact_reader="event", state_model="belief"
    )
    cold_rng = torch.get_rng_state().clone()
    torch.manual_seed(23)
    warm = recipe.build_model(
        width=16,
        facts=True,
        fact_reader="event",
        state_model="belief",
        fact_encoder_weights=path,
    )
    assert torch.equal(cold_rng, torch.get_rng_state())
    encoder = warm.agent.encoders["text"]
    for key, value in donor.encoder.state_dict().items():
        torch.testing.assert_close(value, encoder.state_dict()[key], rtol=0, atol=0)
    for key, value in cold.state_dict().items():
        if not key.startswith("agent.encoders.text."):
            torch.testing.assert_close(value, warm.state_dict()[key], rtol=0, atol=0)
    assert all(p.requires_grad for p in encoder.parameters())
    assert warm.encoder_initialization["sha256"] == file_hash(path)
    assert warm.encoder_initialization["encoder_sha256"] == state_hash(encoder)
    before = state_hash(encoder)
    data = recipe.make_data(config(), "train")
    learner = recipe.LearningState(warm, len(data))
    optimizer = torch.optim.AdamW(trainable_parameters(learner), lr=0.0003)
    assert not optimizer.state
    recipe.update(learner, optimizer, data, [0, 7, 10, 22], config())
    assert state_hash(encoder) != before
    with pytest.raises(ValueError, match="event"):
        recipe.build_model(width=16, facts=True, fact_encoder_weights=path)


def test_warm_encoder_resume_cache_and_changed_donor_rejection(tmp_path, monkeypatch):
    import experiments.multimodal as recipe
    from tests.test_runs import equal_tree

    path, _ = donor_checkpoint(tmp_path)
    settings = dict(
        config(),
        fact_reader="event",
        state_model="belief",
        fact_encoder_weights=str(path),
    )
    recipe.train(settings, tmp_path / "resumed", stop_after=1)
    recipe.train(settings, tmp_path / "resumed", resume=True)
    recipe.train(settings, tmp_path / "full")
    states = [
        torch.load(tmp_path / n / "last.pt", weights_only=True)
        for n in ("resumed", "full")
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

    def forbidden(*args, **kwargs):
        raise AssertionError("Completed resume must reuse cached predictions")

    monkeypatch.setattr(recipe, "fact_predictions", forbidden)
    recipe.train(settings, tmp_path / "resumed", resume=True)
    before = (tmp_path / "resumed/last.pt").read_bytes()
    donor = torch.load(path, weights_only=True)
    donor["model"]["agent.encoder.stem.embedding.weight"] += 1
    torch.save(donor, path)
    with pytest.raises(ValueError, match="Incompatible resume"):
        recipe.train(settings, tmp_path / "resumed", resume=True)
    assert (tmp_path / "resumed/last.pt").read_bytes() == before
