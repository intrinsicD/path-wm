import pytest
import torch

from pathwm.models.tasks import Actor, TaskRequest
from pathwm.models.recall import (
    SeenRecord,
    RecallQuery,
    RecallDecision,
    historical_target,
    select_recall,
    verify_recall,
)
from pathwm.evaluation.recall import fit_temperature, recall_metrics


def query(cutoff=3, entity=7):
    return RecallQuery(
        TaskRequest("q", "Where was it last seen?", Actor("user", "test")),
        "session",
        entity,
        cutoff,
    )


def test_historical_cutoff_and_verification_do_not_conflate_absence_with_abstention():
    records = (SeenRecord(1, 7, 2), SeenRecord(2, 1, 0), SeenRecord(3, 7, 1))
    assert historical_target(records, query(1)) == 2
    assert historical_target(records, query()) == 1
    assert historical_target(records, query(entity=8)) == 4
    with pytest.raises(ValueError, match="ordered"):
        historical_target(records[::-1], query())
    decision = select_recall(torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0]), query())
    assert (
        verify_recall(query(), decision, records, "session")["verification"]
        == "contradicted"
    )
    with pytest.raises(ValueError, match="session"):
        verify_recall(query(), decision, records, "different")


def test_exact_cost_boundary_invalid_probabilities_and_zero_answer_denominator():
    q = query()
    tied = select_recall(torch.tensor([0.75, 0.25, 0.0, 0.0, 0.0]), q)
    assert tied.answer is None and tied.reason == "cost"
    assert select_recall(torch.tensor([0.751, 0.249, 0.0, 0.0, 0.0]), q).answer == 0
    assert select_recall(torch.tensor([float("nan")] * 5), q).reason == "invalid"
    metrics = recall_metrics(torch.zeros(2, 5), torch.tensor([0, 4]))
    assert metrics["coverage"] == 0 and metrics["answered_error"] is None
    assert metrics["task_loss"] == 0.25


def test_temperature_fits_only_supplied_labels_and_preserves_factual_argmax():
    logits = torch.tensor(
        [
            [4.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 4.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 4.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 4.0, 0.0],
        ]
    )
    labels = torch.tensor([0, 2, 2, 0])
    before = logits.clone()
    fit = fit_temperature(logits, labels)
    assert fit["status"] == "fitted" and fit["nll_after"] <= fit["nll_before"]
    assert torch.equal(logits, before)
    assert torch.equal(logits.argmax(-1), (logits / fit["temperature"]).argmax(-1))
    assert not logits.requires_grad
    failed = fit_temperature(torch.full((2, 5), float("nan")), labels[:2])
    assert failed["status"] == "failed" and failed["temperature"] == 1.0


def test_complete_history_and_safe_contract_roundtrip(tmp_path):
    q = query()
    decision = select_recall(torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0]), q)
    torch.save(
        dict(query=q.to_dict(), decision=decision.to_dict()), tmp_path / "result.pt"
    )
    data = torch.load(tmp_path / "result.pt", weights_only=True)
    assert RecallQuery.from_dict(data["query"]) == q
    assert RecallDecision.from_dict(data["decision"]) == decision
    with pytest.raises(ValueError, match="complete"):
        verify_recall(q, decision, (SeenRecord(3, 7, 0),), "session")


def fixture(split="train"):
    from experiments.multimodal import RecallEpisodes

    return RecallEpisodes(
        split=split, count=15, history=16, recent=2, block=2, blocks=2, truncate=4
    )


def test_episode_population_and_inference_cannot_read_evaluator_fields(monkeypatch):
    from dataclasses import replace
    import experiments.multimodal as recipe
    from tests.test_runs import equal_tree

    data, other = fixture(), fixture("test")
    assert data.identity["class_counts"] == [3] * 5
    assert data.identity["groups"] == dict(
        recent=4, compressed=4, consolidated=4, not_observed=3
    )
    assert {x["query"].session_id for x in data.items}.isdisjoint(
        x["query"].session_id for x in other.items
    )
    model = recipe.build_model(
        width=16,
        state_model="belief",
        recall=True,
        memory_recent=2,
        memory_block=2,
        memory_blocks=2,
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("Evaluator was called by inference")

    monkeypatch.setattr(recipe, "historical_target", forbidden)
    original = data.items[0]
    torch.manual_seed(31)
    with torch.no_grad():
        first, state, _ = recipe.recall_forward(model, original)
    data.hidden_locations = [[99]] * 15
    data.groups = ["altered"] * 15
    task = replace(
        original["query"].task, task_id="different", instruction="Ignore all evidence"
    )
    changed = dict(original, query=replace(original["query"], task=task))
    torch.manual_seed(31)
    with torch.no_grad():
        second, after, _ = recipe.recall_forward(model, changed)
    assert torch.equal(first, second)
    equal_tree(state.memory.to_dict(), after.memory.to_dict())


def test_truncation_preserves_forward_memory_and_final_segment_gradient_routes():
    from experiments.multimodal import build_model, recall_forward
    from tests.test_runs import equal_tree

    model = build_model(
        width=16,
        state_model="belief",
        recall=True,
        memory_recent=2,
        memory_block=2,
        memory_blocks=2,
    )
    inputs = fixture().items[0]
    torch.manual_seed(91)
    full, a, _ = recall_forward(model, inputs, truncate=0)
    torch.manual_seed(91)
    short, b, aux = recall_forward(model, inputs, truncate=4, auxiliary=True)
    assert torch.equal(full, short)
    equal_tree(a.memory.to_dict(), b.memory.to_dict())
    assert b.memory.consolidated is not None
    assert len(b.memory.recent) == 2 and len(b.memory.compressed) == 2
    assert model.memory.storage_bytes(b.memory) <= model.memory.capacity_bytes(
        16, 8, 8, 8
    )
    (short.square().mean() + aux.sum()).backward()
    for module in (
        model.recall_head,
        model.task_interpreter,
        model.thinker,
        model.updater,
        model.dynamics,
        model.encoders["text"],
        model.evidence_encoder,
        model.memory.compressors,
        model.memory.consolidators,
    ):
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters()
        ), type(module).__name__
    assert all(
        p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()
    )


def test_recall_resume_selects_same_weights_and_caches_heldout_results(
    tmp_path, monkeypatch
):
    import json
    import experiments.multimodal as recipe
    from tests.test_multimodal_training import settings
    from tests.test_runs import equal_tree

    config = dict(
        settings(),
        dataset="recall",
        state_model="belief",
        history=16,
        memory_recent=2,
        memory_block=2,
        memory_blocks=2,
        recall_truncate=4,
        train_windows=15,
        validation_windows=15,
        calibration_windows=15,
        test_windows=15,
        improve_every=0,
        batch_size=1,
        abstain_cost=0.25,
    )
    calls = []
    predict = recipe.recall_predictions

    def recording(model, data, settings):
        calls.append(data.split)
        return predict(model, data, settings)

    monkeypatch.setattr(recipe, "recall_predictions", recording)
    recipe.train(config, tmp_path / "resume", stop_after=1)
    assert set(calls) == {"validation"}
    assert not (tmp_path / "resume/recall_predictions.pt").exists()
    calls.clear()
    recipe.train(config, tmp_path / "resume", resume=True)
    assert calls.count("calibration") == calls.count("test") == 1
    recipe.train(config, tmp_path / "full")
    full, resumed = [
        torch.load(tmp_path / name / "last.pt", weights_only=True)
        for name in ("full", "resume")
    ]
    for key in (
        "model",
        "optimizer",
        "sampler",
        "torch",
        "step",
        "rows",
        "random",
        "numpy",
    ):
        equal_tree(full[key], resumed[key])
    results = [
        json.loads((tmp_path / name / "recall_results.json").read_text())
        for name in ("full", "resume")
    ]
    assert results[0] == results[1]
    calls.clear()
    original_report = recipe.write_report

    def broken(*args, **kwargs):
        raise RuntimeError("injected report failure")

    monkeypatch.setattr(recipe, "write_report", broken)
    with pytest.raises(RuntimeError, match="injected"):
        recipe.train(config, tmp_path / "resume", resume=True)
    status = json.loads((tmp_path / "resume/status.json").read_text())
    assert status["result"] == "completed" and status["report"] == "failed"
    monkeypatch.setattr(recipe, "write_report", original_report)
    recipe.train(config, tmp_path / "resume", resume=True)
    assert not calls
    assert (
        json.loads((tmp_path / "resume/status.json").read_text())["report"]
        == "structural_verified"
    )
    assert (
        "Selective historical recall" in (tmp_path / "resume/report.html").read_text()
    )


def diagnostic_settings():
    from tests.test_multimodal_training import settings

    return dict(
        settings(), dataset="recall", state_model="belief", recall_mode="current-recent",
        history=4, memory_recent=2, memory_block=2, memory_blocks=2,
        recall_truncate=4, train_windows=10, validation_windows=10,
        improve_every=0, batch_size=1, evaluate_every=1, max_seconds=900.0,
    )


def test_current_recent_fixtures_balance_bindings_and_exclude_split_duplicates():
    import experiments.multimodal as recipe

    train, dev = [recipe.make_data(diagnostic_settings(), s) for s in ("train", "validation")]
    signatures = []
    for data in (train, dev):
        assert data.identity["class_counts"] == [2] * 5
        assert data.identity["cohorts"] == {"current": 5, "recent": 5}
        signatures.append(set())
        for item, cohort in zip(data.items, data.cohorts):
            records, q = item["records"], item["query"]
            label = historical_target(records, q)
            matches = [r for r in records if r.entity == q.entity]
            if label < 4:
                assert matches[-1].ordinal == (4 if cohort == "current" else 3)
                assert matches[0].location != matches[-1].location
                if cohort == "recent":
                    assert records[-1].entity != q.entity
            else:
                assert not matches
            signatures[-1].add((q.entity, tuple((r.entity, r.location) for r in records)))
        assert len(signatures[-1]) == len(data)
    assert signatures[0].isdisjoint(signatures[1])
    with pytest.raises(ValueError, match="diagnostic"):
        recipe.make_data(diagnostic_settings(), "test")


def test_diagnostic_final_weights_resume_and_no_calibration_or_test(tmp_path, monkeypatch):
    import json
    import experiments.multimodal as recipe
    from pathwm.io import state_hash
    from tests.test_runs import equal_tree

    config = diagnostic_settings()
    calls = []
    predict = recipe.recall_predictions

    def recording(model, data, settings, **kwargs):
        calls.append((data.split, state_hash(model)))
        assert data.split in ("train", "validation")
        return predict(model, data, settings, **kwargs)

    monkeypatch.setattr(recipe, "recall_predictions", recording)
    monkeypatch.setattr(recipe, "fit_temperature", lambda *a: pytest.fail("diagnostic calibrated"))
    recipe.train(config, tmp_path / "paused", stop_after=1)
    assert all(s == "train" for s, _ in calls)
    recipe.train(config, tmp_path / "paused", resume=True)
    result = json.loads((tmp_path / "paused/recall_diagnostic.json").read_text())
    assert result["final_step"] == 2
    assert [h for s, h in calls if s == "validation"] == [result["model_sha256"]]
    calls.clear()
    recipe.train(config, tmp_path / "full")
    full, resumed = [torch.load(tmp_path / n / "last.pt", weights_only=True) for n in ("full", "paused")]
    for state in (full, resumed):
        state["model"].pop("diagnostic_elapsed_seconds")
    for key in ("model", "optimizer", "sampler", "torch", "rows", "step", "numpy", "random"):
        equal_tree(full[key], resumed[key])
    calls.clear()
    monkeypatch.setattr(recipe, "write_report", lambda *a: (_ for _ in ()).throw(RuntimeError("report failure")))
    with pytest.raises(RuntimeError, match="report failure"):
        recipe.train(config, tmp_path / "paused", resume=True)
    status = json.loads((tmp_path / "paused/status.json").read_text())
    assert status["result"] == "completed" and status["report"] == "failed"
    assert not calls
    assert not (tmp_path / "paused/recall_results.json").exists()
    assert result["gates"]["advance_to_memory_comparison"] is False


def test_diagnostic_budget_stops_and_cannot_reset_on_resume(tmp_path):
    import json
    import experiments.multimodal as recipe

    config = dict(diagnostic_settings(), max_seconds=1e-12)
    recipe.train(config, tmp_path / "limited")
    for resume in (False, True):
        if resume:
            recipe.train(config, tmp_path / "limited", resume=True)
        status = json.loads((tmp_path / "limited/status.json").read_text())
        assert status["result"] == "stopped" and status["step"] == 0
        assert status["report"] == "structural_verified"
        assert not (tmp_path / "limited/recall_diagnostic_predictions.pt").exists()
