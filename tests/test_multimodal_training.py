import copy
import json

import pytest
import torch

from experiments.multimodal import (
    build_model,
    SyntheticEpisodes,
    LearningState,
    objective,
    observe_history,
    train,
)
from pathwm.models.modalities import Observation
from pathwm.training.improvement import try_improvement, replay_probabilities
from tests.test_runs import equal_tree


def settings():
    return dict(
        seed=17,
        dataset="synthetic",
        data_root="unused",
        width=16,
        image_size=16,
        audio_samples=32,
        history=2,
        horizon=2,
        train_windows=4,
        validation_windows=3,
        batch_size=2,
        steps=2,
        evaluate_every=2,
        improve_every=2,
        ema_decay=0.99,
        learning_rate=3e-4,
        device="cpu",
        purpose="development",
    )


def test_training_target_is_detached_and_inference_never_reads_future_observations():
    torch.manual_seed(31)
    data = SyntheticEpisodes(count=2)
    batch = data.batch([0, 1])
    learner = LearningState(build_model(width=16), len(data))
    losses, _, _ = objective(learner, batch)
    sum(losses.values()).backward()
    assert all(p.grad is None for p in learner.target.parameters())
    for name in ("updater", "dynamics", "thinker", "action_head", "monitor"):
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in getattr(learner.agent, name).parameters()
        ), name
    batch["images"].requires_grad_()
    state = observe_history(learner.agent, batch, 2)
    future = learner.agent.imagine(state, batch["actions"][:, 1])
    future.tokens.square().mean().backward()
    assert batch["images"].grad[:, :2].abs().sum() > 0
    assert batch["images"].grad[:, 2:].abs().sum() == 0
    altered = {
        k: v.clone() if isinstance(v, torch.Tensor) else list(v)
        for k, v in batch.items()
    }
    for name in ("images", "audio", "text"):
        altered[name][:, 2:] = 0
    second = learner.agent.imagine(
        observe_history(learner.agent, altered, 2), batch["actions"][:, 1]
    )
    torch.testing.assert_close(future.tokens, second.tokens, rtol=0, atol=0)


def test_masked_observations_have_zero_gradient_and_memory_is_bounded_per_stream():
    model = build_model(width=16)
    values = torch.rand(2, 2, 3, 16, 16, requires_grad=True)
    valid = torch.tensor([[True, False], [False, True]])
    state = model.initial_state(2)
    with pytest.raises(ValueError, match="unobserved"):
        model.remember(state, source="initial is not an observation")
    state = model.observe(
        state, {"video": Observation(values, torch.zeros(2, 2), valid)}, time=0
    )
    state.tokens.square().mean().backward()
    assert values.grad[~valid].count_nonzero() == 0
    assert values.grad[valid].abs().sum() > 0
    model.memory.capacity = 2
    for t in range(3):
        state = model.observe(
            state,
            {"image": Observation(values[:, :1], torch.full((2, 1), float(t)))},
            time=t,
        )
        state = model.remember(state, source=f"frame-{t}")
    assert state.memory.sources == ("frame-1", "frame-2")
    assert state.memory.values.shape[1] == 2
    assert state.memory.times.tolist() == [[1.0, 2.0], [1.0, 2.0]]
    assert not torch.equal(state.memory.values[0], state.memory.values[1])
    before = state.memory.to_dict()
    model.think(state, steps=3)
    model.imagine(state, None)
    equal_tree(state.memory.to_dict(), before)


def test_noop_and_exception_proposals_restore_teacher_replay_rng_and_gradients():
    learner = LearningState(build_model(width=16), 4)
    optimizer = torch.optim.AdamW(learner.agent.parameters())
    generator = torch.Generator().manual_seed(52)
    before = copy.deepcopy(learner.state_dict())
    rng = generator.get_state().clone()
    monitor_input = torch.randn(2, 30, 16, requires_grad=True)
    learner.agent.monitor(monitor_input).sum().backward()
    assert monitor_input.grad is None
    gradients = [
        None if p.grad is None else p.grad.clone() for p in learner.parameters()
    ]

    def evaluate():
        return {"error": 1.0}

    result = try_improvement(
        learner,
        optimizer,
        lambda: None,
        evaluate,
        primary="error",
        min_improvement=0.01,
        tolerances={"error": 0.0},
    )
    assert not result["accepted"]

    def fail():
        with torch.no_grad():
            learner.target.initial.add_(1)
            learner.replay_errors.add_(3)
            learner.agent.initial.add_(2)
        torch.rand(3, generator=generator)
        learner.zero_grad(set_to_none=True)
        raise RuntimeError("interrupted proposal")

    with pytest.raises(RuntimeError, match="interrupted"):
        try_improvement(
            learner,
            optimizer,
            fail,
            evaluate,
            primary="error",
            min_improvement=0.01,
            tolerances={"error": 0.0},
            generators=(generator,),
        )
    equal_tree(before, learner.state_dict())
    assert torch.equal(rng, generator.get_state())
    equal_tree(gradients, [p.grad for p in learner.parameters()])
    priority = replay_probabilities(torch.tensor([0.0, 0.0, 100.0, 0.0]))
    assert priority[2] > priority[0] > 0 and priority.sum() == 1


def test_multimodal_resume_includes_teacher_replay_admission_and_reports(tmp_path):
    config = settings()
    train(config, tmp_path / "full")
    train(config, tmp_path / "resumed", stop_after=1)
    train(config, tmp_path / "resumed", resume=True)
    full, resumed = [
        torch.load(tmp_path / name / "last.pt", weights_only=True)
        for name in ("full", "resumed")
    ]
    for name in ("model", "optimizer", "sampler", "torch", "step"):
        equal_tree(full[name], resumed[name])
    assert [r for r in full["rows"] if r["split"] == "train"] == [
        r for r in resumed["rows"] if r["split"] == "train"
    ]
    assert [r for r in full["rows"] if r["split"] == "proposal"] == [
        r for r in resumed["rows"] if r["split"] == "proposal"
    ]
    for name in (
        "report.html",
        "inspection.pt",
        "inspection.json",
        "imagined.gif",
        "imagined.wav",
    ):
        assert (tmp_path / "resumed" / name).is_file()
    assert (
        json.loads((tmp_path / "resumed/status.json").read_text())["result"]
        == "completed"
    )
    inspection = torch.load(tmp_path / "resumed/inspection.pt", weights_only=True)
    assert all(state["imagined"] for state in inspection["future"])
    assert inspection["state"]["observation_count"] == 2
    with pytest.raises(FileExistsError):
        train(config, tmp_path / "full")
    with pytest.raises(ValueError, match="Incompatible resume"):
        train(dict(config, horizon=1), tmp_path / "full", resume=True)


def test_multimodal_report_failure_preserves_completed_result(tmp_path, monkeypatch):
    import experiments.multimodal as recipe

    def fail(*args, **kwargs):
        raise RuntimeError("injected report failure")

    config = dict(settings(), steps=1, improve_every=0)
    with monkeypatch.context() as scoped:
        scoped.setattr(recipe, "save_examples", fail)
        with pytest.raises(RuntimeError, match="injected"):
            train(config, tmp_path / "report_failure")
    status = json.loads((tmp_path / "report_failure/status.json").read_text())
    assert status["result"] == "completed" and status["report"] == "failed"
    assert (tmp_path / "report_failure/last.pt").is_file()
    train(config, tmp_path / "report_failure", resume=True)
    assert (
        json.loads((tmp_path / "report_failure/status.json").read_text())["report"]
        == "structural_verified"
    )
