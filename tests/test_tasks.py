import io
from dataclasses import replace

import pytest
import torch

from experiments.multimodal import build_model, InstructionEpisodes, LearningState, objective
from pathwm.models.agent_state import LatentState
from pathwm.models.modalities import Observation
from pathwm.models.tasks import (
    Actor, OutputControl, TaskRequest, TaskSession, OutputRequest,
    TaskPrediction, OPERATIONS,
)
from tests.test_runs import equal_tree


USER = Actor("user", "alex")
AGENT = Actor("agent", "model-1")


def task(*controls):
    return TaskSession(TaskRequest("task-1", "Explain this scene", USER, controls))


def request(modality, actor=AGENT, purpose="answer"):
    return OutputRequest("out-" + modality, modality, actor, "task-1", purpose)


def observed(model):
    return model.observe(model.initial_state(1), {
        "image": Observation(torch.rand(1, 1, 3, 16, 16), torch.zeros(1, 1))
    }, time=0)


def test_requester_is_independent_from_producer_and_automatic_control_author():
    model = build_model(width=16).eval()
    session = task(OutputControl("image", "required", USER),
                   OutputControl("audio", "automatic", USER))
    prediction = TaskPrediction(torch.tensor([[0., 0, 0, 0, 8, 0, 0]]),
                                torch.tensor([[2., 2., -2., -2.]]), torch.zeros(1),
                                ("image", "audio", "text", "video"))
    decision = prediction.select(session, AGENT)
    assert {r.modality: r.requested_by for r in decision.requests} == {
        "image": USER, "audio": AGENT}
    result = model.emit(observed(model), session, decision.requests, produced_by=AGENT)
    assert not result.errors and not result.session.remaining
    assert all(o.provenance.produced_by == AGENT for o in result.outputs)
    assert result.outputs[0].provenance.request.in_response_to == "task-1"
    assert session.remaining == ("image",)  # caller was not mutated


def test_full_prevalidation_and_partial_failure_with_exact_pending_outputs():
    model = build_model(width=16).eval()
    state = observed(model)
    session = task(OutputControl("image", "required", USER),
                   OutputControl("audio", "required", USER),
                   OutputControl("text", "disabled", USER))
    calls = []
    handle = model.decoders["image"].register_forward_hook(lambda *args: calls.append(1))
    with pytest.raises(ValueError, match="disabled"):
        model.emit(state, session, (request("image", USER), request("text")), produced_by=AGENT)
    assert not calls
    with pytest.raises(ValueError, match="requester"):
        model.emit(state, session, (request("image"),), produced_by=AGENT)
    class Broken(torch.nn.Module):
        def forward(self, *args, **kwargs):
            raise RuntimeError("decoder failed")
    model.decoders["audio"] = Broken()
    result = model.emit(state, session, (request("image", USER), request("audio", USER)), produced_by=AGENT)
    handle.remove()
    assert len(result.outputs) == 1 and "out-audio" in result.errors
    assert result.session.remaining == ("audio",)
    assert session.remaining == ("image", "audio")
    candidate = model.emit(state, session, (request("image", USER, "candidate"),), produced_by=AGENT)
    assert candidate.session.remaining == ("image", "audio")


def test_video_internal_image_dependency_and_new_output_adapter():
    model = build_model(width=16).eval()
    state = observed(model)
    session = task(OutputControl("image", "disabled", USER), OutputControl("video", "required", USER))
    with pytest.raises(ValueError, match="trajectory"):
        model.emit(state, session, (request("video", USER),), produced_by=AGENT)
    result = model.emit(state, session, (request("video", USER),), produced_by=AGENT,
                        video_states=(state, model.imagine(state)))
    assert result.outputs[0].values.shape == (1, 2, 3, 16, 16)
    assert not result.session.remaining
    class Force(torch.nn.Module):
        def forward(self, tokens, trace=None):
            return tokens.mean(1)[:, :3]
    model.decoders["force"] = Force()
    force = model.emit(state, task(), (request("force"),), produced_by=AGENT)
    assert force.outputs[0].values.shape == (1, 3)


def test_gates_hold_for_arbitrary_logits_and_finish_requires_fulfillment():
    torch.manual_seed(19)
    session = task(OutputControl("image", "required", USER), OutputControl("audio", "disabled", USER))
    for _ in range(32):
        raw = TaskPrediction(torch.randn(1, 7), torch.randn(1, 4) * 10,
                             torch.randn(1), ("image", "audio", "text", "video"))
        selection = raw.select(session, AGENT)
        assert "audio" not in [r.modality for r in selection.requests]
        assert selection.operation != "finish"
        if selection.operation == "emit":
            assert "image" in [r.modality for r in selection.requests]
        assert selection.raw_operation == OPERATIONS[int(raw.operation_logits.argmax(-1))]


def test_loopback_preserves_world_state_and_cannot_be_laundered_into_memory():
    model = build_model(width=16).eval()
    state = model.remember(observed(model), source="camera")
    session = task()
    output = model.emit(state, session, (request("image"),), produced_by=AGENT).outputs[0]
    obs = output.loopback(state.time)
    derived = obs.derive(obs.values.mean(-1, keepdim=True))
    assert derived.provenance.origin == "derived_generated"
    for item in (obs, derived):
        with pytest.raises(ValueError, match="generated"):
            model.observe(state, {"image": item}, time=state.time)
    trace = {}
    reflected = model.reflect(state, {"image": obs}, trace=trace)
    for name in ("time", "observed_time", "log_scale"):
        torch.testing.assert_close(getattr(state, name), getattr(reflected, name), rtol=0, atol=0)
    assert state.imagined == reflected.imagined and state.observation_count == reflected.observation_count
    for role in ("sensory", "entities", "context"):
        torch.testing.assert_close(state.tokens[:, model.layout[role]], reflected.tokens[:, model.layout[role]], rtol=0, atol=0)
    assert not torch.equal(state.tokens, reflected.tokens)
    assert reflected.generated_ancestry[0] == output.provenance
    assert "reflect.provenance" in trace
    with pytest.raises(ValueError, match="generated"):
        model.remember(model.think(reflected), source="summary")
    # A later camera frame does not erase generated ancestry.
    updated = model.observe(reflected, {"image": Observation(torch.rand(1, 1, 3, 16, 16), torch.ones(1, 1))}, time=1)
    with pytest.raises(ValueError, match="generated"):
        model.remember(updated, source="camera plus reflection")
    equal_tree(reflected.to_dict(), LatentState.from_dict(reflected.to_dict()).to_dict())


def test_task_checkpoint_restores_partial_fulfillment_and_continuation():
    model = build_model(width=16).eval()
    state = observed(model)
    session = task(OutputControl("image", "required", USER), OutputControl("audio", "required", USER))
    partial = model.emit(state, session, (request("image", USER),), produced_by=AGENT).session
    buffer = io.BytesIO()
    torch.save(partial.to_dict(), buffer)
    buffer.seek(0)
    restored = TaskSession.from_dict(torch.load(buffer, weights_only=True))
    assert restored.remaining == ("audio",)
    equal_tree(restored.to_dict(), partial.to_dict())
    a = model.emit(state, partial, (request("audio", USER),), produced_by=AGENT)
    b = model.emit(state, restored, (request("audio", USER),), produced_by=AGENT)
    equal_tree(a.session.to_dict(), b.session.to_dict())
    with pytest.raises(ValueError, match="duplicate"):
        model.emit(state, restored, (request("image", USER),), produced_by=AGENT)


def test_task_heads_learn_from_text_and_curriculum_splits_are_disjoint():
    training, validation = InstructionEpisodes(count=64), InstructionEpisodes(split="validation", count=64)
    assert set(x.request.instruction for x in training.tasks).isdisjoint(x.request.instruction for x in validation.tasks)
    assert set(training.operation_targets.tolist()) == set(range(len(OPERATIONS)))
    learner = LearningState(build_model(width=16), len(training))
    batch = training.batch([0, 1, 2, 3])
    losses, _, raw = objective(learner, batch)
    assert {"task_operation_ce", "task_modality_bce", "task_completion_bce"} <= losses.keys()
    sum(losses.values()).backward()
    for name in ("task_interpreter", "task_policy", "metadata_encoder"):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in getattr(learner.agent, name).parameters()), name
    assert all(p.grad is None for p in learner.target.parameters())
    model = learner.agent.eval()
    state = model.initial_state(1)
    a = model.task_predictions(state, [task()])
    b = model.task_predictions(state, [TaskSession(replace(task().request, instruction="Ask me what I mean"))])
    assert not torch.equal(a.operation_logits, b.operation_logits)
    assert "task_operation_error" in raw
