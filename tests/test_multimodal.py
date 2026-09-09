from dataclasses import replace
import io

import pytest
import torch

from experiments.multimodal import build_model
from pathwm.models.modalities import Observation, VectorEncoder, bytes_batch
from pathwm.models.agent_state import LatentState
from pathwm.evaluation.agent import plan
from pathwm.training.improvement import try_improvement
from tests.test_runs import equal_tree


def small():
    torch.manual_seed(12)
    return build_model(width=16, image_size=16, audio_samples=32)


def image_observation(batch=2, time=0.0):
    return Observation(torch.rand(batch, 1, 3, 16, 16), torch.full((batch, 1), time))


def test_all_modalities_outputs_and_gradients():
    model = small()
    text, valid = bytes_batch(["ball", "hit"])
    obs = {
        "image": image_observation(),
        "video": Observation(
            torch.rand(2, 2, 3, 16, 16), torch.tensor([[-1.0, 0.0]]).expand(2, -1)
        ),
        "audio": Observation(
            torch.randn(2, 2, 32), torch.tensor([[-1.0, 0.0]]).expand(2, -1)
        ),
        "text": Observation(text, torch.zeros_like(text, dtype=torch.float), valid),
    }
    trace = {}
    state = model.observe(model.initial_state(2, time=-1), obs, time=0, trace=trace)
    output = model.decode(state, text_prefix=text[:, :-1])
    assert output["image"].shape == (2, 3, 16, 16)
    assert output["audio"].shape == (2, 32)
    assert output["text"].shape == (*text[:, :-1].shape, 259)
    future = model.imagine(state, torch.zeros(2, 2), dt=1)
    assert model.decode_video([state, future]).shape == (2, 2, 3, 16, 16)
    loss = (
        sum(x.square().mean() for x in output.values()) + future.tokens.square().mean()
    )
    loss.backward()
    for name, module in model.encoders.items():
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters()
        ), name
    for name, module in model.decoders.items():
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters()
        ), name
    assert "observe.attention" in trace and "observe.tokens" in trace
    assert all(
        not x.requires_grad for x in trace.values() if isinstance(x, torch.Tensor)
    )


def test_masks_hide_values_and_future_observations_are_rejected():
    model = small().eval()
    state = model.initial_state(2)
    values = torch.rand(2, 2, 3, 16, 16)
    times = torch.tensor([[0.0, 10.0], [0.0, 10.0]])
    valid = torch.tensor([[True, False], [True, False]])
    a = model.observe(state, {"video": Observation(values, times, valid)}, time=0)
    changed = values.clone()
    changed[:, 1] = float("nan")
    b = model.observe(state, {"video": Observation(changed, times, valid)}, time=0)
    torch.testing.assert_close(a.tokens, b.tokens, rtol=0, atol=0)
    with pytest.raises(ValueError, match="future"):
        model.observe(state, {"video": Observation(values, times)}, time=0)
    with pytest.raises(ValueError, match="valid observation"):
        model.observe(
            state, {"video": Observation(values, times, valid & False)}, time=0
        )
    with pytest.raises(ValueError, match="backward"):
        model.observe(
            replace(state, time=state.time + 1), {"image": image_observation()}, time=0
        )


def test_thinking_memory_imagination_and_state_roundtrip():
    model = small().eval()
    observed = model.observe(
        model.initial_state(2), {"image": image_observation()}, time=0
    )
    before = observed.tokens.clone()
    remembered = model.remember(observed, source="episode-7/frame-0")
    assert observed.memory is None
    assert remembered.memory.sources == ("episode-7/frame-0",)
    assert not remembered.memory.values.requires_grad
    trace = {}
    thought = model.think(remembered, steps=2, trace=trace)
    assert torch.equal(thought.time, remembered.time)
    assert thought.thinking_steps == 2
    for role in ("sensory", "entities", "context"):
        s = model.layout[role]
        torch.testing.assert_close(thought.tokens[:, s], before[:, s], rtol=0, atol=0)
    assert not torch.equal(
        thought.tokens[:, model.layout["reasoning"]],
        before[:, model.layout["reasoning"]],
    )
    assert "think.0.memory_indices" in trace
    a = model.imagine(thought, torch.zeros(2, 2), dt=0.5)
    b = model.imagine(thought, torch.ones(2, 2), dt=0.5)
    assert a.imagined and not thought.imagined
    assert torch.equal(a.observed_time, thought.observed_time)
    assert torch.equal(a.time, thought.time + 0.5)
    assert not torch.equal(a.tokens, b.tokens)
    assert not torch.equal(a.tokens, model.imagine(thought, None, dt=0.5).tokens)
    torch.testing.assert_close(observed.tokens, before, rtol=0, atol=0)
    with pytest.raises(ValueError, match="imagined"):
        model.remember(a, source="not observed")
    with pytest.raises(ValueError, match="positive"):
        model.imagine(thought, None, dt=0)
    buffer = io.BytesIO()
    torch.save(thought.to_dict(), buffer)
    buffer.seek(0)
    restored = LatentState.from_dict(torch.load(buffer, weights_only=True))
    equal_tree(thought.to_dict(), restored.to_dict())
    torch.manual_seed(3)
    sampled = model.imagine(thought, None, dt=1, sample=True)
    torch.manual_seed(3)
    equal_tree(
        sampled.to_dict(), model.imagine(thought, None, dt=1, sample=True).to_dict()
    )


def test_text_decoder_is_causal_and_new_input_needs_only_an_adapter():
    model = small().eval()
    model.encoders["sensor"] = VectorEncoder(5, 16)
    state = model.observe(
        model.initial_state(2),
        {
            "sensor": Observation(torch.randn(2, 3, 5), torch.zeros(2, 3)),
        },
        time=0,
    )
    prefix = torch.tensor([[1, 10, 11, 12], [1, 10, 11, 12]])
    altered = prefix.clone()
    altered[:, 2:] = 20
    a = model.decoders["text"](state.tokens, prefix)
    b = model.decoders["text"](state.tokens, altered)
    torch.testing.assert_close(a[:, :2], b[:, :2], rtol=0, atol=0)
    generated = model.generate_text(state, max_tokens=5)
    assert generated.shape[0] == 2 and generated.shape[1] <= 6
    changed = model.intervene(state, "entities", 0.0)
    assert torch.count_nonzero(changed.tokens[:, model.layout["entities"]]) == 0
    assert torch.count_nonzero(state.tokens[:, model.layout["entities"]]) > 0

    class ForceDecoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.output = torch.nn.Linear(16, 3)

        def forward(self, tokens, trace=None):
            return self.output(tokens.mean(1))

    model.decoders["force"] = ForceDecoder()
    force = model.decode(state, modalities=["force"])["force"]
    assert force.shape == (2, 3)
    force.square().mean().backward()
    assert model.decoders["force"].output.weight.grad.abs().sum() > 0


def test_large_timestamps_preserve_short_intervals_and_same_time_updates():
    model = small()
    start = 1_800_000_000.0
    state = model.initial_state(2, time=start)
    obs = image_observation(time=0.0)
    obs = replace(obs, times=torch.full((2, 1), start, dtype=torch.float64))
    state = model(state, {"image": obs}, time=start)
    updated = model.observe(state, {"image": obs}, time=start)
    assert updated.observation_count == 2 and torch.equal(state.time, updated.time)
    future = model.imagine(updated, None, dt=0.125)
    assert (future.time - state.time).tolist() == [0.125, 0.125]
    assert future.tokens.dtype == torch.float32 and future.time.dtype == torch.float64


def test_planner_matches_known_transition_and_preserves_caller():
    model = small().eval()
    state = model.initial_state(2)

    class Toy(torch.nn.Module):
        action_width = 2

        def imagine(self, state, action, dt=1.0, **kwargs):
            return replace(
                state,
                tokens=state.tokens + action[:, :1, None] * dt,
                time=state.time + dt,
                imagined=True,
            )

    candidates = torch.tensor([[[[0.0, 0.0]], [[1.0, 0.0]], [[-1.0, 0.0]]]]).expand(
        2, -1, -1, -1
    )
    target = state.tokens + 1
    original = state.tokens.clone()
    result = plan(
        Toy(),
        state,
        candidates,
        lambda s: (s.tokens - target).square().mean((1, 2)),
        lower=-1,
        upper=1,
    )
    assert result.indices.tolist() == [1, 1]
    assert result.scores[:, 1].tolist() == [0.0, 0.0]
    torch.testing.assert_close(state.tokens, original, rtol=0, atol=0)
    with pytest.raises(ValueError, match="bounds"):
        plan(
            Toy(),
            state,
            candidates * 2,
            lambda s: s.tokens.mean((1, 2)),
            lower=-1,
            upper=1,
        )
    with pytest.raises(ValueError, match="finite"):
        plan(
            Toy(), state, candidates, lambda s: s.time * float("nan"), lower=-1, upper=1
        )


def test_learning_proposal_accepts_improvement_and_rolls_back_regression():
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)

    def evaluate():
        return {"error": float(model.weight.detach().square().mean())}

    def proposal():
        optimizer.zero_grad(set_to_none=True)
        model.weight.square().sum().backward()
        optimizer.step()

    accepted = try_improvement(
        model,
        optimizer,
        proposal,
        evaluate,
        primary="error",
        min_improvement=0.01,
        tolerances={"error": 0.0},
    )
    assert accepted["accepted"]
    weight = model.weight.detach().clone()
    import copy

    opt = copy.deepcopy(optimizer.state_dict())
    rng = torch.get_rng_state().clone()

    def bad():
        proposal()
        with torch.no_grad():
            model.weight.add_(10)
        torch.rand(5)

    rejected = try_improvement(
        model,
        optimizer,
        bad,
        evaluate,
        primary="error",
        min_improvement=0.01,
        tolerances={"error": 0.0},
    )
    assert not rejected["accepted"]
    torch.testing.assert_close(model.weight, weight, rtol=0, atol=0)
    equal_tree(optimizer.state_dict(), opt)
    assert torch.equal(torch.get_rng_state(), rng)
