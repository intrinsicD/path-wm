"""Structural checks for the agreed latent/observer/planning contract."""

import itertools

import pytest
import torch
from torch import nn

from world_model.paddle.models import (
    Decoder, Encoder, MemoryUpdater, PositionReadout, Predictor, StateReadout,
)
from world_model.paddle.planner import ExhaustivePlanner, PlanningFailure, sequence_score
from world_model.paddle.rollout import rollout, rollout_step
from world_model.paddle.types import ObservationLatent, PlanningState, action_one_hot


def latent(batch=1):
    return ObservationLatent(torch.randn(batch, 256, 64), torch.randn(batch, 64, 64))


def test_ordered_latent_roundtrip_and_branch_storage():
    tokens = torch.arange(2 * 320 * 64).reshape(2, 320, 64).float()
    s = ObservationLatent.from_tokens(tokens)
    assert torch.equal(s.fine, tokens[:, :256])
    assert torch.equal(s.coarse, tokens[:, 256:])
    assert torch.equal(ObservationLatent.cat([s[:1], s[1:]]).tokens(), tokens)
    clone = PlanningState(s, torch.zeros(2, 128)).clone()
    clone.observation.fine.add_(1)
    clone.memory.add_(1)
    assert torch.equal(s.tokens(), tokens)
    assert torch.equal(action_one_hot(torch.tensor([0, 1, 2])), torch.eye(3))
    with pytest.raises(ValueError):
        ObservationLatent.from_tokens(torch.zeros(1, 319, 64))


def test_exact_shapes_zero_predictor_copy_and_readout_order():
    torch.manual_seed(1)
    e, d, h, u, p, r = Encoder(), Decoder(), PositionReadout(), MemoryUpdater(), Predictor(), StateReadout()
    s = e(torch.rand(2, 3, 64, 64))
    assert s.fine.shape == (2, 256, 64)
    assert s.coarse.shape == (2, 64, 64)
    image = d(s)
    assert image.shape == (2, 3, 64, 64)
    assert image.min() >= 0 and image.max() <= 1
    assert h(s).shape == (2, 3)
    memory = u(torch.zeros(2, 128), s, torch.zeros(2, 3))
    assert memory.shape == (2, 128)
    assert r(memory).shape == (2, 5)
    predicted = p(s, memory, action_one_hot(torch.tensor([0, 2])))
    torch.testing.assert_close(predicted.tokens(), s.tokens(), rtol=0, atol=0)
    # A spatially distinct coordinate must reach H at its documented flattened index.
    with torch.no_grad():
        h.linear.weight.zero_()
        h.linear.bias.zero_()
        h.linear.weight[0, 256 * 64 + 17] = 1
    torch.testing.assert_close(h(s)[:, 0], s.coarse[:, 0, 17])


def test_encoder_cross_scale_directions_use_incoming_branches():
    torch.manual_seed(2)
    encoder = Encoder().eval()
    frame = torch.rand(1, 3, 64, 64)
    before = encoder(frame)
    # Altering only fine-from-coarse must not change this exchange's coarse output.
    with torch.no_grad():
        encoder.fine_from_coarse.output_projection.bias.add_(3)
    after = encoder(frame)
    assert not torch.allclose(before.fine, after.fine)
    torch.testing.assert_close(before.coarse, after.coarse, rtol=0, atol=0)


class CountingPredictor(nn.Module):
    def forward(self, observation, memory, action):
        delta = memory[:, :1, None] + action[:, :1, None]
        return ObservationLatent(observation.fine + delta, observation.coarse + delta)


class CountingUpdater(nn.Module):
    def __init__(self):
        super().__init__()
        self.seen = []

    def forward(self, memory, observation, action):
        self.seen.append((memory.clone(), observation.tokens().clone(), action.clone()))
        return memory + observation.fine[:, 0, :1] + action[:, 2:3]


def test_five_step_recurrence_uses_predicted_features_and_updated_shared_u():
    s = ObservationLatent(torch.ones(1, 256, 64), torch.ones(1, 64, 64))
    original = PlanningState(s, torch.ones(1, 128))
    updater = CountingUpdater()
    actions = action_one_hot(torch.tensor([[0, 2, 1, 0, 1]]))
    result = rollout(original, actions, CountingPredictor(), updater)
    expected_s, expected_m = 1., 1.
    for index, predicted in enumerate(result):
        expected_s += expected_m + float(actions[0, index, 0])
        expected_m += expected_s + float(actions[0, index, 2])
        torch.testing.assert_close(predicted.observation.fine, torch.full_like(s.fine, expected_s))
        torch.testing.assert_close(predicted.memory, torch.full_like(original.memory, expected_m))
        assert updater.seen[index][1][0, 0, 0] == expected_s
    assert len(updater.seen) == 5
    assert original.memory.eq(1).all() and original.observation.fine.eq(1).all()


def test_frozen_u_keeps_path_from_later_prediction_to_earlier_prediction():
    torch.manual_seed(3)
    p, u = Predictor(), MemoryUpdater()
    u.requires_grad_(False)
    nn.init.normal_(p.output_projection.weight, std=.02)
    state = PlanningState(latent(), torch.randn(1, 128))
    a = action_one_hot(torch.tensor([2]))
    first = rollout_step(state, a, p, u)
    first.observation.fine.retain_grad()
    # Detach the direct feature path: any earlier feature gradient must go through U.
    second = p(first.observation.detach(), first.memory, a)
    second.tokens().square().mean().backward()
    assert first.observation.fine.grad is not None
    assert first.observation.fine.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in u.parameters())


def test_future_targets_are_isolated_and_only_predictor_parameters_change():
    torch.manual_seed(4)
    frozen = [Encoder(), MemoryUpdater(), Decoder(), PositionReadout(), StateReadout()]
    for module in frozen:
        module.eval().requires_grad_(False)
    e, u, _, _, _ = frozen
    p = Predictor()
    nn.init.normal_(p.output_projection.weight, std=.01)
    optimizer = torch.optim.AdamW(p.parameters(), lr=1e-3)
    with torch.no_grad():
        s = e(torch.rand(1, 3, 64, 64))
        state = PlanningState(s, u(torch.zeros(1, 128), s, torch.zeros(1, 3)))
        targets = [e(torch.rand(1, 3, 64, 64)) for _ in range(2)]
    before = [{k: v.clone() for k, v in module.state_dict().items()} for module in frozen]
    p_before = p.output_projection.weight.detach().clone()
    actions = action_one_hot(torch.tensor([[0, 2]]))
    predictions = rollout(state, actions, p, u)
    other_predictions = rollout(state, actions, p, u)
    loss = sum((v.observation.tokens() - target.tokens()).square().mean() for v, target in zip(predictions, targets))
    changed_loss = sum((v.observation.tokens() - target.tokens() - 3).square().mean() for v, target in zip(other_predictions, targets))
    assert not torch.isclose(loss, changed_loss)
    for a, b in zip(predictions, other_predictions):
        torch.testing.assert_close(a.observation.tokens(), b.observation.tokens(), rtol=0, atol=0)
    loss.backward()
    optimizer.step()
    for module, snapshot in zip(frozen, before):
        for name, value in module.state_dict().items():
            torch.testing.assert_close(value, snapshot[name], rtol=0, atol=0)
    assert not torch.equal(p_before, p.output_projection.weight)


class ToyDynamics(nn.Module):
    def forward(self, observation, memory, action):
        fine = observation.fine.clone()
        fine[:, 0, 0] += memory[:, 1]
        fine[:, 0, 1] += 3
        fine[:, 0, 2] += 4 * (action[:, 2] - action[:, 0])
        return ObservationLatent(fine, observation.coarse.clone())


class ToyMemory(nn.Module):
    def forward(self, memory, observation, action):
        return memory.clone()


class ToyReadout(nn.Module):
    def forward(self, observation):
        return observation.fine[:, 0, :3] / 64


def toy_state(x=32., y=46., paddle=32., vx=6.):
    s = ObservationLatent(torch.zeros(1, 256, 64), torch.zeros(1, 64, 64))
    s.fine[0, 0, :3] = torch.tensor([x, y, paddle])
    memory = torch.zeros(1, 128)
    memory[0, 1] = vx
    return PlanningState(s, memory)


def scalar_reference(state):
    results = []
    x0, y0, paddle0 = state.observation.fine[0, 0, :3].tolist()
    vx = float(state.memory[0, 1])
    for tie, sequence in enumerate(itertools.product((1, 0, 2), repeat=5)):
        x, y, paddle = x0, y0, paddle0
        squared, movements, miss = [], 0, 0
        for step, action in enumerate(sequence, 1):
            x, y, paddle = x + vx, y + 3, paddle + (action - 1) * 4
            squared.append(((x-paddle)/64)**2)
            movements += action != 1
            if y >= 61:
                miss = step
                break
        score = (int(miss > 0), 5-miss if miss else 0, sum(squared)/len(squared), movements, tie)
        results.append((score, sequence))
    return min(results)


@pytest.mark.parametrize("state", [toy_state(), toy_state(vx=-6), toy_state(y=58, vx=0), toy_state(y=20, vx=0)])
def test_batched_planner_matches_scalar_reference_and_preserves_actual_state(state):
    actual = state.clone()
    score, sequence = scalar_reference(state)
    for batch_size in (243, 17):
        plan = ExhaustivePlanner(ToyDynamics(), ToyMemory(), ToyReadout(), candidate_batch_size=batch_size).plan(state)
        assert plan.sequence == sequence
        assert plan.action_id == sequence[0]
        assert plan.score == pytest.approx(score)
        torch.testing.assert_close(state.observation.tokens(), actual.observation.tokens(), rtol=0, atol=0)
        torch.testing.assert_close(state.memory, actual.memory, rtol=0, atol=0)


def test_terminal_prefix_and_latest_miss_lexicographic_priority():
    positions = torch.tensor([[40., 58., 32.], [40., 61., 32.], [0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
    score = sequence_score(positions, (1, 2, 0, 0, 0))
    assert score[:4] == (1, 3, (8/64)**2, 1)
    later_bad_alignment = positions.clone()
    later_bad_alignment[:3] = torch.tensor([100., 58., 0.])
    later_bad_alignment[3] = torch.tensor([100., 61., 0.])
    assert sequence_score(later_bad_alignment, (1, 1, 1, 1, 1)) < score
    plan = ExhaustivePlanner(ToyDynamics(), ToyMemory(), ToyReadout()).plan(toy_state(y=58, vx=0))
    assert plan.sequence == (1, 1, 1, 1, 1)
    assert all(float(v.observation.fine[0, 0, 1]) == 61 for v in plan.states)


def test_all_invalid_candidates_raise_explicit_failure():
    class InvalidReadout(nn.Module):
        def forward(self, observation):
            return torch.full((observation.fine.shape[0], 3), float("nan"))
    with pytest.raises(PlanningFailure, match="243"):
        ExhaustivePlanner(ToyDynamics(), ToyMemory(), InvalidReadout()).plan(toy_state())


def test_save_load_preserves_outputs_and_planned_action(tmp_path):
    torch.manual_seed(5)
    e, u, p, h = Encoder().eval(), MemoryUpdater().eval(), Predictor().eval(), PositionReadout().eval()
    frame = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        s = e(frame)
        state = PlanningState(s, u(torch.zeros(1, 128), s, torch.zeros(1, 3)))
    path = tmp_path / "weights.pt"
    torch.save({key: module.state_dict() for key, module in zip(("E", "U", "P", "H"), (e, u, p, h))}, path)
    weights = torch.load(path, weights_only=True)
    loaded = [Encoder().eval(), MemoryUpdater().eval(), Predictor().eval(), PositionReadout().eval()]
    for key, module in zip(("E", "U", "P", "H"), loaded):
        module.load_state_dict(weights[key])
    e2, u2, p2, h2 = loaded
    with torch.no_grad():
        s2 = e2(frame)
        state2 = PlanningState(s2, u2(torch.zeros(1, 128), s2, torch.zeros(1, 3)))
    torch.testing.assert_close(s.tokens(), s2.tokens(), rtol=0, atol=0)
    torch.testing.assert_close(state.memory, state2.memory, rtol=0, atol=0)
    # A tiny chunk keeps the real architecture check within ordinary CPU memory.
    first = ExhaustivePlanner(p, u, h, candidate_batch_size=27).plan(state)
    second = ExhaustivePlanner(p2, u2, h2, candidate_batch_size=27).plan(state2)
    assert first.action_id == second.action_id
    assert first.sequence == second.sequence
    assert first.score == second.score
