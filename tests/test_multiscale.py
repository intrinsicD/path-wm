from dataclasses import replace

import pytest
import torch

from experiments.multimodal import build_model, SyntheticEpisodes, observations
from pathwm.models.modalities import Observation
from pathwm.models.multiscale import (
    FeatureScale,
    MultiScaleImageEncoder,
    MultiScaleAudioEncoder,
    MultiScaleTextEncoder,
    pool_scale,
)


def examples():
    return observations(SyntheticEpisodes(count=2).batch([0, 1]), 1)


def test_all_modalities_export_processed_scales_and_live_code_gradients():
    model = build_model(width=16)
    expected = {
        "image": [16, 4, 1],
        "video": [32, 4, 1],
        "audio": [8, 4, 2],
        "text": [12, 6, 3],
    }
    for name, observation in examples().items():
        encoder = model.encoders[name]
        code = torch.randn(2, model.feature_controller.code_width, requires_grad=True)
        pyramid = encoder(observation, condition=code)
        assert [scale.values.shape[1] for scale in pyramid.scales] == expected[name]
        for scale in pyramid.scales:
            assert scale.values.shape[-1] == 16
            gradient = torch.autograd.grad(
                scale.values.square().mean(), code, retain_graph=True
            )[0]
            assert gradient.abs().sum() > 0
        combined = pyramid.as_tokens()
        torch.testing.assert_close(
            combined.values, torch.cat([s.values for s in pyramid.scales], 1)
        )


def test_scale_processing_precedes_every_consumer_and_gradients_only_go_up():
    encoder = MultiScaleImageEncoder(16, code_width=8)
    observation = examples()["image"]
    events, finished = [], {}
    handles = []
    for i, stage in enumerate(encoder.pyramid.stages):

        def record(module, args, output, i=i):
            events.append(f"stage{i}")
            finished[i] = output

        handles.append(stage.register_forward_hook(record))
    for i, merge in enumerate(encoder.pyramid.merges):

        def check(module, args, i=i):
            assert args[0] is finished[i]
            events.append(f"merge{i}")

        handles.append(merge.register_forward_pre_hook(check))
    pyramid = encoder(observation)
    assert events == ["stage0", "merge0", "stage1", "merge1", "stage2"]
    assert all(s is finished[i] for i, s in enumerate(pyramid.scales))
    for handle in handles:
        handle.remove()
    fine_parameters = list(encoder.pyramid.stages[0].parameters())
    up = torch.autograd.grad(
        pyramid.scales[-1].values.square().mean(), fine_parameters, retain_graph=True
    )
    assert any(g.abs().sum() > 0 for g in up)
    down = torch.autograd.grad(
        pyramid.scales[0].values.square().mean(),
        list(encoder.pyramid.stages[-1].parameters()),
        allow_unused=True,
    )
    assert all(g is None for g in down)
    baseline = pyramid.scales[-1].values.detach()

    def intervene(module, args, output):
        return replace(
            output, values=output.values + torch.arange(16).to(output.values)
        )

    handle = encoder.pyramid.stages[0].register_forward_hook(intervene)
    changed = encoder(observation).scales[-1].values
    handle.remove()
    assert not torch.allclose(changed, baseline)


def test_pooling_matches_support_ancestors_and_handles_odd_masked_groups():
    fine = FeatureScale(
        torch.tensor([[[1.0], [3.0], [9.0], [7.0], [5.0]]]),
        torch.tensor([[1.0, 4.0, 6.0, 8.0, 10.0]], dtype=torch.float64),
        torch.tensor([[True, True, False, False, True]]),
        torch.arange(5)[None],
        (5,),
    )
    coarse, membership = pool_scale(fine, (2,))
    assert coarse.values.flatten().tolist() == [2.0, 0.0, 5.0]
    assert coarse.times.tolist() == [[4.0, 0.0, 10.0]]
    assert coarse.valid.tolist() == [[True, False, True]]
    assert coarse.ends.tolist() == [[1, -1, 4]]
    assert membership.tolist() == [
        [True, True, False, False, False],
        [False, False, True, True, False],
        [False, False, False, False, True],
    ]
    last, _ = pool_scale(coarse, (2,))
    assert last.times.tolist() == [[4.0, 10.0]]


@pytest.mark.parametrize("kind", ["video", "audio", "text"])
def test_temporal_and_equal_timestamp_prefix_causality(kind):
    if kind == "video":
        encoder = MultiScaleImageEncoder(16, video=True, code_width=8)
        values = torch.rand(1, 3, 3, 12, 20)
    elif kind == "audio":
        encoder = MultiScaleAudioEncoder(9, 16, patch_size=4, code_width=8)
        values = torch.rand(1, 3, 9)
    else:
        encoder = MultiScaleTextEncoder(16, code_width=8)
        values = torch.tensor([[1, 8, 9]])
    for times in (torch.tensor([[0.0, 1.0, 2.0]]), torch.zeros(1, 3)):
        before = encoder(Observation(values, times))
        altered = values.clone()
        altered[:, -1] += 2
        after = encoder(Observation(altered, times))
        end = (values.shape[1] - 1) * (3 if kind == "audio" else 1)
        for a, b in zip(before.scales, after.scales):
            earlier = a.valid & (a.ends < end)
            torch.testing.assert_close(
                a.values[earlier], b.values[earlier], rtol=0, atol=0
            )
            torch.testing.assert_close(a.times, b.times, rtol=0, atol=0)
        if times.max() > 0:
            for a in before.scales:
                # Attention never assigns an earlier availability than its latest supporting input.
                assert (
                    a.times[a.valid]
                    >= times[0, (a.ends[a.valid] // (3 if kind == "audio" else 1))]
                ).all()


def test_masks_singletons_cross_attention_footprints_and_neutral_code_after_update():
    encoder = MultiScaleTextEncoder(16, code_width=8)
    tokens = torch.tensor([[1, 5, 7, 9, 2], [0, 0, 0, 0, 0]])
    valid = tokens != 0
    times = torch.zeros(2, 5)
    trace = {}
    output = encoder(Observation(tokens, times, valid), trace=trace)
    for scale in output.scales:
        assert torch.isfinite(scale.values).all()
        assert not scale.valid[1].any()
        assert scale.values[1].count_nonzero() == 0
    for i in range(2):
        _, membership = pool_scale(output.scales[i], (2,))
        weights = trace[f"merge.{i}.attention"]
        assert weights[:, :, ~membership].count_nonzero() == 0
        assert weights[1].count_nonzero() == 0
    output.as_tokens().values.square().mean().backward()
    optimizer = torch.optim.SGD(encoder.parameters(), lr=0.01)
    optimizer.step()
    for parameter in encoder.parameters():
        assert parameter.grad is None or torch.isfinite(parameter.grad).all()
    a = encoder(Observation(tokens, times, valid))
    b = encoder(Observation(tokens, times, valid), condition=torch.zeros(2, 8))
    torch.testing.assert_close(
        a.as_tokens().values, b.as_tokens().values, rtol=0, atol=0
    )
    singleton = encoder(
        Observation(torch.ones(1, 1, dtype=torch.long), torch.zeros(1, 1))
    )
    assert all(s.values.shape == (1, 1, 16) for s in singleton.scales)
    with pytest.raises(ValueError, match="condition"):
        encoder(
            Observation(tokens, times, valid),
            condition=torch.full((2, 8), float("nan")),
        )


def test_agent_control_uses_prior_state_user_override_and_cutoff_before_encoders():
    model = build_model(width=16)
    state = model.initial_state(2, time=1.0)
    inputs = examples()
    before = state.tokens.clone()
    expected = model.propose_feature_code(state)
    trace = {}
    updated = model.observe(state, inputs, time=1.0, trace=trace)
    torch.testing.assert_close(
        trace["encode.feature_code"], expected.detach(), rtol=0, atol=0
    )
    torch.testing.assert_close(state.tokens, before, rtol=0, atol=0)
    override = torch.zeros_like(expected)
    neutral = model.observe(state, inputs, time=1.0, feature_code=override)
    assert not torch.allclose(updated.tokens, neutral.tokens)
    updated.tokens.square().mean().backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.feature_controller.parameters()
    )
    encoded = model.encode(state, inputs, time=2.0, feature_code=override)
    assert all(
        (s.times[s.valid] == 2).all() for p in encoded.values() for s in p.scales
    )
    called = []
    handle = model.encoders["image"].register_forward_pre_hook(
        lambda *args: called.append(True)
    )
    future = dict(
        inputs,
        text=replace(inputs["text"], times=torch.full_like(inputs["text"].times, 3.0)),
    )
    with pytest.raises(ValueError, match="future"):
        model.observe(state, future, time=1.0)
    handle.remove()
    assert not called


def test_masked_video_nans_do_not_reach_any_scale_or_gradient():
    encoder = MultiScaleImageEncoder(16, video=True, code_width=8)
    values = torch.rand(2, 3, 3, 12, 20)
    valid = torch.tensor([[True, False, True], [False, False, False]])
    values[~valid] = float("nan")
    values.requires_grad_()
    output = encoder(Observation(values, torch.zeros(2, 3), valid))
    output.as_tokens().values.square().mean().backward()
    assert torch.isfinite(values.grad).all()
    assert values.grad[~valid].count_nonzero() == 0
    assert values.grad[valid].abs().sum() > 0
    assert all(torch.isfinite(s.values).all() for s in output.scales)
