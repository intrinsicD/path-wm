"""Physical local packing versus independently constructed dense attention."""

from dataclasses import replace
import copy
import math

import pytest
import torch

from pathwm.models.multiscale import (
    FeatureScale, ScaleMerge, ScaleProcessor, MultiScaleImageEncoder, pool_scale,
)
from pathwm.models.modalities import Observation


def scale(grid):
    n = math.prod(grid)
    values = torch.randn(2, n, 16, requires_grad=True)
    valid = torch.ones(2, n, dtype=torch.bool)
    valid[0, 1::3] = False
    valid[1] = False
    return FeatureScale(values.masked_fill(~valid[..., None], float("nan")),
                        torch.arange(n)[None].expand(2, -1).double(), valid,
                        torch.arange(n)[None].expand(2, -1), grid), values


@pytest.mark.parametrize("grid,factors", [((7,), (2,)), ((3, 5, 7), (2, 2, 2)), ((1, 1, 1), (1, 2, 2))])
def test_packed_merge_matches_dense_outputs_metadata_and_gradients(grid, factors):
    dense = ScaleMerge(16, 8, factors)
    packed = ScaleMerge(16, 8, factors, packed=True)
    packed.load_state_dict(dense.state_dict(), strict=True)
    fine, values = scale(grid)
    code = torch.randn(2, 8, requires_grad=True)
    expected = dense(fine, code)
    actual = packed(fine, code)
    for field in ("values", "times", "valid", "ends", "content_times"):
        torch.testing.assert_close(getattr(actual, field), getattr(expected, field), atol=2e-6, rtol=2e-5)
    inputs = [values, code]
    left = torch.autograd.grad(expected.values.square().sum(), inputs + list(dense.parameters()), retain_graph=True)
    right = torch.autograd.grad(actual.values.square().sum(), inputs + list(packed.parameters()))
    for a, b in zip(left, right):
        torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-4)
    assert actual.values[1].count_nonzero() == 0


def test_windows_match_dense_local_mask_keep_all_positions_and_gradients():
    stage = ScaleProcessor(16, 8, depth=2, window=(2, 2, 3))
    fine, values = scale((3, 5, 7))
    code = torch.randn(2, 8, requires_grad=True)
    # Geometric membership reference independent of the implementation's index packer.
    xyz = torch.cartesian_prod(torch.arange(3), torch.arange(5), torch.arange(7))
    group = xyz // torch.tensor([2, 2, 3])
    allowed = (group[:, None] == group[None]).all(-1)
    expected = replace(fine, values=(fine.values.masked_fill(~fine.valid[..., None], 0) + stage.identity).masked_fill(~fine.valid[..., None], 0))
    for block in stage.blocks:
        expected = block(expected, code, footprint=allowed)
    trace = {}
    actual = stage(fine, code, trace=trace)
    torch.testing.assert_close(actual.values, expected.values, atol=2e-6, rtol=2e-5)
    assert actual.grid == fine.grid and actual.values.shape == fine.values.shape
    assert torch.equal(actual.times, fine.times)
    assert trace['scale.attention.0'].shape == (2, 4, 105, 105)
    assert trace['scale.attention.0'][:, :, ~allowed].count_nonzero() == 0
    inputs = [values, code, *stage.parameters()]
    left = torch.autograd.grad(expected.values.square().sum(), inputs, retain_graph=True)
    right = torch.autograd.grad(actual.values.square().sum(), inputs)
    for a, b in zip(left, right):
        torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-4)


def test_local_video_preserves_causality_trace_neutrality_and_weight_shapes():
    encoder = MultiScaleImageEncoder(16, video=True, code_width=8, window_size=4, packed_merges=True)
    reference = MultiScaleImageEncoder(16, video=True, code_width=8)
    reference.load_state_dict(encoder.state_dict(), strict=True)
    assert encoder.pyramid.stages[-1].window is None
    values = torch.randn(1, 3, 3, 20, 28)
    observation = Observation(values, torch.zeros(1, 3))
    before = torch.get_rng_state()
    actual = encoder(observation)
    traced = encoder(observation, trace={})
    assert torch.equal(before, torch.get_rng_state())
    assert torch.equal(actual.as_tokens().values, traced.as_tokens().values)
    changed = values.clone(); changed[:, -1] += 5
    future = encoder(Observation(changed, observation.times))
    for a, b in zip(actual.scales, future.scales):
        earlier = a.ends < 2
        assert torch.equal(a.values[earlier], b.values[earlier])


def test_workload_counts_all_windows_as_batch_work():
    from pathwm.evaluation.workload import Workload
    stage = ScaleProcessor(16, 8, window=(4,))
    fine, _ = scale((16,))
    with Workload(stage) as work:
        stage(fine, torch.zeros(2, 8))
    record = work.summary()
    # Two samples, four windows each, sixteen pair interactions per window.
    assert record["batch_self_attention_pairs"] == 2 * 4 * 4 * 4
    assert record["batch_query_token_evaluations"] == 2 * 16
