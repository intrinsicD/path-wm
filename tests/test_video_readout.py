import copy
import json

import pytest
import torch

from pathwm.models.modalities import position
from pathwm.models.readout import TemporalImageDecoder


def test_time_query_placement_matches_reference_and_preserves_masks_gradients():
    torch.manual_seed(22)
    legacy = TemporalImageDecoder(16)
    torch.manual_seed(22)
    model = TemporalImageDecoder(16, time_conditioning="query")
    assert all(
        torch.equal(v, model.state_dict()[k]) for k, v in legacy.state_dict().items()
    )
    context = torch.randn(2, 5, 16, requires_grad=True)
    valid = torch.tensor([[True, True, True, False, False]]).expand(2, -1)
    dirty = context.masked_fill(~valid[..., None], float("nan"))
    times = torch.tensor([0.0, 0.7, 3.0])
    original = context.detach().clone()
    actual = model(dirty, times, valid=valid)
    expected = torch.stack(
        [
            model.image(
                context[:, :3],
                query_offset=model.time_projection(position(t[None], 16))[:, None],
            )
            for t in times
        ],
        1,
    )
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        model(dirty, times[[0, 2]], valid=valid), actual[:, [0, 2]]
    )
    assert torch.equal(context.detach(), original)
    actual.square().mean().backward()
    assert context.grad[:, :3].abs().sum() > 0
    assert context.grad[:, 3:].count_nonzero() == 0
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
    )
    with pytest.raises(ValueError, match="conditioning"):
        TemporalImageDecoder(16, time_conditioning="bad")
    with pytest.raises(ValueError, match="offset"):
        model.image(context, query_offset=torch.zeros(3, 1, 16))


def test_legacy_video_numerical_path_unchanged():
    torch.manual_seed(23)
    model = TemporalImageDecoder(16)
    context, times = torch.randn(2, 5, 16), torch.arange(4.0)
    code = model.time_projection(position(times, 16))
    old = model.image((context[:, None] + code[None, :, None]).reshape(8, 5, 16))
    assert torch.equal(model(context, times), old.reshape(2, 4, 3, 16, 16))


def test_motion_metrics_reject_static_and_reversed_clips():
    from pathwm.data.modality_readout import canonical
    from pathwm.evaluation.modality_readout import video_motion_metrics

    target = canonical([[0, 1, 0], [1, 1, 1]])["video"]
    exact = video_motion_metrics(target, target)
    assert exact["centroid_error_pixels"] == 0
    assert exact["displacement_mae_pixels"] == 0
    assert exact["motion_direction_accuracy"] == 1
    for wrong in (
        target.flip(1),
        target[:, :1].expand_as(target),
        torch.zeros_like(target),
    ):
        metrics = video_motion_metrics(wrong, target)
        assert metrics["motion_direction_accuracy"] == 0
        assert metrics["displacement_mae_pixels"] >= 2


def test_initialization_restores_saved_video_conditioning(tmp_path):
    from experiments.modality_readout import Model, load_initial

    model = Model("native", video_conditioning="query")
    torch.save({"model": copy.deepcopy(model.state_dict())}, tmp_path / "last.pt")
    (tmp_path / "run.json").write_text(
        json.dumps({"identity": {"settings": {"video_conditioning": "query"}}})
    )
    target = Model("native")
    load_initial(target, tmp_path, "cpu")
    assert target.outputs.decoders["video"].time_conditioning == "query"
