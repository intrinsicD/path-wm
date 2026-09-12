import copy

import numpy as np
import pytest
import torch


def test_visual_perturbations_keep_paired_controls_and_reflection_labels():
    from pathwm.data.visual_memory import VisualMemoryEpisodes
    from pathwm.evaluation.capabilities import transform_visual

    data = VisualMemoryEpisodes(pairs=2, seed=106301)
    before = data.images.copy()
    for condition in (
        "original",
        "mirror",
        "swap_red_blue",
        "grayscale",
        "dim",
        "noise",
    ):
        changed = transform_visual(data, condition, seed=106401)
        np.testing.assert_array_equal(
            changed.images[::2, -2:], changed.images[1::2, -2:]
        )
        np.testing.assert_array_equal(
            changed.labels, data.labels ^ (condition == "mirror")
        )
        assert changed.identity["condition"] == condition
        assert changed.images.min() >= 0 and changed.images.max() <= 255
    np.testing.assert_array_equal(data.images, before)
    mirrored = transform_visual(data, "mirror")
    np.testing.assert_array_equal(mirrored.images, data.images[:, :, :, ::-1])


def test_scores_and_comparison_never_promote_unmeasured_or_changed_cases():
    from pathwm.evaluation.capabilities import score, compare_baselines

    assert score(None, 0.9)["status"] == "not_measured"
    assert score(0.5, 0.9)["status"] == "fail"
    assert score(1.0, 0.9)["status"] == "pass"
    with pytest.raises(ValueError):
        score(float("nan"), 0.9)
    before = dict(
        protocol_sha256="a",
        cases=[
            dict(
                id="recall",
                input_sha256="b",
                checkpoint="old",
                metrics=dict(accuracy=0.5),
            )
        ],
    )
    after = copy.deepcopy(before)
    after["cases"][0].update(checkpoint="new", metrics=dict(accuracy=0.75))
    delta = compare_baselines(before, after)
    assert delta[0]["delta"]["accuracy"] == 0.25
    for broken in ("protocol_sha256", "input_sha256"):
        candidate = copy.deepcopy(after)
        (candidate if broken == "protocol_sha256" else candidate["cases"][0])[
            broken
        ] = "changed"
        with pytest.raises(ValueError):
            compare_baselines(before, candidate)


def test_output_metrics_are_observable_and_keep_raw_targets():
    from pathwm.evaluation.capabilities import prediction_metrics

    current = torch.zeros(2, 3, 4, 4)
    target = torch.ones(2, 2, 3, 4, 4)
    predicted = torch.full_like(target, 0.5)
    result = prediction_metrics(predicted, target, current)
    assert result == dict(image_mse=0.25, copy_image_mse=1.0, gray_image_mse=0.25)
