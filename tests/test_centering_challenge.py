"""Integrity of a fixed-checkpoint scene/lighting challenge."""

from copy import deepcopy
import hashlib
import numpy as np
import pytest
import torch
from pathwm.data.memory_output import MemoryOutputEpisodes, COLORS


def base(seed=93):
    return MemoryOutputEpisodes(16, seed=seed, split="test", curriculum="relocation")


def test_scene_neutral_and_source_provenance():
    d = base()
    identity = deepcopy(d.identity)
    result = d.with_scene()
    assert np.array_equal(d.images, result.images)
    assert result.identity["input_transform"]["source_data"] == identity
    assert np.array_equal(result.targets, d.targets)
    assert np.array_equal(result.labels, d.labels)
    result.images[0, 0, 0, 0] = 0
    assert d.identity == identity
    assert hashlib.sha256(d.images.tobytes()).hexdigest() == identity["images_sha256"]


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(frame_offsets=(-12, 12, -8)),
        dict(rgb_offset=(8, -8, 4)),
        dict(background_offset=(12, 0, -8)),
        dict(texture=8),
        dict(background_offset=(48, 48, 48)),
        dict(radius=12),
        dict(clutter=True),
        dict(gain=0.75),
        dict(shadow=12),
    ],
)
def test_scene_preserves_counterfactuals_and_semantic_targets(kwargs):
    d = base()
    result = d.with_scene(**kwargs)
    assert not np.array_equal(result.images, d.images)
    assert np.array_equal(result.targets, d.targets)
    assert np.array_equal(result.labels, d.labels)
    for i in range(0, len(d), 4):
        a = result.images[i : i + 4]
        assert np.array_equal(a[:2, 0], a[2:, 0])
        assert np.array_equal(a[0, 1:], a[1, 1:])
        assert np.array_equal(a[2, 1:], a[3, 1:])
        assert not np.array_equal(a[0, 1], a[2, 1])
    assert result.shortcut_audit() == d.shortcut_audit()
    assert result.identity["input_transform"]["source_data"] == d.identity
    assert (
        result.identity["images_sha256"]
        == hashlib.sha256(result.images.tobytes()).hexdigest()
    )
    assert np.array_equal(base().with_scene(**kwargs).images, result.images)


def test_background_changes_do_not_recolor_objects_or_selection_cue():
    d = base()
    mask = np.any(np.all(d.images[..., None, :] == COLORS, axis=-1), axis=-1)
    mask |= np.all(d.images == 235, axis=-1)
    for kw in [
        dict(background_offset=(12, 0, -8)),
        dict(texture=8),
        dict(clutter=True),
    ]:
        a = d.with_scene(**kw).images
        assert np.array_equal(a[mask], d.images[mask])
    a = d.with_scene(radius=12).images
    for i in range(len(d)):
        for selected in (0, 1):
            color, shape, side = d.labels[(i // 2) * 2 + selected]
            x = 16 + 32 * side
            assert np.array_equal(a[i, 1, 32, x], COLORS[color])
            assert np.array_equal(a[i, 1, 21, x], COLORS[color])
            assert np.array_equal(a[i, 1, 21, x - 11], COLORS[color]) == (shape == 0)
        x = 16 + 32 * (i % 2)
        assert np.all(a[i, 0, 16, x] == 235)
        assert not np.all(a[i, 0, 16, 64 - x] == 235)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(radius=13),
        dict(radius=True),
        dict(gain=float("nan")),
        dict(frame_offsets=(0, 0)),
        dict(rgb_offset=(True, 0, 0)),
        dict(texture=-1),
        dict(clutter=1),
        dict(shadow=-1),
        dict(gain=2),
        dict(background_offset=(-100, 0, 0)),
    ],
)
def test_scene_rejects_invalid_parameters_and_clipping(kwargs):
    with pytest.raises(ValueError):
        base().with_scene(**kwargs)


def test_scene_cannot_retransform_or_use_incompatible_curriculum():
    for d in [
        base().with_scene(texture=8),
        base().with_input_offsets([0]),
        MemoryOutputEpisodes(16, input_offset=8, curriculum="relocation"),
        MemoryOutputEpisodes(4),
    ]:
        with pytest.raises(ValueError):
            d.with_scene()


def test_frame_channel_cancellation_and_discarded_intensity():
    from pathwm.models.memory_output import PixelMedianCentering
    from pathwm.models.modalities import Observation

    class Capture(torch.nn.Module):
        code_width = 8

        def forward(self, obs):
            return obs

    d = base()
    a = d.batch(range(4))["images"]
    b = d.with_scene(frame_offsets=(-12, 12, -8), rgb_offset=(4, -4, 2)).batch(
        range(4)
    )["images"]
    times = torch.arange(3).float().expand(4, -1)
    m = PixelMedianCentering(Capture(), [40 / 255] * 3)
    torch.testing.assert_close(
        m(Observation(a, times)).values,
        m(Observation(b, times)).values,
        atol=1e-7,
        rtol=0,
    )
    # If absolute lightness were a target, this representation cannot distinguish it.
    x = torch.full_like(a, 0.25)
    torch.testing.assert_close(
        m(Observation(x, times)).values, m(Observation(x + 0.25, times)).values
    )


def test_scene_cli_is_evaluation_only(monkeypatch, tmp_path):
    import sys
    import experiments.memory_output as recipe

    weights = tmp_path / "weights.pt"
    torch.save(dict(settings=dict(curriculum="relocation")), weights)
    seen = []
    monkeypatch.setattr(
        recipe, "evaluate_export", lambda weights, data, **kw: seen.append(data)
    )
    args = [
        "memory_output",
        "--weights",
        str(weights),
        "--output",
        str(tmp_path / "out"),
        "--test-seed",
        "93",
        "--scene",
        "temporal-offset",
    ]
    monkeypatch.setattr(sys, "argv", args + ["--evaluate-only"])
    recipe.main()
    assert seen[0].identity["input_transform"]["parameters"]["frame_offsets"] == [
        -12,
        12,
        -8,
    ]
    monkeypatch.setattr(sys, "argv", args)
    with pytest.raises(SystemExit):
        recipe.main()
