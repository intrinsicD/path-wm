"""Scene learning preserves task binding and persisted input preprocessing."""

from copy import deepcopy
import json
import numpy as np
import pytest
import torch
from pathwm.data.memory_output import MemoryOutputEpisodes


def test_scene_blocks_preserve_binding_provenance_and_neutral_control():
    d = MemoryOutputEpisodes(16, seed=81, curriculum="relocation")
    identity = deepcopy(d.identity)
    scenes = [
        {},
        {"background_offset": [12, 0, -8]},
        {"background_offset": [-8, 0, 12]},
    ]
    a, c = d.with_scenes(scenes), d.with_scenes([{}] * 3)
    assert len(a) == len(c) == 96
    assert d.identity == identity
    for i, scene in enumerate(scenes):
        sl = slice(i * len(d), (i + 1) * len(d))
        assert np.array_equal(a.images[sl], d.with_scene(**scene).images)
        assert np.array_equal(c.images[sl], d.images)
        assert np.array_equal(a.targets[sl], d.targets)
        assert np.array_equal(a.labels[sl], d.labels)
    actual, expected = a.shortcut_audit(), d.shortcut_audit()
    assert actual.pop("tuple_counts") == {
        k: 3 * v for k, v in expected.pop("tuple_counts").items()
    }
    assert actual == expected
    assert a.identity["input_augmentation"]["source_data"] == identity
    snapshot = deepcopy(a.identity)
    scenes[1]["background_offset"][0] = 10
    d.identity["seed"] = -1
    assert a.identity == snapshot
    with pytest.raises(ValueError, match="untransformed"):
        a.with_scenes([{}])


@pytest.mark.parametrize(
    "scenes", [[], "neutral", [None], [{"background_offset": [255, 0, 0]}]]
)
def test_scene_blocks_reject_invalid_or_clipping(scenes):
    d = MemoryOutputEpisodes(16, seed=82, curriculum="relocation")
    with pytest.raises(ValueError):
        d.with_scenes(scenes)


