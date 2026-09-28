import numpy as np
import pytest
import torch
from pathwm.data.memory_output import MemoryOutputEpisodes


def test_brightness_variants_keep_complete_histories_targets_and_source():
    from copy import deepcopy

    d = MemoryOutputEpisodes(16, seed=65, curriculum="relocation")
    original = deepcopy(d.identity)
    offsets = [-16, -8, 0, 8, 16]
    a = d.with_input_offsets(offsets)
    control = d.with_input_offsets([0] * 5)
    assert len(a) == 5 * len(d) == len(control)
    assert d.identity == original
    for i, off in enumerate(offsets):
        sl = slice(i * len(d), (i + 1) * len(d))
        assert np.array_equal(a.images[sl].astype("int16") - off, d.images)
        assert np.array_equal(control.images[sl], d.images)
        assert np.array_equal(a.labels[sl], d.labels)
        assert np.array_equal(a.targets[sl], d.targets)
    assert a.identity["input_augmentation"]["source_data"] == original
    assert a.identity["input_augmentation"]["offsets_uint8"] == offsets
    assert a.identity["pairs"] == 5 * original["pairs"]
    assert a.shortcut_audit()["hidden_frame_only_joint_accuracy"] == 0.25
    assert np.array_equal(a.labels, control.labels)


@pytest.mark.parametrize("offsets", [[], [True], [1.5], [21], [-31]])
def test_brightness_variants_reject_invalid_or_clipping(offsets):
    d = MemoryOutputEpisodes(16, seed=65, curriculum="relocation")
    with pytest.raises(ValueError, match="offset"):
        d.with_input_offsets(offsets)


