from dataclasses import replace

import numpy as np
import pytest
import torch


def test_relocation_curriculum_breaks_appearance_and_initial_frame_shortcuts():
    from collections import Counter
    import hashlib
    from pathwm.data.memory_output import MemoryOutputEpisodes, image_labels

    datasets = [
        MemoryOutputEpisodes(16, seed=s, curriculum="relocation") for s in (17, 18)
    ]
    frame_sets = []
    for data in datasets:
        b = data.batch(range(len(data)))
        assert torch.equal(image_labels(b["target"]), b["labels"])
        counts = Counter(map(tuple, data.labels.tolist()))
        assert len(counts) == 16 and set(counts.values()) == {2}
        for start in range(0, len(data), 4):
            frames = b["images"][start : start + 4]
            labels = b["labels"][start : start + 4]
            assert torch.equal(frames[:2, 0], frames[2:, 0])
            assert torch.equal(frames[0, 1:], frames[1, 1:])
            assert torch.equal(frames[2, 1:], frames[3, 1:])
            assert not torch.equal(frames[0, 1], frames[2, 1])
            assert torch.equal(labels[:2, :2], labels[2:, :2])
            assert (labels[:2, 2] != labels[2:, 2]).all()
        audit = data.shortcut_audit()
        assert audit["appearance_only_side_accuracy"] == 0.5
        assert audit["initial_frame_only_side_accuracy"] == 0.5
        assert audit["final_frame_only_joint_accuracy"] == 0.5
        assert audit["hidden_frame_only_side_accuracy"] == 0.5
        assert audit["hidden_frame_only_joint_accuracy"] == 0.25
        frame_sets.append(
            {hashlib.sha256(x.tobytes()).hexdigest() for x in data.images[:, 0]}
        )
    assert not frame_sets[0] & frame_sets[1]
    with pytest.raises(ValueError, match="complete"):
        MemoryOutputEpisodes(17, curriculum="relocation")


def test_pairs_require_history_and_target_combinations_are_disjoint():
    from pathwm.data.memory_output import MemoryOutputEpisodes, templates, image_labels

    train = MemoryOutputEpisodes(16, seed=17, split="train")
    test = MemoryOutputEpisodes(16, seed=18, split="test")
    tuples = []
    for data, parity in ((train, 0), (test, 1)):
        b = data.batch(range(len(data)))
        assert torch.equal(b["images"][0::2, 1:], b["images"][1::2, 1:])
        assert (b["labels"][0::2] != b["labels"][1::2]).all()
        assert not torch.equal(b["images"][0::2, 0], b["images"][1::2, 0])
        assert torch.equal(image_labels(b["target"]), b["labels"])
        assert torch.equal(image_labels(templates()[0]), templates()[1])
        c, shape, side = b["labels"].T
        assert ((c % 2) ^ shape ^ side == parity).all()
        tuples.append(set(map(tuple, b["labels"].tolist())))
    assert len(tuples[0]) == len(tuples[1]) == 8
    assert not tuples[0] & tuples[1]
    assert train.identity != test.identity
    with pytest.raises(ValueError):
        MemoryOutputEpisodes(0)


@pytest.mark.parametrize("offset", [-16, 0, 16])
def test_input_offset_preserves_identity_targets_and_counterfactual_ambiguity(offset):
    import hashlib
    from pathwm.data.memory_output import MemoryOutputEpisodes, COLORS

    base = MemoryOutputEpisodes(16, seed=63, split="test", curriculum="relocation")
    shifted = MemoryOutputEpisodes(
        16, seed=63, split="test", curriculum="relocation", input_offset=offset
    )
    assert np.array_equal(shifted.images.astype(np.int16) - offset, base.images)
    assert np.array_equal(shifted.targets, base.targets)
    assert np.array_equal(shifted.labels, base.labels)
    assert shifted.shortcut_audit() == base.shortcut_audit()
    for start in range(0, len(shifted), 4):
        x = shifted.images[start : start + 4]
        assert np.array_equal(x[0, 1:], x[1, 1:])
        assert np.array_equal(x[2, 1:], x[3, 1:])
        assert np.array_equal(x[:2, 0], x[2:, 0])
    colors = COLORS.astype(np.int16) + offset
    distances = ((colors[:, None].astype(float) - COLORS[None]) ** 2).sum(2)
    assert np.array_equal(distances.argmin(1), np.arange(4))
    if offset:
        assert shifted.identity["input_transform"] == dict(
            kind="additive-rgb-offset-v1",
            offset_uint8=offset,
            source_images_sha256=hashlib.sha256(base.images.tobytes()).hexdigest(),
            targets="unchanged canonical rendering",
        )
        assert shifted.identity["images_sha256"] != base.identity["images_sha256"]
    else:
        assert shifted.identity == base.identity


@pytest.mark.parametrize("offset", [-31, 21, 65536, True, 1.5])
def test_input_offset_rejects_clipping_or_noninteger_values(offset):
    from pathwm.data.memory_output import MemoryOutputEpisodes

    with pytest.raises(ValueError, match="offset"):
        MemoryOutputEpisodes(16, seed=63, input_offset=offset)


