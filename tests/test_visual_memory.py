import json

import numpy as np
import pytest
import torch
from torch.nn import functional as F


def test_rendered_pairs_are_visible_balanced_and_split_isolated():
    from pathwm.data.visual_memory import VisualMemoryEpisodes

    a = VisualMemoryEpisodes(8, seed=3101)
    b = VisualMemoryEpisodes(8, seed=3201)
    assert set(a.identity["scene_sha256"]).isdisjoint(b.identity["scene_sha256"])
    assert set(a.identity["final_sha256"]).isdisjoint(b.identity["final_sha256"])
    batch = a.batch(range(16))
    assert batch["images"].shape == (16, 4, 3, 32, 32)
    assert batch["labels"].tolist() == [0, 1] * 8
    for i in range(0, 16, 2):
        left, right = batch["images"][i : i + 2]
        if (i // 2) % 2 == 0:
            assert torch.equal(left[0], right[0])
        else:
            assert torch.equal(left[0], right[1])
            assert torch.equal(right[0], left[1])
        assert torch.equal(left[2:], right[2:])
        delta = (left[1] - right[1]).abs().sum(0)
        assert delta[:, :16].sum() > 0 and delta[:, 16:].sum() > 0
        # The target's bright red center must actually be visible on the named side.
        for side in (0, 1):
            frame = batch["images"][i + side, 1]
            red = (frame[0] > 0.75) & (frame[1] < 0.35) & (frame[2] < 0.35)
            assert red.any()
            assert (
                (red.nonzero()[:, 1] < 16).all()
                if side == 0
                else (red.nonzero()[:, 1] >= 16).all()
            )


