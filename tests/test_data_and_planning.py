import numpy as np
import pytest
import torch
from pathwm.data.sequences import PushTSequences
from pathwm.models.features import FeatureSpec


def test_sequence_windows_align_actions_without_crossing_episode(monkeypatch):
    import pathwm.data.sequences as module

    frames = np.stack([np.full((4, 4, 3), i, dtype="uint8") for i in range(8)])
    episodes = [
        dict(
            split="train",
            source_rows=np.arange(5),
            actions=np.arange(8, dtype="float32").reshape(4, 2),
        ),
        dict(
            split="train",
            source_rows=np.arange(5, 8),
            actions=np.zeros((2, 2), dtype="float32"),
        ),
    ]
    monkeypatch.setattr(
        module, "pusht_records", lambda root: (frames, episodes, {"source": "fixture"})
    )
    ds = PushTSequences("fixture", history=2, horizon=2)
    assert ds.windows == [(0, 0), (0, 1)]
    b = ds.batch([1])
    torch.testing.assert_close(
        b["history"][0, :, 0, 0, 0] * 255, torch.tensor([1.0, 2.0])
    )
    torch.testing.assert_close(
        b["future"][0, :, 0, 0, 0] * 255, torch.tensor([3.0, 4.0])
    )
    assert torch.equal(b["initial_previous_action"], torch.tensor([[0.0, 1.0]]))
    assert torch.equal(b["history_actions"], torch.tensor([[[2.0, 3.0]]]))
    assert torch.equal(b["actions"], torch.tensor([[[4.0, 5.0], [6.0, 7.0]]]))


