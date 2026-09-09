import numpy as np
import pytest
import torch
from pathwm.data.sequences import PushTSequences
from pathwm.models.features import FeatureSpec
from pathwm.models.temporal import MemoryUpdater, Predictor
from pathwm.evaluation.planning import choose_sequence


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


def test_candidate_planner_rejects_nonfinite_and_keeps_caller_state():
    spec = {"map": FeatureSpec(8, (2, 2), "test")}
    u, p = (
        MemoryUpdater(spec, action_width=2, memory_width=16),
        Predictor(spec, action_width=2, memory_width=16),
    )
    features = {"map": torch.zeros(1, 8, 2, 2)}
    memory = torch.zeros(1, 16)
    candidates = torch.tensor([[[0.8, 0.5]], [[0.2, 0.5]], [[float("nan"), 0.5]]])
    chosen, info = choose_sequence(
        features, memory, candidates, p, u, lambda states, actions: actions[:, 0, 0]
    )
    assert info["index"] == 1 and info["invalid_candidates"] == 1
    assert torch.equal(chosen, candidates[1])
    assert not memory.any() and not features["map"].any()
    with pytest.raises(ValueError, match="nonfinite"):
        choose_sequence(
            features,
            memory,
            candidates[2:],
            p,
            u,
            lambda states, actions: actions[:, 0, 0],
        )
