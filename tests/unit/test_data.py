"""The tiny collector writes the registered episode format and samples aligned windows."""
from __future__ import annotations

import copy

import torch

from training.data import collect_episodes, ensure_episode_store
from world_state.abi import ROOT


def test_collect_and_reload_episode_store(cfg, tmp_path):
    spec = copy.deepcopy(cfg)
    spec["abi"] = str(ROOT / spec["abi"])
    spec["env"].update(
        n_worlds=2,
        episode_len=3,
        n_transitions=12,
        resolution=16,
        dataset="dataset",
    )
    store = collect_episodes(spec, tmp_path / "dataset", seed=7)
    assert store.observations.shape == (4, 4, 3, 16, 16)
    assert store.observations.dtype == torch.uint8
    assert store.actions.shape == (4, 3, 2)
    assert store.actions.dtype == torch.float32
    assert store.transitions == 12

    loaded = ensure_episode_store(spec, tmp_path, seed=999)
    assert torch.equal(loaded.observations, store.observations)
    observations, actions = loaded.sample(5, 2, torch.Generator().manual_seed(1))
    assert observations.shape == (5, 3, 3, 16, 16)
    assert actions.shape == (5, 2, 2)
