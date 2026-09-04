"""The tiny collector writes the registered episode format and samples aligned windows."""
from __future__ import annotations

import copy

import torch

from envs import build_env
from training.data import (
    collect_episodes,
    ensure_episode_store,
    ensure_paired_intervention_store,
    generate_paired_interventions,
)
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


def test_paired_interventions_replay_every_action_from_the_same_snapshot(cfg):
    spec = copy.deepcopy(cfg)
    spec["abi"] = str(ROOT / spec["abi"])
    spec["env"].update(n_worlds=2, episode_len=3, resolution=16)
    spec["losses"]["counterfactual"]["k"] = 3

    store = generate_paired_interventions(spec, groups=2, seed=7)

    assert store.initial_observations.shape == (2, 3, 16, 16)
    assert store.next_observations.shape == (2, 3, 3, 16, 16)
    assert store.actions.shape == (2, 3, 2)
    assert store.initial_observations.dtype == store.next_observations.dtype == torch.uint8
    assert store.actions.dtype == torch.float32

    # The first collector batch has zero warm-up steps. Replaying each stored action from one reset
    # snapshot must reproduce every stored branch exactly; stepping branches in sequence would fail.
    env = build_env(spec)
    assert torch.equal(store.initial_observations, env.reset(7))
    snapshot = env.save()
    for branch in range(store.branches):
        env.restore(snapshot)
        assert torch.equal(store.next_observations[:, branch], env.step(store.actions[:, branch]))

    initial, following, actions = store.sample(3, torch.Generator().manual_seed(11))
    assert initial.shape == (3, 3, 16, 16)
    assert following.shape == (3, 3, 3, 16, 16)
    assert actions.shape == (3, 3, 2)


def test_paired_intervention_store_is_persisted_and_validated(cfg, tmp_path):
    spec = copy.deepcopy(cfg)
    spec["abi"] = str(ROOT / spec["abi"])
    spec["env"].update(n_worlds=2, episode_len=2, resolution=16, dataset="dataset")
    spec["counterfactual_data"].update(groups=3, seed=17)
    spec["losses"]["counterfactual"]["k"] = 2

    collected = ensure_paired_intervention_store(spec, tmp_path)
    reloaded = ensure_paired_intervention_store(spec, tmp_path)

    assert (tmp_path / "dataset" / "counterfactual.pt").exists()
    assert torch.equal(reloaded.initial_observations, collected.initial_observations)
    assert torch.equal(reloaded.next_observations, collected.next_observations)
    assert torch.equal(reloaded.actions, collected.actions)
    assert reloaded.seed == collected.seed == 17
