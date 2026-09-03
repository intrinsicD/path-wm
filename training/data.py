"""Reproducible episode collection and window sampling for the first E1 slice (DDR §13.15).

What: `<env.dataset>/episodes.pt` stores uint8 observation episodes and float32 actions in the fixed
layouts (episodes, L+1, 3, H, W) and (episodes, L, action_dims).
How: each collector reset yields N independent episodes; uniform actions come from a private seeded
generator, and the configured transition count must divide exactly into reset batches.
Why: E1 training and fixed-seed evaluation need the same raw-observation/action interface without
letting environment labels or ground truth enter the world-state objective (Invariant 11).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from envs import build_env
from world_state.abi import load_abi


@dataclass(frozen=True)
class EpisodeStore:
    observations: torch.Tensor
    actions: torch.Tensor

    def __post_init__(self) -> None:
        if self.observations.ndim != 5 or self.actions.ndim != 3:
            raise ValueError("episode tensors must be (E,L+1,C,H,W) and (E,L,A)")
        if self.observations.shape[0] != self.actions.shape[0] or self.observations.shape[1] != self.actions.shape[1] + 1:
            raise ValueError("observation and action episode dimensions do not align")
        if self.observations.dtype != torch.uint8 or self.actions.dtype != torch.float32:
            raise ValueError("episode observations must be uint8 and actions float32")

    @property
    def episode_length(self) -> int:
        return self.actions.shape[1]

    @property
    def transitions(self) -> int:
        return self.actions.shape[0] * self.actions.shape[1]

    def sample(
        self, batch_size: int, horizon: int, generator: torch.Generator
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not 1 <= horizon <= self.episode_length:
            raise ValueError(f"horizon {horizon} outside episode length {self.episode_length}")
        episodes = torch.randint(self.actions.shape[0], (batch_size,), generator=generator)
        starts = torch.randint(self.episode_length - horizon + 1, (batch_size,), generator=generator)
        observation_offsets = torch.arange(horizon + 1)
        action_offsets = torch.arange(horizon)
        observations = self.observations[episodes[:, None], starts[:, None] + observation_offsets]
        actions = self.actions[episodes[:, None], starts[:, None] + action_offsets]
        return observations, actions


def _load(path: Path) -> EpisodeStore:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return EpisodeStore(payload["observations"], payload["actions"])


def collect_episodes(cfg: dict, dataset_dir: Path, seed: int = 0) -> EpisodeStore:
    env_cfg = cfg["env"]
    n_worlds = int(env_cfg["n_worlds"])
    episode_length = int(env_cfg["episode_len"])
    transitions = int(env_cfg["n_transitions"])
    per_reset = n_worlds * episode_length
    if transitions % per_reset:
        raise ValueError(
            f"env.n_transitions {transitions} must equal an integer number of "
            f"n_worlds*episode_len batches ({per_reset})"
        )
    if env_cfg.get("exploration") != "uniform_random":
        raise ValueError("the first-slice collector implements only env.exploration='uniform_random'")

    env = build_env(cfg)
    action_generator = torch.Generator().manual_seed(seed + 1_000_003)
    observation_episodes, action_episodes = [], []
    for batch_index in range(transitions // per_reset):
        frames = [env.reset(seed + batch_index)]
        actions = []
        for _ in range(episode_length):
            action = torch.rand(n_worlds, env.abi.action_dims, generator=action_generator) * 2.0 - 1.0
            actions.append(action)
            frames.append(env.step(action))
        observation_episodes.append(torch.stack(frames, dim=1))
        action_episodes.append(torch.stack(actions, dim=1))

    store = EpisodeStore(torch.cat(observation_episodes), torch.cat(action_episodes))
    if store.transitions != transitions:
        raise RuntimeError(f"collector produced {store.transitions} transitions, expected {transitions}")
    dataset_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "observations": store.observations,
            "actions": store.actions,
            "seed": seed,
            "n_worlds": n_worlds,
            "episode_length": episode_length,
        },
        dataset_dir / "episodes.pt",
    )
    return store


def ensure_episode_store(cfg: dict, root: Path, seed: int = 0) -> EpisodeStore:
    dataset_dir = root / cfg["env"]["dataset"]
    path = dataset_dir / "episodes.pt"
    store = _load(path) if path.exists() else collect_episodes(cfg, dataset_dir, seed)
    expected = int(cfg["env"]["n_transitions"])
    if store.transitions != expected:
        raise ValueError(f"dataset has {store.transitions} transitions, spec requires {expected}")
    expected_length = int(cfg["env"]["episode_len"])
    if store.episode_length != expected_length:
        raise ValueError(f"dataset episode length {store.episode_length} != spec {expected_length}")
    expected_shape = (3, int(cfg["env"]["resolution"]), int(cfg["env"]["resolution"]))
    if tuple(store.observations.shape[2:]) != expected_shape:
        raise ValueError(f"dataset observation shape {tuple(store.observations.shape[2:])} != {expected_shape}")
    action_dims = load_abi(root / cfg["abi"]).action_dims
    if store.actions.shape[-1] != action_dims:
        raise ValueError(f"dataset action width {store.actions.shape[-1]} != ABI {action_dims}")
    return store
