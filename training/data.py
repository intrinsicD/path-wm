"""Reproducible episode and paired-intervention data for the first E1 slice (DDR §13.15).

What: `<env.dataset>/episodes.pt` stores uint8 observation episodes and float32 actions in the fixed
layouts (episodes, L+1, 3, H, W) and (episodes, L, action_dims). `counterfactual.pt` stores one initial
observation, K action branches and K next observations per intervention group.
How: each collector reset yields N independent episodes; uniform actions come from a private seeded
generator. Each intervention action is stepped only after restoring the same snapshot.
Why: E1 training and fixed-seed evaluation need raw observations/actions without environment labels
entering W (Invariant 11), while §6.4 needs exact shared-cause branches to anchor action semantics.
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


@dataclass(frozen=True)
class PairedInterventionStore:
    initial_observations: torch.Tensor
    next_observations: torch.Tensor
    actions: torch.Tensor
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.initial_observations.ndim != 4 or self.next_observations.ndim != 5 or self.actions.ndim != 3:
            raise ValueError("paired tensors must be (G,C,H,W), (G,K,C,H,W), and (G,K,A)")
        groups = self.initial_observations.shape[0]
        if groups < 1 or self.next_observations.shape[:2] != self.actions.shape[:2]:
            raise ValueError("paired group and branch dimensions do not align")
        if self.next_observations.shape[0] != groups:
            raise ValueError("paired initial and next observations have different group counts")
        if self.next_observations.shape[1] < 2:
            raise ValueError("counterfactual data needs at least two action branches")
        if self.next_observations.shape[2:] != self.initial_observations.shape[1:]:
            raise ValueError("paired initial and next observation shapes do not align")
        if (
            self.initial_observations.dtype != torch.uint8
            or self.next_observations.dtype != torch.uint8
            or self.actions.dtype != torch.float32
        ):
            raise ValueError("paired observations must be uint8 and actions float32")

    @property
    def groups(self) -> int:
        return self.actions.shape[0]

    @property
    def branches(self) -> int:
        return self.actions.shape[1]

    def sample(
        self, batch_size: int, generator: torch.Generator
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        groups = torch.randint(self.groups, (batch_size,), generator=generator)
        return (
            self.initial_observations[groups],
            self.next_observations[groups],
            self.actions[groups],
        )


def _load(path: Path) -> EpisodeStore:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return EpisodeStore(payload["observations"], payload["actions"])


def _load_paired(path: Path) -> PairedInterventionStore:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return PairedInterventionStore(
        payload["initial_observations"],
        payload["next_observations"],
        payload["actions"],
        payload.get("seed"),
    )


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


def generate_paired_interventions(cfg: dict, groups: int, seed: int) -> PairedInterventionStore:
    """Generate K one-step branches from exact E0 snapshots without reading ground truth."""
    if groups < 1:
        raise ValueError(f"counterfactual groups must be positive, got {groups}")
    branches = int(cfg["losses"]["counterfactual"]["k"])
    if branches < 2:
        raise ValueError(f"losses.counterfactual.k must be at least two, got {branches}")

    env = build_env(cfg)
    episode_length = int(cfg["env"]["episode_len"])
    if episode_length < 1:
        raise ValueError(f"env.episode_len must be positive, got {episode_length}")
    action_generator = torch.Generator().manual_seed(seed + 1_000_003)
    initial_batches, next_batches, action_batches = [], [], []
    batches = (groups + env.n_worlds - 1) // env.n_worlds
    for batch_index in range(batches):
        env.reset(seed + batch_index)
        # Cycling across the episode supplies position/velocity contexts without another data-policy knob.
        for _ in range(batch_index % episode_length):
            warmup_action = (
                torch.rand(env.n_worlds, env.abi.action_dims, generator=action_generator) * 2.0 - 1.0
            )
            env.step(warmup_action)
        initial = env.render()
        snapshot = env.save()
        actions = torch.rand(
            env.n_worlds,
            branches,
            env.abi.action_dims,
            generator=action_generator,
        ) * 2.0 - 1.0
        following = []
        for branch in range(branches):
            env.restore(snapshot)
            following.append(env.step(actions[:, branch]))
        initial_batches.append(initial)
        next_batches.append(torch.stack(following, dim=1))
        action_batches.append(actions)

    return PairedInterventionStore(
        torch.cat(initial_batches)[:groups],
        torch.cat(next_batches)[:groups],
        torch.cat(action_batches)[:groups],
        seed,
    )


def collect_paired_interventions(cfg: dict, dataset_dir: Path) -> PairedInterventionStore:
    data_cfg = cfg["counterfactual_data"]
    groups, seed = int(data_cfg["groups"]), int(data_cfg["seed"])
    store = generate_paired_interventions(cfg, groups, seed)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "initial_observations": store.initial_observations,
            "next_observations": store.next_observations,
            "actions": store.actions,
            "groups": groups,
            "branches": store.branches,
            "seed": store.seed,
        },
        dataset_dir / "counterfactual.pt",
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


def ensure_paired_intervention_store(cfg: dict, root: Path) -> PairedInterventionStore:
    dataset_dir = root / cfg["env"]["dataset"]
    path = dataset_dir / "counterfactual.pt"
    store = _load_paired(path) if path.exists() else collect_paired_interventions(cfg, dataset_dir)
    data_cfg = cfg["counterfactual_data"]
    expected_groups = int(data_cfg["groups"])
    expected_seed = int(data_cfg["seed"])
    expected_branches = int(cfg["losses"]["counterfactual"]["k"])
    if store.groups != expected_groups:
        raise ValueError(f"paired dataset has {store.groups} groups, spec requires {expected_groups}")
    if store.branches != expected_branches:
        raise ValueError(f"paired dataset has {store.branches} branches, spec requires {expected_branches}")
    if store.seed != expected_seed:
        raise ValueError(f"paired dataset seed {store.seed} != spec {expected_seed}")
    expected_shape = (3, int(cfg["env"]["resolution"]), int(cfg["env"]["resolution"]))
    if tuple(store.initial_observations.shape[1:]) != expected_shape:
        raise ValueError(
            f"paired dataset observation shape {tuple(store.initial_observations.shape[1:])} "
            f"!= {expected_shape}"
        )
    action_dims = load_abi(root / cfg["abi"]).action_dims
    if store.actions.shape[-1] != action_dims:
        raise ValueError(f"paired dataset action width {store.actions.shape[-1]} != ABI {action_dims}")
    return store
