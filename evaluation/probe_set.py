"""Regenerate the fixed transition probe set named by ABI v1.

The checkpoint's probe seed, count, and horizon completely determine raw RGB/action trajectories.
Ground truth is deliberately not returned: first-slice conformance measures only learned ABI dynamics.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from envs import build_env


@dataclass(frozen=True)
class ProbeSet:
    observations: torch.Tensor  # (count, H+1, 3, resolution, resolution) uint8
    actions: torch.Tensor  # (count, H, action_dims) float32


def generate_probe_set(cfg: dict) -> ProbeSet:
    probe = cfg["probe_set"]
    count, horizon, seed = int(probe["count"]), int(probe["horizon"]), int(probe["seed"])
    if count < 1 or horizon < 1:
        raise ValueError("probe_set.count and probe_set.horizon must be positive")
    env = build_env(cfg)
    generator = torch.Generator().manual_seed(seed + 3_000_003)
    observation_batches, action_batches = [], []
    batches = (count + env.n_worlds - 1) // env.n_worlds
    for batch_index in range(batches):
        frames = [env.reset(seed + batch_index)]
        actions = []
        for _ in range(horizon):
            action = torch.rand(env.n_worlds, env.abi.action_dims, generator=generator) * 2.0 - 1.0
            actions.append(action)
            frames.append(env.step(action))
        observation_batches.append(torch.stack(frames, dim=1))
        action_batches.append(torch.stack(actions, dim=1))
    return ProbeSet(torch.cat(observation_batches)[:count], torch.cat(action_batches)[:count])
