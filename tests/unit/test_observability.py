"""E1-a observability diagnostics for the deterministic causal world.

What: prove that one RGB frame aliases hidden velocity, then show that two consecutive positions
recover enough velocity to predict the next agent position far better than the identity shortcut.
How: fork one rendered state with opposite velocities for the impossibility witness; on fresh random
trajectories compare finite-difference history against identity and an action-only zero-velocity model.
Why: a Markov predictor can only be correct if W is a sufficient state. These ground-truth-only probes
decide whether another loss sweep is justified or E1-b needs its planned history-bearing updater.
"""
from __future__ import annotations

from dataclasses import replace

import torch

from envs.causal_world import physics


def test_single_rgb_frame_aliases_hidden_velocity(build, cfg, abi):
    env = build("env")
    env.reset(seed=17)
    snapshot = env.save()
    velocity = torch.zeros_like(snapshot.agent_velocity)
    velocity[:, 0] = 0.8
    positive = replace(snapshot, agent_velocity=velocity)
    negative = replace(snapshot, agent_velocity=-velocity)
    action = torch.zeros(cfg["env"]["n_worlds"], abi.action_dims)

    env.restore(positive)
    positive_current = env.render()
    positive_next = env.step(action)
    env.restore(negative)
    negative_current = env.render()
    negative_next = env.step(action)

    assert torch.equal(positive_current, negative_current)
    assert (positive_next != negative_next).flatten(1).any(dim=1).all()


def test_two_frame_velocity_estimate_beats_single_frame_shortcuts(build, cfg, abi):
    env = build("env")
    generator = torch.Generator().manual_seed(9_000_003)
    errors = {name: [] for name in ("identity", "action_only", "two_frame")}

    for reset_seed in range(2):
        env.reset(10_000 + reset_seed)
        previous_position = env.ground_truth()["full_state"][:, :2]
        first_action = torch.rand(cfg["env"]["n_worlds"], abi.action_dims, generator=generator) * 2.0 - 1.0
        env.step(first_action)
        current_position = env.ground_truth()["full_state"][:, :2]
        for _ in range(15):
            action = torch.rand(cfg["env"]["n_worlds"], abi.action_dims, generator=generator) * 2.0 - 1.0
            estimated_velocity = (current_position - previous_position) / physics.DT
            action_only_velocity = physics.integrate_velocity(
                torch.zeros_like(estimated_velocity), action, physics.AGENT_FRICTION
            )
            history_velocity = physics.integrate_velocity(
                estimated_velocity, action, physics.AGENT_FRICTION
            )
            predictions = {
                "identity": current_position,
                "action_only": current_position + physics.DT * action_only_velocity,
                "two_frame": current_position + physics.DT * history_velocity,
            }
            env.step(action)
            next_position = env.ground_truth()["full_state"][:, :2]
            for name, prediction in predictions.items():
                errors[name].append((prediction - next_position).square().sum(dim=-1))
            previous_position, current_position = current_position, next_position

    means = {name: torch.cat(values).mean() for name, values in errors.items()}
    assert means["two_frame"] < 0.05 * means["identity"]
    assert means["two_frame"] < 0.05 * means["action_only"]
