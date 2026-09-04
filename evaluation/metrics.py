"""First instrument-panel metrics: action sensitivity and latent transition error (§10, §14).

What: s(w) compares opposite valid actions at the same W; transition error reports one-step and the
configured-H free-running endpoint MSE on a fixed regenerated probe set. Identity, zero-action and
deterministically shuffled-action errors test correctness; paired-branch accuracy tests §6.4 directly.
How: checkpoint modules run in eval/no-grad mode; observations are encoded in bounded batches, and all
latent distances are computed in fp32 on normalized ABI tokens. One-step errors share all count x H
transitions; shuffling rotates trajectories across probe examples while preserving each time index.
Why: opposite-action separation alone can grow even when action semantics are wrong. The controls make
that failure visible and are diagnostics, not frozen thresholds or experiment results under configs/dev/.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from evaluation.probe_set import ProbeSet, generate_probe_set
from losses.counterfactual import counterfactual_loss
from training.data import PairedInterventionStore, generate_paired_interventions
from training.model import ModelBundle, load_checkpoint


def _encode(models: ModelBundle, observations: torch.Tensor, device: torch.device, batch_size: int = 128) -> torch.Tensor:
    shape = observations.shape
    flat = observations.flatten(0, 1)
    encoded = []
    for start in range(0, flat.shape[0], batch_size):
        batch = flat[start : start + batch_size].to(device)
        encoded.append(models.adapter.adapt(models.encoder.encode(batch)).cpu())
    W = torch.cat(encoded)
    return W.reshape(shape[0], shape[1], *W.shape[1:])


def _predict(
    models: ModelBundle,
    states: torch.Tensor,
    actions: torch.Tensor,
    delta_t: int,
    batch_size: int = 128,
) -> torch.Tensor:
    """Run predictor probes in bounded batches without changing their sample-wise mean."""
    predicted = []
    for start in range(0, states.shape[0], batch_size):
        predicted.append(
            models.predictor.predict(
                states[start : start + batch_size],
                actions[start : start + batch_size],
                delta_t,
            )
        )
    return torch.cat(predicted)


def _mse(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (predicted.float() - target.float()).square().mean()


def _counterfactual_accuracy(
    models: ModelBundle,
    probe: PairedInterventionStore,
    device: torch.device,
    kappa: float,
    batch_size: int = 16,
) -> float:
    observations = torch.cat((probe.initial_observations[:, None], probe.next_observations), dim=1)
    states = _encode(models, observations, device)
    correct, total = 0.0, 0
    for start in range(0, probe.groups, batch_size):
        stop = min(start + batch_size, probe.groups)
        values = counterfactual_loss(
            models.predictor,
            states[start:stop, 0].to(device),
            states[start:stop, 1:].to(device),
            probe.actions[start:stop].to(device),
            kappa=kappa,
        )
        decisions = (stop - start) * probe.branches
        correct += float(values["accuracy"]) * decisions
        total += decisions
    return correct / total


def evaluate_models(
    models: ModelBundle,
    probe: ProbeSet,
    device: torch.device,
    counterfactual_probe: PairedInterventionStore | None = None,
    counterfactual_kappa: float | None = None,
) -> dict[str, float | int]:
    models.eval()
    with torch.no_grad():
        states = _encode(models, probe.observations, device)
        actions = probe.actions.to(device)
        W0 = states[:, 0].to(device)
        first_action = actions[:, :1]
        positive = _predict(models, W0, first_action, 1)
        negative = _predict(models, W0, -first_action, 1)
        numerator = (positive.float() - negative.float()).flatten(1).norm(dim=-1)
        denominator = W0.float().flatten(1).norm(dim=-1).clamp_min(1e-12)
        sensitivity = (numerator / denominator).mean()

        # Every control uses exactly the same count x H current/target pairs. Rolling the probe-example
        # axis is a fixed-point-free permutation when count > 1 and preserves the action's time index.
        current = states[:, :-1].flatten(0, 1).to(device)
        target = states[:, 1:].flatten(0, 1).to(device)
        correct_actions = actions.flatten(0, 1).unsqueeze(1)
        zero_actions = torch.zeros_like(correct_actions)
        shuffled_actions = torch.roll(actions, shifts=1, dims=0).flatten(0, 1).unsqueeze(1)
        correct = _predict(models, current, correct_actions, 1)
        zero = _predict(models, current, zero_actions, 1)
        shuffled = _predict(models, current, shuffled_actions, 1)

        one_step_error = _mse(correct, target)
        identity_error = _mse(current, target)
        zero_action_error = _mse(zero, target)
        shuffled_action_error = _mse(shuffled, target)
        predicted = W0
        for step in range(actions.shape[1]):
            predicted = _predict(models, predicted, actions[:, step : step + 1], 1)
        horizon_error = _mse(predicted, states[:, -1].to(device))
        counterfactual_accuracy = None
        if counterfactual_probe is not None:
            if counterfactual_kappa is None:
                raise ValueError("counterfactual probe requires its configured kappa")
            counterfactual_accuracy = _counterfactual_accuracy(
                models,
                counterfactual_probe,
                device,
                counterfactual_kappa,
            )
    metrics: dict[str, float | int] = {
        "action_sensitivity_ratio": float(sensitivity),
        "transition_error": float(horizon_error),
        "transition_error_one_step": float(one_step_error),
        "transition_error_identity": float(identity_error),
        "transition_error_zero_action": float(zero_action_error),
        "transition_error_shuffled_action": float(shuffled_action_error),
        "probe_count": probe.observations.shape[0],
        "probe_horizon": probe.actions.shape[1],
    }
    if counterfactual_accuracy is not None:
        metrics["counterfactual_accuracy"] = counterfactual_accuracy
        metrics["counterfactual_probe_count"] = counterfactual_probe.groups
        metrics["counterfactual_branches"] = counterfactual_probe.branches
    return metrics


def evaluate_checkpoint(
    checkpoint_path: Path,
    run_dir: Path,
    device: torch.device,
) -> dict[str, float | int]:
    cfg, models, checkpoint = load_checkpoint(checkpoint_path, device)
    data_cfg = cfg.get("counterfactual_data", {})
    transition_probe = cfg["probe_set"]
    counterfactual_probe = generate_paired_interventions(
        cfg,
        groups=int(data_cfg.get("probe_groups", transition_probe["count"])),
        seed=int(data_cfg.get("probe_seed", int(transition_probe["seed"]) + 8_000_003)),
    )
    metrics = evaluate_models(
        models,
        generate_probe_set(cfg),
        device,
        counterfactual_probe,
        float(cfg["losses"]["counterfactual"]["kappa"]),
    )
    record = {
        "status": "development" if cfg.get("status") == "dev" else cfg.get("status", "unknown"),
        "seed": int(checkpoint["seed"]),
        "step": int(checkpoint["step"]),
        "checkpoint": str(checkpoint_path),
        "parameter_counts": checkpoint["parameter_counts"],
        "metrics": metrics,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metrics
