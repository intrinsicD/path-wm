"""First instrument-panel metrics: action sensitivity and latent transition error (§10, §14).

What: s(w) compares opposite valid actions at the same W; transition error reports one-step and the
configured-H free-running endpoint MSE on a fixed regenerated probe set.
How: checkpoint modules run in eval/no-grad mode; observations are encoded in bounded batches, and all
latent distances are computed in fp32 on normalized ABI tokens.
Why: these two numbers distinguish an unwired/identity predictor from inaccurate learned dynamics and
are the first slice's diagnostic output, not a frozen experiment result when run from configs/dev/.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from evaluation.probe_set import ProbeSet, generate_probe_set
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


def evaluate_models(
    models: ModelBundle,
    probe: ProbeSet,
    device: torch.device,
) -> dict[str, float | int]:
    models.eval()
    with torch.no_grad():
        states = _encode(models, probe.observations, device)
        actions = probe.actions.to(device)
        W0 = states[:, 0].to(device)
        first_action = actions[:, :1]
        positive = models.predictor.predict(W0, first_action, 1)
        negative = models.predictor.predict(W0, -first_action, 1)
        numerator = (positive.float() - negative.float()).flatten(1).norm(dim=-1)
        denominator = W0.float().flatten(1).norm(dim=-1).clamp_min(1e-12)
        sensitivity = (numerator / denominator).mean()

        one_step_error = (positive.float() - states[:, 1].to(device).float()).square().mean()
        predicted = W0
        for step in range(actions.shape[1]):
            predicted = models.predictor.predict(predicted, actions[:, step : step + 1], 1)
        horizon_error = (predicted.float() - states[:, -1].to(device).float()).square().mean()
    return {
        "action_sensitivity_ratio": float(sensitivity),
        "transition_error": float(horizon_error),
        "transition_error_one_step": float(one_step_error),
        "probe_count": probe.observations.shape[0],
        "probe_horizon": probe.actions.shape[1],
    }


def evaluate_checkpoint(
    checkpoint_path: Path,
    run_dir: Path,
    device: torch.device,
) -> dict[str, float | int]:
    cfg, models, checkpoint = load_checkpoint(checkpoint_path, device)
    metrics = evaluate_models(models, generate_probe_set(cfg), device)
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
