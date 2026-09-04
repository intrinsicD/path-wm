"""Evaluate fixed held-out probes at diagnostic E1 checkpoints.

What: turn snapshots selected by ``train.diagnostic_checkpoint_steps`` into a learning-onset curve,
including the ordinary panel and representation-scale diagnostics.
How: regenerate each fixed probe once, load snapshots in step order, and write one authoritative JSON
record per checkpoint under the run directory.
Why: final metrics cannot distinguish a deliberately delayed objective from representation bootstrap
or late optimization. This diagnostic serves E1_reference and does not set ABI thresholds.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from evaluation.metrics import _encode, evaluate_models
from evaluation.probe_set import generate_probe_set
from losses.counterfactual import counterfactual_loss
from training.data import PairedInterventionStore, generate_paired_interventions
from training.model import ModelBundle, load_checkpoint


def representation_diagnostics(
    models: ModelBundle,
    probe: PairedInterventionStore,
    device: torch.device,
    kappa: float,
) -> dict[str, float]:
    """Measure whether W carries scene variation and whether P has reached transition scale."""
    models.eval()
    observations = torch.cat((probe.initial_observations[:, None], probe.next_observations), dim=1)
    with torch.no_grad():
        states = _encode(models, observations, device)
        initial = states[:, 0].to(device)
        targets = states[:, 1:].to(device)
        actions = probe.actions.to(device)
        values = counterfactual_loss(models.predictor, initial, targets, actions, kappa=kappa)
        predictions = torch.stack(
            [
                models.predictor.predict(initial, actions[:, branch : branch + 1], 1)
                for branch in range(probe.branches)
            ],
            dim=1,
        )
        target_separation = torch.stack(
            [
                (targets[:, left].float() - targets[:, right].float()).square().mean()
                for left in range(probe.branches)
                for right in range(left + 1, probe.branches)
            ]
        ).mean()
        prediction_separation = torch.stack(
            [
                (predictions[:, left].float() - predictions[:, right].float()).square().mean()
                for left in range(probe.branches)
                for right in range(left + 1, probe.branches)
            ]
        ).mean()
        centered = initial.float() - initial.float().mean(dim=0, keepdim=True)
        identity = (initial[:, None].float() - targets.float()).square().mean()
    return {
        "state_rms": float(initial.float().square().mean().sqrt()),
        "state_across_example_variance": float(centered.square().mean()),
        "paired_identity_mse": float(identity),
        "paired_positive_mse": float(values["positive_mse"]),
        "target_branch_separation": float(target_separation),
        "predicted_branch_separation": float(prediction_separation),
    }


def evaluate_learning_curve(run_dir: Path, device: torch.device) -> list[dict]:
    checkpoints = sorted((run_dir / "checkpoints").glob("step_*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"no diagnostic checkpoints under {run_dir / 'checkpoints'}")
    cfg, _, _ = load_checkpoint(checkpoints[0], device)
    transition_probe = generate_probe_set(cfg)
    data_cfg = cfg.get("counterfactual_data", {})
    paired_probe = generate_paired_interventions(
        cfg,
        groups=int(data_cfg.get("probe_groups", cfg["probe_set"]["count"])),
        seed=int(data_cfg.get("probe_seed", int(cfg["probe_set"]["seed"]) + 8_000_003)),
    )
    kappa = float(cfg["losses"]["counterfactual"]["kappa"])
    records = []
    for checkpoint_path in checkpoints:
        checkpoint_cfg, models, checkpoint = load_checkpoint(checkpoint_path, device)
        if checkpoint_cfg != cfg:
            raise ValueError(f"checkpoint config changed within learning curve: {checkpoint_path}")
        metrics = evaluate_models(models, transition_probe, device, paired_probe, kappa)
        records.append(
            {
                "step": int(checkpoint["step"]),
                "checkpoint": str(checkpoint_path),
                "metrics": metrics,
                "representation": representation_diagnostics(models, paired_probe, device, kappa),
            }
        )
    output = run_dir / "learning_curve.jsonl"
    output.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in records), encoding="utf-8")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    records = evaluate_learning_curve(args.run_dir, device)
    for record in records:
        metrics, representation = record["metrics"], record["representation"]
        print(
            f"step={record['step']:4d} cf={metrics['counterfactual_accuracy']:.6f} "
            f"one={metrics['transition_error_one_step']:.7f} "
            f"scene_var={representation['state_across_example_variance']:.6f} "
            f"paired={representation['paired_positive_mse']:.7f}"
        )


if __name__ == "__main__":
    main()
