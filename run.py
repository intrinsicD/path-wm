"""Run one PATH-WM experiment or development spec: collect, train, evaluate, record, visualize.

Usage: `python run.py configs/dev/first_slice.yaml`. A config under configs/dev/X.yaml writes to
runs/dev/X/<seed>/; generated datasets and run artifacts are gitignored and dev numbers are labeled.
Every completed seed also refreshes the self-contained visual instrument panel at
``runs/experiment_dashboard.html`` from every completed run ledger.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from evaluation import evaluate_checkpoint
from training import ensure_episode_store, ensure_paired_intervention_store, train_e1
from viewer.dashboard import write_experiment_dashboard

ROOT = Path(__file__).resolve().parent


def run_key(spec_path: Path) -> Path:
    relative = spec_path.resolve().relative_to(ROOT)
    without_suffix = relative.with_suffix("")
    if without_suffix.parts[:2] == ("configs", "dev"):
        return Path(*without_suffix.parts[1:])
    if without_suffix.parts[:1] == ("experiments",):
        return Path(without_suffix.name)
    return Path(without_suffix.name)


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", type=Path)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or a concrete device such as cuda:0")
    args = parser.parse_args()
    spec_path = args.spec if args.spec.is_absolute() else ROOT / args.spec
    cfg = yaml.safe_load(spec_path.read_text())
    device = select_device(args.device)
    store = ensure_episode_store(cfg, ROOT, seed=int(cfg["probe_set"]["seed"]))
    counterfactual_store = None
    counterfactual_cfg = cfg["losses"]["counterfactual"]
    if float(counterfactual_cfg["weight"]) != 0.0 or float(counterfactual_cfg.get("positive_weight", 0.0)) != 0.0:
        counterfactual_store = ensure_paired_intervention_store(cfg, ROOT)

    for seed in cfg["seeds"]:
        run_dir = ROOT / "runs" / run_key(spec_path) / str(seed)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "spec.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        checkpoint, final_training = train_e1(
            cfg,
            store,
            run_dir,
            int(seed),
            device,
            counterfactual_store=counterfactual_store,
        )
        metrics = evaluate_checkpoint(checkpoint, run_dir, device)
        summary = {
            "run_dir": str(run_dir),
            "device": str(device),
            "final_training": final_training,
            "metrics": metrics,
        }
        (run_dir / "run_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        _, dashboard_path, receipt = write_experiment_dashboard(ROOT / "runs")
        print(
            f"Experiment dashboard: {dashboard_path} "
            f"({receipt.get('stages', {}).get('verification', 'unknown')} verification)"
        )


if __name__ == "__main__":
    main()
