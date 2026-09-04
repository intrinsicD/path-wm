"""Thin real-data R0/R1 optimizer and ledger for the ABI-v2 common base.

What: build the common core/EMA learner, record a matched held-out panel before and after optimization,
evaluate the configured fail-closed gate, and save one reproducible checkpoint.
How: train and evaluation use independent seeded streams and manifest splits. Only stage-declared
parameters enter AdamW; EMA teachers update after each optimizer step.
Why: iteration 2 needs one honest number on the instrument panel before B0/D0 work can begin. A dev run
on TAU examples checks plumbing only; the complete corpus is required for promotion (DDR §22).
"""
from __future__ import annotations

import json
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from contracts import RepresentationBatch, TemporalObservation
from evaluation.representation import evaluate_representation
from training.base_model import build_common_world_model
from training.curriculum import CommonBaseCurriculum, GateResult
from training.representation import build_representation_learner


@dataclass(frozen=True)
class CommonBaseTrainingResult:
    checkpoint: Path
    final_training: dict[str, float | int]
    metrics: dict[str, float | int]
    parameter_counts: dict[str, int]
    gate: GateResult


def _move_batch(batch: RepresentationBatch, device: torch.device) -> RepresentationBatch:
    def move(views: Mapping[str, TemporalObservation]) -> dict[str, TemporalObservation]:
        return {
            modality: TemporalObservation(
                observation.values.to(device),
                observation.timestamps.to(device),
                observation.valid_mask.to(device),
            )
            for modality, observation in views.items()
        }

    return RepresentationBatch(move(batch.current), move(batch.future), move(batch.shifted))


def _counts(learner: torch.nn.Module) -> dict[str, int]:
    return {
        "video_encoder": sum(parameter.numel() for parameter in learner.core.encoders["video"].parameters()),
        "audio_encoder": sum(parameter.numel() for parameter in learner.core.encoders["audio"].parameters()),
        "evidence_adapters": sum(parameter.numel() for parameter in learner.core.adapters.parameters()),
        "representation_heads": sum(
            parameter.numel()
            for group in (learner.masked_heads, learner.future_heads, learner.av_projectors)
            for parameter in group.parameters()
        ),
        "common_base_total": sum(parameter.numel() for parameter in learner.parameters()),
        "trainable": sum(parameter.numel() for parameter in learner.parameters() if parameter.requires_grad),
    }


def _panel(
    learner: torch.nn.Module,
    data: Any,
    train_cfg: Mapping[str, Any],
    stage: str,
    seed: int,
) -> dict[str, float]:
    return evaluate_representation(
        learner,
        data,
        stage=stage,
        batches=int(train_cfg["held_out_batches"]),
        batch_size=int(train_cfg["batch_size"]),
        generator=torch.Generator().manual_seed(seed + 8_000_003),
    )


def train_common_base(
    cfg: Mapping[str, Any],
    data: Any,
    run_dir: Path,
    seed: int,
    device: torch.device,
    *,
    resume: bool = False,
) -> CommonBaseTrainingResult:
    """Run one R0/R1 seed, or explicitly resume its atomic optimizer/EMA/RNG snapshot."""
    train_cfg = cfg["train"]
    stage = str(train_cfg["stage"])
    if stage not in {"representation_unimodal", "representation_av"}:
        raise ValueError("the current common-base runner implements only R0/R1")
    if train_cfg["optimizer"] != "adamw":
        raise ValueError("the current common-base runner implements only AdamW")
    max_steps = int(train_cfg["max_steps"])
    batch_size = int(train_cfg["batch_size"])
    log_every = int(train_cfg["log_every"])
    checkpoint_every = int(train_cfg.get("checkpoint_every", max_steps))
    if min(max_steps, batch_size, log_every, checkpoint_every) < 1:
        raise ValueError("max_steps, batch_size, log_every and checkpoint_every must be positive")
    checkpoint = run_dir / "checkpoint.pt"
    training_path = run_dir / "training.jsonl"
    fingerprint = getattr(data, "fingerprint", None)
    saved = None
    if resume:
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if saved["config"] != dict(cfg) or saved["seed"] != seed or saved["stage"] != stage:
            raise ValueError("resume config, seed and stage must exactly match the checkpoint")
        if "data_fingerprint" not in saved or saved["data_fingerprint"] != fingerprint:
            raise ValueError("resume manifest fingerprint differs from the checkpoint")
        if saved.get("device_type") != device.type:
            raise ValueError("resume device type must match for random-stream reproducibility")
    elif checkpoint.exists() or training_path.exists():
        raise FileExistsError(f"run already exists: {run_dir}; use --resume or a new dev spec")

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    core = build_common_world_model(dict(cfg))
    learner = build_representation_learner(dict(cfg), core).to(device)
    learner.set_stage(stage)
    parameter_counts = _counts(learner)
    parameters = [parameter for parameter in learner.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=float(train_cfg["lr"]), weight_decay=float(train_cfg["weight_decay"]))
    data_generator = torch.Generator().manual_seed(seed + 1_000_003)
    corruption_generator = torch.Generator().manual_seed(seed + 2_000_003)
    run_dir.mkdir(parents=True, exist_ok=True)
    start_step = 0
    final_training: dict[str, float | int] = {}
    previous_seconds = 0.0
    if saved is None:
        initial_metrics = _panel(learner, data, train_cfg, stage, seed)
    else:
        learner.load_state_dict(saved["learner"])
        optimizer.load_state_dict(saved["optimizer"])
        data_generator.set_state(saved["data_rng"])
        corruption_generator.set_state(saved["corruption_rng"])
        torch.set_rng_state(saved["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        initial_metrics = saved["initial_metrics"]
        start_step = int(saved["step"])
        final_training = saved["final_training"]
        previous_seconds = float(saved.get("training_seconds", 0.0))
        # A log flush may precede the latest durable optimizer snapshot. Replay those steps once.
        retained = []
        if training_path.exists():
            lines = training_path.read_text().splitlines()
            for index, line in enumerate(lines):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    if index != len(lines) - 1:
                        raise
                    break
                if int(row["step"]) <= start_step:
                    retained.append(json.dumps(row, sort_keys=True) + "\n")
        temporary_log = training_path.with_suffix(".jsonl.tmp")
        temporary_log.write_text("".join(retained), encoding="utf-8")
        temporary_log.replace(training_path)

    panel_path = run_dir / "representation_panel.jsonl"
    initial_row = json.dumps({"step": 0, "metrics": initial_metrics}, sort_keys=True) + "\n"
    if saved is None:
        panel_path.write_text(initial_row, encoding="utf-8")
    learner.train()
    learner.set_stage(stage)
    started = time.monotonic()

    def save_snapshot(step: int, metrics: Mapping[str, float | int] | None = None) -> None:
        snapshot = {
            "config": dict(cfg), "learner": learner.state_dict(), "optimizer": optimizer.state_dict(),
            "stage": stage, "step": step, "seed": seed, "parameter_counts": parameter_counts,
            "metrics": dict(metrics or {}), "initial_metrics": initial_metrics,
            "final_training": final_training, "data_fingerprint": fingerprint, "device_type": device.type,
            "data_rng": data_generator.get_state(), "corruption_rng": corruption_generator.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
            "training_seconds": previous_seconds + time.monotonic() - started,
        }
        temporary_checkpoint = checkpoint.with_suffix(".pt.tmp")
        torch.save(snapshot, temporary_checkpoint)
        temporary_checkpoint.replace(checkpoint)

    if saved is None:
        save_snapshot(0)
    with training_path.open("a", encoding="utf-8") as ledger:
        for step in range(start_step + 1, max_steps + 1):
            batch = _move_batch(data.sample("train", stage, batch_size, data_generator), device)
            optimizer.zero_grad(set_to_none=True)
            autocast = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if device.type == "cuda" and train_cfg["precision"] == "bf16" else nullcontext()
            )
            with autocast:
                losses = learner.loss(batch, stage=stage, generator=corruption_generator)
            if not torch.isfinite(losses["total"]):
                raise ValueError(f"non-finite representation loss at step {step}")
            losses["total"].backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                parameters, float(train_cfg["grad_clip"]), error_if_nonfinite=True,
            )
            optimizer.step()
            learner.update_teachers()
            if step % log_every == 0 or step == max_steps:
                final_training = {
                    "step": step, **{name: float(value.detach()) for name, value in losses.items()},
                    "gradient_norm": float(gradient_norm),
                }
                ledger.write(json.dumps(final_training, sort_keys=True) + "\n")
                ledger.flush()
                print(f"R0/R1 seed={seed} step={step}/{max_steps} loss={final_training['total']:.6f} "
                      f"elapsed={time.monotonic() - started:.1f}s", flush=True)
            if step % checkpoint_every == 0 or step == max_steps:
                save_snapshot(step)

    final_metrics = _panel(learner, data, train_cfg, stage, seed)
    curriculum = CommonBaseCurriculum.from_config(cfg["curriculum"])
    gate = curriculum.evaluate(stage, final_metrics)
    metrics: dict[str, float | int] = {
        **final_metrics, "gate_passed": int(gate.passed), "gate_failure_count": len(gate.failures),
        "held_out_batches": int(train_cfg["held_out_batches"]),
        "held_out_examples": int(train_cfg["held_out_batches"]) * batch_size,
    }
    temporary_panel = panel_path.with_suffix(".jsonl.tmp")
    temporary_panel.write_text(initial_row + json.dumps({"step": max_steps, "metrics": final_metrics}, sort_keys=True) + "\n")
    temporary_panel.replace(panel_path)
    (run_dir / "threshold_record.json").write_text(
        json.dumps({"metrics": metrics, "thresholds": {
            "gate": gate.gate, "passed": gate.passed, "failures": list(gate.failures),
        }}, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    save_snapshot(max_steps, metrics)
    return CommonBaseTrainingResult(checkpoint, final_training, metrics, parameter_counts, gate)
