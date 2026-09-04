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
) -> CommonBaseTrainingResult:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    train_cfg = cfg["train"]
    stage = str(train_cfg["stage"])
    if stage not in {"representation_unimodal", "representation_av"}:
        raise ValueError("the current common-base runner implements only R0/R1")
    if train_cfg["optimizer"] != "adamw":
        raise ValueError("the current common-base runner implements only AdamW")
    max_steps = int(train_cfg["max_steps"])
    batch_size = int(train_cfg["batch_size"])
    log_every = int(train_cfg["log_every"])
    if min(max_steps, batch_size, log_every) < 1:
        raise ValueError("max_steps, batch_size and log_every must be positive")

    core = build_common_world_model(dict(cfg))
    learner = build_representation_learner(dict(cfg), core).to(device)
    learner.set_stage(stage)
    parameter_counts = _counts(learner)
    parameters = [parameter for parameter in learner.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    initial_metrics = _panel(learner, data, train_cfg, stage, seed)
    learner.train()
    learner.set_stage(stage)
    data_generator = torch.Generator().manual_seed(seed + 1_000_003)
    corruption_generator = torch.Generator().manual_seed(seed + 2_000_003)
    training_path = run_dir / "training.jsonl"
    final_training: dict[str, float | int] = {}
    with training_path.open("w", encoding="utf-8") as ledger:
        for step in range(1, max_steps + 1):
            batch = _move_batch(data.sample("train", stage, batch_size, data_generator), device)
            optimizer.zero_grad(set_to_none=True)
            autocast = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if device.type == "cuda" and train_cfg["precision"] == "bf16"
                else nullcontext()
            )
            with autocast:
                losses = learner.loss(batch, stage=stage, generator=corruption_generator)
            losses["total"].backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, float(train_cfg["grad_clip"]))
            optimizer.step()
            learner.update_teachers()
            if step % log_every == 0 or step == max_steps:
                final_training = {
                    "step": step,
                    **{name: float(value.detach()) for name, value in losses.items()},
                    "gradient_norm": float(gradient_norm),
                }
                ledger.write(json.dumps(final_training, sort_keys=True) + "\n")
                ledger.flush()

    final_metrics = _panel(learner, data, train_cfg, stage, seed)
    curriculum = CommonBaseCurriculum.from_config(cfg["curriculum"])
    gate = curriculum.evaluate(stage, final_metrics)
    metrics: dict[str, float | int] = {
        **final_metrics,
        "gate_passed": int(gate.passed),
        "gate_failure_count": len(gate.failures),
        "held_out_batches": int(train_cfg["held_out_batches"]),
        "held_out_examples": int(train_cfg["held_out_batches"]) * batch_size,
    }
    panel_path = run_dir / "representation_panel.jsonl"
    panel_path.write_text(
        json.dumps({"step": 0, "metrics": initial_metrics}, sort_keys=True)
        + "\n"
        + json.dumps({"step": max_steps, "metrics": final_metrics}, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "threshold_record.json").write_text(
        json.dumps(
            {
                "metrics": metrics,
                "thresholds": {
                    "gate": gate.gate,
                    "passed": gate.passed,
                    "failures": list(gate.failures),
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    checkpoint = run_dir / "checkpoint.pt"
    torch.save(
        {
            "config": dict(cfg),
            "learner": learner.state_dict(),
            "optimizer": optimizer.state_dict(),
            "stage": stage,
            "step": max_steps,
            "seed": seed,
            "parameter_counts": parameter_counts,
            "metrics": metrics,
        },
        checkpoint,
    )
    return CommonBaseTrainingResult(checkpoint, final_training, metrics, parameter_counts, gate)
