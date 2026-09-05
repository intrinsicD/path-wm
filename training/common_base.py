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
import hashlib
import io
import math
import time
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from contracts import RepresentationBatch, TemporalObservation
from evaluation.representation import evaluate_representation
from encoders.temporal import CoordinateEmbedding
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
    r0_initialization: dict[str, Any] | None = None


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


def _training_batches(
    data: Any, stage: str, batch_size: int, generator: torch.Generator,
    *, count: int, prefetch: bool,
) -> Generator[RepresentationBatch, None, None]:
    """Yield ordered CPU batches; checkpoints see only randomness already consumed."""
    if not prefetch:
        for _ in range(count):
            yield data.sample("train", stage, batch_size, generator)
        return
    if count <= 0:
        return

    def draw(state: torch.Tensor) -> tuple[RepresentationBatch, torch.Tensor]:
        worker_generator = torch.Generator().set_state(state)
        batch = data.sample("train", stage, batch_size, worker_generator)
        return batch, worker_generator.get_state()

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="av-batch") as pool:
        pending = pool.submit(draw, generator.get_state())
        for index in range(count):
            batch, state = pending.result()
            generator.set_state(state)  # The saved stream belongs to this consumed batch only.
            if index + 1 < count:
                pending = pool.submit(draw, state)
            yield batch



def _r0_initialization_entry(train_cfg: Mapping[str, Any], seed: int) -> dict[str, str] | None:
    sources = train_cfg.get("r0_initialization")
    if train_cfg["stage"] != "representation_av":
        if sources is not None:
            raise ValueError("r0_initialization is only valid for R1 / representation_av")
        return None
    if not isinstance(sources, Mapping) or seed not in sources:
        raise ValueError("fresh R1 requires train.r0_initialization for its seed")
    entry = sources[seed]
    if not isinstance(entry, Mapping) or set(entry) != {"checkpoint", "sha256"}:
        raise ValueError("R0 initialization needs checkpoint and sha256")
    digest = str(entry["sha256"]).lower()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError("R0 initialization sha256 must be a full hexadecimal digest")
    path = Path(str(entry["checkpoint"]))
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[1] / path
    return {"checkpoint": str(path.resolve()), "sha256": digest}


def _load_r0_initialization(
    cfg: Mapping[str, Any], seed: int, fingerprint: str | None, entry: Mapping[str, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Bind the completed R0 measurement and weights to the exact configured bytes (DDR §32)."""
    raw = Path(entry["checkpoint"]).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != entry["sha256"]:
        raise ValueError("R0 source checkpoint SHA-256 differs from initialization")
    source = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    source_cfg = source["config"]
    if source.get("stage") != "representation_unimodal" or source_cfg["train"]["stage"] != "representation_unimodal":
        raise ValueError("R0 source must be representation_unimodal")
    if source.get("step") != source_cfg["train"]["max_steps"] or not source.get("metrics"):
        raise ValueError("R0 source must complete its configured budget and final held-out panel")
    if source.get("seed") != seed:
        raise ValueError("R0 source seed differs from the R1 seed")
    if not fingerprint or source.get("data_fingerprint") != fingerprint:
        raise ValueError("R0 source manifest fingerprint differs from R1 data")
    for name in ("abi", "modalities", "action_adapter", "predictor", "updater", "representation", "data"):
        if source_cfg[name] != cfg[name]:
            raise ValueError(f"R0 source model/data config differs at {name}")
    gate_name = "unimodal_representation_ready"
    if source_cfg["curriculum"]["gates"][gate_name] != cfg["curriculum"]["gates"][gate_name]:
        raise ValueError("R0 source thresholds differ from the R1 prerequisite thresholds")
    gate = CommonBaseCurriculum.from_config(source_cfg["curriculum"]).evaluate(
        "representation_unimodal", source["metrics"])
    if not gate.passed or source["metrics"].get("gate_passed") != 1:
        raise ValueError(f"R0 source gate failed: {gate.failures}")
    expected_batches = int(source_cfg["train"]["held_out_batches"])
    if (source["metrics"].get("held_out_batches") != expected_batches or
            source["metrics"].get("held_out_examples") != expected_batches * int(source_cfg["train"]["batch_size"])):
        raise ValueError("R0 source panel cohort differs from its configured evaluation budget")
    provenance = {**entry, "stage": source["stage"], "step": source["step"], "seed": seed,
                  "data_fingerprint": fingerprint, "metrics": dict(source["metrics"]),
                  "gate": {"name": gate.gate, "passed": gate.passed, "failures": list(gate.failures)}}
    return source, provenance


@torch.no_grad()
def _shift_time_embedding(embedding: CoordinateEmbedding, offset: float) -> None:
    """Re-express f(t) as f_new(t-offset), keeping other coordinates intact (DDR §34)."""
    if not isinstance(embedding, CoordinateEmbedding):
        raise TypeError("R0 time transport requires a CoordinateEmbedding")
    projection = embedding.projection
    if projection.bias is None or not math.isfinite(offset):
        raise ValueError("R0 time transport requires a bias and finite offset")
    bands = embedding.bands
    old = projection.weight.detach().double().clone()
    # Match the runtime's fp32 frequencies before doing the coefficient rotation in fp64.
    frequencies = ((2.0 ** torch.arange(bands, device=old.device, dtype=torch.float32)) * math.pi).double()
    phase = frequencies * offset
    sine, cosine = old[:, 1:1 + bands], old[:, 1 + bands:1 + 2 * bands]
    projection.weight[:, 1:1 + bands].copy_(sine * phase.cos() - cosine * phase.sin())
    projection.weight[:, 1 + bands:1 + 2 * bands].copy_(sine * phase.sin() + cosine * phase.cos())
    projection.bias.copy_(projection.bias.double() + old[:, 0] * offset)


def _transport_r0_time_reference(learner: torch.nn.Module, cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Convert online and EMA time features once, only on fresh R0-to-R1 entry."""
    offsets = {"video": 1.0 / float(cfg["data"]["video"]["frames_per_second"]),
               "audio": 1.0 / float(cfg["data"]["audio"]["sample_rate"])}
    modules = []
    for modality, offset in offsets.items():
        for name in (f"core.encoders.{modality}.position", f"core.adapters.{modality}.time_embedding",
                     f"teachers.{modality}.module.encoder.position", f"teachers.{modality}.module.adapter.time_embedding"):
            _shift_time_embedding(learner.get_submodule(name), offset)
            modules.append(name)
    return {"source_reference": "last_sample", "target_reference": "window_end",
            "offset_seconds": offsets, "modules": modules}


def train_common_base(
    cfg: Mapping[str, Any],
    data: Any,
    run_dir: Path,
    seed: int,
    device: torch.device,
    *,
    resume: bool = False,
) -> CommonBaseTrainingResult:
    """Run R0 or a gated R0-initialized R1; resume owns its optimizer/EMA/RNG snapshot."""
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
    prefetch_batches = train_cfg.get("prefetch_batches", 0)
    if prefetch_batches not in (0, 1):
        raise ValueError("prefetch_batches must be 0 or 1")
    if min(max_steps, batch_size, log_every, checkpoint_every) < 1:
        raise ValueError("max_steps, batch_size, log_every and checkpoint_every must be positive")
    checkpoint = run_dir / "checkpoint.pt"
    training_path = run_dir / "training.jsonl"
    fingerprint = getattr(data, "fingerprint", None)
    entry = _r0_initialization_entry(train_cfg, seed)
    source = None
    initialization = None
    saved = None
    if resume:
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if saved["config"] != dict(cfg) or saved["seed"] != seed or saved["stage"] != stage:
            raise ValueError("resume config, seed and stage must exactly match the checkpoint")
        if "data_fingerprint" not in saved or saved["data_fingerprint"] != fingerprint:
            raise ValueError("resume manifest fingerprint differs from the checkpoint")
        if saved.get("device_type") != device.type:
            raise ValueError("resume device type must match for random-stream reproducibility")
        if entry is not None:
            initialization = saved.get("r0_initialization")
            if (not isinstance(initialization, Mapping) or
                    any(initialization.get(key) != value for key, value in entry.items()) or
                    initialization.get("seed") != seed or initialization.get("data_fingerprint") != fingerprint or
                    initialization.get("stage") != "representation_unimodal" or
                    initialization.get("gate", {}).get("passed") is not True):
                raise ValueError("R1 resume snapshot lacks matching R0 initialization provenance")
    elif checkpoint.exists() or training_path.exists():
        raise FileExistsError(f"run already exists: {run_dir}; use --resume or a new dev spec")
    elif entry is not None:
        source, initialization = _load_r0_initialization(cfg, seed, fingerprint, entry)
        # Validate inherited non-collapse guards before doing any R1 optimization.
        CommonBaseCurriculum.from_config(cfg["curriculum"])

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    core = build_common_world_model(dict(cfg))
    learner = build_representation_learner(dict(cfg), core).to(device)
    if source is not None:
        learner.load_state_dict(source["learner"])  # Transfer online/EMA weights, never R0 optimizer/RNG.
        initialization["time_reference_transport"] = _transport_r0_time_reference(learner, cfg)
        del source
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
        if initialization is not None:
            snapshot["r0_initialization"] = dict(initialization)
        temporary_checkpoint = checkpoint.with_suffix(".pt.tmp")
        torch.save(snapshot, temporary_checkpoint)
        temporary_checkpoint.replace(checkpoint)

    if saved is None:
        save_snapshot(0)
    batches = _training_batches(data, stage, batch_size, data_generator,
                                count=max_steps - start_step, prefetch=bool(prefetch_batches))
    # Explicit close joins any pending read even when optimization or checkpoint writing fails.
    with training_path.open("a", encoding="utf-8") as ledger, closing(batches):
        for step, cpu_batch in enumerate(batches, start=start_step + 1):
            batch = _move_batch(cpu_batch, device)
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
    return CommonBaseTrainingResult(checkpoint, final_training, metrics, parameter_counts, gate, initialization)
