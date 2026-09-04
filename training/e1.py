"""Single-seed E1-a training loop for the first vertical slice.

Raw observations are encoded per sampled window; stage 2 also samples exact paired interventions and
encodes their targets behind stop-gradient. The context encoding can remain trainable or be detached to
route the auxiliary gradient only through P. Independent streams keep ordinary sampling matched.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from losses import build_objective
from losses.e1 import curriculum_horizon
from training.data import EpisodeStore, PairedInterventionStore
from training.model import ModelBundle, build_models, check_parameter_targets, parameter_counts, save_checkpoint
from world_state.abi import ROOT, load_abi


def _encode_window(models: ModelBundle, observations: torch.Tensor) -> torch.Tensor:
    batch, frames = observations.shape[:2]
    flat = observations.flatten(0, 1)
    z = models.encoder.encode(flat)
    W = models.adapter.adapt(z)
    return W.reshape(batch, frames, *W.shape[1:])


def _encode_counterfactuals(
    models: ModelBundle,
    initial: torch.Tensor,
    following: torch.Tensor,
    *,
    context_gradient: bool = True,
) -> torch.Tensor:
    if context_gradient:
        initial_states = models.adapter.adapt(models.encoder.encode(initial))
    else:
        with torch.no_grad():
            initial_states = models.adapter.adapt(models.encoder.encode(initial))
    batch, branches = following.shape[:2]
    # Prediction targets are constants (§6.1). Avoid retaining an encoder graph that detach would discard.
    with torch.no_grad():
        flat = following.flatten(0, 1)
        target_states = models.adapter.adapt(models.encoder.encode(flat))
    target_states = target_states.reshape(batch, branches, *target_states.shape[1:])
    return torch.cat((initial_states[:, None], target_states), dim=1)


def train_e1(
    cfg: dict,
    store: EpisodeStore,
    run_dir: Path,
    seed: int,
    device: torch.device,
    counterfactual_store: PairedInterventionStore | None = None,
) -> tuple[Path, dict[str, float | int]]:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    abi = load_abi(ROOT / cfg["abi"])
    models = build_models(cfg, device)
    counts = parameter_counts(models)
    check_parameter_targets(cfg, counts)
    objective = build_objective(cfg, abi).to(device)
    counterfactual_enabled = (
        objective.counterfactual_weight != 0.0 or objective.counterfactual_positive_weight != 0.0
    )
    if counterfactual_enabled and counterfactual_store is None:
        raise ValueError("an enabled counterfactual objective requires a paired-intervention store")
    optimizer = torch.optim.AdamW(
        models.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )
    total_steps = int(cfg["train"]["steps"])
    batch_size = int(cfg["train"]["batch_size"])
    max_horizon = int(cfg["losses"]["rollout"]["h_train"])
    sample_generator = torch.Generator().manual_seed(seed + 2_000_003)
    counterfactual_generator = torch.Generator().manual_seed(seed + 6_000_003)
    models.train()
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "training.jsonl"
    last_record: dict[str, float | int] = {}

    with log_path.open("w", encoding="utf-8") as log:
        for step in range(total_steps):
            horizon = curriculum_horizon(
                step,
                total_steps,
                max_horizon,
                float(cfg["curriculum"]["stage0_fraction"]),
                float(cfg["curriculum"]["horizon_growth_fraction"]),
            )
            observations, actions = store.sample(batch_size, horizon, sample_generator)
            observations = observations.to(device)
            actions = actions.to(device)
            optimizer.zero_grad(set_to_none=True)
            use_autocast = device.type == "cuda" and cfg["train"]["precision"] == "bf16"
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_autocast):
                states = _encode_window(models, observations)
                counterfactual_states = None
                counterfactual_actions = None
                if objective.counterfactual_active(step, total_steps):
                    assert counterfactual_store is not None
                    paired_batch_size = int(cfg["losses"]["counterfactual"]["batch_size"])
                    paired_initial, paired_following, counterfactual_actions = counterfactual_store.sample(
                        paired_batch_size,
                        counterfactual_generator,
                    )
                    paired_initial = paired_initial.to(device)
                    paired_following = paired_following.to(device)
                    counterfactual_actions = counterfactual_actions.to(device)
                    counterfactual_states = _encode_counterfactuals(
                        models,
                        paired_initial,
                        paired_following,
                        context_gradient=objective.counterfactual_context_gradient,
                    )
                values = objective(
                    models.predictor,
                    models.inverse,
                    states,
                    actions,
                    step=step,
                    total_steps=total_steps,
                    generator=sample_generator,
                    counterfactual_states=counterfactual_states,
                    counterfactual_actions=counterfactual_actions,
                )
            loss = values["total"]
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {step}: {float(loss.detach())}")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                list(models.parameters()), float(cfg["train"]["grad_clip"])
            )
            optimizer.step()

            last_record = {
                key: float(value.detach()) if isinstance(value, torch.Tensor) else (-1 if value is None else int(value))
                for key, value in values.items()
            }
            last_record.update(step=step + 1, gradient_norm=float(gradient_norm))
            if step == 0 or (step + 1) % 50 == 0 or step + 1 == total_steps:
                log.write(json.dumps(last_record, sort_keys=True) + "\n")
                log.flush()

    checkpoint_path = run_dir / "checkpoint.pt"
    save_checkpoint(checkpoint_path, cfg, models, optimizer, total_steps, seed, counts)
    return checkpoint_path, last_record
