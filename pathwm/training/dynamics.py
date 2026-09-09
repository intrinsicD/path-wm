"""Explicit multi-step fitting with a frozen encoder and detached future targets."""

import numpy as np
import torch
from pathwm.io import Run, training_mode, evaluation_mode, trainable_parameters
from pathwm.evaluation.report import write_report, render_report


def predictions_and_targets(model, batch):
    if any(p.requires_grad for p in model.encoder.parameters()):
        raise ValueError(
            "This loop requires frozen E; joint E training needs an explicit target-encoder policy"
        )
    if batch["future"].shape[1] != batch["actions"].shape[1]:
        raise ValueError("Future targets must align with every candidate action")
    states = model(
        batch["history"],
        batch["history_actions"],
        batch["actions"],
        batch["initial_previous_action"],
    )
    with torch.no_grad(), evaluation_mode(model.encoder):
        targets = [
            model.encoder(batch["future"][:, t])
            for t in range(batch["future"].shape[1])
        ]
    return states, targets


def step(model, batch, objective, optimizer, clip=1.0):
    training_mode(model)
    optimizer.zero_grad(set_to_none=True)
    states, targets = predictions_and_targets(model, batch)
    losses = objective(states, targets, batch)
    loss = sum(losses.values())
    if not torch.isfinite(loss):
        raise ValueError("Nonfinite dynamics loss")
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(
        trainable_parameters(model), clip, error_if_nonfinite=True
    )
    optimizer.step()
    return {
        **{k: float(v.detach()) for k, v in losses.items()},
        "loss": float(loss.detach()),
        "grad_norm": float(norm),
    }


@torch.no_grad()
def evaluate(model, data, objective, device, batch_size=4):
    sums = {}
    seen = 0
    with evaluation_mode(model):
        for start in range(0, len(data), batch_size):
            ids = np.arange(start, min(start + batch_size, len(data)))
            batch = data.batch(ids, device)
            if len(batch["history"]) != len(ids):
                raise ValueError("Dataset returned a truncated validation batch")
            seen += len(batch["history"])
            states, targets = predictions_and_targets(model, batch)
            metrics = objective(states, targets, batch)
            metrics["loss"] = sum(metrics.values())
            current = model.encoder(batch["history"][:, -1])
            for t, ((features, _), target) in enumerate(zip(states, targets), 1):
                metrics[f"latent_mse_h{t}"] = torch.stack(
                    [(features[k] - target[k]).square().mean() for k in target]
                ).mean()
                metrics[f"copy_mse_h{t}"] = torch.stack(
                    [(current[k] - target[k]).square().mean() for k in target]
                ).mean()
            for k, v in metrics.items():
                sums[k] = sums.get(k, 0.0) + float(v) * len(ids)
    if seen != len(data):
        raise ValueError("Evaluated population differs from declared size")
    return {**{k: v / len(data) for k, v in sums.items()}, "evaluated_windows": seen}


def check(model, data, objective, device="cpu"):
    training_mode(model)
    model.zero_grad(set_to_none=True)
    batch = data.batch(np.arange(min(2, len(data))), device)
    states, targets = predictions_and_targets(model, batch)
    losses = objective(states, targets, batch)
    sum(losses.values()).backward()
    result = {
        "history": list(batch["history"].shape),
        "history_actions": list(batch["history_actions"].shape),
        "candidate_actions": list(batch["actions"].shape),
        "horizon": len(states),
        "final_features": {k: list(v.shape) for k, v in states[-1][0].items()},
        "targets_detached": all(
            v.grad_fn is None for target in targets for v in target.values()
        ),
        "losses": {k: float(v.detach()) for k, v in losses.items()},
        "modules": {
            k: {
                "parameters": sum(p.numel() for p in m.parameters()),
                "trainable": sum(p.numel() for p in m.parameters() if p.requires_grad),
                "receives_gradient": any(
                    p.grad is not None and bool(p.grad.abs().sum())
                    for p in m.parameters()
                ),
            }
            for k, m in [
                ("encoder", model.encoder),
                ("memory", model.updater),
                ("predictor", model.predictor),
            ]
        },
    }
    model.zero_grad(set_to_none=True)
    return result


def train_dynamics(
    model,
    train,
    validation,
    objective,
    optimizer,
    *,
    settings,
    recipe,
    output,
    device="cpu",
    resume=False,
    stop_after=None,
    decoder=None,
):
    run = Run(
        output,
        settings=settings,
        data={"train": train.identity, "validation": validation.identity},
        recipe=recipe,
        model=model,
        optimizer=optimizer,
        device=device,
        resume=resume,
    )
    try:
        if not run.rows:
            run.log(
                dict(
                    step=0,
                    split="validation",
                    **evaluate(model, validation, objective, device),
                )
            )
            run.save()
        end = (
            min(settings["steps"], run.step + stop_after)
            if stop_after is not None
            else settings["steps"]
        )
        for update in range(run.step + 1, end + 1):
            batch = train.batch(run.sample(len(train), settings["batch_size"]), device)
            metrics = step(model, batch, objective, optimizer, settings["grad_clip"])
            run.step = update
            run.log(
                dict(
                    step=update,
                    split="train",
                    examples=update * settings["batch_size"],
                    **metrics,
                )
            )
            if update % settings["evaluate_every"] == 0 or update == end:
                val = evaluate(model, validation, objective, device)
                run.log(dict(step=update, split="validation", **val))
                print(
                    f"step {update}/{settings['steps']}  validation loss {val['loss']:.6f}",
                    flush=True,
                )
            run.save()
        result = "completed" if run.step == settings["steps"] else "paused"
        run.status(result, "pending")
    except BaseException as exc:
        run.status("failed", "pending", f"{type(exc).__name__}: {exc}")
        try:
            render_report(run.path)
        except Exception as report_error:
            run.status("failed", "failed", f"{exc}; report: {report_error}")
        raise
    try:
        batch = validation.batch(np.arange(min(6, len(validation))), device)
        with torch.no_grad(), evaluation_mode(model):
            states, _ = predictions_and_targets(model, batch)
            # Render predicted state only. Ground-truth future is the comparison image.
            features = states[-1][0]
            outputs = {"rgb": decoder(features)} if decoder is not None else {}
        write_report(run.path, {"rgb": batch["future"][:, -1]}, outputs, features)
    except BaseException as exc:
        run.status(result, "failed", f"{type(exc).__name__}: {exc}")
        raise
    return run.path
