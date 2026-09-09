"""Perception fitting with visible losses and deterministic direct batch sampling."""

import numpy as np
import torch
from pathwm.io import Run, training_mode, trainable_parameters, evaluation_mode
from pathwm.evaluation.report import write_report, render_report


def step(model, batch, objective, optimizer, clip=1.0):
    training_mode(model)
    optimizer.zero_grad(set_to_none=True)
    losses = objective(model(batch["rgb"]), batch)
    loss = sum(losses.values())
    if not torch.isfinite(loss):
        raise ValueError("Nonfinite perception loss")
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
def evaluate(model, data, objective, device, batch_size=16):
    sums, counts, loss_names = {}, {}, set()
    seen = 0
    with evaluation_mode(model):
        for start in range(0, len(data), batch_size):
            ids = np.arange(start, min(start + batch_size, len(data)))
            batch = data.batch(ids, device)
            if len(batch["rgb"]) != len(ids):
                raise ValueError("Dataset returned a truncated validation batch")
            seen += len(batch["rgb"])
            output = model(batch["rgb"])
            losses = objective(output, batch)
            loss_names.update(losses)
            metrics = dict(losses)
            valid_count = len(ids)
            if "pose" in output:
                metrics["position_mae_normalized"] = (
                    (output["pose"][:, :4] - batch["pose"][:, :4]).abs().mean()
                )
            if "mask" in output:
                valid = batch["valid"] > 0
                keep = valid.flatten(1).any(1)
                valid_count = int(keep.sum())
                predicted = (output["mask"] >= 0) & valid
                target = (batch["mask"] > 0) & valid
                intersection = (predicted & target).flatten(1).sum(1)
                union = (predicted | target).flatten(1).sum(1)
                iou = torch.where(
                    union > 0,
                    intersection / union.clamp_min(1),
                    torch.ones_like(union, dtype=torch.float32),
                )
                metrics["mask_iou"] = iou[keep].mean() if valid_count else iou.sum() * 0
            for key, value in metrics.items():
                weight = valid_count if key in ("mask_bce", "mask_iou") else len(ids)
                sums[key] = sums.get(key, 0.0) + float(value) * weight
                counts[key] = counts.get(key, 0) + weight
    result = {k: sums[k] / counts[k] if counts[k] else 0.0 for k in sums}
    if seen != len(data):
        raise ValueError("Evaluated population differs from declared size")
    result["evaluated_frames"] = seen
    result["loss"] = sum(result[k] for k in sorted(loss_names))
    if "mask_iou" in counts:
        result["mask_valid_frames"] = counts["mask_iou"]
        if not counts["mask_iou"]:
            result["mask_iou"] = None
    return result


def check(model, data, objective, device="cpu", batch_size=2):
    """Real forward/backward without optimizer updates. Caller constructs fresh modules."""
    training_mode(model)
    model.zero_grad(set_to_none=True)
    batch = data.batch(np.arange(min(batch_size, len(data))), device)
    output = model(batch["rgb"])
    losses = objective(output, batch)
    sum(losses.values()).backward()
    result = {
        "features": {k: list(v.shape) for k, v in model.encoder(batch["rgb"]).items()},
        "outputs": {k: list(v.shape) for k, v in output.items()},
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
            for k, m in [("encoder", model.encoder), *model.heads.items()]
        },
    }
    model.zero_grad(set_to_none=True)
    return result


def train_perception(
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
        if run.step == 0 and not run.rows:
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
            ids = run.sample(len(train), settings["batch_size"])
            metrics = step(
                model,
                train.batch(ids, device),
                objective,
                optimizer,
                settings["grad_clip"],
            )
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
        model.eval()
        batch = validation.batch(np.arange(min(6, len(validation))), device)
        with torch.no_grad(), evaluation_mode(model):
            outputs = model(batch["rgb"])
            features = model.encoder(batch["rgb"])
        write_report(run.path, batch, outputs, features)
    except BaseException as exc:
        run.status(result, "failed", f"{type(exc).__name__}: {exc}")
        raise
    return run.path
