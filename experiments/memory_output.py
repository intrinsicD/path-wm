"""Small selected-object history -> factual answer and state-produced RGB experiment."""

import argparse
import json
import math
import hashlib
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F

from pathwm.data.memory_output import MemoryOutputEpisodes, image_labels, BACKGROUND
from pathwm.evaluation.report import write_report
from pathwm.io import (
    Run,
    atomic_json,
    file_hash,
    seed_everything,
    state_hash,
    source_record,
    environment,
)
from pathwm.models.memory_output import (
    make_codec,
    build_model,
    load_model as load_model,
    configure_recall_repair,
    frozen_tensors,
    load_workspace_reference,
    configure_output_readout,
)
from pathwm.models.modalities import Observation


def fact_labels(logits):
    return torch.stack([x.argmax(1) for x in logits.split((4, 2, 2), -1)], -1)


def factual_loss(logits, target):
    return (
        sum(
            F.cross_entropy(x, target[:, i])
            for i, x in enumerate(logits.split((4, 2, 2), -1))
        )
        / 3
    )


def weighted_error(image, target):
    weights = 1 + 9 * (target - BACKGROUND / 255).abs().amax(1, keepdim=True).gt(0.01)
    return ((image - target).square() * weights).sum((1, 2, 3)) / (
        3 * weights.sum((1, 2, 3))
    )


def objective(model, batch, reset, repair=False, reference_weight=0.0):
    if not math.isfinite(reference_weight) or reference_weight < 0:
        raise ValueError("Reference loss weight must be finite and nonnegative")
    if reference_weight and (not repair or model.workspace_reference is None):
        raise ValueError("Reference supervision requires repair and a frozen reader")
    history = model.observe_history(batch["images"])
    state = model.query(
        history["final"], batch["images"][:, -1], "reset" if reset else "ordinary"
    )
    output = model.output(state)
    with torch.no_grad():
        targets = model.teacher(batch["target"])
    stored = model.facts(model.output_normalization(model.working(history["stored"])))
    facts = factual_loss(output["facts"], batch["labels"])
    write = factual_loss(stored, batch["labels"])
    pixels = weighted_error(output["image"], batch["target"]).mean()
    latent = model.agent.decoders["image"].latent_loss(output["features"], targets)
    reference = (
        factual_loss(output["reference_facts"], batch["labels"])
        if model.workspace_reference is not None
        else facts.new_zeros(())
    )
    loss = facts + (0.0 if repair else 0.5) * write + pixels + 0.1 * latent
    if reference_weight:
        loss = loss + reference_weight * reference
    return loss, dict(
        reference_loss=float(reference.detach()),
        factual_loss=float(facts.detach()),
        write_loss=float(write.detach()),
        rgb_loss=float(pixels.detach()),
        latent_loss=float(latent.detach()),
    )


@torch.no_grad()
def cache_readout(model, data, device, batch_size=16, mode="reset"):
    """Detached training inputs; labels/teacher targets never enter state formation."""
    values = {k: [] for k in ("tokens", "labels", "target")}
    if mode == "mixed":
        values["ordinary_tokens"] = []
    features = {}
    for start in range(0, len(data), batch_size):
        b = data.batch(range(start, min(start + batch_size, len(data))), device)
        h = model.observe_history(b["images"])
        tokens = model.working(
            model.query(
                h["final"], b["images"][:, -1], "reset" if mode == "mixed" else mode
            )
        )
        if mode == "mixed":
            ordinary = model.working(
                model.query(h["final"], b["images"][:, -1], "ordinary")
            )
            values["ordinary_tokens"].append(ordinary.detach().clone())
        for k, v in dict(tokens=tokens, labels=b["labels"], target=b["target"]).items():
            values[k].append(v.detach().clone())
        for k, v in model.teacher(b["target"]).items():
            features.setdefault(k, []).append(v.detach().clone())
    return dict(
        **{k: torch.cat(v) for k, v in values.items()},
        features={k: torch.cat(v) for k, v in features.items()},
    )


def readout_objective(model, cache, ids, *, context="reset", step=1):
    """Equal total presentations; mixed batches replace half with ordinary states."""
    if context not in ("reset", "mixed"):
        raise ValueError("Readout context must be reset or mixed")
    tokens = cache["tokens"][ids]
    ordinary = torch.zeros(len(ids), dtype=torch.bool, device=tokens.device)
    if context == "mixed":
        if not len(ids) or len(ids) % 2:
            raise ValueError("Mixed readout requires a nonempty even batch")
        ordinary = (torch.arange(len(ids), device=tokens.device) + step) % 2 == 0
        tokens = torch.where(
            ordinary[:, None, None], cache["ordinary_tokens"][ids], tokens
        )
    out = model.output_tokens(tokens)
    facts = factual_loss(out["facts"], cache["labels"][ids])
    pixels = weighted_error(out["image"], cache["target"][ids]).mean()
    latent = model.agent.decoders["image"].latent_loss(
        out["features"], {k: v[ids] for k, v in cache["features"].items()}
    )
    return facts + pixels + 0.1 * latent, dict(
        ordinary_examples=int(ordinary.sum()),
        reset_examples=len(ids) - int(ordinary.sum()),
        first_ordinary=int(ordinary[0]),
        factual_loss=float(facts.detach()),
        rgb_loss=float(pixels.detach()),
        latent_loss=float(latent.detach()),
    )


def live_readout_objective(model, batch, *, step):
    """Recompute differentiable working states after every observer update."""
    images = batch["images"]
    history = model.observe_history(images, memory_grad=True)
    values = dict(labels=batch["labels"], target=batch["target"])
    for mode, name in (("reset", "tokens"), ("ordinary", "ordinary_tokens")):
        values[name] = model.working(model.query(history["final"], images[:, -1], mode))
    with torch.no_grad():
        values["features"] = model.teacher(batch["target"])
    loss, metrics = readout_objective(
        model, values, range(len(images)), context="mixed", step=step
    )
    metrics["memory_replay_verified"] = 1  # observe_history checked each actual write
    return loss, metrics


def default_settings(seed=7801):
    return dict(
        seed=seed,
        steps=1536,
        batch_size=16,
        wall_seconds=300,
        width=32,
        levels=3,
        depth=2,
        fusion_depth=2,
        objective="factor CE + 0.5 write CE + foreground-weighted RGB MSE + 0.1 standardized feature MSE; independent direct-reader CE",
        example_labels={
            "input": "Held-out targets (never query inputs)",
            "rgb": "Reset-recall image outputs",
        },
    )


@torch.no_grad()
def evaluate(model, data, device, modes, batch_size=16):
    # no_grad alone does not normalize trainability-dependent GPU execution.
    # Keep validation/export inference comparable across training policies.
    flags = [(p, p.requires_grad) for p in model.parameters()]
    try:
        model.requires_grad_(False)
        arrays = {
            "target": [],
            "labels": [],
            "history": [],
            "direct_logits": [],
            "stored_logits": [],
            "teacher_image": [],
        }
        if model.workspace_reference is not None:
            arrays["reference_stored_logits"] = []
            for mode in modes:
                arrays[mode + "_reference_logits"] = []
        for mode in modes:
            arrays[mode + "_logits"], arrays[mode + "_image"] = [], []
        for start in range(0, len(data), batch_size):
            batch = data.batch(range(start, min(start + batch_size, len(data))), device)
            images = batch["images"]
            history = model.observe_history(images)
            if model.workspace_reference is not None:
                arrays["reference_stored_logits"].append(
                    model.workspace_reference(model.working(history["stored"])).cpu()
                )
            for key, value in dict(
                target=batch["target"],
                labels=batch["labels"],
                history=images,
                direct_logits=model.direct(model.direct_tokens(images)),
                stored_logits=model.facts(
                    model.output_normalization(model.working(history["stored"]))
                ),
                teacher_image=model.agent.decoders["image"].head(
                    model.teacher(batch["target"])
                ),
            ).items():
                arrays[key].append(value.cpu())
            for mode in modes:
                if mode in ("erased_history", "cue_erased", "last_seen_erased"):
                    output = model(images, mode)
                else:
                    output = model.output(
                        model.query(history["final"], images[:, -1], mode)
                    )
                if model.workspace_reference is not None:
                    arrays[mode + "_reference_logits"].append(
                        output["reference_facts"].cpu()
                    )
                arrays[mode + "_logits"].append(output["facts"].cpu())
                arrays[mode + "_image"].append(output["image"].cpu())
        tensors = {key: torch.cat(value) for key, value in arrays.items()}
        return score(tensors, modes, relocation=data.curriculum == "relocation"), {
            k: v.numpy() for k, v in tensors.items()
        }
    finally:
        for parameter, trainable in flags:
            parameter.requires_grad_(trainable)


def score(arrays, modes, *, relocation=False):
    target, labels = arrays["target"], arrays["labels"]
    paired = torch.arange(len(labels)) ^ 1
    result = {}
    for name in ("direct", "stored"):
        result[name + "_joint_accuracy"] = float(
            (fact_labels(arrays[name + "_logits"]) == labels).all(1).float().mean()
        )
    result["teacher_mse"] = float((arrays["teacher_image"] - target).square().mean())
    result["teacher_image_accuracy"] = float(
        (image_labels(arrays["teacher_image"]) == labels).all(1).float().mean()
    )
    result["background_weighted_mse"] = float(
        weighted_error(torch.full_like(target, BACKGROUND / 255), target).mean()
    )
    result["pair_mean_weighted_mse"] = float(
        weighted_error((target + target[paired]) / 2, target).mean()
    )
    if "reference_stored_logits" in arrays:
        result["reference_stored_accuracy"] = float(
            (fact_labels(arrays["reference_stored_logits"]) == labels)
            .all(1)
            .float()
            .mean()
        )
    for mode in modes:
        pixels, facts = arrays[mode + "_image"], fact_labels(arrays[mode + "_logits"])
        visual = image_labels(pixels)
        fc, vc = (facts == labels).all(1), (visual == labels).all(1)
        result[mode] = dict(
            factual_accuracy=float(fc.float().mean()),
            image_accuracy=float(vc.float().mean()),
            factual_pair_accuracy=float(fc.reshape(-1, 2).all(1).float().mean()),
            image_pair_accuracy=float(vc.reshape(-1, 2).all(1).float().mean()),
            agreement=float((facts == visual).all(1).float().mean()),
            rgb_mse=float((pixels - target).square().mean()),
            weighted_mse=float(weighted_error(pixels, target).mean()),
            alternate_factual_accuracy=float(
                (facts == labels[paired]).all(1).float().mean()
            ),
            alternate_image_accuracy=float(
                (visual == labels[paired]).all(1).float().mean()
            ),
        )
        if mode + "_reference_logits" in arrays:
            correct = fact_labels(arrays[mode + "_reference_logits"]) == labels
            result[mode]["reference_accuracy"] = float(correct.all(1).float().mean())
            result[mode]["reference_side_accuracy"] = float(
                correct[:, 2].float().mean()
            )
        for kind, prediction in (("factual", facts), ("image", visual)):
            for i, name in enumerate(("color", "shape", "side")):
                result[mode][f"{kind}_{name}_accuracy"] = float(
                    (prediction[:, i] == labels[:, i]).float().mean()
                )
        if relocation:
            for kind, correct in (("factual", fc), ("image", vc)):
                quartet = correct.reshape(-1, 2, 2)  # group, movement, selection
                result[mode][f"{kind}_relocation_pair_accuracy"] = float(
                    quartet.all(1).float().mean()
                )
                for moving, name in enumerate(("static", "moved")):
                    result[mode][f"{name}_{kind}_accuracy"] = float(
                        quartet[:, moving].float().mean()
                    )
    return result


def passes(metrics):
    for mode in ("ordinary", "reset"):
        m = metrics[mode]
        if "factual_relocation_pair_accuracy" in m:
            if any(
                m[f"{k}_relocation_pair_accuracy"] < 0.8 for k in ("factual", "image")
            ):
                return False
            if any(
                metrics["reset"][key] - metrics[control][key] < 0.3
                for control in ("cue_erased", "last_seen_erased")
                for key in ("factual_accuracy", "image_accuracy")
            ):
                return False
        if not (
            m["factual_accuracy"] >= 0.9
            and m["image_accuracy"] >= 0.9
            and m["factual_pair_accuracy"] >= 0.8
            and m["image_pair_accuracy"] >= 0.8
        ):
            return False
        for key in ("factual_accuracy", "image_accuracy"):
            if m[key] - metrics["erased_history"][key] < 0.3:
                return False
        if m["weighted_mse"] >= min(
            metrics["background_weighted_mse"], metrics["pair_mean_weighted_mse"]
        ):
            return False
    return all(
        metrics["reset"][key] - metrics["reset_erased"][key] >= 0.3
        for key in ("factual_accuracy", "image_accuracy")
    ) and all(
        metrics["reset_swapped"][key] >= 0.8
        for key in ("alternate_factual_accuracy", "alternate_image_accuracy")
    )


def comparison_panel(path, arrays):
    from PIL import Image, ImageDraw

    columns = [
        "Selection",
        "Last seen",
        "Hidden",
        "Target",
        "Ordinary",
        "Recall",
        "Erase bank",
        "Swap bank",
    ]
    count, cell = min(8, len(arrays["target"])), 96
    canvas = Image.new("RGB", (len(columns) * cell, 26 + count * (cell + 18)), "white")
    draw = ImageDraw.Draw(canvas)
    for j, label in enumerate(columns):
        draw.text((j * cell + 3, 5), label, fill="black")
    for i in range(count):
        row = [
            *arrays["history"][i],
            arrays["target"][i],
            *(
                arrays[k + "_image"][i]
                for k in ("ordinary", "reset", "reset_erased", "reset_swapped")
            ),
        ]
        for j, value in enumerate(row):
            pixels = np.rint(np.clip(value.transpose(1, 2, 0), 0, 1) * 255).astype(
                "uint8"
            )
            canvas.paste(
                Image.fromarray(pixels).resize((cell, cell), Image.Resampling.NEAREST),
                (j * cell, 26 + i * (cell + 18)),
            )
        draw.text(
            (3, 26 + i * (cell + 18) + cell),
            f"Example {i + 1}; color/shape/side {arrays['labels'][i].tolist()}",
            fill="black",
        )
    canvas.save(path)


def train(
    model,
    training,
    validation,
    test,
    *,
    output,
    settings,
    device="cpu",
    resume=False,
    stop_after=None,
):
    cache = None
    writer_learning = settings.get("writer_learning")
    if settings.get("image_only") and (
        writer_learning != "frozen" or settings.get("train_thinker")
    ):
        raise ValueError("Image-only learning requires the frozen live writer policy")
    if settings.get("train_thinker") and writer_learning != "trainable":
        raise ValueError("Joint thinker learning requires a trainable writer")
    readout_context = settings.get("readout_context", "reset")
    if readout_context not in ("reset", "mixed"):
        raise ValueError("Readout context must be reset or mixed")
    if readout_context == "mixed" and (
        settings.get("readout_stage") != "native" or settings["batch_size"] % 2
    ):
        raise ValueError("Mixed context requires native readout and an even batch")
    preparation = perf_counter()
    if writer_learning is not None:
        if writer_learning not in ("frozen", "trainable") or (
            settings.get("readout_stage") != "native"
            or readout_context != "mixed"
            or settings.get("standardize_output")
            or settings.get("reference_weight", 0)
        ):
            raise ValueError(
                "Writer learning requires native raw mixed readout and no auxiliary loss"
            )
        if any(p.requires_grad for p in model.agent.updater.parameters()) != (
            writer_learning == "trainable"
        ):
            raise ValueError("Writer freeze configuration differs from settings")
        if any(p.requires_grad for p in model.agent.thinker.parameters()) != bool(
            settings.get("train_thinker")
        ):
            raise ValueError("Thinker freeze configuration differs from settings")
    if settings.get("readout_stage") and any(
        p.requires_grad != (not settings.get("image_only", False))
        for p in model.facts.parameters()
    ):
        raise ValueError("Factual head freeze configuration differs from settings")
    if settings.get("readout_stage") and writer_learning is None:
        cache = cache_readout(model, training, device, mode=readout_context)
        if settings.get("standardize_output"):
            calibration = cache["tokens"]
            if readout_context == "mixed":
                calibration = torch.cat([calibration, cache["ordinary_tokens"]])
            model.output_normalization.calibrate(calibration)
    preparation_seconds = perf_counter() - preparation
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=0.001,
        weight_decay=0.0001,
    )
    identity = dict(
        train=training.identity, validation=validation.identity, test=test.identity
    )
    if cache is not None:
        tensors = {k: v for k, v in cache.items() if k != "features"}
        tensors.update({"feature." + k: v for k, v in cache["features"].items()})
        identity["readout_cache"] = {
            k: hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest()
            for k, v in tensors.items()
        }
    run = Run(
        output,
        settings=settings,
        data=identity,
        recipe=__file__,
        model=model,
        optimizer=optimizer,
        device=device,
        resume=resume,
    )
    if cache is not None and not resume:
        torch.save(
            {k: v.cpu() for k, v in tensors.items()}, run.path / "training_cache.pt"
        )
        atomic_json(
            run.path / "cache.json",
            dict(
                sha256=identity["readout_cache"],
                preparation_seconds=preparation_seconds,
                data=training.identity,
                stage=settings["readout_stage"],
                context=readout_context,
                standardize_output=settings.get("standardize_output", False),
            ),
        )
    if cache is not None:
        # Each invocation gets a receipt, including the preparation repeated on resume.
        with (run.path / "cache_preparations.jsonl").open("a") as stream:
            stream.write(
                json.dumps(
                    dict(resume=resume, step=run.step, seconds=preparation_seconds)
                )
                + "\n"
            )
    relocation = training.curriculum == "relocation"
    if relocation:
        atomic_json(
            run.path / "shortcut_audit.json",
            {
                name: data.shortcut_audit()
                for name, data in (
                    ("train", training),
                    ("validation", validation),
                    ("test", test),
                )
            },
        )
    frozen = dict(
        encoder=state_hash(model.teacher),
        decoder=state_hash(model.agent.decoders["image"].head),
        input_adapter=state_hash(model.agent.encoders["image"]),
    )
    repair = bool(settings.get("recall_repair"))
    fixed_values = {
        k: v.detach().cpu().clone() for k, v in frozen_tensors(model).items()
    }
    if not resume:
        atomic_json(
            run.path / "initialization.json",
            dict(
                model_sha256=state_hash(model),
                trainable_sha256=hashlib.sha256(
                    b"".join(
                        n.encode() + p.detach().cpu().numpy().tobytes()
                        for n, p in model.named_parameters()
                        if p.requires_grad
                    )
                ).hexdigest(),
                trainable_parameters=sum(
                    p.numel() for p in model.parameters() if p.requires_grad
                ),
                trainable_names=[
                    n for n, p in model.named_parameters() if p.requires_grad
                ],
            ),
        )
    prior = (
        json.loads((run.path / "runtime.json").read_text())["training_seconds"]
        if resume
        else 0.0
    )
    start = perf_counter()
    try:
        end = (
            settings["steps"]
            if stop_after is None
            else min(settings["steps"], run.step + stop_after)
        )
        for step in range(run.step + 1, end + 1):
            if prior + perf_counter() - start >= settings["wall_seconds"]:
                break
            ids = run.sample(len(training), settings["batch_size"])
            optimizer.zero_grad(set_to_none=True)
            if writer_learning is not None:
                loss, metrics = live_readout_objective(
                    model, training.batch(ids, device), step=step
                )
            elif cache is not None:
                loss, metrics = readout_objective(
                    model, cache, ids, context=readout_context, step=step
                )
            else:
                batch = training.batch(ids, device)
                loss, metrics = objective(
                    model,
                    batch,
                    reset=bool(step % 2),
                    repair=repair,
                    reference_weight=settings.get("reference_weight", 0.0),
                )
            direct = (
                loss.new_zeros(())
                if repair
                else factual_loss(
                    model.direct(model.direct_tokens(batch["images"])), batch["labels"]
                )
            )
            if not torch.isfinite(loss + direct):
                raise ValueError("Nonfinite training loss")
            (loss + direct).backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, error_if_nonfinite=True
            )
            optimizer.step()
            run.step = step
            run.log(
                dict(
                    step=step,
                    split="train",
                    loss=float(loss.detach()),
                    direct_loss=float(direct.detach()),
                    **metrics,
                )
            )
            if step % 128 == 0 or step == end:
                scores, values = evaluate(
                    model, validation, device, ["ordinary", "reset"]
                )
                run.log(
                    dict(
                        step=step,
                        split="diagnostic_development",
                        nll=float(
                            factual_loss(
                                torch.from_numpy(values["reset_logits"]),
                                torch.from_numpy(values["labels"]),
                            )
                        ),
                        factual_accuracy=scores["reset"]["factual_accuracy"],
                        image_accuracy=scores["reset"]["image_accuracy"],
                        ordinary_factual_accuracy=scores["ordinary"][
                            "factual_accuracy"
                        ],
                        ordinary_image_accuracy=scores["ordinary"]["image_accuracy"],
                    )
                )
                print(
                    f"step {step}: validation reset facts {scores['reset']['factual_accuracy']:.3f}, image {scores['reset']['image_accuracy']:.3f}, stored {scores['stored_joint_accuracy']:.3f}",
                    flush=True,
                )
                run.save()
                atomic_json(
                    run.path / "runtime.json",
                    dict(training_seconds=prior + perf_counter() - start),
                )
        run.save()
        elapsed = prior + perf_counter() - start
        atomic_json(run.path / "runtime.json", dict(training_seconds=elapsed))
        completed = run.step == settings["steps"]
        # No held-out test exposure during an interrupted training chunk.
        modes = [
            "ordinary",
            "ordinary_no_bank",
            "reset",
            "reset_erased",
            "reset_swapped",
            "erased_history",
        ]
        if relocation:
            modes += ["cue_erased", "last_seen_erased"]
        if repair:
            modes += ["reset_time_erased", "reset_time_swapped"]
        scores, arrays = evaluate(
            model, test if completed else validation, device, modes
        )
        unchanged = frozen == dict(
            encoder=state_hash(model.teacher),
            decoder=state_hash(model.agent.decoders["image"].head),
            input_adapter=state_hash(model.agent.encoders["image"]),
        )
        all_frozen_unchanged = all(
            torch.equal(v.detach().cpu(), fixed_values[k])
            for k, v in frozen_tensors(model).items()
        )
        unchanged = unchanged and all_frozen_unchanged
        if not unchanged or not all(np.isfinite(v).all() for v in arrays.values()):
            raise RuntimeError("Frozen mutation or nonfinite evaluation output")
        if (cache is not None or writer_learning is not None) and completed:
            training_scores, training_arrays = evaluate(
                model, training, device, ["ordinary", "reset"]
            )
            atomic_json(run.path / "training_fit.json", training_scores)
            np.savez_compressed(
                run.path / "training_predictions.npz", **training_arrays
            )
        atomic_json(
            run.path / "result.json",
            dict(
                completed=completed,
                step=run.step,
                gate=completed and passes(scores),
                evaluation_split=(
                    "fresh-background relocation histories; trained tuple and motion support"
                    if relocation
                    else "held-out combinations"
                )
                if completed
                else "validation only; training incomplete",
                metrics=scores,
                frozen_unchanged=unchanged,
                frozen_hashes=frozen,
                all_frozen_tensors_unchanged=all_frozen_unchanged,
                limitations=[
                    "synthetic familiar tuples and motion; no unseen-combination claim"
                    if relocation
                    else "synthetic familiar factors, unseen combinations",
                    "fixed task; no general language",
                    "supplied latest-snapshot route with zero fallback/tie averaging; not learned retrieval"
                    if settings.get("readout_stage") == "stored"
                    else "two supplied snapshots both retrieved; no search or learned write policy",
                    "frozen state formation and factual head; only image feature production learns"
                    if settings.get("image_only")
                    else "shared observer learns across historical writes and current/query observations; not isolated memory-only learning"
                    if writer_learning == "trainable"
                    else "frozen writer; stored logits use the adapting output head, not an independent probe"
                    if repair
                    else "supervised write readout differs from independent direct encoder diagnostic",
                ],
            ),
        )
        np.savez_compressed(run.path / "predictions.npz", **arrays)
        torch.save(
            dict(model=model.state_dict(), settings=settings), run.path / "weights.pt"
        )
        run.status("completed" if completed else "interrupted", "pending")
        comparison_panel(run.path / "comparison.png", arrays)
        write_report(
            run.path,
            {"rgb": torch.from_numpy(arrays["target"][:8])},
            {"rgb": torch.from_numpy(arrays["reset_image"][:8])},
        )
        run.status("completed" if completed else "interrupted", "structural-only")
        return model
    except BaseException as exc:
        status = json.loads((run.path / "status.json").read_text())
        result_done = status["result"] in ("completed", "interrupted")
        run.status(
            status["result"] if result_done else "failed",
            "failed" if result_done else "pending",
            f"{type(exc).__name__}: {exc}",
        )
        raise


def evaluate_export(weights, data, *, output, device="cpu"):
    """Score an unchanged export with separate training and evaluation identities."""
    weights, output = Path(weights).resolve(), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(
        output / "status.json", dict(result="running", report="pending", step=0)
    )
    completed = False
    try:
        started = perf_counter()
        checkpoint_hash = file_hash(weights)
        origin_path = weights.parent / "run.json"
        original = json.loads(origin_path.read_text())
        settings = torch.load(weights, map_location="cpu", weights_only=True)[
            "settings"
        ]
        if original["identity"]["settings"] != json.loads(json.dumps(settings)):
            raise ValueError("Export settings differ from original training manifest")
        model = load_model(weights, device)
        model_hash = state_hash(model)
        source = source_record(__file__, model)
        atomic_json(
            output / "run.json",
            dict(
                identity=dict(
                    settings=dict(
                        purpose="evaluation only; no optimization",
                        example_labels={
                            "input": "Held-out targets (never query inputs)",
                            "rgb": "Reset-recall image outputs",
                        },
                    ),
                    data=dict(test=data.identity),
                    source_sha256=source["sha256"],
                    environment=environment(device),
                ),
                source=source,
                origin=dict(
                    checkpoint_path=str(weights),
                    checkpoint_sha256=checkpoint_hash,
                    run_sha256=file_hash(origin_path),
                    run=original,
                ),
            ),
        )
        (output / "recipe.py").write_text(Path(__file__).read_text())
        (output / "metrics.jsonl").write_text("")
        modes = [
            "ordinary",
            "ordinary_no_bank",
            "reset",
            "reset_erased",
            "reset_swapped",
            "erased_history",
            "cue_erased",
            "last_seen_erased",
            "reset_time_erased",
            "reset_time_swapped",
        ]
        scores, arrays = evaluate(model, data, device, modes)
        if state_hash(model) != model_hash or file_hash(weights) != checkpoint_hash:
            raise RuntimeError("Evaluation mutated the source model")
        if not all(np.isfinite(v).all() for v in arrays.values()):
            raise RuntimeError("Nonfinite evaluation output")
        temporary = output / "predictions.partial.npz"
        np.savez_compressed(temporary, **arrays)
        temporary.replace(output / "predictions.npz")
        atomic_json(
            output / "runtime.json",
            dict(training_seconds=0.0, evaluation_seconds=perf_counter() - started),
        )
        atomic_json(
            output / "resources.json",
            dict(
                peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(device)
                if torch.device(device).type == "cuda"
                else 0,
                training_performed=False,
            ),
        )
        atomic_json(
            output / "result.json",
            dict(
                completed=True,
                evaluation_only=True,
                step=0,
                gate=passes(scores),
                checkpoint_sha256=checkpoint_hash,
                metrics=scores,
                evaluation_scope="Fresh evaluation; original training provenance is in run.json. Zero optimizer updates.",
            ),
        )
        completed = True
        atomic_json(
            output / "status.json", dict(result="completed", report="pending", step=0)
        )
        comparison_panel(output / "comparison.png", arrays)
        write_report(
            output,
            {"rgb": torch.from_numpy(arrays["target"])},
            {"rgb": torch.from_numpy(arrays["reset_image"])},
        )
        atomic_json(
            output / "status.json",
            dict(result="completed", report="structural-only", step=0),
        )
        return scores
    except BaseException as exc:
        atomic_json(
            output / "status.json",
            dict(
                result="completed" if completed else "failed",
                report="failed" if completed else "pending",
                step=0,
                error=f"{type(exc).__name__}: {exc}",
            ),
        )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=7801)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--normalize-input", action="store_true")
    parser.add_argument("--repair", choices=("identity", "calibrated", "temporal"))
    parser.add_argument("--readout-stage", choices=("native", "stored"))
    parser.add_argument(
        "--readout-context", choices=("reset", "mixed"), default="reset"
    )
    parser.add_argument("--standardize-output", action="store_true")
    parser.add_argument("--writer-learning", choices=("frozen", "trainable"))
    parser.add_argument("--train-thinker", action="store_true")
    parser.add_argument(
        "--image-only",
        action="store_true",
        help="Freeze native state and facts; train only image features",
    )
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument(
        "--input-offset",
        type=int,
        default=0,
        help="Evaluation-only observed RGB offset in 8-bit units",
    )
    parser.add_argument("--workspace-reference", type=Path)
    parser.add_argument("--reference-weight", type=float, default=0.0)
    parser.add_argument("--validation-seed", type=int)
    parser.add_argument("--test-seed", type=int)
    parser.add_argument("--development", action="store_true")
    parser.add_argument(
        "--curriculum", choices=("parity", "relocation"), default="parity"
    )
    args = parser.parse_args()
    if args.input_offset and not args.evaluate_only:
        parser.error("Input offset is an evaluation-only override")
    if args.evaluate_only:
        if (
            args.repair
            or args.readout_stage
            or args.writer_learning
            or args.train_thinker
            or args.image_only
            or args.normalize_input
            or args.standardize_output
            or args.workspace_reference
            or args.reference_weight
            or args.resume
            or args.stop_after is not None
            or args.check
            or args.development
            or args.validation_seed is not None
            or args.readout_context != "reset"
            or args.curriculum != "parity"
        ):
            parser.error(
                "Evaluation-only loads export settings; training overrides are not allowed"
            )
        if args.test_seed is None:
            parser.error("Evaluation-only requires an explicit test seed")
        seed_everything(args.seed)
        if torch.device(args.device).type == "cuda":
            free, total = torch.cuda.mem_get_info(args.device)
            limit = min(4 * 1024**3, free - 1024**3)
            if limit <= 0:
                raise RuntimeError("Insufficient GPU headroom")
            torch.cuda.set_per_process_memory_fraction(
                limit / total, torch.device(args.device).index or 0
            )
            torch.cuda.reset_peak_memory_stats(args.device)
        saved = torch.load(args.weights, map_location="cpu", weights_only=True)[
            "settings"
        ]
        data = MemoryOutputEpisodes(
            64,
            seed=args.test_seed,
            split="test",
            curriculum=saved.get("curriculum", "parity"),
            input_offset=args.input_offset,
        )
        evaluate_export(args.weights, data, output=args.output, device=args.device)
        return
    if (
        not math.isfinite(args.reference_weight)
        or args.reference_weight < 0
        or (args.reference_weight and args.workspace_reference is None)
    ):
        parser.error(
            "Reference weight requires a reader and must be finite/nonnegative"
        )
    if args.readout_stage and (args.repair != "identity" or args.reference_weight):
        parser.error("Direct readout requires identity repair and zero auxiliary loss")
    if args.standardize_output and not args.readout_stage:
        parser.error("Output standardization requires a declared readout stage")
    if args.readout_context == "mixed" and args.readout_stage != "native":
        parser.error("Mixed context requires native readout")
    if args.writer_learning is not None and (
        args.readout_stage != "native"
        or args.readout_context != "mixed"
        or args.standardize_output
    ):
        parser.error("Writer learning requires native raw mixed readout")
    if args.image_only and (args.writer_learning != "frozen" or args.train_thinker):
        parser.error("Image-only learning requires --writer-learning frozen")
    if args.train_thinker and args.writer_learning != "trainable":
        parser.error("Joint thinker learning requires --writer-learning trainable")
    if args.workspace_reference is not None and args.repair != "identity":
        parser.error("Workspace supervision currently requires --repair identity")
    if args.curriculum == "relocation" and not args.normalize_input:
        if not args.repair:
            parser.error("The relocation comparison requires --normalize-input")
    if args.development and not args.repair:
        parser.error("--development is scoped to recall repair")
    if args.repair and args.check:
        parser.error("Use --development for the bounded recall-repair check")
    seed_everything(args.seed)
    if torch.device(args.device).type == "cuda":
        free, total = torch.cuda.mem_get_info(args.device)
        limit = min(4 * 1024**3, free - 1024**3)
        if limit <= 0:
            raise RuntimeError("Insufficient GPU headroom")
        index = torch.device(args.device).index or 0
        torch.cuda.set_per_process_memory_fraction(limit / total, index)
        torch.cuda.reset_peak_memory_stats(args.device)
    if args.repair:
        source = torch.load(args.weights, map_location="cpu", weights_only=True)
        if source["settings"].get("curriculum") != "relocation" or source[
            "settings"
        ].get("recall_repair") not in (None, "identity"):
            parser.error(
                "Repair requires an original or untimed identity-relocation checkpoint"
            )
        model = configure_recall_repair(
            load_model(args.weights, args.device),
            relative_time=args.repair == "temporal",
        )
        if args.workspace_reference is not None:
            model.workspace_reference = load_workspace_reference(
                args.workspace_reference, model.agent.width, args.device
            )
        training = MemoryOutputEpisodes(
            128, seed=17701 if args.development else 7701, curriculum="relocation"
        )
        validation = MemoryOutputEpisodes(
            32,
            seed=args.validation_seed
            if args.validation_seed is not None
            else (17722 if args.development else 7722),
            curriculum="relocation",
        )
        test = MemoryOutputEpisodes(
            64,
            seed=args.test_seed
            if args.test_seed is not None
            else (17723 if args.development else 7723),
            split="test",
            curriculum="relocation",
        )
        calibration_start = perf_counter()
        if args.repair == "calibrated":

            @torch.no_grad()
            def values():
                for start in range(0, len(training), 16):
                    batch = training.batch(range(start, start + 16), args.device)
                    yield model.observe_history(batch["images"])["final"].memory.values

            model.agent.memory.calibrate(values())
        settings = dict(source["settings"])
        settings.update(default_settings(args.seed))
        settings.update(
            {
                k: source["settings"][k]
                for k in ("width", "levels", "depth", "fusion_depth")
            }
        )
        settings.update(
            recall_repair=args.repair,
            source_sha256=file_hash(args.weights),
            objective="frozen writer; factor CE + foreground-weighted RGB MSE + 0.1 standardized feature MSE",
            memory_calibration_data=training.identity,
        )
        if args.workspace_reference is not None:
            settings.update(
                workspace_reference_sha256=file_hash(args.workspace_reference),
                reference_weight=args.reference_weight,
                objective=settings["objective"]
                + "; frozen-reader workspace CE x "
                + str(args.reference_weight),
            )
        elif settings.get("workspace_reference_sha256") and not args.readout_stage:
            parser.error(
                "Continuing a supervised export requires its explicit reference"
            )
        if args.readout_stage:
            configure_output_readout(
                model,
                args.readout_stage,
                train_writer=args.writer_learning == "trainable",
                train_thinker=args.train_thinker,
                image_only=args.image_only,
            )
            settings.update(
                readout_stage=args.readout_stage,
                readout_context=args.readout_context,
                writer_learning=args.writer_learning,
                train_thinker=args.train_thinker,
                image_only=args.image_only,
                standardize_output=args.standardize_output,
                reference_weight=0.0,
                wall_seconds=180,
                objective=f"frozen writer and thinker; cached {args.readout_context} tokens; native factor CE + weighted RGB MSE + 0.1 standardized feature MSE",
            )
            if args.writer_learning is not None:
                settings.update(
                    steps=1024,
                    wall_seconds=360,
                    objective=f"{args.writer_learning} shared observer; {'trainable' if args.train_thinker else 'frozen'} thinker; live mixed task CE + weighted RGB MSE + 0.1 standardized feature MSE; ephemeral value gradients",
                )
        if args.image_only:
            settings["objective"] += (
                "; image feature producer only; factual CE constant"
            )
        if args.development:
            settings.update(steps=16, wall_seconds=60)
        calibration_seconds = perf_counter() - calibration_start
        train(
            model,
            training,
            validation,
            test,
            output=args.output,
            settings=settings,
            device=args.device,
            resume=args.resume,
            stop_after=args.stop_after,
        )
        atomic_json(
            args.output / "resources.json",
            dict(
                peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(args.device)
                if torch.device(args.device).type == "cuda"
                else 0,
                calibration_seconds=calibration_seconds,
            ),
        )
        return
    codec = make_codec(weights=args.weights).to(args.device).eval()
    model = (
        build_model(codec, normalize_input=args.normalize_input).to(args.device).eval()
    )
    training = MemoryOutputEpisodes(128, seed=7701, curriculum=args.curriculum)
    validation = MemoryOutputEpisodes(
        32,
        seed=args.validation_seed if args.validation_seed is not None else 7702,
        curriculum=args.curriculum,
    )
    test = MemoryOutputEpisodes(
        64,
        seed=args.test_seed if args.test_seed is not None else 7703,
        split="test",
        curriculum=args.curriculum,
    )
    # Hold both feature calibrations fixed at the original training population.
    calibration = MemoryOutputEpisodes(128, seed=7701)
    if args.normalize_input:

        def observations():
            for start in range(0, len(calibration), 16):
                images = calibration.batch(range(start, start + 16), args.device)[
                    "images"
                ]
                for t in range(3):
                    yield Observation(
                        images[:, t : t + 1], images.new_full((16, 1), float(t))
                    )

        model.agent.encoders["image"].calibrate(observations())
    with torch.no_grad():
        # Eight unique training targets suffice; no validation/test target calibration.
        targets = calibration.batch(range(8), args.device)["target"]
        model.agent.decoders["image"].calibrate(model.teacher(targets))
    settings = default_settings(args.seed)
    settings["normalize_input"] = args.normalize_input
    settings["donor_sha256"] = file_hash(args.weights)
    settings["curriculum"] = args.curriculum
    settings["calibration_data"] = calibration.identity
    if args.check:
        batch = MemoryOutputEpisodes(16, seed=17701, curriculum=args.curriculum).batch(
            range(4), args.device
        )
        loss, metrics = objective(model, batch, reset=True)
        loss.backward()
        args.output.mkdir(parents=True, exist_ok=False)
        atomic_json(
            args.output / "check.json", dict(loss=float(loss.detach()), **metrics)
        )
        print(json.dumps(metrics), flush=True)
        return
    train(
        model,
        training,
        validation,
        test,
        output=args.output,
        settings=settings,
        device=args.device,
        resume=args.resume,
        stop_after=args.stop_after,
    )
    atomic_json(
        args.output / "resources.json",
        dict(
            peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(args.device)
            if torch.device(args.device).type == "cuda"
            else 0
        ),
    )


if __name__ == "__main__":
    main()
