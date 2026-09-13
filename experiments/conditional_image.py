"""Train a conditional image-feature generator on a frozen selected-object agent.

The held-out compositions concern this new producer only; the upstream checkpoint
has seen the categories. Targets are canonical synthetic images, not photography.
"""

import argparse
import json
import shutil
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from pathwm.data.memory_output import MemoryOutputEpisodes, image_labels, BACKGROUND
from pathwm.models.memory_output import load_model, frozen_tensors
from pathwm.models.conditional_image import configure_generator, flow_pair
from pathwm.io import (
    Run,
    atomic_json,
    file_hash,
    seed_everything,
    digest,
    source_record,
    environment,
)
from pathwm.evaluation.report import write_report

MODES = ("ordinary", "reset", "reset_erased", "reset_swapped")


def settings(seed=41011, objective="flow"):
    return dict(
        seed=seed,
        steps=1024,
        batch_size=8,
        wall_seconds=300,
        eval_wall_seconds=300,
        lr=0.0003,
        weight_decay=0.0001,
        memory_mib=3072,
        disk_free_gib=3,
        feature_generator=dict(
            depth=1,
            fusion_depth=1,
            objective=objective,
            steps=8,
            seed=13,
            hidden_width=64,
        ),
        objective=objective + " on standardized codec features; frozen agent/codec",
        example_labels={
            "input": "Supervised targets, never generation inputs",
            "rgb": "Generated reset outputs",
        },
    )


def weighted_error(image, target):
    weights = 1 + 9 * (target - BACKGROUND / 255).abs().amax(1, keepdim=True).gt(0.01)
    return ((image - target).square() * weights).sum((1, 2, 3)) / (
        3 * weights.sum((1, 2, 3))
    )


@torch.no_grad()
def cache_context(model, data, device, batch_size=8):
    ids = np.flatnonzero(data.labels[:, 0] % 2 == data.labels[:, 1])
    rows = {k: [] for k in ("ordinary", "reset", "target", "labels")}
    features = {}
    for start in range(0, len(ids), batch_size):
        b = data.batch(ids[start : start + batch_size], device)
        h = model.observe_history(b["images"])
        for mode in ("ordinary", "reset"):
            s = model.query(h["final"], b["images"][:, -1], mode)
            rows[mode].append(
                model.output_normalization(model.working(s)).detach().clone()
            )
        for k in ("target", "labels"):
            rows[k].append(b[k])
        for k, v in model.teacher(b["target"]).items():
            features.setdefault(k, []).append(v)
    return dict(
        **{k: torch.cat(v) for k, v in rows.items()},
        features={k: torch.cat(v) for k, v in features.items()},
        ids=ids.tolist(),
    )


def objective(generator, cache, ids, step):
    ordinary = (torch.arange(len(ids), device=cache["reset"].device) + step) % 2 == 0
    context = torch.where(
        ordinary[:, None, None], cache["ordinary"][ids], cache["reset"][ids]
    )
    target = generator.standardize({k: v[ids] for k, v in cache["features"].items()})
    # Draw in both arms: the data sampler and training RNG streams stay matched.
    progress = torch.rand(len(ids), device=context.device)
    noise = {k: torch.randn_like(v) for k, v in target.items()}
    x, velocity = flow_pair(target, noise, progress)
    if generator.objective == "direct":
        x = {k: torch.zeros_like(v) for k, v in target.items()}
        progress = torch.zeros_like(progress)
        velocity = target
    predicted = generator.field(x, progress, context)
    loss = sum((predicted[k] - velocity[k]).square().mean() for k in target) / len(
        target
    )
    estimate = (
        predicted
        if generator.objective == "direct"
        else {
            k: x[k] + (1 - progress[:, None, None, None]) * predicted[k] for k in target
        }
    )
    endpoint = sum((estimate[k] - target[k]).square().mean() for k in target) / len(
        target
    )
    return loss, dict(
        latent_loss=float(loss.detach()),
        endpoint_mse=float(endpoint.detach()),
        ordinary_examples=int(ordinary.sum()),
    )


@torch.no_grad()
def evaluate(model, data, device, seed=13, batch_size=8, wall_seconds=300):
    """Live memory route; sampling IDs only choose noise, never model inputs."""
    started = perf_counter()
    flags = [(p, p.requires_grad) for p in model.parameters()]
    model.requires_grad_(False).eval()
    arrays = {k: [] for k in ("target", "labels", "teacher_image", "history")}
    for mode in MODES:
        arrays[mode + "_image"] = []
        arrays[mode + "_facts"] = []
    try:
        decoder = model.agent.decoders["image"]
        for start in range(0, len(data), batch_size):
            if perf_counter() - started >= wall_seconds:
                raise RuntimeError("Evaluation wall budget reached")
            ids = list(range(start, min(start + batch_size, len(data))))
            b = data.batch(ids, device)
            h = model.observe_history(b["images"])
            arrays["target"].append(b["target"].cpu())
            arrays["labels"].append(b["labels"].cpu())
            arrays["history"].append(b["images"].cpu())
            arrays["teacher_image"].append(
                decoder.head(model.teacher(b["target"])).cpu()
            )
            for mode in MODES:
                s = model.query(h["final"], b["images"][:, -1], mode)
                tokens = model.output_normalization(model.working(s))
                if hasattr(decoder, "objective"):
                    features = decoder.features(
                        tokens, seed=seed, sample_ids=[i // 2 for i in ids]
                    )
                else:
                    features = decoder.features(tokens)
                arrays[mode + "_image"].append(decoder.head(features).cpu())
                logits = model.facts(tokens)
                arrays[mode + "_facts"].append(
                    torch.stack(
                        [v.argmax(1) for v in logits.split((4, 2, 2), -1)], -1
                    ).cpu()
                )
        return {k: torch.cat(v) for k, v in arrays.items()}
    finally:
        for p, flag in flags:
            p.requires_grad_(flag)


def score(arrays):
    labels = arrays["labels"]
    result = {}
    for mode in ("teacher", *MODES):
        image = arrays[mode + "_image"]
        pred = image_labels(image)
        expected = (
            labels[torch.arange(len(labels)) ^ 1] if mode == "reset_swapped" else labels
        )
        expected_image = (
            arrays["target"][torch.arange(len(labels)) ^ 1]
            if mode == "reset_swapped"
            else arrays["target"]
        )
        seen = expected[:, 0] % 2 == expected[:, 1]
        for name, mask in [
            ("all", torch.ones(len(labels), dtype=torch.bool)),
            ("seen", seen),
            ("unseen", ~seen),
        ]:
            if not mask.any():
                continue
            row = dict(
                image_accuracy=float(
                    (pred[mask] == expected[mask]).all(1).float().mean()
                ),
                weighted_mse=float(weighted_error(image, expected_image)[mask].mean()),
                count=int(mask.sum()),
            )
            if mode != "teacher":
                row["factual_accuracy"] = float(
                    (arrays[mode + "_facts"][mask] == expected[mask])
                    .all(1)
                    .float()
                    .mean()
                )
            result[mode + "/" + name] = row
    return result


def capability(metrics):
    return all(
        metrics[m + "/" + s]["image_accuracy"] >= 0.95
        and metrics[m + "/" + s]["weighted_mse"] <= 0.01
        for m in ("ordinary", "reset")
        for s in ("seen", "unseen")
    ) and (
        metrics["reset/all"]["image_accuracy"]
        - metrics["reset_erased/all"]["image_accuracy"]
        >= 0.25
        and metrics["reset_swapped/all"]["image_accuracy"] >= 0.95
    )


def check_budget(output, config, device):
    if shutil.disk_usage(Path(output).parent).free < config["disk_free_gib"] * 1024**3:
        raise RuntimeError("Disk headroom floor reached")
    if torch.device(device).type == "cuda":
        free, _ = torch.cuda.mem_get_info(device)
        if (
            free < 1024**3
            or torch.cuda.max_memory_reserved(device) > config["memory_mib"] * 1024**2
        ):
            raise RuntimeError("GPU headroom/reserved-memory cap reached")


def train(
    model,
    cache,
    training,
    validation,
    *,
    output,
    config,
    device="cpu",
    resume=False,
    stop_after=None,
):
    decoder = model.agent.decoders["image"]
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )
    identity = dict(
        training=training.identity,
        validation=validation.identity,
        train_ids=cache["ids"],
        context_sha256=digest(
            {k: cache[k].cpu().tolist() for k in ("ordinary", "reset")}
        ),
        target_sha256=digest(cache["target"].cpu().tolist()),
    )
    run = Run(
        output,
        settings=config,
        data=identity,
        recipe=__file__,
        model=model,
        optimizer=optimizer,
        device=device,
        resume=resume,
    )
    frozen = {k: v.clone() for k, v in frozen_tensors(model).items()}
    prior = sum(r.get("elapsed_seconds", 0) for r in run.rows if r["split"] == "timing")
    start = perf_counter()
    end = (
        config["steps"]
        if stop_after is None
        else min(config["steps"], run.step + stop_after)
    )
    try:
        for step in range(run.step + 1, end + 1):
            if prior + perf_counter() - start >= config["wall_seconds"]:
                break
            ids = run.sample(len(cache["target"]), config["batch_size"])
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = objective(decoder, cache, ids, step)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite generator objective")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                1,
                error_if_nonfinite=True,
            )
            optimizer.step()
            run.step = step
            run.log(
                dict(
                    step=step,
                    split="train",
                    loss=float(loss.detach()),
                    grad_norm=float(norm),
                    **metrics,
                )
            )
            if step % 128 == 0 or step == end:
                check_budget(run.path, config, device)
                run.save()
                print(
                    f"{decoder.objective} step {step}: objective {float(loss.detach()):.6f}",
                    flush=True,
                )
        run.log(
            dict(step=run.step, split="timing", elapsed_seconds=perf_counter() - start)
        )
        run.save()
        if not all(torch.equal(v, frozen_tensors(model)[k]) for k, v in frozen.items()):
            raise RuntimeError("Frozen agent/codec changed")
        torch.save(
            dict(model=model.state_dict(), settings=config), run.path / "weights.pt"
        )
        arrays = evaluate(model, validation, device)
        metrics = score(arrays)
        completed = run.step == config["steps"]
        partial = run.step == end and end < config["steps"]
        atomic_json(
            run.path / "result.json",
            dict(
                completed=completed,
                paused=partial,
                step=run.step,
                metrics=metrics,
                capability_gate=completed and capability(metrics),
                gate=completed and capability(metrics),
                frozen_equal=True,
                peak_reserved_mib=torch.cuda.max_memory_reserved(device) / 1024**2
                if str(device).startswith("cuda")
                else 0,
                training_seconds=sum(
                    r.get("elapsed_seconds", 0)
                    for r in run.rows
                    if r["split"] == "timing"
                ),
                evaluation_scope="Validation diagnostic only; upstream saw categories; no photographic quality claim",
            ),
        )
        np.savez_compressed(
            run.path / "validation.npz", **{k: v.numpy() for k, v in arrays.items()}
        )
        run.status(
            "completed" if completed else "paused" if partial else "budget-stopped",
            "pending",
        )
        write_report(
            run.path,
            batch={"rgb": arrays["target"][:8]},
            outputs={"rgb": arrays["reset_image"][:8]},
        )
        return run
    except BaseException as e:
        run.save()
        status = json.loads((run.path / "status.json").read_text())
        if status["result"] in ("completed", "paused", "budget-stopped"):
            run.status(status["result"], "failed", str(e))
        else:
            run.status("failed", "pending", str(e))
        raise


def build(source, training, config, device="cpu"):
    seed_everything(config["seed"])
    model = load_model(source, device).requires_grad_(False).eval()
    if hasattr(model.agent.decoders["image"], "objective"):
        raise ValueError("Initialize from the preserved deterministic source")
    cache = cache_context(model, training, device)
    generator = configure_generator(model, config["feature_generator"])
    generator.calibrate(cache["features"])
    # Model initialization/targets/context have no test dependency.
    model.eval()
    return model, cache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--objective", choices=("flow", "direct"), default="flow")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--development", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--test-seed", type=int, default=41073)
    parser.add_argument("--sample-seed", type=int, default=13)
    args = parser.parse_args()
    if args.evaluate_only:
        args.output.mkdir(parents=True, exist_ok=False)
        start = perf_counter()
        seed_everything(41011)
        model = load_model(args.weights, args.device).requires_grad_(False).eval()
        config = torch.load(args.weights, map_location="cpu", weights_only=True)[
            "settings"
        ]
        data = MemoryOutputEpisodes(
            32, seed=args.test_seed, curriculum="relocation", split="test"
        )
        arrays = evaluate(model, data, args.device, args.sample_seed)
        metrics = score(arrays)
        elapsed = perf_counter() - start
        np.savez_compressed(
            args.output / "predictions.npz", **{k: v.numpy() for k, v in arrays.items()}
        )
        atomic_json(
            args.output / "result.json",
            dict(
                completed=True,
                metrics=metrics,
                capability_gate=capability(metrics),
                gate=capability(metrics),
                elapsed_seconds=elapsed,
                weights_sha256=file_hash(args.weights),
                data=data.identity,
                sample_seed=args.sample_seed,
                evaluation_scope="Fresh histories; generator-held-out compositions only; all upstream categories familiar. Swap metrics use counterfactual targets.",
            ),
        )
        atomic_json(
            args.output / "run.json",
            dict(
                identity=dict(
                    settings=config,
                    data=data.identity,
                    environment=environment(args.device),
                ),
                source=source_record(__file__, model),
            ),
        )
        atomic_json(
            args.output / "status.json",
            dict(result="completed", report="pending", step=0, error=None),
        )
        (args.output / "metrics.jsonl").write_text("")
        write_report(
            args.output,
            batch={"rgb": arrays["target"][:8]},
            outputs={"rgb": arrays["reset_image"][:8]},
        )
        print(
            json.dumps(
                dict(
                    capability=capability(metrics),
                    seconds=elapsed,
                    reset=metrics["reset/all"],
                )
            ),
            flush=True,
        )
        return
    source_config = torch.load(args.weights, map_location="cpu", weights_only=True)[
        "settings"
    ]
    config = dict(source_config, **settings(objective=args.objective))
    config.update(
        source_weights_sha256=file_hash(args.weights),
        source_weights_path=str(args.weights.resolve()),
    )
    if args.development:
        config.update(seed=41021, steps=16, wall_seconds=120, disk_free_gib=3)
    training = MemoryOutputEpisodes(
        16 if args.development else 128,
        seed=41003 if args.development else 41001,
        curriculum="relocation",
    )
    validation = MemoryOutputEpisodes(
        16,
        seed=41004 if args.development else 41002,
        curriculum="relocation",
        split="test",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if str(args.device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(args.device)
    check_budget(args.output, config, args.device)
    model, cache = build(args.weights, training, config, args.device)
    train(
        model,
        cache,
        training,
        validation,
        output=args.output,
        config=config,
        device=args.device,
        resume=args.resume,
        stop_after=args.stop_after,
    )


if __name__ == "__main__":
    main()
