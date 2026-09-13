"""Continue the own image generator on real COCO photo recall, at RGB64.

Photos are shown twice then removed. This is a constructed still-image recall
protocol, not video dynamics or text-to-image generation. Edit settings below.
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from pathwm.data.images import CocoFrames, Frames
from pathwm.models.memory_output import load_model, frozen_tensors, PixelMedianCentering
from pathwm.models.conditional_image import flow_pair, integrate
from pathwm.io import (
    Run,
    atomic_json,
    file_hash,
    seed_everything,
    source_record,
    environment,
)
from pathwm.evaluation.report import write_report


def settings():
    return dict(
        seed=45011,
        steps=2048,
        batch_size=8,
        wall_seconds=600,
        memory_mib=3072,
        disk_free_gib=3,
        lr=0.0003,
        weight_decay=0.0001,
        data_seed=45001,
        train_count=1024,
        validation_count=128,
        test_count=256,
        purpose="real-photo continuation",
        real_photo=True,
        objective="standardized velocity MSE + 10 * uniform RGB MSE through Euler sample",
        example_labels={
            "input": "Previously observed photos (64px crops)",
            "rgb": "Generated from reset state and recalled memory",
        },
    )


def photo_data(root, counts=(1024, 128, 256), seed=45001):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    splits = ("train", "validation", "test")
    records = manifest["records"]
    groups = {s: {records[i]["group"] for i in manifest["splits"][s]} for s in splits}
    if any(
        groups[a] & groups[b]
        for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]
    ):
        raise ValueError("Photo duplicate group overlap across splits")
    if len(counts) != 3 or min(counts) < 1:
        raise ValueError("Positive photo counts required")
    rng = np.random.default_rng(seed)
    result = {}
    # Verify the immutable prepared frame store once, not separately per subset.
    base = CocoFrames(root, "train")
    for split, count in zip(splits, counts):
        rows = []
        used = set()
        for i in rng.permutation(manifest["splits"][split]):
            group = records[i]["group"]
            if group not in used:
                rows.append(int(i))
                used.add(group)
                if len(rows) == count:
                    break
        if len(rows) != count:
            raise ValueError("Not enough distinct photo groups")
        result[split] = Frames(
            base.frames,
            rows,
            identity=dict(
                base.identity,
                split=split,
                selection_seed=seed,
                selected=[dict(row=i, **records[i]) for i in rows],
            ),
        )
    return result


def photo_history(rgb):
    if rgb.ndim != 4 or rgb.shape[1:] != (3, 64, 64):
        raise ValueError("Photo history requires RGB64")
    return torch.stack([rgb, rgb, torch.full_like(rgb, 40 / 255)], 1)


def cache_hash(cache):
    h = hashlib.sha256()
    values = {k: cache[k] for k in ("ordinary", "reset", "target")}
    values.update({"feature." + k: v for k, v in cache["features"].items()})
    for k, v in sorted(values.items()):
        h.update(k.encode())
        h.update(str(v.shape).encode())
        h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


@torch.no_grad()
def prepare(source, training, config, device="cpu"):
    seed_everything(config["seed"])
    model = load_model(source, device).requires_grad_(False).eval()
    config = dict(
        config,
        source_weights_path=str(Path(source).resolve()),
        source_weights_sha256=file_hash(source),
        input_centering=None,
    )
    if isinstance(model.agent.encoders["image"], PixelMedianCentering):
        model.agent.encoders["image"] = model.agent.encoders["image"].base
    g = model.agent.decoders["image"]
    if not hasattr(g, "objective") or g.objective != "flow":
        raise ValueError("Continue an existing conditional flow generator export")
    rows = {k: [] for k in ("ordinary", "reset", "target")}
    features = {}
    for start in range(0, len(training), 8):
        rgb = training.batch(range(start, min(start + 8, len(training))), device)["rgb"]
        history = photo_history(rgb)
        h = model.observe_history(history)
        for mode in ("ordinary", "reset"):
            state = model.query(h["final"], history[:, -1], mode)
            rows[mode].append(model.output_normalization(model.working(state)).detach())
        rows["target"].append(rgb)
        for k, v in model.teacher(rgb).items():
            features.setdefault(k, []).append(v.detach())
    cache = {k: torch.cat(v) for k, v in rows.items()}
    cache["features"] = {k: torch.cat(v) for k, v in features.items()}
    cache["mean"] = cache["target"].mean(0)
    g.calibrate(cache["features"])
    g.requires_grad_(True)
    g.head.requires_grad_(False)
    if not all(
        torch.isfinite(v).all() for v in cache.values() if isinstance(v, torch.Tensor)
    ):
        raise ValueError("Nonfinite real-photo context")
    return model, cache, config


def objective(g, cache, ids, step):
    ordinary = (torch.arange(len(ids), device=cache["target"].device) + step) % 2 == 0
    c = torch.where(
        ordinary[:, None, None], cache["ordinary"][ids], cache["reset"][ids]
    )
    target = g.standardize({k: v[ids] for k, v in cache["features"].items()})
    draw = torch.rand(len(ids), device=c.device)
    t = torch.where(draw < 0.5, 0, 2 * draw - 1)
    noise = {k: torch.randn_like(v) for k, v in target.items()}
    x, v = flow_pair(target, noise, t)
    pred = g.field(x, t, c)
    latent = sum((pred[k] - v[k]).square().mean() for k in target) / len(target)
    sampled = integrate(lambda z, tau: g.field(z, tau, c), noise, g.steps)
    image = g.head(g.unstandardize(sampled))
    rgb = (image - cache["target"][ids]).square().mean()
    return latent + 10 * rgb, dict(
        latent_loss=float(latent.detach()), rgb_loss=float(rgb.detach())
    )


@torch.no_grad()
def evaluate(model, data, mean, device="cpu", sample_seed=13):
    if len(data) % 2:
        raise ValueError("Photo swap evaluation needs even counts")
    flags = [(p, p.requires_grad) for p in model.parameters()]
    model.requires_grad_(False).eval()
    rows = {
        k: []
        for k in (
            "target",
            "teacher",
            "mean",
            "ordinary",
            "reset",
            "erased",
            "swapped",
            "blind",
        )
    }
    started = perf_counter()
    try:
        g = model.agent.decoders["image"]
        for start in range(0, len(data), 8):
            if perf_counter() - started > 180:
                raise RuntimeError("Photo evaluation wall cap")
            ids = list(range(start, min(start + 8, len(data))))
            rgb = data.batch(ids, device)["rgb"]
            history = photo_history(rgb)
            h = model.observe_history(history)
            blind = model.observe_history(photo_history(history[:, -1]))
            rows["target"].append(rgb.cpu())
            rows["teacher"].append(g.head(model.teacher(rgb)).cpu())
            rows["mean"].append(mean.to(device)[None].expand_as(rgb).cpu())
            for name, mode in [
                ("ordinary", "ordinary"),
                ("reset", "reset"),
                ("erased", "reset_erased"),
                ("swapped", "reset_swapped"),
                ("blind", "reset"),
            ]:
                state = model.query(
                    (blind if name == "blind" else h)["final"], history[:, -1], mode
                )
                context = model.output_normalization(model.working(state))
                image = g(context, seed=sample_seed, sample_ids=[i // 2 for i in ids])
                rows[name].append(image.cpu())
        return {k: torch.cat(v) for k, v in rows.items()}
    finally:
        for p, flag in flags:
            p.requires_grad_(flag)


def score(arrays):
    target = arrays["target"]
    result = {}
    for name, image in arrays.items():
        if name == "target":
            continue
        for label, expected in (
            [
                ("swapped_original", target),
                ("swapped_target", target[torch.arange(len(target)) ^ 1]),
            ]
            if name == "swapped"
            else [(name, target)]
        ):
            if not torch.isfinite(image).all():
                raise ValueError("Nonfinite photo output")
            mse = (image - expected).square().mean((1, 2, 3))
            edge = (
                (image.diff(dim=-1) - expected.diff(dim=-1)).square().mean()
                + (image.diff(dim=-2) - expected.diff(dim=-2)).square().mean()
            ) / 2
            result[label] = dict(
                rgb_mse=float(mse.mean()),
                mean_psnr=float((-10 * mse.clamp_min(1e-12).log10()).mean()),
                edge_mse=float(edge),
                boundary_pixel_fraction=float(
                    ((image <= 0) | (image >= 1)).float().mean()
                ),
                count=len(image),
            )
    return result


def screens(before, after):
    learning = all(
        after[k]["rgb_mse"] <= 0.9 * before[k]["rgb_mse"]
        and after[k]["rgb_mse"] <= 0.9 * after["mean"]["rgb_mse"]
        for k in ("ordinary", "reset")
    )
    context = (
        all(
            after["reset"]["rgb_mse"] <= 0.9 * after[k]["rgb_mse"]
            for k in ("erased", "swapped_original", "blind")
        )
        and after["swapped_target"]["rgb_mse"] <= 1.1 * after["reset"]["rgb_mse"]
    )
    return dict(learning=learning, context=context, passed=learning and context)


def budget(path, config, device):
    if shutil.disk_usage(Path(path).parent).free < config["disk_free_gib"] * 1024**3:
        raise RuntimeError("Disk reserve floor reached")
    if str(device).startswith("cuda"):
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
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )
    run = Run(
        output,
        settings=config,
        data=dict(
            training=training.identity,
            validation=validation.identity,
            cache_sha256=cache_hash(cache),
        ),
        recipe=__file__,
        model=model,
        optimizer=optimizer,
        device=device,
        resume=resume,
    )
    frozen = {k: v.clone() for k, v in frozen_tensors(model).items()}
    mean = cache["mean"].detach().cpu()
    prior = sum(r.get("elapsed_seconds", 0) for r in run.rows if r["split"] == "timing")
    try:
        if not resume:
            torch.save(
                dict(model=model.state_dict(), settings=config, train_mean=mean),
                run.path / "initial_weights.pt",
            )
            before = evaluate(model, validation, mean, device)
            np.savez_compressed(
                run.path / "initial_validation.npz",
                **{k: v.numpy() for k, v in before.items()},
            )
            atomic_json(run.path / "initial_validation.json", score(before))
        start = perf_counter()
        end = (
            config["steps"]
            if stop_after is None
            else min(config["steps"], run.step + stop_after)
        )
        for step in range(run.step + 1, end + 1):
            if prior + perf_counter() - start >= config["wall_seconds"]:
                break
            ids = run.sample(len(training), config["batch_size"])
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = objective(model.agent.decoders["image"], cache, ids, step)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite photo training objective")
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
                budget(output, config, device)
                run.save()
                print(f"photo step {step}: RGB {metrics['rgb_loss']:.6f}", flush=True)
        run.log(
            dict(step=run.step, split="timing", elapsed_seconds=perf_counter() - start)
        )
        run.save()
        if not all(torch.equal(v, frozen_tensors(model)[k]) for k, v in frozen.items()):
            raise RuntimeError("Frozen photo pipeline changed")
        torch.save(
            dict(model=model.state_dict(), settings=config, train_mean=mean),
            run.path / "weights.pt",
        )
        arrays = evaluate(model, validation, mean, device)
        metrics = score(arrays)
        completed = run.step == config["steps"]
        paused = run.step == end and end < config["steps"]
        before = json.loads((run.path / "initial_validation.json").read_text())
        gates = screens(before, metrics)
        atomic_json(
            run.path / "result.json",
            dict(
                completed=completed,
                paused=paused,
                step=run.step,
                metrics=metrics,
                initial_metrics=before,
                gate=completed and gates["passed"],
                screens=gates,
                frozen_equal=True,
                training_seconds=sum(
                    r["elapsed_seconds"] for r in run.rows if r["split"] == "timing"
                ),
                peak_reserved_mib=torch.cuda.max_memory_reserved(device) / 1024**2
                if str(device).startswith("cuda")
                else 0,
                evaluation_scope="Validation diagnostic: real64px photos, constructed recall, frozen upstream; not text-to-image or real video.",
            ),
        )
        np.savez_compressed(
            run.path / "validation.npz", **{k: v.numpy() for k, v in arrays.items()}
        )
        run.status(
            "completed" if completed else "paused" if paused else "budget-stopped",
            "pending",
        )
        write_report(
            run.path,
            batch={"rgb": arrays["target"][:8]},
            outputs={"rgb": arrays["reset"][:8]},
        )
        return run
    except BaseException as e:
        run.save()
        status = json.loads((run.path / "status.json").read_text())
        run.status(
            status["result"]
            if status["result"] in ("completed", "paused", "budget-stopped")
            else "failed",
            "failed",
            str(e),
        )
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--data-root", type=Path, default=Path("data/curriculum/coco_v1"))
    p.add_argument("--device", default="cpu")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--stop-after", type=int)
    p.add_argument("--development", action="store_true")
    p.add_argument("--evaluate-only", action="store_true")
    args = p.parse_args()
    if args.stop_after is not None and args.stop_after < 1:
        p.error("--stop-after must be positive")
    payload = torch.load(args.weights, map_location="cpu", weights_only=True)
    config = dict(payload["settings"], **settings())
    if args.evaluate_only:
        config = payload["settings"]
        seed_everything(config["seed"])
        data = photo_data(
            args.data_root,
            (config["train_count"], config["validation_count"], config["test_count"]),
            config["data_seed"],
        )["test"]
        args.output.mkdir(parents=True, exist_ok=False)
        model = load_model(args.weights, args.device)
        arrays = evaluate(model, data, payload["train_mean"], args.device)
        metrics = score(arrays)
        atomic_json(
            args.output / "run.json",
            dict(
                identity=dict(
                    settings=config,
                    data=data.identity,
                    environment=environment(args.device),
                ),
                source=source_record(__file__, model),
                weights_sha256=file_hash(args.weights),
            ),
        )
        atomic_json(
            args.output / "result.json",
            dict(
                completed=True,
                metrics=metrics,
                evaluation_scope="Heldout from this continuation: real COCO RGB64 photographs; constructed recall; fixed sample seed13; upstream has prior COCO/synthetic exposure.",
            ),
        )
        np.savez_compressed(
            args.output / "predictions.npz", **{k: v.numpy() for k, v in arrays.items()}
        )
        atomic_json(
            args.output / "status.json",
            dict(result="completed", report="pending", step=0, error=None),
        )
        (args.output / "metrics.jsonl").write_text("")
        write_report(
            args.output,
            batch={"rgb": arrays["target"][:8]},
            outputs={"rgb": arrays["reset"][:8]},
        )
        print(json.dumps(metrics["reset"]), flush=True)
        return
    if args.development:
        selected = photo_data(args.data_root, (1040, 144, 256), 45001)
        training = Frames(
            selected["train"].frames,
            selected["train"].rows[1024:],
            identity=dict(
                selected["train"].identity,
                selected=selected["train"].identity["selected"][1024:],
                reserved_development=True,
            ),
        )
        validation = Frames(
            selected["validation"].frames,
            selected["validation"].rows[128:],
            identity=dict(
                selected["validation"].identity,
                selected=selected["validation"].identity["selected"][128:],
                reserved_development=True,
            ),
        )
        config.update(seed=45021, steps=16, wall_seconds=120)
    else:
        selected = photo_data(args.data_root)
        training = selected["train"]
        validation = selected["validation"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if str(args.device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(args.device)
    budget(args.output, config, args.device)
    model, cache, config = prepare(args.weights, training, config, args.device)
    budget(args.output, config, args.device)
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
