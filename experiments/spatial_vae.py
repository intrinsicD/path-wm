"""Three explicit-scale VAE variants on real photos. No work occurs on import.

python -m experiments.spatial_vae --variant base --output runs/spatial_vae_v1/base --device cuda
Use --evaluate-only --weights ... for the frozen multi-task suite. Settings below
are the predeclared recipe; library modules contain no experiment orchestration.
"""

import argparse
import copy
import json
import shutil
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from pathwm.data.images import Frames
from pathwm.data.photo_recall import photo_data
from pathwm.io import (
    Run,
    atomic_json,
    file_hash,
    source_record,
    seed_everything,
    environment,
    evaluation_mode,
)
from pathwm.evaluation.report import write_report
from pathwm.models.spatial_vae import SpatialVAE, build_variant, vae_loss
from pathwm.models.spatial_vae_v2 import build_hierarchy


def settings(variant="base"):
    return dict(
        seed=56101,
        variant=variant,
        steps=512,
        batch_size=8,
        lr=3e-4,
        weight_decay=1e-4,
        beta=1.0,
        variance=0.5,
        wall_seconds=300,
        memory_mib=3072,
        disk_free_gib=3,
        data_seed=56001,
        counts=[1024, 128, 192],
        purpose="spatial VAE comparison",
        objective="RGB-summed squared error + beta * summed KL, each per original pixel",
        initialization="shared seeded projections; identity local mixers; neutral attention output",
        example_labels={
            "input": "Source photo",
            "rgb": "Posterior-mean reconstruction",
            "sampled": "Sampled reconstruction",
            "prior": "Unconditional prior sample (quality unvalidated)",
        },
    )


def data_sets(root, config, development=False):
    counts = config["counts"]
    extra = [counts[0] + 16, counts[1] + 16, counts[2]]
    selected = photo_data(root, extra, config["data_seed"])
    result = {}
    for split, count in zip(("train", "validation", "test"), counts):
        source = selected[split]
        ids = (
            source.rows[count:]
            if development and split != "test"
            else source.rows[:count]
        )
        meta = (
            source.identity["selected"][count:]
            if development and split != "test"
            else source.identity["selected"][:count]
        )
        result[split] = Frames(
            source.frames,
            ids,
            identity=dict(
                source.identity, selected=meta, reserved_development=development
            ),
        )
    return result


def resources(path, config, device):
    if shutil.disk_usage(Path(path).parent).free < config["disk_free_gib"] * 1024**3:
        raise RuntimeError("Disk reserve reached; preserve earlier artifacts")
    if str(device).startswith("cuda"):
        if (
            torch.cuda.max_memory_reserved(device) > config["memory_mib"] * 1024**2
            or torch.cuda.mem_get_info(device)[0] < 1024**3
        ):
            raise RuntimeError("GPU reserve limit exceeded")


def image_metrics(pred, target):
    raw = (pred - target).square().mean((1, 2, 3))
    p = pred.clamp(0, 1)
    mse = (p - target).square().mean((1, 2, 3))
    edge = (
        (p.diff(dim=-1) - target.diff(dim=-1)).square().mean((1, 2, 3))
        + (p.diff(dim=-2) - target.diff(dim=-2)).square().mean((1, 2, 3))
    ) / 2
    highp = p - F.avg_pool2d(p, 3, 1, 1, count_include_pad=False)
    hight = target - F.avg_pool2d(target, 3, 1, 1, count_include_pad=False)
    detail = (highp - hight).square().mean((1, 2, 3))
    color = (p.mean((2, 3)) - target.mean((2, 3))).square().mean(1)
    scores = dict(
        mse=float(mse.mean()),
        raw_mse=float(raw.mean()),
        psnr_from_mean_mse=float(-10 * mse.mean().clamp_min(1e-12).log10()),
        mean_psnr=float((-10 * mse.clamp_min(1e-12).log10()).mean()),
        edge_mse=float(edge.mean()),
        detail_mse=float(detail.mean()),
        color_mse=float(color.mean()),
        count=len(target),
    )
    return scores, dict(
        mse=mse.numpy(),
        raw_mse=raw.numpy(),
        edge_mse=edge.numpy(),
        detail_mse=detail.numpy(),
        color_mse=color.numpy(),
    )


@torch.no_grad()
def evaluate_images(model, rgb, device="cpu", seed=56171, batch_size=8, controls=True):
    rows = {k: [] for k in ("target", "mean", "sampled", "mu", "logvar")}
    with evaluation_mode(model):
        for start in range(0, len(rgb), batch_size):
            x = rgb[start : start + batch_size].to(device)
            p = model.encode(x)
            # One independent stream per example keeps sampling invariant to batching.
            z = torch.cat(
                [
                    p.mu[i : i + 1]
                    + (0.5 * p.logvar[i : i + 1]).exp()
                    * torch.randn(
                        p.mu[i : i + 1].shape,
                        device=device,
                        generator=torch.Generator(device=device).manual_seed(
                            seed + start + i
                        ),
                    )
                    for i in range(len(x))
                ]
            )
            for name, v in [
                ("target", x),
                ("mean", model.decode(p.mu, tuple(x.shape[-2:]))),
                ("sampled", model.decode(z, tuple(x.shape[-2:]))),
                ("mu", p.mu),
                ("logvar", p.logvar),
            ]:
                rows[name].append(v.cpu())
        rows = {k: torch.cat(v) for k, v in rows.items()}
        if controls:
            for name in ["wrong", "zero", "spatial_shuffle"]:
                out = []
                for start in range(0, len(rgb), batch_size):
                    ids = torch.arange(start, min(start + batch_size, len(rgb)))
                    z = (
                        rows["mu"][(ids + 1) % len(rgb)].to(device)
                        if name == "wrong"
                        else rows["mu"][ids].to(device)
                    )
                    if name == "zero":
                        z = torch.zeros_like(z)
                    if name == "spatial_shuffle":
                        order = torch.randperm(
                            z.shape[-2] * z.shape[-1],
                            generator=torch.Generator().manual_seed(seed),
                        )
                        z = z.flatten(2)[:, :, order].reshape_as(z)
                    out.append(model.decode(z, tuple(rgb.shape[-2:])).cpu())
                rows[name] = torch.cat(out)
    metrics = {}
    per = {}
    for name, pred in rows.items():
        if name in ("target", "mu", "logvar"):
            continue
        scores, errors = image_metrics(pred, rows["target"])
        metrics[name] = scores
        per.update({name + "_" + k: v for k, v in errors.items()})
    mu, lv = rows["mu"], rows["logvar"]
    kl = 0.5 * (mu.square() + lv.exp() - 1 - lv)
    nats = kl.sum((1, 2, 3)).mean()
    area = rgb.shape[-2] * rgb.shape[-1]
    low, high = model.config["logvar_bounds"]
    metrics["posterior"] = dict(
        kl_nats_per_sample=float(nats),
        kl_bits_per_sample=float(nats / np.log(2)),
        kl_bits_per_latent_position=float(
            nats / (np.log(2) * mu.shape[-2] * mu.shape[-1])
        ),
        kl_bits_per_original_pixel=float(nats / (np.log(2) * area)),
        original_area=area,
        latent_positions=mu.shape[-2] * mu.shape[-1],
        rate_interpretation="KL rate proxy; no entropy coder",
        rate_nats_per_pixel=float(
            kl.sum((1, 2, 3)).mean() / (rgb.shape[-2] * rgb.shape[-1])
        ),
        active_channels=int((mu.var((0, 2, 3), unbiased=False) > 0.01).sum()),
        mean_std=float((0.5 * lv).exp().mean()),
        mu_rms=float(mu.square().mean().sqrt()),
        logvar_saturated_fraction=float(((lv <= low) | (lv >= high)).float().mean()),
    )
    per["kl_per_channel"] = kl.mean((0, 2, 3)).numpy()
    per["kl_per_position"] = kl.mean((0, 1)).numpy()
    return rows, metrics, per


@torch.no_grad()
def retrieval(model, rgb, device):
    query = (torch.roll(rgb, 1, -1) * 0.9 + 0.05).clamp(0, 1)
    with evaluation_mode(model):
        gallery = torch.cat(
            [model.encode(x.to(device)).mu.flatten(1).cpu() for x in rgb.split(8)]
        )
        features = torch.cat(
            [model.encode(x.to(device)).mu.flatten(1).cpu() for x in query.split(8)]
        )
    predictions = {}
    scores = {}
    for name, g, q in [
        ("latent", gallery, features),
        ("pixels", rgb.flatten(1), query.flatten(1)),
        ("color_mean", rgb.mean((2, 3)), query.mean((2, 3))),
    ]:
        ids = torch.cdist(q.double(), g.double()).argmin(1)
        predictions[name] = ids.numpy()
        scores[name + "_accuracy"] = float(
            (ids == torch.arange(len(rgb))).float().mean()
        )
    scores["chance_accuracy"] = 1 / len(rgb)
    scores["count"] = len(rgb)
    return scores, predictions


def native_crops(root, data, size, count):
    manifest = json.loads((Path(root) / "manifest.json").read_text())
    source = Path(manifest["source"])
    images = []
    used = []
    h, w = size
    for row in data.rows:
        record = manifest["records"][int(row)]
        if record["source_size"][0] < w or record["source_size"][1] < h:
            continue
        path = source / record["file"]
        if file_hash(path) != record["source_sha256"]:
            raise ValueError("Original photo hash changed")
        with Image.open(path) as im:
            im = im.convert("RGB")
            left = (im.width - w) // 2
            top = (im.height - h) // 2
            images.append(
                torch.from_numpy(
                    np.asarray(im.crop((left, top, left + w, top + h))).copy()
                )
                .permute(2, 0, 1)
                .float()
                / 255
            )
        used.append(
            dict(
                row=int(row),
                file=record["file"],
                sha256=record["source_sha256"],
                crop=[left, top, w, h],
            )
        )
        if len(images) == count:
            break
    if len(images) != count:
        raise ValueError("Insufficient native photo crops")
    return torch.stack(images), used


def patterns(count=32, size=64):
    y, x = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    images = []
    rng = torch.Generator().manual_seed(56099)
    for i in range(count):
        c = torch.rand(3, 1, 1, generator=rng) * 0.8 + 0.1
        if i % 4 == 0:
            mask = (x + i) % 2 == 0
        elif i % 4 == 1:
            mask = (x // (1 + i % 7) + y // (1 + i % 5)) % 2 == 0
        elif i % 4 == 2:
            mask = (x - size // 2) ** 2 + (y - size // 2) ** 2 < (size // 4) ** 2
        else:
            mask = (
                (x > size // 4 + i % 8)
                & (x < 3 * size // 4)
                & (y > size // 4)
                & (y < 3 * size // 4)
            )
        images.append(torch.where(mask[None], c, torch.full((3, 1, 1), 0.1)))
    return torch.stack(images)


def example_panel(path, arrays, labels, count=8):
    """Scientific comparison panel for the existing report renderer's PNG slot.

    The shared gallery only displays its standard input/rgb keys, so custom
    sampled/prior/variant arrays must be explicitly included in this panel.
    """
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    count = min(count, *(len(a) for a in arrays.values()))
    fig = Figure(figsize=(2.4 * len(arrays), 2.1 * count), layout="constrained")
    FigureCanvasAgg(fig)
    axes = fig.subplots(count, len(arrays), squeeze=False)
    for j, (key, values) in enumerate(arrays.items()):
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy()
        for i in range(count):
            axes[i, j].imshow(
                np.clip(values[i].transpose(1, 2, 0), 0, 1), interpolation="nearest"
            )
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
            if i == 0:
                axes[i, j].set_title(labels.get(key, key), fontsize=10)
            if j == 0:
                axes[i, j].set_ylabel(f"Example {i + 1}")
    fig.savefig(path, dpi=130)


def report_evaluation(path, config, identity, model, metrics, rows, device, scope):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    atomic_json(
        path / "run.json",
        dict(
            identity=dict(
                settings=config, data=identity, environment=environment(device)
            ),
            source=source_record(__file__, model),
        ),
    )
    atomic_json(
        path / "result.json",
        dict(completed=True, metrics=metrics, evaluation_scope=scope),
    )
    atomic_json(
        path / "status.json",
        dict(result="completed", report="pending", step=0, error=None),
    )
    (path / "metrics.jsonl").write_text("")
    try:
        example_panel(
            path / "comparison.png",
            {
                **{k: rows[k] for k in ("target", "mean", "sampled")},
                **(
                    {"absolute_error": (rows["mean"] - rows["target"]).abs()}
                    if "model_config" in config
                    else {}
                ),
            },
            dict(
                target="Original",
                mean="Posterior mean",
                sampled="Posterior sample",
                absolute_error="Absolute RGB error [0,1]",
            ),
        )
        write_report(
            path,
            batch={"rgb": rows["target"][:8]},
            outputs={"rgb": rows["mean"][:8], "sampled": rows["sampled"][:8]},
        )
    except BaseException as e:
        atomic_json(
            path / "status.json",
            dict(result="completed", report="failed", step=0, error=str(e)),
        )
        raise


def train(
    output,
    training,
    validation,
    config,
    device="cpu",
    resume=False,
    stop_after=None,
    source=None,
    deterministic=False,
    trainable="all",
):
    seed_everything(config["seed"])
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    model = (
        SpatialVAE.load(source, device)
        if source
        else (
            build_hierarchy(config["variant"], config["seed"], **config["model_config"])
            if "model_config" in config
            else build_variant(config["variant"], config["seed"])
        ).to(device)
    )
    if model.config.get("variant", model.config.get("ablation")) != config["variant"]:
        raise ValueError("Continuation variant must match the saved architecture")
    if trainable not in ("all", "encoder", "decoder"):
        raise ValueError("Choose all, encoder or decoder trainable parameters")
    if trainable != "all":
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith(trainable + "."))
        config = dict(config, trainable_parts=trainable)
    # Initialization/load must not alter the matched training-noise stream.
    seed_everything(config["seed"])
    config = dict(
        config,
        deterministic_control=deterministic,
        source_sha256=file_hash(source) if source else None,
        architecture=model.config,
    )
    resources(output, config, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )
    run = Run(
        output,
        settings=config,
        data=dict(training=training.identity, validation=validation.identity),
        recipe=__file__,
        model=model,
        optimizer=optimizer,
        device=device,
        resume=resume,
    )
    train_rgb = training.batch(range(len(training)))["rgb"]
    val_rgb = validation.batch(range(len(validation)))["rgb"]
    mean = train_rgb.mean(0)
    prior = sum(r.get("elapsed_seconds", 0) for r in run.rows if r["split"] == "timing")
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)
    try:
        if not resume:
            _, metrics, _ = evaluate_images(model, val_rgb, device, controls=False)
            atomic_json(output / "initial_validation.json", metrics)
            if deterministic:
                _, initial_train, _ = evaluate_images(
                    model, train_rgb, device, controls=False
                )
                atomic_json(output / "initial_training.json", initial_train)
        end = (
            config["steps"]
            if stop_after is None
            else min(config["steps"], run.step + stop_after)
        )
        start = perf_counter()
        for step in range(run.step + 1, end + 1):
            if prior + perf_counter() - start >= config["wall_seconds"]:
                break
            ids = run.sample(len(training), config["batch_size"])
            x = train_rgb[ids].to(device)
            optimizer.zero_grad(set_to_none=True)
            y, p = model(x, sample=not deterministic)
            loss, terms = vae_loss(y, x, p, config["beta"], config["variance"])
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite VAE objective")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1, error_if_nonfinite=True
            )
            optimizer.step()
            run.step = step
            run.log(
                dict(
                    step=step,
                    split="train",
                    loss=float(loss.detach()),
                    grad_norm=float(norm),
                    **{k: float(v.detach()) for k, v in terms.items()},
                )
            )
            if step % 128 == 0 or step == end:
                if str(device).startswith("cuda"):
                    torch.cuda.synchronize(device)
                resources(output, config, device)
                run.save()
                print(
                    f"{config['variant']} beta={config['beta']} step {step}: D={float(terms['distortion'].detach()):.6f} KL={float(terms['rate'].detach()):.6f}",
                    flush=True,
                )
        if str(device).startswith("cuda"):
            torch.cuda.synchronize(device)
        run.log(
            dict(step=run.step, split="timing", elapsed_seconds=perf_counter() - start)
        )
        run.save()
        model.save(output / "weights.pt")
        rows, metrics, _ = evaluate_images(model, val_rgb, device, controls=False)
        metrics["train_mean"], _ = image_metrics(mean[None].expand_as(val_rgb), val_rgb)
        if deterministic:
            _, final_train, _ = evaluate_images(
                model, train_rgb, device, controls=False
            )
            initial_train = json.loads((output / "initial_training.json").read_text())
            atomic_json(
                output / "overfit.json",
                dict(
                    initial=initial_train,
                    final=final_train,
                    passed=final_train["mean"]["raw_mse"]
                    <= 0.25 * initial_train["mean"]["raw_mse"],
                    scope="Eight training photos, deterministic mean decoding, beta zero; not a VAE or generalization claim.",
                ),
            )
        state = (
            "completed"
            if run.step == config["steps"]
            else "paused"
            if run.step == end and end < config["steps"]
            else "budget-stopped"
        )
        atomic_json(
            output / "result.json",
            dict(
                completed=state == "completed",
                paused=state == "paused",
                step=run.step,
                metrics=metrics,
                parameters=sum(p.numel() for p in model.parameters()),
                samples_seen=run.step * config["batch_size"],
                peak_allocated_mib=torch.cuda.max_memory_allocated(device) / 1024**2
                if str(device).startswith("cuda")
                else 0,
                training_seconds=sum(
                    r["elapsed_seconds"] for r in run.rows if r["split"] == "timing"
                ),
                peak_reserved_mib=torch.cuda.max_memory_reserved(device) / 1024**2
                if str(device).startswith("cuda")
                else 0,
                weights_sha256=file_hash(output / "weights.pt"),
                evaluation_scope="Validation only; independent codec reconstruction. General generation and world-state integration untested.",
            ),
        )
        run.status(state, "pending")
        example_panel(
            output / "comparison.png",
            {
                **{k: rows[k] for k in ("target", "mean", "sampled")},
                **(
                    {"absolute_error": (rows["mean"] - rows["target"]).abs()}
                    if "model_config" in config
                    else {}
                ),
            },
            dict(
                target="Validation original",
                absolute_error="Absolute RGB error [0,1]",
                mean="Posterior mean",
                sampled="Posterior sample",
            ),
        )
        write_report(
            output,
            batch={"rgb": rows["target"][:8]},
            outputs={"rgb": rows["mean"][:8], "sampled": rows["sampled"][:8]},
        )
        return run
    except BaseException as e:
        run.save()
        status = json.loads((output / "status.json").read_text())
        run.status(
            status["result"]
            if status["result"] in ("completed", "paused", "budget-stopped")
            else "failed",
            "failed",
            str(e),
        )
        raise


def capability(metrics):
    p = metrics["photo64"]
    r = metrics["retrieval"]
    return dict(
        photo_mean=p["mean"]["mse"] <= 0.01,
        photo_sampled=p["sampled"]["mse"] <= 0.01,
        mean_baseline=p["mean"]["mse"] <= 0.8 * p["train_mean"]["mse"]
        and p["sampled"]["mse"] <= 0.8 * p["train_mean"]["mse"],
        retrieval=r["latent_accuracy"] >= 0.8,
        native128=metrics["native128"]["mean"]["mse"] <= 0.015,
        odd=metrics["odd63x79"]["mean"]["mse"] <= 0.015,
    )


def evaluate_suite(output, weights, data, root, config, device):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    try:
        return _evaluate_suite(output, weights, data, root, config, device)
    except BaseException as e:
        if output.exists():
            status_path = output / "status.json"
            status = json.loads(status_path.read_text()) if status_path.exists() else {}
            atomic_json(
                status_path,
                dict(
                    result="completed"
                    if status.get("result") == "completed"
                    else "failed",
                    report="failed",
                    step=0,
                    error=str(e),
                ),
            )
        raise


def _evaluate_suite(output, weights, data, root, config, device):
    seed_everything(config["seed"])
    model = SpatialVAE.load(weights, device).eval()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(
        output / "status.json",
        dict(result="running", report="pending", step=0, error=None),
    )
    atomic_json(
        output / "run.json",
        dict(
            identity=dict(
                settings=config,
                data=data["test"].identity,
                environment=environment(device),
            ),
            source=source_record(__file__, model),
            weights_sha256=file_hash(weights),
        ),
    )
    began = perf_counter()
    datasets = {}
    rgb = data["test"].batch(range(len(data["test"])))["rgb"]
    datasets["photo64"] = (rgb, data["test"].identity)
    for name, size, count in [
        ("native64", (64, 64), 32),
        ("native128", (128, 128), 32),
        ("odd63x79", (63, 79), 32),
        ("rectangle96x160", (96, 160), 16),
        ("native192x256", (192, 256), 8),
    ]:
        datasets[name] = native_crops(root, data["test"], size, count)
    datasets["patterns"] = (
        patterns(),
        dict(seed=56099, scope="Synthetic out-of-training diagnostic"),
    )
    datasets["phase_shift"] = (torch.roll(patterns(), 1, -1), dict(seed=56099, shift=1))
    metrics = {}
    images = {}
    checks = {}
    resources(output, config, device)
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)
    for name, (target, identity) in datasets.items():
        batch_size = (
            8 if target.shape[-1] <= 80 else 2 if target.shape[-1] <= 160 else 1
        )
        rows, m, per = evaluate_images(
            model, target, device, batch_size=batch_size, controls=name == "photo64"
        )
        if name == "photo64":
            train_mean = data["train"].batch(range(len(data["train"])))["rgb"].mean(0)
            m["train_mean"], _ = image_metrics(
                train_mean[None].expand_as(target), target
            )
            m["image_color_mean"], _ = image_metrics(
                target.mean((2, 3), keepdim=True).expand_as(target), target
            )
            retrieval_scores, ids = retrieval(model, target, device)
            metrics["retrieval"] = retrieval_scores
            np.savez_compressed(output / "retrieval.npz", **ids)
            # The copy is an independent decoder object, no encoder or image inputs.
            decoder = copy.deepcopy(model.decoder).eval()
            with torch.no_grad():
                decoded = decoder(rows["mu"][:8].to(device), (64, 64)).cpu()
            checks["saved_latent_decoder_only_exact"] = torch.equal(
                decoded, rows["mean"][:8]
            )
            del decoder
            torch.save(
                dict(z=rows["mu"][:8], output_size=[64, 64]),
                output / "saved_latents.pt",
            )
        report_evaluation(
            output / name,
            config,
            identity,
            model,
            m,
            rows,
            device,
            "Held-out independent codec: "
            + name
            + ". Native crops contain original source pixels, not enlarged RGB64.",
        )
        np.savez_compressed(
            output / name / "predictions.npz",
            **{k: v.numpy() for k, v in rows.items()},
            **{"error_" + k: v for k, v in per.items()},
        )
        metrics[name] = m
        images[name] = rows["mean"][:8]
        resources(output, config, device)
        if perf_counter() - began > 300:
            raise RuntimeError("Evaluation wall cap")
    with torch.no_grad():
        z = torch.randn(
            8,
            8,
            8,
            8,
            device=device,
            generator=torch.Generator(device=device).manual_seed(56181),
        )
        prior = model.decode(z, (64, 64)).cpu()
    metrics["prior_diagnostic"] = dict(
        pixel_std_across_samples=float(prior.std(0).mean()),
        raw_min=float(prior.min()),
        raw_max=float(prior.max()),
        quality_validated=False,
    )
    np.savez_compressed(output / "prior_samples.npz", prior=prior.numpy())
    gates = capability(metrics)
    atomic_json(
        output / "result.json",
        dict(
            completed=True,
            metrics=metrics,
            screens=gates,
            gate=all(gates.values()),
            checks=checks,
            weights_sha256=file_hash(weights),
            parameters=sum(p.numel() for p in model.parameters()),
            evaluation_seconds=perf_counter() - began,
            peak_reserved_mib=torch.cuda.max_memory_reserved(device) / 1024**2
            if str(device).startswith("cuda")
            else 0,
            evaluation_scope="Independent image codec: reconstruction, shape flexibility, bounded instance retrieval and latent-only decoding. No world-state generation, semantics or other-modal capability claim.",
        ),
    )
    atomic_json(
        output / "run.json",
        dict(
            identity=dict(
                settings=config,
                data=data["test"].identity,
                environment=environment(device),
            ),
            source=source_record(__file__, model),
        ),
    )
    atomic_json(
        output / "status.json",
        dict(result="completed", report="pending", step=0, error=None),
    )
    (output / "metrics.jsonl").write_text("")
    # Flatten the metrics for the common report, preserving nested authoritative result.
    readable = {
        name + "." + sub: value
        for name, condition in metrics.items()
        for sub, value in condition.items()
        if isinstance(value, dict)
    }
    readable.update(
        {k: v for k, v in metrics.items() if k in ("retrieval", "prior_diagnostic")}
    )
    original = json.loads((output / "result.json").read_text())
    atomic_json(output / "suite.json", original)
    atomic_json(output / "result.json", dict(original, metrics=readable))
    example_panel(
        output / "comparison.png",
        dict(original=rgb[:8], mean=images["photo64"], prior=prior),
        dict(
            original="Original",
            mean="Posterior mean",
            prior="Unconditional prior\n(unpaired; quality unvalidated)",
        ),
    )
    write_report(
        output,
        batch={"rgb": rgb[:8]},
        outputs={"rgb": images["photo64"], "prior": prior},
    )
    if not all(checks.values()):
        raise RuntimeError("Decoder-only replay failed")
    return original


def compare(root, cohort):
    root = Path(root)
    results = {
        v: json.loads((root / cohort / v / "evaluation/suite.json").read_text())
        for v in ["base", "attention", "reversible"]
    }
    training = {
        v: json.loads((root / cohort / v / "result.json").read_text()) for v in results
    }
    benefits = {}
    for name, a, b in [
        ("attention_vs_base", "base", "attention"),
        ("reversible_vs_attention", "attention", "reversible"),
    ]:
        ma, mb = results[a]["metrics"], results[b]["metrics"]
        pa = root / cohort / a / "evaluation/photo64/predictions.npz"
        pb = root / cohort / b / "evaluation/photo64/predictions.npz"
        with np.load(pa) as za, np.load(pb) as zb:
            delta = za["error_mean_mse"] - zb["error_mean_mse"]
        rng = np.random.default_rng(56191)
        means = delta[rng.integers(0, len(delta), (1000, len(delta)))].mean(1)
        interval = np.quantile(means, [0.025, 0.975]).tolist()
        checks = dict(
            rgb_improves=mb["photo64"]["mean"]["mse"]
            <= 0.95 * ma["photo64"]["mean"]["mse"],
            positive_interval=interval[0] > 0,
            edge_retention=mb["photo64"]["mean"]["edge_mse"]
            <= 1.05 * ma["photo64"]["mean"]["edge_mse"],
            native_retention=mb["native128"]["mean"]["mse"]
            <= 1.05 * ma["native128"]["mean"]["mse"],
            odd_retention=mb["odd63x79"]["mean"]["mse"]
            <= 1.05 * ma["odd63x79"]["mean"]["mse"],
            retrieval_retention=1 - mb["retrieval"]["latent_accuracy"]
            <= max(1.05 * (1 - ma["retrieval"]["latent_accuracy"]), 1 / 192),
        )
        benefits[name] = dict(
            checks=checks,
            passed=all(checks.values()),
            paired_mse_improvement_interval=interval,
        )
    directory = root / cohort / "comparison"
    directory.mkdir(exist_ok=False)
    tables = {
        v: dict(
            mean_mse=r["metrics"]["photo64"]["mean"]["mse"],
            sampled_mse=r["metrics"]["photo64"]["sampled"]["mse"],
            edge_mse=r["metrics"]["photo64"]["mean"]["edge_mse"],
            native128_mse=r["metrics"]["native128"]["mean"]["mse"],
            odd_mse=r["metrics"]["odd63x79"]["mean"]["mse"],
            retrieval=r["metrics"]["retrieval"]["latent_accuracy"],
            rate=r["metrics"]["photo64"]["posterior"]["rate_nats_per_pixel"],
            parameters=r["parameters"],
            training_seconds=training[v]["training_seconds"],
            capability_passed=r["gate"],
        )
        for v, r in results.items()
    }
    atomic_json(
        directory / "result.json",
        dict(
            completed=True,
            metrics=tables,
            benefits=benefits,
            evaluation_scope=cohort
            + ": equal data/updates, not equal achieved KL rate or compute. Each task gate retains its scope.",
        ),
    )
    atomic_json(
        directory / "run.json",
        dict(
            identity=dict(
                settings=dict(
                    purpose="spatial VAE comparison",
                    example_labels={
                        "input": "Original",
                        "base": "Base VAE",
                        "attention": "Cross-scale attention",
                        "reversible": "Attention + reversible local mixing",
                    },
                ),
                data={},
            ),
            source={},
        ),
    )
    atomic_json(
        directory / "status.json",
        dict(result="completed", report="pending", step=512, error=None),
    )
    (directory / "metrics.jsonl").write_text("")
    arrays = {
        v: np.load(root / cohort / v / "evaluation/photo64/predictions.npz")
        for v in results
    }
    example_panel(
        directory / "comparison.png",
        dict(
            original=arrays["base"]["target"],
            **{v: z["mean"] for v, z in arrays.items()},
        ),
        dict(
            original="Original",
            base="Base",
            attention="Cross-scale attention",
            reversible="Attention + reversible",
        ),
    )
    write_report(
        directory,
        batch={"rgb": torch.from_numpy(arrays["base"]["target"][:8])},
        outputs={v: torch.from_numpy(z["mean"][:8]) for v, z in arrays.items()},
    )
    for z in arrays.values():
        z.close()
    return tables, benefits


def hierarchy_settings(ablation="C", beta=1.0):
    return dict(
        settings(ablation),
        seed=57101,
        counts=[512, 64, 96],
        beta=beta,
        wall_seconds=120,
        memory_mib=2048,
        model_config={},
        purpose="R/P/M/C spatial VAE sanity and descriptive rate-distortion study",
        initialization="Seeded C reference common tensors; zero local residual ends; A_local different stride kernel; attention standard initialization",
    )


def hierarchy_data(root, config, development=False):
    """New groups after the explicitly excluded v1 study population."""
    offsets = [1040, 144, 192]
    counts = config["counts"]
    extra = [16, 16, 0]
    source = photo_data(
        root, [a + b + c for a, b, c in zip(offsets, counts, extra)], 56001
    )
    data = {}
    for i, split in enumerate(("train", "validation", "test")):
        start = offsets[i] + (counts[i] if development and i < 2 else 0)
        end = start + (16 if development and i < 2 else counts[i])
        subset = source[split]
        data[split] = Frames(
            subset.frames,
            subset.rows[start:end],
            identity=dict(
                subset.identity,
                selected=subset.identity["selected"][start:end],
                excluded_prefix_groups=offsets[i],
                development=development,
            ),
        )
    return data


def hierarchy_evaluate(output, weights, data, root, config, device):
    try:
        return _hierarchy_evaluate(output, weights, data, root, config, device)
    except FileExistsError:
        raise
    except BaseException as error:
        directory = Path(output)
        directory.mkdir(parents=True, exist_ok=True)
        atomic_json(
            directory / "status.json",
            dict(result="failed", report="pending", step=0, error=str(error)),
        )
        atomic_json(
            directory / "failure.json",
            dict(
                error=str(error),
                scope="Incomplete hierarchy evaluation; preserve partial artifacts",
            ),
        )
        raise


def _hierarchy_evaluate(output, weights, data, root, config, device):
    from pathwm.evaluation.spatial_vae import stage_probes, profile_codec

    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Preserve prior evaluation: {output}")
    model = SpatialVAE.load(weights, device)
    if model.config.get("ablation") != config["variant"]:
        raise ValueError("Evaluation variant must match the saved hierarchy")
    rgb = data["test"].batch(range(len(data["test"])))["rgb"]
    rows, metrics, per = evaluate_images(model, rgb, device)
    train_rgb = data["train"].batch(range(len(data["train"])))["rgb"]
    val_rgb = data["validation"].batch(range(len(data["validation"])))["rgb"]
    metrics["train_mean"], _ = image_metrics(
        train_rgb.mean(0)[None].expand_as(rgb), rgb
    )
    metrics["resources"] = profile_codec(model, rgb[:8].to(device))
    metrics["quality_screen"] = dict(
        mean_mse=metrics["mean"]["mse"] <= 0.01,
        sampled_mse=metrics["sampled"]["mse"] <= 0.01,
    )
    report_evaluation(
        output,
        config,
        data["test"].identity,
        model,
        metrics,
        rows,
        device,
        "Heldout real COCO reconstruction, not general generation or semantic retention.",
    )
    # Compact raw outputs and independent per-image numbers, not all diagnostic traces.
    np.savez_compressed(
        output / "predictions.npz", **{k: v.numpy() for k, v in rows.items()}, **per
    )
    probes = stage_probes(model, train_rgb[:64], val_rgb[:16], rgb[:16], device)
    atomic_json(output / "probes.json", probes)
    submetrics = {}
    for label, size in [
        ("native96x128", (96, 128)),
        ("native65x79", (65, 79)),
        ("patterns", None),
    ]:
        if size:
            images, identity = native_crops(root, data["test"], size, 16)
        else:
            images, identity = (
                patterns(),
                {"synthetic": "deterministic fine-pattern control"},
            )
        out, scores, numbers = evaluate_images(model, images, device)
        submetrics[label] = scores
        report_evaluation(
            output / label,
            config,
            identity,
            model,
            scores,
            out,
            device,
            "Native-pixel geometry/detail diagnostic"
            if size
            else "Synthetic detail retention control",
        )
        np.savez_compressed(output / label / "per_image.npz", **numbers)
    metrics["diagnostics"] = submetrics
    metrics["probe_seconds"] = probes["seconds"]
    metrics["probe_summary"] = {
        name: {
            key: values["test"][key]["linear"]["normalized_mse"]
            for key in ("feature_recovery", "rgb_before", "rgb_after")
        }
        for name, values in probes["stages"].items()
    }
    result = json.loads((output / "result.json").read_text())
    atomic_json(
        output / "result.json",
        dict(
            result,
            metrics=metrics,
            probes_file="probes.json",
            weights_sha256=file_hash(weights),
        ),
    )
    write_report(output)
    return metrics


def hierarchy_comparison(output, records):
    """Existing HTML renderer with an explicitly labelled scientific figure."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    fig = Figure(figsize=(12, 5), layout="constrained")
    FigureCanvasAgg(fig)
    ax, bars = fig.subplots(1, 2)
    for variant in ("A_local", "C"):
        values = sorted(
            [r for r in records if r["variant"] == variant],
            key=lambda r: r["bits_per_pixel"],
        )
        ax.plot(
            [r["bits_per_pixel"] for r in values],
            [r["sampled_mse"] for r in values],
            "o-",
            label=variant,
        )
        for r in values:
            ax.annotate(
                f"beta={r['beta']}",
                (r["bits_per_pixel"], r["sampled_mse"]),
                xytext=(4, 9 if variant == "A_local" else -13),
                textcoords="offset points",
                fontsize=8,
            )
    ax.set(
        xlabel="KL bits / original pixel (rate proxy)",
        ylabel="Sampled RGB MSE",
        title="Fixed beta points; one seed, unequal resources",
    )
    ax.margins(x=0.15, y=0.15)
    ax.legend()
    single = [r for r in records if r["beta"] == 1.0]
    bars.bar([r["variant"] for r in single], [r["mean_mse"] for r in single])
    bars.axhline(0.01, color="red", linestyle="--", label="Photo quality screen")
    bars.set(
        ylabel="Posterior-mean RGB MSE",
        title="Same beta and updates; not matched rate/compute",
    )
    bars.legend()
    fig.savefig(output / "comparison.png", dpi=130)
    atomic_json(
        output / "result.json",
        dict(
            completed=True,
            metrics={"runs": records},
            evaluation_scope="One-seed real-photo sanity study; no superiority, semantic or general-generation claim. See child reports for inputs/reconstructions/errors/probes.",
        ),
    )
    atomic_json(
        output / "run.json",
        dict(
            identity=dict(
                settings={"protocol": "docs/spatial-vae-v2-plan.md"}, data={}
            ),
            source=source_record(__file__, torch.nn.Identity()),
        ),
    )
    atomic_json(
        output / "status.json",
        dict(result="completed", report="pending", step=512, error=None),
    )
    (output / "metrics.jsonl").write_text("")
    write_report(output)


def hierarchy_study(output, root, device):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Use a new study directory; preserve {output}")
    output.mkdir(parents=True)
    data = hierarchy_data(root, hierarchy_settings())
    records = []
    arms = [
        ("A_local", 1.0),
        ("C", 1.0),
        ("B", 1.0),
        ("C_after", 1.0),
        ("D", 1.0),
        ("E", 1.0),
    ]
    arms += [(v, b) for b in (0.1, 0.01) for v in ("A_local", "C")]
    start = perf_counter()
    for variant, beta in arms:
        if perf_counter() - start > 1200:
            raise RuntimeError("Study wall-clock cap reached; preserve completed runs")
        size = sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
        if size > 900 * 1024**2:
            raise RuntimeError("Study artifact budget reached")
        config = hierarchy_settings(variant, beta)
        path = output / f"{variant}_beta{beta:g}"
        train(path / "training", data["train"], data["validation"], config, device)
        fit = json.loads((path / "training/result.json").read_text())
        initial = json.loads((path / "training/initial_validation.json").read_text())
        valid = fit["metrics"]
        gates = dict(
            completed=fit["completed"],
            improves_initial=valid["mean"]["raw_mse"]
            <= 0.8 * initial["mean"]["raw_mse"],
            beats_mean=valid["mean"]["raw_mse"] <= 0.9 * valid["train_mean"]["raw_mse"],
            positive_kl=valid["posterior"]["rate_nats_per_pixel"] > 0,
            active_latent=valid["posterior"]["active_channels"] >= 1,
        )
        atomic_json(path / "sanity.json", dict(gates=gates, passed=all(gates.values())))
        if not fit["completed"]:
            raise RuntimeError("Training did not finish its fixed update budget")
        scores = hierarchy_evaluate(
            path / "evaluation",
            path / "training/weights.pt",
            data,
            root,
            config,
            device,
        )
        records.append(
            dict(
                variant=variant,
                beta=beta,
                sanity=all(gates.values()),
                mean_mse=scores["mean"]["mse"],
                sampled_mse=scores["sampled"]["mse"],
                bits_per_pixel=scores["posterior"]["kl_bits_per_original_pixel"],
                parameters=fit["parameters"],
                training_seconds=fit["training_seconds"],
                peak_reserved_mib=fit["peak_reserved_mib"],
                peak_allocated_mib=fit["peak_allocated_mib"],
                memory_scope="Training peak; reserved includes allocator cache from preceding runs in this process",
                resources=scores["resources"],
                photo_quality=all(scores["quality_screen"].values()),
                report=str(path / "evaluation/report.html"),
            )
        )
        atomic_json(output / "progress.json", records)
        if len(records) == 2 and not all(r["sanity"] for r in records):
            hierarchy_comparison(output, records)
            raise RuntimeError(
                "Sanity gate failed; no expanded grid permitted by protocol"
            )
    hierarchy_comparison(output, records)
    return records


def color_diagnosis(output, weights, root, device):
    from pathwm.evaluation.spatial_vae import (
        color_grid_metrics,
        color_readouts,
        constant_decoder_diagnostic,
    )

    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Preserve existing diagnosis {output}")
    output.mkdir(parents=True)
    config = dict(
        hierarchy_settings("C", 0.1),
        seed=57201,
        purpose="Frozen color/grid attribution and fixed-budget component reparability",
    )
    data = hierarchy_data(root, config)
    tensors = {s: d.batch(range(len(d)))["rgb"] for s, d in data.items()}
    model = SpatialVAE.load(weights, device)
    if model.config.get("ablation") != "C":
        raise ValueError("This protocol uses the C reference")
    frozen = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    rows, _, _ = evaluate_images(model, tensors["test"], device)
    constants, pictures = constant_decoder_diagnostic(model, device)
    readouts = color_readouts(
        model, tensors["train"][:256], tensors["validation"], tensors["test"], device
    )
    metrics = {
        key: color_grid_metrics(rows[key], rows["target"])
        for key in ["mean", "sampled"]
    }
    native, identity = native_crops(root, data["test"], (96, 128), 16)
    native_rows, _, _ = evaluate_images(model, native, device)
    metrics["native_mean"] = color_grid_metrics(native_rows["mean"], native)
    metrics["constants"] = constants
    metrics["color_readouts"] = readouts
    report_evaluation(
        output / "reference",
        config,
        data["test"].identity,
        model,
        metrics,
        rows,
        device,
        "Diagnostic re-use of previously evaluated test groups; color and grid measurements, no pristine selection claim.",
    )
    report_evaluation(
        output / "reference/native",
        config,
        identity,
        model,
        metrics["native_mean"],
        native_rows,
        device,
        "Saved model native-pixel crop diagnostics, fixed first16 eligible test sources.",
    )
    np.savez_compressed(
        output / "reference/predictions.npz", **{k: v.numpy() for k, v in rows.items()}
    )
    np.savez_compressed(
        output / "reference/constants.npz",
        **{k: v.numpy() for k, v in pictures.items()},
    )
    selections = [0, 4, 8, 13, 18, 22, 26]
    example_panel(
        output / "reference/palette.png",
        dict(
            original=pictures["palette_input"][selections],
            mean=pictures["palette_output"][selections],
        ),
        dict(original="Constant RGB input", mean="Posterior-mean output"),
        count=7,
    )
    atomic_json(
        output / "reference/frozen.json",
        dict(
            passed=all(
                torch.equal(v, model.state_dict()[k].cpu()) for k, v in frozen.items()
            )
        ),
    )
    print(
        "REFERENCE",
        json.dumps(
            {
                k: metrics["mean"][k]
                for k in ["raw_mse", "global_chroma_mse", "chroma_gain"]
            }
        ),
        flush=True,
    )
    print("CONSTANT TRACE", json.dumps(constants["trace"]), flush=True)
    print(
        "READOUTS",
        json.dumps({k: v["test"]["linear"]["mse"] for k, v in readouts.items()}),
        flush=True,
    )
    del model
    records = {}
    comparisons = dict(original=rows["target"], reference=rows["mean"])
    for part in ["decoder", "encoder", "all"]:
        path = output / part
        run = train(
            path / "training",
            data["train"],
            data["validation"],
            config,
            device,
            source=weights,
            trainable=part,
        )
        if run.step != config["steps"]:
            raise RuntimeError("Component continuation did not finish")
        model = SpatialVAE.load(path / "training/weights.pt", device)
        result = json.loads((path / "training/result.json").read_text())
        state = model.state_dict()
        intact = all(
            torch.equal(v, state[k].cpu())
            for k, v in frozen.items()
            if part != "all" and not k.startswith(part + ".")
        )
        if not intact:
            raise RuntimeError("Frozen component changed")
        after, _, _ = evaluate_images(model, tensors["test"], device)
        color = {
            key: color_grid_metrics(after[key], after["target"])
            for key in ["mean", "sampled"]
        }
        current_const, _ = constant_decoder_diagnostic(model, device)
        color["constant_output_rms"] = current_const["output_rms"]
        nrows, _, _ = evaluate_images(model, native, device)
        color["native_mean"] = color_grid_metrics(nrows["mean"], native)
        base, now = metrics["mean"], color["mean"]
        guards = dict(
            rgb_nonregression=now["raw_mse"] <= 1.05 * base["raw_mse"],
            color_improvement=now["global_chroma_mse"]
            <= 0.8 * base["global_chroma_mse"],
            grid_improvement=now["period4"]["residual_rms"]
            <= 0.5 * base["period4"]["residual_rms"],
        )
        color.update(
            guards=guards,
            frozen_component_unchanged=intact,
            trainable=part,
            reference_sha256=file_hash(weights),
            weights_sha256=file_hash(path / "training/weights.pt"),
            training_seconds=result["training_seconds"],
        )
        report_evaluation(
            path / "evaluation",
            config,
            data["test"].identity,
            model,
            color,
            after,
            device,
            "Fixed512-update diagnostic from reference weights; fresh optimizer. Frozen-component outcome is local reparability evidence.",
        )
        report_evaluation(
            path / "native",
            config,
            identity,
            model,
            color["native_mean"],
            nrows,
            device,
            "Native color/grid diagnostic after the same continuation.",
        )
        np.savez_compressed(
            path / "evaluation/predictions.npz",
            **{k: v.numpy() for k, v in after.items()},
        )
        records[part] = color
        comparisons[part] = after["mean"]
        atomic_json(output / "progress.json", records)
        print(
            part,
            json.dumps(
                {k: now[k] for k in ["raw_mse", "global_chroma_mse", "chroma_gain"]}
            ),
            flush=True,
        )
        del model, run
    atomic_json(
        output / "run.json",
        dict(
            identity=dict(
                settings=config, data={k: v.identity for k, v in data.items()}
            ),
            source=source_record(__file__, torch.nn.Identity()),
        ),
    )
    atomic_json(
        output / "result.json",
        dict(
            completed=True,
            metrics=dict(reference=metrics, continuations=records),
            evaluation_scope="Controlled diagnosis: known test population, one seed, brief fresh-optimizer continuation; no general quality or unique asymptotic cause claim.",
        ),
    )
    atomic_json(
        output / "status.json",
        dict(result="completed", report="pending", step=512, error=None),
    )
    (output / "metrics.jsonl").write_text("")
    example_panel(
        output / "comparison.png",
        comparisons,
        dict(
            original="Original",
            reference="Reference",
            decoder="Decoder trained",
            encoder="Encoder trained",
            all="Both trained",
        ),
        count=6,
    )
    write_report(
        output,
        batch={"rgb": rows["target"][:6]},
        outputs={k: v[:6] for k, v in comparisons.items() if k != "original"},
    )
    return records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--variant",
        choices=[
            "base",
            "attention",
            "reversible",
            "A_local",
            "A_exact",
            "B",
            "C",
            "C_after",
            "D",
            "E",
        ],
        default="base",
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--data-root", type=Path, default=Path("data/curriculum/coco_v1"))
    p.add_argument("--device", default="cpu")
    p.add_argument("--weights", type=Path)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--stop-after", type=int)
    p.add_argument("--development", action="store_true")
    p.add_argument("--overfit", action="store_true")
    p.add_argument("--evaluate-only", action="store_true")
    p.add_argument("--low-kl", action="store_true")
    p.add_argument("--hierarchy-study", action="store_true")
    p.add_argument("--color-diagnosis", action="store_true")
    args = p.parse_args()
    if args.stop_after is not None and args.stop_after <= 0:
        p.error("--stop-after must be positive")
    if args.low_kl and not args.weights:
        p.error("--low-kl requires the corresponding beta-one --weights")
    if sum([args.development, args.overfit, args.low_kl]) > 1:
        p.error("Development, overfit and low-KL modes are separate protocols")
    if args.color_diagnosis:
        if not args.weights:
            p.error("--color-diagnosis requires reference --weights")
        color_diagnosis(args.output, args.weights, args.data_root, args.device)
        return
    if args.hierarchy_study:
        hierarchy_study(args.output, args.data_root, args.device)
        return
    hierarchy = args.variant in ("A_local", "A_exact", "B", "C", "C_after", "D", "E")
    c = hierarchy_settings(args.variant) if hierarchy else settings(args.variant)
    if args.low_kl:
        c.update(beta=0.001, seed=56201)
    if args.development:
        c.update(steps=8, wall_seconds=90)
    if args.overfit:
        c.update(steps=128, beta=0.0, wall_seconds=180)
    data = (hierarchy_data if hierarchy else data_sets)(
        args.data_root, c, args.development or args.overfit
    )
    if args.overfit:
        data["train"] = data["train"].take(8)
    if args.evaluate_only:
        if not args.weights:
            p.error("--weights required")
        (hierarchy_evaluate if hierarchy else evaluate_suite)(
            args.output, args.weights, data, args.data_root, c, args.device
        )
    else:
        train(
            args.output,
            data["train"],
            data["validation"],
            c,
            args.device,
            args.resume,
            args.stop_after,
            args.weights,
            args.overfit,
        )


if __name__ == "__main__":
    main()
