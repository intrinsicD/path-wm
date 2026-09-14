"""Frozen photo-detail localization: patch audit plus exact kernel readers."""

import argparse
import json
import shutil
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F

from pathwm.data.images import Frames
from pathwm.data.photo_recall import photo_data
from pathwm.evaluation.photo_detail import stage_values, patch_audit, grid_score
from pathwm.evaluation.report import write_report
from pathwm.models.memory_output import load_model, FrozenFeatureNormalization
from pathwm.models.photo_probe import RidgeReader
from pathwm.io import (
    atomic_json,
    file_hash,
    source_record,
    state_hash,
    seed_everything,
    environment,
)


def settings():
    return dict(
        seed=46101,
        data_seed=45001,
        purpose="frozen photo-detail diagnostic",
        ridge=[1e-6, 1e-4, 0.01, 0.1],
        bandwidth=[0.5, 1.0, 2.0],
        cpu_seconds=600,
        cache_seconds=180,
        memory_mib=3072,
        disk_free_gib=3,
        bootstrap_seed=46199,
        bootstrap_count=1000,
        objective="Validation-selected closed-form RGB16 kernel ridge; no agent training",
        example_labels={
            "input": "Previously observed photos — fresh held-out group suffix",
            "rgb": "Each column: encoder reader → first-state reader → final-workspace reader (16px grids enlarged)",
        },
    )


def subset(data, start, end):
    return Frames(
        data.frames,
        data.rows[start:end],
        identity=dict(data.identity, selected=data.identity["selected"][start:end]),
    )


def budget(output, config, device):
    if shutil.disk_usage(output).free < config["disk_free_gib"] * 1024**3:
        raise RuntimeError("Disk reserve floor reached")
    if str(device).startswith("cuda"):
        free, _ = torch.cuda.mem_get_info(device)
        if (
            free < 1024**3
            or torch.cuda.max_memory_reserved(device) > config["memory_mib"] * 1024**2
        ):
            raise RuntimeError("GPU reserve/headroom budget reached")


@torch.no_grad()
def extract(model, data, device, *, native=False):
    rows, images = {}, {}
    audit = None
    for start in range(0, len(data), 8):
        rgb = data.batch(range(start, min(start + 8, len(data))), device)["rgb"]
        values, controls, audit = stage_values(model, rgb, native=native, start=start)
        for key, value in values.items():
            rows.setdefault(key, []).append(value.cpu())
        for key, value in controls.items():
            if native or key == "grid":
                images.setdefault(key, []).append(value.cpu())
    return dict(
        features={k: torch.cat(v) for k, v in rows.items()},
        images={k: torch.cat(v) for k, v in images.items()},
        audit=audit,
    )


def select_reader(x, y, vx, vy, kind, config):
    best, state = None, None
    rows = []
    for width in config["bandwidth"] if kind == "rbf" else [1.0]:
        factor = RidgeReader.factor(x, y, kernel=kind, bandwidth=width)
        for ridge in config["ridge"]:
            m = RidgeReader.solve(factor, ridge=ridge)
            pred = RidgeReader.predict(m, vx).clamp(0, 1)
            error = float((pred - vy).square().mean())
            rows.append(
                dict(kernel=kind, bandwidth=width, ridge=ridge, validation_mse=error)
            )
            if best is None or error < best:
                best, state = error, m
    return state, rows


def transition_tests(predictions, target, config):
    pairs = [
        ("stem", "encoder"),
        ("encoder", "observed_all"),
        ("observed_all", "stored_all"),
        ("stored_all", "stored_working"),
        ("stored_all", "recall_all"),
        ("stored_working", "recall_working"),
        ("recall_all", "recall_working"),
    ]
    rng = np.random.default_rng(config["bootstrap_seed"])
    draws = rng.integers(0, len(target), (config["bootstrap_count"], len(target)))
    result = []
    for a, b in pairs:
        row = dict(before=a, after=b, families={})
        for family in ("linear", "rbf"):
            one = predictions[f"{a}.{family}"].double()
            two = predictions[f"{b}.{family}"].double()
            t = target.double()
            e1 = (one - t).square().mean((1, 2, 3))
            e2 = (two - t).square().mean((1, 2, 3))
            diff = (e2 - e1).numpy()
            ci = np.quantile(diff[draws].mean(1), [0.025, 0.975]).tolist()
            row["families"][family] = dict(
                error_ratio=float(e2.mean() / e1.mean().clamp_min(1e-12)),
                paired_difference=float(diff.mean()),
                conditional_bootstrap95=ci,
                degraded=bool(e2.mean() > 1.2 * e1.mean() and ci[0] > 0),
            )
        row["both_readers_degraded"] = all(
            x["degraded"] for x in row["families"].values()
        )
        result.append(row)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--data-root", type=Path, default=Path("data/curriculum/coco_v1"))
    p.add_argument("--device", default="cpu")
    p.add_argument("--development", action="store_true")
    args = p.parse_args()
    config = settings()
    seed_everything(config["seed"])
    if args.development:
        config.update(purpose="development", cpu_seconds=120)
    args.output.mkdir(parents=True, exist_ok=False)
    atomic_json(
        args.output / "status.json",
        dict(result="running", report="pending", step=0, error=None),
    )
    (args.output / "metrics.jsonl").write_text("")
    try:
        selected = photo_data(args.data_root, (1040, 144, 512), config["data_seed"])
        if args.development:
            train_data = subset(selected["train"], 1024, 1040)
            validation = subset(selected["validation"], 128, 144)
            test = validation
        else:
            train_data = subset(selected["train"], 0, 1024)
            validation = subset(selected["validation"], 0, 128)
            test = subset(selected["test"], 256, 512)
        model = load_model(args.weights, args.device).requires_grad_(False).eval()
        if str(args.device).startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(args.device)
        before = state_hash(model)
        source = source_record(__file__, model)
        atomic_json(
            args.output / "run.json",
            dict(
                identity=dict(
                    settings=config,
                    data={
                        k: v.identity
                        for k, v in dict(
                            train=train_data, validation=validation, test=test
                        ).items()
                    },
                    environment=environment(args.device),
                ),
                source=source,
                weights_path=str(args.weights.resolve()),
                weights_sha256=file_hash(args.weights),
            ),
        )
        snapshots = {}
        for i, (name, expected) in enumerate(source["files"].items()):
            local = f"source/{i:03d}_{Path(name).name}"
            q = args.output / local
            q.parent.mkdir(exist_ok=True)
            q.write_bytes(Path(name).read_bytes())
            assert file_hash(q) == expected
            snapshots[name] = local
        atomic_json(args.output / "source_index.json", snapshots)
        encoder = model.agent.encoders["image"]
        base = (
            encoder.base if isinstance(encoder, FrozenFeatureNormalization) else encoder
        )
        patch = patch_audit(base.stem.patch)
        atomic_json(args.output / "patch.json", patch)
        budget(args.output, config, args.device)
        start = perf_counter()
        caches = {}
        for name, data in [
            ("train", train_data),
            ("validation", validation),
            ("test", test),
        ]:
            caches[name] = extract(model, data, args.device, native=name == "test")
            if perf_counter() - start > config["cache_seconds"]:
                raise RuntimeError("Cache wall budget reached")
            budget(args.output, config, args.device)
        cache_seconds = perf_counter() - start
        assert state_hash(model) == before
        peak = (
            torch.cuda.max_memory_reserved(args.device) / 1024**2
            if str(args.device).startswith("cuda")
            else 0
        )
        del model
        if str(args.device).startswith("cuda"):
            torch.cuda.empty_cache()
        for name, cache in caches.items():
            torch.save(cache, args.output / f"{name}_cache.pt")
        atomic_json(
            args.output / "cache.json",
            dict(
                frozen_state_sha256=before,
                frozen_equal=True,
                cache_seconds=cache_seconds,
                peak_reserved_mib=peak,
                hashes={n: file_hash(args.output / f"{n}_cache.pt") for n in caches},
                audit=caches["test"]["audit"],
            ),
        )
        t = caches["train"]
        v = caches["validation"]
        fresh = caches["test"]
        target = fresh["images"]["grid"]
        y = t["images"]["grid"].flatten(1).double()
        vy = v["images"]["grid"].flatten(1).double()
        predictions = {}
        readers = {}
        selection = {}
        metrics = {}
        start = perf_counter()
        rows = []
        for name, x in t["features"].items():
            for family in ("linear", "rbf"):
                key = f"{name}.{family}"
                m, trials = select_reader(x, y, v["features"][name], vy, family, config)
                pred = (
                    RidgeReader.predict(m, fresh["features"][name])
                    .clamp(0, 1)
                    .reshape_as(target)
                )
                train_pred = (
                    RidgeReader.predict(m, x).clamp(0, 1).reshape(-1, 3, 16, 16)
                )
                readers[key] = m
                selection[key] = trials
                predictions[key] = pred.float()
                metrics[key] = dict(
                    grid_score(pred, target),
                    validation_rgb_mse=min(r["validation_mse"] for r in trials),
                    train_rgb_mse=float(
                        (train_pred - t["images"]["grid"]).square().mean()
                    ),
                    selected_ridge=m["ridge"],
                    selected_bandwidth=m["bandwidth"],
                    dimensions=x.shape[1],
                )
                rows.append(
                    dict(
                        step=len(rows) + 1,
                        split="validation",
                        reader=key,
                        **metrics[key],
                    )
                )
                print(f"{key}: grid MSE {metrics[key]['rgb_mse']:.6f}", flush=True)
                if perf_counter() - start > config["cpu_seconds"]:
                    raise RuntimeError("Reader fit wall budget reached")
                budget(args.output, config, "cpu")
        permutation = torch.randperm(
            len(y), generator=torch.Generator().manual_seed(46123)
        )
        m, trials = select_reader(
            t["features"]["recall_working"],
            y[permutation],
            v["features"]["recall_working"],
            vy,
            "linear",
            config,
        )
        readers["shuffled_working"] = m
        selection["shuffled_working"] = trials
        predictions["shuffled_working"] = (
            RidgeReader.predict(m, fresh["features"]["recall_working"])
            .clamp(0, 1)
            .reshape_as(target)
            .float()
        )
        predictions["mean"] = y.mean(0).reshape(1, 3, 16, 16).expand_as(target).float()
        predictions["oracle_color_only"] = target.mean(
            (-2, -1), keepdim=True
        ).expand_as(target)
        predictions["native"] = F.avg_pool2d(fresh["images"]["native"], 4)
        for k in ("shuffled_working", "mean", "oracle_color_only", "native"):
            metrics[k] = grid_score(predictions[k], target)
        full_metrics = {
            k: grid_score(fresh["images"][k], fresh["images"]["target"])
            for k in ("codec", "codec_no_detail", "native")
        }
        transitions = transition_tests(predictions, target, config)
        positive = metrics["raw_grid.linear"]["rgb_mse"] < 1e-4
        for k in readers:
            metrics[k]["rgb_access"] = (
                metrics[k]["rgb_mse"] < 0.8 * metrics["mean"]["rgb_mse"]
            )
            metrics[k]["spatial_access"] = (
                metrics[k]["spatial_mse"] < 0.8 * metrics["mean"]["spatial_mse"]
            )
        torch.save(
            dict(schema="photo-readers-v1", readers=readers, settings=config),
            args.output / "weights.pt",
        )
        atomic_json(args.output / "selection.json", selection)
        result = dict(
            completed=True,
            positive_control_passed=positive,
            metrics=metrics,
            full_image_controls=full_metrics,
            transitions=transitions,
            patch=patch,
            fit_seconds=perf_counter() - start,
            cache_seconds=cache_seconds,
            peak_reserved_mib=peak,
            frozen_equal=True,
            evaluation_scope="Development validation only."
            if args.development
            else "Fresh256 group-disjoint COCO crops, test suffix256:512. Readers recover RGB16 patch means from frozen stages, not full-resolution photographs. Two exact-solve reader families; validation selection only; no exhaustive information-loss or SGD-duration conclusion. Codec control has a separate raw-detail branch.",
        )
        np.savez_compressed(
            args.output / "predictions.npz",
            target=target.numpy(),
            **{k: v.numpy() for k, v in predictions.items()},
        )
        atomic_json(args.output / "result.json", result)
        (args.output / "reader_metrics.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows)
        )
        atomic_json(
            args.output / "status.json",
            dict(result="completed", report="pending", step=0, error=None),
        )
        strip = torch.cat(
            [
                predictions[k][:8]
                for k in (
                    "encoder.linear",
                    "observed_all.linear",
                    "recall_working.linear",
                )
            ],
            dim=-2,
        )
        write_report(
            args.output,
            batch={"rgb": fresh["images"]["target"][:8]},
            outputs={"rgb": strip},
        )
        print(
            json.dumps(
                dict(positive_control=positive, fit_seconds=result["fit_seconds"])
            ),
            flush=True,
        )
    except BaseException as e:
        completed = (args.output / "result.json").exists()
        atomic_json(
            args.output / "status.json",
            dict(
                result="completed" if completed else "failed",
                report="failed",
                step=0,
                error=str(e),
            ),
        )
        raise


if __name__ == "__main__":
    main()
