"""Bounded COCO hierarchy comparison; edit the visible composition and protocol here.

python -m experiments.hierarchy_fusion --arm deep_fusion --seed 7401 --device cuda
Uses the ordinary perception trainer, checkpoint format and report. No work on import.
"""

import argparse
from html import escape
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from pathwm.data.images import CocoMasks
from pathwm.io import (
    atomic_json,
    file_hash,
    resume_arguments,
    seed_everything,
    state_hash,
    trainable_parameters,
)
from pathwm.models.decoders import DenseHead
from pathwm.models.encoders import PyramidEncoder
from pathwm.models.perception import Perception
from pathwm.training.perception import check, train_perception


ARMS = {
    "shallow": (1, 0, 384),
    "deep": (2, 0, 384),
    "deep_fusion": (2, 2, 384),
    "shallow_long": (1, 0, 768),
}


def build_model(seed, arm):
    def create(depth, fusion):
        encoder = PyramidEncoder(width=32, levels=3, depth=depth, fusion_depth=fusion)
        levels = tuple(encoder.feature_spec)
        return Perception(
            encoder,
            {
                "rgb": DenseHead(
                    encoder.feature_spec,
                    channels=3,
                    levels=levels,
                    activation="sigmoid",
                    retain_statistics=True,
                ),
                "mask": DenseHead(
                    encoder.feature_spec, levels=levels, retain_statistics=True
                ),
            },
        )

    seed_everything(seed)
    anchor = create(1, 0)
    seed_everything(seed + 100000)
    depth, fusion, _ = ARMS[arm]
    model = create(depth, fusion)
    state = model.state_dict()
    for name, value in anchor.state_dict().items():
        if name not in state or value.shape != state[name].shape:
            raise ValueError(f"Shared initialization mismatch: {name}")
        state[name] = value
    model.load_state_dict(state, strict=True)
    seed_everything(seed)
    return model, state_hash(anchor)


def objective(outputs, batch):
    valid = batch["valid"]
    counts = valid.sum((1, 2, 3))
    keep = counts > 0
    bce = F.binary_cross_entropy_with_logits(
        outputs["mask"], batch["mask"], reduction="none"
    )
    mask = (
        ((bce * valid).sum((1, 2, 3))[keep] / counts[keep]).mean()
        if keep.any()
        else bce.sum() * 0
    )
    return {"rgb_mse": F.mse_loss(outputs["rgb"], batch["rgb"]), "mask_bce": mask}


def data_splits(root, masks):
    result = {}
    for split, count in [("train", 512), ("validation", 128), ("test", 128)]:
        result[split] = CocoMasks(root, masks, split).take(count)
        if len(result[split]) != count:
            raise ValueError(f"Insufficient {split} rows")
    for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        if set(result[a].rows) & set(result[b].rows):
            raise ValueError("Comparison split overlap")
    return result


@torch.no_grad()
def score(model, data, output, device):
    """Raw final outputs and fixed controls; no fitting or selection on this split."""
    model.eval()
    before = state_hash(model)
    arrays = {}
    shifted = np.roll(np.arange(len(data)), 1)
    for start in range(0, len(data), 16):
        ids = np.arange(start, min(start + 16, len(data)))
        batch = data.batch(ids, device)
        features = model.encoder(batch["rgb"])
        other = model.encoder(data.batch(shifted[ids], device)["rgb"])
        values = {
            "target_rgb": batch["rgb"],
            "target_mask": batch["mask"],
            "valid": batch["valid"],
        }
        for name, tokens in [
            ("observed", features),
            ("shuffled", other),
            ("zero", {k: torch.zeros_like(v) for k, v in features.items()}),
        ]:
            values[f"{name}_rgb"] = model.heads["rgb"](tokens)
            values[f"{name}_mask"] = model.heads["mask"](tokens)
        for name, value in values.items():
            arrays.setdefault(name, []).append(value.cpu().numpy())
    arrays = {k: np.concatenate(v) for k, v in arrays.items()}
    arrays.update(rows=data.rows, shuffled_rows=data.rows[shifted])
    valid = arrays["valid"] > 0
    target = (arrays["target_mask"] > 0) & valid
    counts = valid.sum((1, 2, 3))
    keep = counts > 0

    def iou(predicted):
        predicted = predicted & valid
        union = (predicted | target).sum((1, 2, 3))
        intersection = (predicted & target).sum((1, 2, 3))
        return float(np.where(union > 0, intersection / union.clip(1), 1)[keep].mean())

    metrics = {}
    for name in ("observed", "shuffled", "zero"):
        logits = arrays[f"{name}_mask"].astype(np.float64)
        bce = np.logaddexp(0, logits) - arrays["target_mask"] * logits
        metrics[name] = dict(
            rgb_mse=float(
                np.square(
                    arrays[f"{name}_rgb"].astype(np.float64) - arrays["target_rgb"]
                ).mean()
            ),
            mask_bce=float(((bce * valid).sum((1, 2, 3))[keep] / counts[keep]).mean()),
            mask_iou=iou(logits >= 0),
        )
    metrics["baselines"] = dict(
        gray_rgb_mse=float(
            np.square(arrays["target_rgb"].astype(np.float64) - 0.5).mean()
        ),
        empty_mask_iou=iou(np.zeros_like(target)),
        full_mask_iou=iou(valid),
    )
    if before != state_hash(model):
        raise RuntimeError("Scoring changed model weights or buffers")
    np.savez_compressed(output / "test_outputs.npz", **arrays)
    result = dict(
        metrics=metrics,
        data=data.identity,
        checkpoint_sha256=file_hash(output / "last.pt"),
        arrays_sha256=file_hash(output / "test_outputs.npz"),
        scope="128 reused internal test images; foreground union, not entity identity",
    )
    atomic_json(output / "test_metrics.json", result)
    return result


def append_test_table(output, result):
    rows = "".join(
        f"<tr><td>{escape(name)}</td><td>{v['rgb_mse']:.6f}</td>"
        f"<td>{v['mask_iou']:.4f}</td></tr>"
        for name, v in result["metrics"].items()
        if name != "baselines"
    )
    section = (
        "<section><h2>Final held-out screen</h2><p>128 reused internal test images. "
        "Foreground masks measure coverage, not persistent identity.</p><table>"
        "<tr><th>Features</th><th>RGB MSE</th><th>Mask IoU</th></tr>"
        + rows
        + "</table></section>"
    )
    path = output / "report.html"
    html = path.read_text()
    if "</main>" not in html:
        raise ValueError("Missing report container")
    path.write_text(html.replace("</main>", section + "</main>"))
    final = path.read_text()
    if "<script" in final or 'src="http' in final:
        raise ValueError("Report must remain self-contained and script-free")
    qa = json.loads((output / "report.qa.json").read_text())
    qa.update(
        report_sha256=file_hash(path),
        appendix_recipe_sha256=file_hash(__file__),
        browser_checked=False,
        status="structural_verified",
    )
    atomic_json(output / "report.qa.json", qa)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, default="shallow")
    parser.add_argument("--seed", type=int, default=7401)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--data-root", type=Path, default=Path("data/coco64"))
    parser.add_argument("--masks", type=Path, default=Path("data/assets/coco_masks"))
    args = resume_arguments(parser, parser.parse_args())
    if args.stop_after is not None and args.stop_after < 1:
        parser.error("stop-after must be positive")
    seed_everything(args.seed)
    gpu = torch.device(args.device).type == "cuda"
    limit = None
    if gpu:
        free, total = torch.cuda.mem_get_info(args.device)
        limit = min(4 * 1024**3, free - 1024**3)
        if limit <= 0:
            raise RuntimeError("Insufficient GPU headroom")
        device_index = torch.device(args.device).index
        if device_index is None:
            device_index = torch.cuda.current_device()
        torch.cuda.set_per_process_memory_fraction(limit / total, device_index)
        torch.cuda.reset_peak_memory_stats(args.device)
    data = data_splits(args.data_root, args.masks)
    model, anchor_hash = build_model(args.seed, args.arm)
    model.to(args.device)
    if args.check:
        result = check(model, data["train"], objective, args.device, batch_size=4)
        with torch.no_grad():
            model(data["validation"].batch(np.arange(16), args.device)["rgb"])
        result["peak_reserved_bytes"] = (
            torch.cuda.max_memory_reserved(args.device) if gpu else None
        )
        print(json.dumps(result, indent=2))
        return
    depth, fusion, steps = ARMS[args.arm]
    settings = dict(
        arm=args.arm,
        seed=args.seed,
        device=args.device,
        data_root=str(args.data_root),
        masks=str(args.masks),
        width=32,
        levels=3,
        stage_depth=depth,
        fusion_depth=fusion,
        steps=steps,
        batch_size=4,
        evaluate_every=64,
        grad_clip=1.0,
        learning_rate=0.0003,
        weight_decay=0.0001,
        precision="fp32",
        purpose="paired architecture screen",
        initial_anchor_sha256=anchor_hash,
        test_identity=data["test"].identity,
    )
    optimizer = torch.optim.AdamW(
        trainable_parameters(model), lr=0.0003, weight_decay=0.0001
    )
    started = time.monotonic()
    output = (
        args.resume
        or args.output
        or Path(f"runs/hierarchy_fusion_v1/seed_{args.seed}/{args.arm}")
    )
    train_perception(
        model,
        data["train"],
        data["validation"],
        objective,
        optimizer,
        settings=settings,
        recipe=__file__,
        output=output,
        device=args.device,
        resume=args.resume is not None,
        stop_after=args.stop_after,
    )
    status = json.loads((output / "status.json").read_text())
    if status["result"] == "completed":
        result = score(model, data["test"], output, args.device)
        try:
            append_test_table(output, result)
        except BaseException as exc:
            atomic_json(
                output / "status.json", dict(status, report="failed", error=str(exc))
            )
            raise
    resources = dict(
        seconds=time.monotonic() - started,
        parameters=sum(p.numel() for p in model.parameters()),
        encoder_parameters=sum(p.numel() for p in model.encoder.parameters()),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(args.device)
        if gpu
        else None,
        peak_reserved_bytes=torch.cuda.max_memory_reserved(args.device)
        if gpu
        else None,
        free_device_bytes=torch.cuda.mem_get_info(args.device)[0] if gpu else None,
        reserved_limit_bytes=limit,
    )
    # Resume resources describe only this invocation, never replace the first receipt.
    resource_path = output / (
        "resume_resources.json" if args.resume else "resources.json"
    )
    atomic_json(resource_path, resources)
    if gpu and (
        resources["peak_reserved_bytes"] > limit
        or resources["free_device_bytes"] < 1024**3
    ):
        atomic_json(
            output / "status.json",
            dict(status, result="failed", error="GPU resource gate failed"),
        )
        raise RuntimeError("GPU resource gate failed; partial results retained")
    print(json.dumps(resources), flush=True)
    print(f"Results: {output}/report.html")


if __name__ == "__main__":
    main()
