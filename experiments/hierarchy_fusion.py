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
    Run,
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
from pathwm.models.multiscale import ConditionedBlock
from pathwm.evaluation.report import write_report
from pathwm.training.perception import check, evaluate, train_perception


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


@torch.no_grad()
def construct_weights(model):
    """Write a color-transport circuit, not pretrained visual semantics.

    All numbers are explicit; no data, fitting, RNG or forward-code replacement.
    Carriers make normalization approximately linear for the smaller color signals.
    """
    encoder = model.encoder.encoder
    if encoder.width != 32 or len(encoder.pyramid.stages) != 3:
        raise ValueError("This construction requires width32 and three scales")
    for parameter in model.parameters():
        parameter.zero_()
    for module in model.modules():
        if isinstance(module, (torch.nn.LayerNorm, torch.nn.GroupNorm)):
            module.weight.fill_(1)
        if isinstance(module, ConditionedBlock):
            identity = torch.eye(32, device=module.attention.in_proj_weight.device)
            module.attention.in_proj_weight.copy_(identity.repeat(3, 1))
            module.attention.out_proj.weight.copy_(8 * identity)
            module.mlp[0].weight.copy_(torch.cat((identity, -identity)))
            module.mlp[2].weight.copy_(0.01 * torch.cat((identity, -identity), 1))
    patch = encoder.stem.patch
    for color in range(3):
        for sign, channel in [(1, 2 * color), (-1, 2 * color + 1)]:
            patch.weight[channel, color].fill_(sign * 256 / 16)
            patch.bias[channel] = -sign * 128
    patch.bias[6], patch.bias[7] = 4096, -4096
    for name, head in model.heads.items():
        for projection in head.input.projections.values():
            for color in range(3):
                projection.weight[color, 2 * color] = 2
                projection.weight[color, 2 * color + 1] = -2
        previous_width = None
        for conv_index, norm_index in [(0, 2), (5, 6), (9, 10)]:
            conv, norm = head.trunk[conv_index], head.trunk[norm_index]
            group = conv.out_channels // 8
            for g in range(8):
                conv.bias[g * group + group - 2] = 64
                conv.bias[g * group + group - 1] = -64
            norm.weight.fill_(64 * (2 / group) ** 0.5)
            for color in range(3):
                pos, neg = color * group, color * group + 1
                if previous_width is None:
                    for scale, weight in enumerate((0.80, 0.15, 0.05)):
                        conv.weight[pos, 128 * scale + color, 1, 1] = weight
                        conv.weight[neg, 128 * scale + color, 1, 1] = -weight
                else:
                    source = color * (previous_width // 8)
                    for target, sign in [(pos, 1), (neg, -1)]:
                        conv.weight[target, source, 1, 1] = sign * 0.5
                        conv.weight[target, source + 1, 1, 1] = -sign * 0.5
                norm.bias[pos : neg + 1] = (
                    0 if name == "mask" and conv_index == 9 else 4
                )
            previous_width = conv.out_channels
        final = head.trunk[-1]
        if name == "rgb":
            for color in range(3):
                final.weight[color, 4 * color, 0, 0] = 2
                final.weight[color, 4 * color + 1, 0, 0] = -2
        elif name == "mask":
            for color in range(3):
                final.weight[0, 4 * color : 4 * color + 2, 0, 0] = 4
            final.bias.fill_(-0.5)
        else:
            raise ValueError("Construction supports the declared RGB/mask heads")
    return dict(method="handwritten_color_transport", data_fitted_parameters=0)


@torch.no_grad()
def fit_readouts(model, data, device, *, deadline=float("inf")):
    """Supervised ridge fit of final convolutions only; never an optimizer update."""
    model.eval()
    grams = {
        k: torch.zeros(33, 33, dtype=torch.float64, device=device) for k in model.heads
    }
    rhs = {
        k: torch.zeros(33, h.trunk[-1].out_channels, dtype=torch.float64, device=device)
        for k, h in model.heads.items()
    }
    counts = {k: 0 for k in model.heads}
    for start in range(0, len(data), 16):
        if time.monotonic() > deadline:
            raise TimeoutError("Readout fitting budget exceeded")
        batch = data.batch(np.arange(start, min(start + 16, len(data))), device)
        features = model.encoder(batch["rgb"])
        for name, head in model.heads.items():
            hidden = (
                head.trunk[:-1](head.input(features))
                .permute(0, 2, 3, 1)
                .reshape(-1, 32)
                .double()
            )
            if name == "rgb":
                target = torch.logit(batch["rgb"].clamp(0.01, 0.99))
            else:
                target = 4 * batch["mask"] - 2
            target = (
                target.permute(0, 2, 3, 1)
                .reshape(-1, head.trunk[-1].out_channels)
                .double()
            )
            if name == "mask":
                valid = batch["valid"].flatten() > 0
                hidden, target = hidden[valid], target[valid]
            x = torch.cat((hidden, torch.ones_like(hidden[:, :1])), 1)
            grams[name] += x.T @ x
            rhs[name] += x.T @ target
            counts[name] += len(x)
    for name, head in model.heads.items():
        if not counts[name]:
            raise ValueError(f"No training pixels for {name}")
        penalty = 0.001 * torch.eye(33, dtype=torch.float64, device=device)
        penalty[-1, -1] = 0
        weights = torch.linalg.solve(
            grams[name] / counts[name] + penalty, rhs[name] / counts[name]
        )
        if not torch.isfinite(weights).all():
            raise ValueError("Nonfinite fitted readout")
        head.trunk[-1].weight.copy_(weights[:-1].T[:, :, None, None])
        head.trunk[-1].bias.copy_(weights[-1])
    return dict(
        method="supervised_ridge_final_convolutions",
        fitted_parameters=132,
        training_rows=data.rows.tolist(),
        training_pixels=counts,
        ridge=0.001,
    )


@torch.no_grad()
def direct_weights(model, data, settings, output, device, *, resume=False):
    method = settings["weight_method"]
    if method in ("constructed", "ridge"):
        construct_weights(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0)
    run = Run(
        output,
        settings=settings,
        data={k: v.identity for k, v in data.items()},
        recipe=__file__,
        model=model,
        optimizer=optimizer,
        device=device,
        resume=resume,
    )
    receipt_path = run.path / "direct_weights.json"
    try:
        if resume:
            receipt = json.loads(receipt_path.read_text())
            if not receipt.get("report_files"):
                raise ValueError("Cached direct-weight report is incomplete")
            for name, expected in {
                **receipt["files"],
                **receipt["report_files"],
            }.items():
                if file_hash(run.path / name) != expected:
                    raise ValueError(f"Cached direct-weight file changed: {name}")
            run.status("completed", "structural_verified")
            return
        deadline = time.monotonic() + 300
        initial = state_hash(model)
        fit = (
            fit_readouts(model, data["train"], device, deadline=deadline)
            if method == "ridge"
            else None
        )
        payload = dict(
            schema="pathwm-hierarchy-weights-v1",
            method=method,
            architecture={
                k: settings[k]
                for k in ("arm", "width", "levels", "stage_depth", "fusion_depth")
            },
            optimizer_updates=0,
            backward_calls=0,
            fit=fit,
            model={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        )
        torch.save(payload, run.path / "weights.pt")
        expected = state_hash(model)
        model.load_state_dict(
            torch.load(run.path / "weights.pt", map_location=device, weights_only=True)[
                "model"
            ],
            strict=True,
        )
        if state_hash(model) != expected:
            raise RuntimeError("Binary reload changed constructed state")
        run.log(
            dict(
                step=0,
                split="validation",
                **evaluate(model, data["validation"], objective, device),
            )
        )
        run.save()
        result = score(model, data["test"], run.path, device)
        if time.monotonic() > deadline:
            raise TimeoutError("Direct weight evaluation budget exceeded")
        receipt = dict(
            method=method,
            fit=fit,
            initial_sha256=initial,
            final_sha256=expected,
            optimizer_updates=0,
            backward_calls=0,
            files={
                name: file_hash(run.path / name)
                for name in (
                    "weights.pt",
                    "last.pt",
                    "test_outputs.npz",
                    "test_metrics.json",
                )
            },
            parameters=sum(p.numel() for p in model.parameters()),
            nonzero_parameters=sum(
                int(torch.count_nonzero(p)) for p in model.parameters()
            ),
        )
        atomic_json(receipt_path, receipt)
        run.status("completed", "pending")
    except BaseException as exc:
        run.status("failed", "pending", str(exc))
        raise
    try:
        model.eval()
        batch = data["validation"].batch(
            np.arange(min(6, len(data["validation"]))), device
        )
        write_report(run.path, batch, model(batch["rgb"]), model.encoder(batch["rgb"]))
        append_test_table(run.path, result)
        receipt["report_files"] = {
            name: file_hash(run.path / name)
            for name in ("report.html", "report.qa.json")
        }
        atomic_json(receipt_path, receipt)
    except BaseException as exc:
        run.status("completed", "failed", str(exc))
        raise


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
        evaluated_frames=len(data),
        scope=f"{len(data)} reused internal test images; foreground union, not entity identity",
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
        f"<section><h2>Final held-out screen</h2><p>{result['evaluated_frames']} reused internal test images. "
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
    parser.add_argument(
        "--weight-method",
        choices=("trained", "untrained", "constructed", "ridge"),
        default="trained",
    )
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
        if args.weight_method == "trained":
            result = check(model, data["train"], objective, args.device, batch_size=4)
        else:
            if args.weight_method in ("constructed", "ridge"):
                construct_weights(model)
            result = {"weight_method": args.weight_method, "backward_calls": 0}
        with torch.no_grad():
            outputs = model(data["validation"].batch(np.arange(16), args.device)["rgb"])
            if not all(torch.isfinite(v).all() for v in outputs.values()):
                raise ValueError("Nonfinite preflight outputs")
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
    started = time.monotonic()
    output = (
        args.resume
        or args.output
        or (
            Path(f"runs/hierarchy_fusion_v1/seed_{args.seed}/{args.arm}")
            if args.weight_method == "trained"
            else Path(f"runs/hierarchy_weights_v1/{args.weight_method}_{args.seed}")
        )
    )
    if args.weight_method == "trained":
        optimizer = torch.optim.AdamW(
            trainable_parameters(model), lr=0.0003, weight_decay=0.0001
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
    else:
        if args.stop_after is not None:
            parser.error("Direct weights do not have optimizer steps")
        settings.update(
            weight_method=args.weight_method,
            steps=0,
            purpose="direct numerical weights; reused real-image screen",
            optimizer_updates=0,
            backward_calls=0,
            max_seconds=300,
            learning_rate=0,
            weight_decay=0,
            readout_batch_size=16,
        )
        direct_weights(
            model, data, settings, output, args.device, resume=args.resume is not None
        )
    status = json.loads((output / "status.json").read_text())
    if status["result"] == "completed" and args.weight_method == "trained":
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
