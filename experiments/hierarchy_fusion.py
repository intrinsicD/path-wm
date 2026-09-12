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
    evaluation_mode,
)
from pathwm.models.blocks import SpatialResidual
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


def rgb_objective(outputs, batch):
    return {"rgb_mse": F.mse_loss(outputs["rgb"], batch["rgb"])}


def build_training_optimizer(model, *, decoder_learning_rate=None):
    if decoder_learning_rate is None:
        parameters = trainable_parameters(model)
    else:
        if not np.isfinite(decoder_learning_rate) or decoder_learning_rate <= 0:
            raise ValueError("Decoder learning rate must be finite and positive")
        parameters = [
            dict(params=trainable_parameters(module), lr=rate, name=name)
            for name, module, rate in [
                ("encoder", model.encoder, 0.0003),
                ("decoder", model.heads, decoder_learning_rate),
            ]
            if trainable_parameters(module)
        ]
    return torch.optim.AdamW(parameters, lr=0.0003, weight_decay=0.0001)


def configure_training(
    model,
    *,
    initial_weights=None,
    train_part="both",
    loss_mode="joint",
    open_residual_branches=False,
    seed=7501,
):
    """Strict initialization and explicit freeze rules, before optimizer creation."""
    if train_part not in ("both", "encoder", "decoder") or loss_mode not in (
        "joint",
        "rgb",
    ):
        raise ValueError("Unsupported train part or loss mode")
    source = None
    if initial_weights is not None:
        path = Path(initial_weights).resolve()
        payload = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(payload["model"], strict=True)
        source = dict(
            path=str(path),
            sha256=file_hash(path),
            method=payload.get("method", payload.get("schema")),
        )
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError("Nonfinite initialization")
    loaded_hash = state_hash(model)
    model.requires_grad_(True)
    if train_part == "decoder":
        model.encoder.requires_grad_(False)
    if train_part == "encoder":
        model.heads.requires_grad_(False)
    if loss_mode == "rgb":
        model.heads["mask"].requires_grad_(False)
    opened = []
    if open_residual_branches:
        if (
            source is None
            or source["method"] != "constructed"
            or train_part == "encoder"
        ):
            raise ValueError(
                "Opening requires a handwritten file and a trainable decoder"
            )
        generator = torch.Generator().manual_seed(seed + 200000)
        with torch.no_grad():
            for name, module in model.heads["rgb"].named_modules():
                if isinstance(module, SpatialResidual):
                    first, last = module.net[2], module.net[5]
                    if any(
                        torch.count_nonzero(p)
                        for layer in (first, last)
                        for p in layer.parameters()
                    ):
                        raise ValueError("Expected an exactly zero dormant branch")
                    first.weight.copy_(
                        0.001
                        * torch.randn(first.weight.shape, generator=generator).to(
                            first.weight
                        )
                    )
                    opened.append("heads.rgb." + name)
        if not opened:
            raise ValueError("No dormant RGB decoder branch found")
    modules = dict(
        encoder=model.encoder, rgb=model.heads["rgb"], mask=model.heads["mask"]
    )
    return dict(
        source=source,
        loaded_model_sha256=loaded_hash,
        initial_model_sha256=state_hash(model),
        opened_branches=opened,
        module_hashes={k: state_hash(m) for k, m in modules.items()},
        frozen_module_hashes={
            k: state_hash(m)
            for k, m in modules.items()
            if not any(p.requires_grad for p in m.parameters())
        },
    )


@torch.no_grad()
def information_probe(model):
    """Known equal-mean ambiguity, independent of real train/evaluation images."""
    patch = model.encoder.encoder.stem.patch
    coordinates = torch.arange(64, device=patch.weight.device)
    checker = ((coordinates[:, None] + coordinates[None, :]) % 2).to(patch.weight.dtype)
    images = patch.weight.new_full((2, 3, 64, 64), 0.5)
    images[0, 0], images[1, 0] = checker, 1 - checker
    with evaluation_mode(model):
        features = model.encoder(images)
        rgb = model.heads["rgb"](features)
    singular = torch.linalg.svdvals(patch.weight.flatten(1).detach().double().cpu())
    threshold = float(singular.max()) * 1e-8
    return dict(
        input_pair_mse=float((images[0].double() - images[1].double()).square().mean()),
        scale_pair_max_difference={
            k: float((v[0] - v[1]).abs().max()) for k, v in features.items()
        },
        rgb_pair_max_difference=float((rgb[0] - rgb[1]).abs().max()),
        patch_projection_rank=int((singular > threshold).sum()),
        singular_values=singular.tolist(),
        rank_relative_tolerance=1e-8,
        rgb_residual_weight_norms={
            name: float(p.detach().double().norm())
            for name, p in model.heads["rgb"].trunk[1].named_parameters()
            if name in ("net.2.weight", "net.5.weight")
        },
    )


def gradient_probe(model, batch, loss):
    """One non-updating training-data backward; clear gradients and restore RNG/mode."""
    with evaluation_mode(model):
        model.zero_grad(set_to_none=True)
        sum(loss(model(batch["rgb"]), batch).values()).backward()
        modules = dict(
            encoder=model.encoder, rgb=model.heads["rgb"], mask=model.heads["mask"]
        )
        result = {
            k: dict(
                trainable_parameters=sum(
                    p.numel() for p in m.parameters() if p.requires_grad
                ),
                gradient_l2=sum(
                    float(p.grad.double().square().sum())
                    for p in m.parameters()
                    if p.grad is not None
                )
                ** 0.5,
            )
            for k, m in modules.items()
        }
        result["rgb_residual_weight_gradients"] = {
            name: float(p.grad.double().norm()) if p.grad is not None else None
            for name, p in model.heads["rgb"].trunk[1].named_parameters()
            if name in ("net.2.weight", "net.5.weight")
        }
        model.zero_grad(set_to_none=True)
    return result


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
            "patch_mean_rgb": F.interpolate(
                F.avg_pool2d(batch["rgb"], 4), scale_factor=4, mode="nearest"
            ),
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
        patch_mean_rgb_mse=float(
            np.square(
                arrays["patch_mean_rgb"].astype(np.float64) - arrays["target_rgb"]
            ).mean()
        ),
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
    parser.add_argument("--initial-weights", type=Path)
    parser.add_argument(
        "--train-part", choices=("both", "encoder", "decoder"), default="both"
    )
    parser.add_argument("--loss-mode", choices=("joint", "rgb"), default="joint")
    parser.add_argument("--open-residual-branches", action="store_true")
    parser.add_argument("--decoder-learning-rate", type=float)
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
    adaptation = (
        args.initial_weights is not None
        or args.train_part != "both"
        or args.loss_mode != "joint"
        or args.open_residual_branches
        or args.decoder_learning_rate is not None
    )
    if adaptation and args.weight_method != "trained":
        parser.error("Initialization/freeze settings are for optimizer training")
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
    setup = (
        configure_training(
            model,
            initial_weights=args.initial_weights,
            train_part=args.train_part,
            loss_mode=args.loss_mode,
            open_residual_branches=args.open_residual_branches,
            seed=args.seed,
        )
        if adaptation
        else None
    )
    loss = rgb_objective if args.loss_mode == "rgb" else objective
    model.to(args.device)
    if args.check:
        if args.weight_method == "trained":
            result = check(model, data["train"], loss, args.device, batch_size=4)
            if adaptation:
                result["information"] = information_probe(model)
                result["gradients"] = gradient_probe(
                    model, data["train"].batch(np.arange(4), args.device), loss
                )
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
    if adaptation:
        settings.update(
            initial_weights=str(args.initial_weights) if args.initial_weights else None,
            train_part=args.train_part,
            loss_mode=args.loss_mode,
            open_residual_branches=args.open_residual_branches,
            training_setup=setup,
            purpose="handwritten initialization and reconstruction diagnosis",
        )
        if args.decoder_learning_rate is not None:
            settings["decoder_learning_rate"] = args.decoder_learning_rate
    started = time.monotonic()
    output = (
        args.resume
        or args.output
        or (
            (
                Path(
                    f"runs/hierarchy_training_v1/seed_{args.seed}/{'hand' if args.initial_weights else 'ordinary'}_{args.train_part}{'_open' if args.open_residual_branches else ''}"
                )
                if adaptation
                else Path(f"runs/hierarchy_fusion_v1/seed_{args.seed}/{args.arm}")
            )
            if args.weight_method == "trained"
            else Path(f"runs/hierarchy_weights_v1/{args.weight_method}_{args.seed}")
        )
    )
    if args.weight_method == "trained":
        before = None
        if adaptation:
            if args.resume and (output / "adaptation.json").exists():
                before = json.loads((output / "adaptation.json").read_text())["before"]
            else:
                before = dict(
                    information=information_probe(model),
                    gradients=gradient_probe(
                        model, data["train"].batch(np.arange(4), args.device), loss
                    ),
                )
        optimizer = build_training_optimizer(
            model, decoder_learning_rate=args.decoder_learning_rate
        )
        train_perception(
            model,
            data["train"],
            data["validation"],
            loss,
            optimizer,
            settings=settings,
            recipe=__file__,
            output=output,
            device=args.device,
            resume=args.resume is not None,
            stop_after=args.stop_after,
        )
        if adaptation:
            modules = dict(
                encoder=model.encoder, rgb=model.heads["rgb"], mask=model.heads["mask"]
            )
            hashes = {k: state_hash(m) for k, m in modules.items()}
            if any(hashes[k] != v for k, v in setup["frozen_module_hashes"].items()):
                atomic_json(
                    output / "status.json",
                    dict(
                        result="failed",
                        report="pending",
                        error="Frozen component changed",
                    ),
                )
                raise RuntimeError("Frozen component changed")
            atomic_json(
                output / "adaptation.json",
                dict(
                    initialization=setup,
                    before=before,
                    after=information_probe(model),
                    final_module_hashes=hashes,
                    changed_modules={
                        k: v != setup["module_hashes"][k] for k, v in hashes.items()
                    },
                    frozen_modules_verified=True,
                ),
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
        if adaptation:
            torch.save(
                dict(
                    schema="pathwm-hierarchy-weights-v1",
                    method="trained_from_handwritten"
                    if args.initial_weights
                    else "trained_ordinary",
                    loss_mode=args.loss_mode,
                    train_part=args.train_part,
                    initialization=setup,
                    optimizer_updates=status["step"],
                    model={
                        k: v.detach().cpu().clone()
                        for k, v in model.state_dict().items()
                    },
                ),
                output / "weights.pt",
            )
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
