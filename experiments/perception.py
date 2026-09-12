"""Copy this file to try an encoder, head or objective. Edit build_model/objective.

Run from the repository after: python -m pip install -e '.[dev]'
No work happens on import. --check uses a real batch; --stop-after pauses safely.
"""

import argparse
from pathlib import Path
import json
import torch
from torch.nn import functional as F
from pathwm.models.encoders import CNNEncoder, DinoEncoder, PyramidEncoder
from pathwm.models.decoders import ReconstructionDecoder, DenseHead, PushTPoseHead
from pathwm.models.perception import Perception
from pathwm.data.images import PushTFrames, CocoFrames, CocoMasks
from pathwm.io import (
    seed_everything,
    file_hash,
    trainable_parameters,
    resume_arguments,
    load_component,
)
from pathwm.training.perception import check, train_perception


# Loss computation is deliberately here, next to the experiment composition.
def objective(outputs, batch):
    losses = {"rgb_mse": F.mse_loss(outputs["rgb"], batch["rgb"])}
    if "pose" in outputs:
        losses["pose_mse"] = F.mse_loss(outputs["pose"], batch["pose"])
    if "mask" in outputs:
        valid = batch["valid"]
        counts = valid.sum((1, 2, 3))
        keep = counts > 0
        pixels = (
            F.binary_cross_entropy_with_logits(
                outputs["mask"], batch["mask"], reduction="none"
            )
            * valid
        )
        losses["mask_bce"] = (
            (pixels.sum((1, 2, 3))[keep] / counts[keep]).mean()
            if keep.any()
            else outputs["mask"].sum() * 0
        )
    return losses


def build_model(args):
    if args.encoder == "cnn":
        encoder = CNNEncoder(depth=args.depth)
        if args.encoder_weights:
            load_component(encoder, args.encoder_weights, "encoder")
        rgb = ReconstructionDecoder(encoder.feature_spec)
    elif args.encoder == "pyramid":
        encoder = PyramidEncoder(
            width=args.width,
            levels=args.levels,
            depth=args.stage_depth,
            fusion_depth=args.fusion_depth,
        )
        if args.encoder_weights:
            load_component(encoder, args.encoder_weights, "encoder")
        rgb = DenseHead(
            encoder.feature_spec,
            channels=3,
            levels=tuple(encoder.feature_spec),
            activation="sigmoid",
            retain_statistics=True,
        )
    else:
        encoder = DinoEncoder(
            args.dino_source, args.encoder_weights or "data/assets/dinov2/weights.pth"
        )
        rgb = DenseHead(
            encoder.feature_spec,
            channels=3,
            levels=("local", "fine", "coarse"),
            activation="sigmoid",
            retain_statistics=True,
        )
    encoder.requires_grad_(not args.freeze_encoder)
    heads = {"rgb": rgb}
    spatial_levels = (
        tuple(encoder.feature_spec) if args.encoder == "pyramid" else ("fine", "coarse")
    )
    if args.dataset == "pusht" and not args.rgb_only:
        heads["pose"] = PushTPoseHead(encoder.feature_spec, levels=spatial_levels)
    if args.dataset == "coco" and args.masks and not args.rgb_only:
        heads["mask"] = DenseHead(
            encoder.feature_spec,
            levels=("local", "fine", "coarse")
            if args.encoder == "dino"
            else spatial_levels,
            retain_statistics=args.encoder in ("dino", "pyramid"),
        )
    # Diagnostic RGB cannot update E; task heads can still train it.
    return Perception(
        encoder, heads, detached_heads=("rgb",) if args.diagnostic_rgb else ()
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default="runs/my_perception")
    p.add_argument(
        "--resume",
        type=Path,
        help="Existing run directory; restores its experiment settings",
    )
    p.add_argument("--check", action="store_true")
    p.add_argument(
        "--stop-after",
        type=int,
        help="Pause after this many additional updates; keep the full step budget",
    )
    p.add_argument("--dataset", choices=["pusht", "coco"], default="pusht")
    p.add_argument("--data-root", type=Path)
    p.add_argument(
        "--masks",
        type=Path,
        help="Prepared COCO mask directory, e.g. data/assets/coco_masks",
    )
    p.add_argument("--encoder", choices=["cnn", "dino", "pyramid"], default="cnn")
    p.add_argument("--width", type=int, default=32, help="Pyramid feature width")
    p.add_argument("--levels", type=int, default=3, help="Pyramid scale count")
    p.add_argument(
        "--stage-depth",
        type=int,
        default=1,
        help="Transformer blocks per pyramid scale",
    )
    p.add_argument(
        "--fusion-depth", type=int, default=0, help="Final all-scale transformer blocks"
    )
    p.add_argument(
        "--encoder-weights",
        type=Path,
        help="Explicit weights; CNN starts fresh when omitted",
    )
    p.add_argument(
        "--dino-source", type=Path, default=Path("data/assets/dinov2/source")
    )
    p.add_argument("--depth", type=int, default=0, help="Residual blocks per CNN grid")
    p.add_argument("--freeze-encoder", action="store_true")
    p.add_argument(
        "--diagnostic-rgb",
        action="store_true",
        help="Detach RGB-head inputs from encoder gradients",
    )
    p.add_argument("--rgb-only", action="store_true")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--train-frames", type=int, default=256)
    p.add_argument("--validation-frames", type=int, default=32)
    p.add_argument("--evaluate-every", type=int, default=20)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu")
    args = resume_arguments(p, p.parse_args())
    if args.encoder != "pyramid" and (
        args.width,
        args.levels,
        args.stage_depth,
        args.fusion_depth,
    ) != (32, 3, 1, 0):
        p.error(
            "Width, levels, stage-depth and fusion-depth options require --encoder pyramid"
        )
    if min(
        args.steps,
        args.batch_size,
        args.train_frames,
        args.validation_frames,
        args.evaluate_every,
    ) < 1 or (args.stop_after is not None and args.stop_after < 1):
        p.error("Step, batch, population and interval counts must be positive")
    seed_everything(args.seed)
    root = args.data_root or Path(
        "data/pusht64" if args.dataset == "pusht" else "data/coco64"
    )
    if args.dataset == "pusht":
        train, validation = PushTFrames(root, "train"), PushTFrames(root, "validation")
    elif args.masks:
        train, validation = (
            CocoMasks(root, args.masks, "train"),
            CocoMasks(root, args.masks, "validation"),
        )
    else:
        train, validation = CocoFrames(root, "train"), CocoFrames(root, "validation")
    train, validation = (
        train.take(args.train_frames),
        validation.take(args.validation_frames),
    )
    model = build_model(args).to(args.device)
    if args.check:
        print(json.dumps(check(model, train, objective, args.device), indent=2))
        return
    # Settings record every CLI scientific choice; runtime location/pause are separate.
    settings = {
        k: str(v) if isinstance(v, Path) else v
        for k, v in vars(args).items()
        if k not in ("output", "resume", "check", "stop_after")
    }
    settings.update(
        grad_clip=1.0,
        weight_decay=1e-4,
        precision="float32",
        sampling="with replacement, workers=0",
        purpose="development",
    )
    weight_path = args.encoder_weights or (
        Path("data/assets/dinov2/weights.pth") if args.encoder == "dino" else None
    )
    settings["initial_weights_sha256"] = file_hash(weight_path) if weight_path else None
    optimizer = torch.optim.AdamW(
        trainable_parameters(model), lr=args.learning_rate, weight_decay=1e-4
    )
    output = train_perception(
        model,
        train,
        validation,
        objective,
        optimizer,
        settings=settings,
        recipe=__file__,
        output=args.resume or args.output,
        device=args.device,
        resume=args.resume is not None,
        stop_after=args.stop_after,
    )
    print(f"Results: {output}/report.html")


if __name__ == "__main__":
    main()
