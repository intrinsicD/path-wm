"""Edit an action-conditioned world-model experiment with frozen visual features.

The preserved CNN initializes E. U and P start fresh. Short default training is
for experimentation, not an assertion that predictions or control are reliable.
"""

import argparse
from pathlib import Path
import json
import torch
from pathwm.models.encoders import CNNEncoder
from pathwm.models.decoders import ReconstructionDecoder
from pathwm.models.temporal import MemoryUpdater, Predictor, WorldModel
from pathwm.data.sequences import PushTSequences
from pathwm.io import (
    seed_everything,
    file_hash,
    trainable_parameters,
    resume_arguments,
    load_component,
)
from pathwm.training.dynamics import check, train_dynamics


def objective(states, targets, batch):
    # Equal mean across future steps and declared spatial levels, in one fixed E space.
    terms = [
        (features[k] - target[k]).square().mean()
        for (features, _), target in zip(states, targets)
        for k in target
    ]
    return {"latent_mse": torch.stack(terms).mean()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default="runs/my_dynamics")
    p.add_argument(
        "--resume",
        type=Path,
        help="Existing directory; restores its scientific settings",
    )
    p.add_argument("--check", action="store_true")
    p.add_argument("--stop-after", type=int)
    p.add_argument("--data-root", type=Path, default=Path("data/pusht64"))
    p.add_argument(
        "--encoder-weights", type=Path, default=Path("data/assets/cnn_reference.pt")
    )
    p.add_argument(
        "--decoder-weights", type=Path, default=Path("data/assets/rgb_reference.pt")
    )
    p.add_argument("--encoder-depth", type=int, default=2)
    p.add_argument("--predictor-depth", type=int, default=2)
    p.add_argument("--memory-width", type=int, default=128)
    p.add_argument("--history", type=int, default=2)
    p.add_argument("--horizon", type=int, default=3)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--train-windows", type=int, default=128)
    p.add_argument("--validation-windows", type=int, default=16)
    p.add_argument("--evaluate-every", type=int, default=20)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu")
    args = resume_arguments(p, p.parse_args())
    if min(
        args.steps,
        args.batch_size,
        args.history,
        args.horizon,
        args.train_windows,
        args.validation_windows,
        args.evaluate_every,
        args.predictor_depth,
        args.memory_width,
    ) < 1 or (args.stop_after is not None and args.stop_after < 1):
        p.error("Counts, widths and intervals must be positive")
    seed_everything(args.seed)
    encoder = CNNEncoder(depth=args.encoder_depth)
    load_component(encoder, args.encoder_weights, "encoder")
    encoder.requires_grad_(False).eval()
    updater = MemoryUpdater(
        encoder.feature_spec, action_width=2, memory_width=args.memory_width
    )
    predictor = Predictor(
        encoder.feature_spec,
        action_width=2,
        memory_width=args.memory_width,
        depth=args.predictor_depth,
    )
    model = WorldModel(encoder, updater, predictor).to(args.device)
    train = PushTSequences(
        args.data_root, "train", args.history, args.horizon, args.train_windows
    )
    validation = PushTSequences(
        args.data_root,
        "validation",
        args.history,
        args.horizon,
        args.validation_windows,
    )
    if args.check:
        print(json.dumps(check(model, train, objective, args.device), indent=2))
        return
    decoder = None
    if args.decoder_weights is not None:
        decoder = ReconstructionDecoder(encoder.feature_spec).to(args.device)
        load_component(decoder, args.decoder_weights, "heads.rgb")
        decoder.requires_grad_(False).eval()  # Optional diagnostic consumer.
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
        encoder_sha256=file_hash(args.encoder_weights),
        decoder_sha256=file_hash(args.decoder_weights)
        if args.decoder_weights
        else None,
    )
    optimizer = torch.optim.AdamW(
        trainable_parameters(model), lr=args.learning_rate, weight_decay=1e-4
    )
    output = train_dynamics(
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
        decoder=decoder,
    )
    print(f"Results: {output}/report.html")


if __name__ == "__main__":
    main()
