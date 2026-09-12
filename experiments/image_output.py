"""Image detail capacity and request-only state producer; two distinct screens.

The request fit uses four training patterns, never image observations. It is a
memorization/development check, not a text-to-image generalization benchmark.
"""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F

from pathwm.data.images import CocoMasks
from pathwm.evaluation.report import write_report
from pathwm.io import Run, atomic_json, digest, file_hash, seed_everything, state_hash
from pathwm.models.agent import (
    MultimodalAgent,
    ObservationUpdate,
    LatentDynamics,
    Thinker,
    ActionHead,
    ErrorMonitor,
)
from pathwm.models.agent_state import EpisodicMemory
from pathwm.models.decoders import DenseHead, PatchDetailHead, StateFeatureDecoder
from pathwm.models.encoders import PyramidEncoder, PatchDetailEncoder
from pathwm.models.modalities import TextEncoder, Observation, bytes_batch
from pathwm.models.perception import Perception


def base_model():
    encoder = PyramidEncoder(width=32, levels=3, depth=2, fusion_depth=2)
    options = dict(levels=tuple(encoder.feature_spec), retain_statistics=True)
    base = Perception(
        encoder,
        {
            "rgb": DenseHead(
                encoder.feature_spec, channels=3, activation="sigmoid", **options
            ),
            "mask": DenseHead(encoder.feature_spec, **options),
        },
    )
    return base


def load_codec(weights):
    base = base_model()
    payload = torch.load(weights, map_location="cpu", weights_only=True)
    base.load_state_dict(payload["model"], strict=True)
    if not all(torch.isfinite(v).all() for v in base.state_dict().values()):
        raise ValueError("Nonfinite donor checkpoint")
    return Perception(
        PatchDetailEncoder(base.encoder),
        {
            "rgb": PatchDetailHead(base.heads["rgb"]),
            "mask": base.heads["mask"],
        },
    ).requires_grad_(False)


def load_request_agent(checkpoint, device="cpu"):
    """Standalone inference: no donor file, target data or image encoder required."""
    base = base_model()
    spec = PatchDetailEncoder(base.encoder).feature_spec
    head = PatchDetailHead(base.heads["rgb"]).requires_grad_(False)
    agent = build_agent(StateFeatureDecoder(32, spec, head))
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    agent.load_state_dict(payload["model"], strict=True)
    return agent.to(device).eval()


def build_agent(decoder, width=32):
    return MultimodalAgent(
        width=width,
        encoders={"text": TextEncoder(width)},
        decoders={"image": decoder},
        updater=ObservationUpdate(width),
        dynamics=LatentDynamics(width),
        thinker=Thinker(width),
        memory=EpisodicMemory(),
        action_head=ActionHead(width),
        monitor=ErrorMonitor(width),
    )


def patterns(size=64):
    requests = [
        "horizontal light",
        "horizontal dark",
        "vertical light",
        "vertical dark",
    ]
    y, x = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    values = torch.stack([y % 2, 1 - y % 2, x % 2, 1 - x % 2]).float()
    return requests, (0.2 + 0.6 * values[:, None]).expand(-1, 3, -1, -1).clone()


def request_state(agent, requests):
    device = agent.initial.device
    ids, valid = bytes_batch(requests, device)
    observation = Observation(ids, torch.zeros_like(ids, dtype=torch.float64), valid)
    state = agent.observe(
        agent.initial_state(len(requests)), {"text": observation}, time=0
    )
    return agent.think(state)


def request_loss(agent, requests, pixels, targets):
    decoder = agent.decoders["image"]
    features = decoder.features(request_state(agent, requests).tokens)
    output = decoder.head(features)
    rgb = F.mse_loss(output, pixels)
    latent = decoder.latent_loss(features, targets)
    return rgb + 0.1 * latent, rgb, latent, output


def train_request(
    agent,
    requests,
    pixels,
    targets,
    *,
    output,
    settings,
    device="cpu",
    resume=False,
    stop_after=None,
    prior_seconds=0.0,
):
    optimizer = torch.optim.AdamW(
        [p for p in agent.parameters() if p.requires_grad],
        lr=0.001,
        weight_decay=0.0001,
    )
    identity = dict(
        requests=requests,
        pixels_sha256=digest(pixels.detach().cpu().tolist()),
        target_hash=digest({k: v.detach().cpu().tolist() for k, v in targets.items()}),
    )
    run = Run(
        output,
        settings=settings,
        data=identity,
        recipe=__file__,
        model=agent,
        optimizer=optimizer,
        device=device,
        resume=resume,
    )
    frozen_hash = state_hash(agent.decoders["image"].head)
    start = perf_counter()
    try:
        end = (
            settings["steps"]
            if stop_after is None
            else min(settings["steps"], run.step + stop_after)
        )
        if not run.rows:
            with torch.no_grad():
                loss, rgb, latent, _ = request_loss(agent, requests, pixels, targets)
            run.log(
                dict(
                    step=0,
                    split="fit",
                    loss=float(loss),
                    rgb_mse=float(rgb),
                    latent_mse=float(latent),
                )
            )
        for step in range(run.step + 1, end + 1):
            if prior_seconds + perf_counter() - start >= settings["wall_seconds"]:
                break
            ids = run.sample(len(requests), settings["batch_size"])
            optimizer.zero_grad(set_to_none=True)
            loss, rgb, latent, _ = request_loss(
                agent,
                [requests[i] for i in ids],
                pixels[ids],
                {k: v[ids] for k, v in targets.items()},
            )
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite request training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                agent.parameters(), 1.0, error_if_nonfinite=True
            )
            optimizer.step()
            run.step = step
            run.log(
                dict(
                    step=step,
                    split="train",
                    loss=float(loss.detach()),
                    rgb_mse=float(rgb.detach()),
                    latent_mse=float(latent.detach()),
                )
            )
            if step % 64 == 0 or step == end:
                with torch.no_grad():
                    loss, rgb, latent, _ = request_loss(
                        agent, requests, pixels, targets
                    )
                run.log(
                    dict(
                        step=step,
                        split="fit",
                        loss=float(loss),
                        rgb_mse=float(rgb),
                        latent_mse=float(latent),
                    )
                )
                print(f"step {step}: fit RGB MSE {float(rgb):.6f}", flush=True)
                run.save()
        run.save()
        completed = run.step == settings["steps"]
        seconds = prior_seconds + perf_counter() - start
        with torch.no_grad():
            _, rgb, latent, prediction = request_loss(agent, requests, pixels, targets)
            shuffled = agent.decode(
                request_state(agent, requests[1:] + requests[:1]), modalities=["image"]
            )["image"]
            erased = agent.decode(
                request_state(agent, [""] * len(requests)), modalities=["image"]
            )["image"]
            teacher = agent.decoders["image"].head(targets)
        metrics = dict(
            rgb_mse=float(rgb),
            latent_mse=float(latent),
            shuffled_mse=float(F.mse_loss(shuffled, pixels)),
            erased_mse=float(F.mse_loss(erased, pixels)),
            teacher_mse=float(F.mse_loss(teacher, pixels)),
        )
        frozen_equal = frozen_hash == state_hash(agent.decoders["image"].head)
        if not frozen_equal:
            raise RuntimeError("Frozen decoder changed")
        atomic_json(
            run.path / "result.json",
            dict(
                metrics=metrics,
                step=run.step,
                completed=completed,
                gate=completed
                and metrics["rgb_mse"] <= 0.01
                and all(
                    metrics["rgb_mse"] <= 0.5 * metrics[k]
                    for k in ("shuffled_mse", "erased_mse")
                ),
                frozen_decoder_unchanged=frozen_equal,
                image_observations=0,
                evaluation_scope="same four training requests; no generalization claim",
            ),
        )
        atomic_json(run.path / "runtime.json", dict(training_seconds=seconds))
        np.savez_compressed(
            run.path / "predictions.npz",
            target=pixels.cpu().numpy(),
            prediction=prediction.cpu().numpy(),
            shuffled=shuffled.cpu().numpy(),
            erased=erased.cpu().numpy(),
            teacher=teacher.cpu().numpy(),
        )
        run.status("completed" if completed else "paused", "pending")
        write_report(run.path, {"rgb": pixels}, {"rgb": prediction})
        return run
    except BaseException as exc:
        status = json.loads((run.path / "status.json").read_text())
        report_failed = status["result"] in ("completed", "paused")
        run.status(
            status["result"] if report_failed else "failed",
            "failed" if report_failed else "pending",
            f"{type(exc).__name__}: {exc}",
        )
        raise


@torch.no_grad()
def capacity(codec, data, output, settings, device):
    optimizer = torch.optim.SGD(codec.parameters(), lr=0.0)
    run = Run(
        output,
        settings=settings,
        data=data.identity,
        recipe=__file__,
        model=codec,
        optimizer=optimizer,
        device=device,
    )
    arrays = {k: [] for k in ("target", "base", "detail", "zero", "shuffled")}
    before = state_hash(codec)
    for start in range(0, len(data), 8):
        ids = np.arange(start, min(start + 8, len(data)))
        rgb = data.batch(ids, device)["rgb"]
        features = codec.encoder(rgb)
        other = codec.encoder(data.batch((ids + 1) % len(data), device)["rgb"])
        values = dict(
            target=rgb,
            base=codec.heads["rgb"].base(features),
            detail=codec.heads["rgb"](features),
            zero=codec.heads["rgb"](
                {**features, "detail": torch.zeros_like(features["detail"])}
            ),
            shuffled=codec.heads["rgb"]({**features, "detail": other["detail"]}),
        )
        for key, value in values.items():
            arrays[key].append(value.cpu().numpy())
    arrays = {k: np.concatenate(v) for k, v in arrays.items()}
    metrics = {
        k: float(np.square(v.astype("float64") - arrays["target"]).mean())
        for k, v in arrays.items()
        if k != "target"
    }
    unchanged = before == state_hash(codec)
    if not unchanged:
        raise RuntimeError("Capacity evaluation changed weights")
    run.log(dict(step=0, split="test", loss=metrics["detail"], **metrics))
    run.save()
    run.status("completed", "pending")
    np.savez_compressed(run.path / "predictions.npz", **arrays)
    atomic_json(
        run.path / "result.json",
        dict(
            metrics=metrics,
            gate=metrics["detail"] <= 0.75 * metrics["base"]
            and all(metrics[k] > metrics["detail"] for k in ("zero", "shuffled")),
            evaluated_images=len(data),
            weights_unchanged=unchanged,
            optimizer_updates=0,
            scope="high-bandwidth encoder transport control; reused internal test; not agent output",
        ),
    )
    write_report(
        run.path,
        {"rgb": torch.from_numpy(arrays["target"][:6])},
        {"rgb": torch.from_numpy(arrays["detail"][:6])},
    )
    return run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["capacity", "request"], required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=7601)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--data-root", type=Path, default=Path("data/coco64"))
    parser.add_argument("--masks", type=Path, default=Path("data/assets/coco_masks"))
    args = parser.parse_args()
    if args.mode == "capacity" and (args.resume or args.stop_after):
        parser.error("Capacity evaluation has no updates to resume")
    seed_everything(args.seed)
    torch.set_num_threads(2)
    if torch.device(args.device).type == "cuda":
        free, total = torch.cuda.mem_get_info(args.device)
        limit = min(4 * 1024**3, free - 1024**3)
        if limit <= 0:
            raise RuntimeError("Insufficient GPU headroom")
        index = torch.device(args.device).index
        torch.cuda.set_per_process_memory_fraction(
            limit / total, index if index is not None else torch.cuda.current_device()
        )
        torch.cuda.reset_peak_memory_stats(args.device)
    codec = load_codec(args.weights).to(args.device).eval()
    settings = dict(
        seed=args.seed,
        mode=args.mode,
        weights=str(args.weights.resolve()),
        weights_sha256=file_hash(args.weights),
        steps=512,
        batch_size=4,
        wall_seconds=300,
        objective="pixel MSE + 0.1 standardized latent MSE",
        example_labels={
            "input": "Training targets — never model inputs"
            if args.mode == "request"
            else "Input images",
            "rgb": "Request-only outputs (four training requests)"
            if args.mode == "request"
            else "Detail-transport reconstructions",
        },
    )
    if args.mode == "capacity":
        data = CocoMasks(args.data_root, args.masks, "test").take(128)
        if len(data) != 128:
            raise ValueError("Need all128 declared test images")
        run = capacity(codec, data, args.output, settings, args.device)
    else:
        requests, pixels = patterns()
        pixels = pixels.to(args.device)
        with torch.no_grad():
            targets = {k: v.detach() for k, v in codec.encoder(pixels).items()}
        decoder = StateFeatureDecoder(
            32, codec.encoder.feature_spec, codec.heads["rgb"]
        ).to(args.device)
        decoder.calibrate(targets)
        agent = build_agent(decoder).to(args.device)
        if args.check:
            loss, rgb, latent, prediction = request_loss(
                agent, requests, pixels, targets
            )
            loss.backward()
            result = dict(
                loss=float(loss.detach()),
                rgb_mse=float(rgb.detach()),
                latent_mse=float(latent.detach()),
                shape=list(prediction.shape),
                image_encoders=0,
                frozen_decoder_gradients=any(
                    p.grad is not None for p in decoder.head.parameters()
                ),
            )
            args.output.mkdir(parents=True, exist_ok=False)
            atomic_json(args.output / "check.json", result)
            print(json.dumps(result), flush=True)
            return
        runtime = args.output / "runtime.json"
        prior = (
            json.loads(runtime.read_text())["training_seconds"] if args.resume else 0
        )
        run = train_request(
            agent,
            requests,
            pixels,
            targets,
            output=args.output,
            settings=settings,
            device=args.device,
            resume=args.resume,
            stop_after=args.stop_after,
            prior_seconds=prior,
        )
    atomic_json(
        run.path / "resources.json",
        dict(
            peak_reserved_bytes=torch.cuda.max_memory_reserved(args.device)
            if torch.device(args.device).type == "cuda"
            else None
        ),
    )
    print((run.path / "result.json").read_text(), flush=True)


if __name__ == "__main__":
    main()
