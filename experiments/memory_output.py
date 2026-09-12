"""Small selected-object history -> factual answer and state-produced RGB experiment."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F

from pathwm.data.memory_output import MemoryOutputEpisodes, image_labels, BACKGROUND
from pathwm.evaluation.report import write_report
from pathwm.io import Run, atomic_json, file_hash, seed_everything, state_hash
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
from pathwm.models.memory_output import MemoryOutput
from pathwm.models.perception import Perception


def make_codec(width=32, levels=3, depth=2, fusion_depth=2, weights=None):
    encoder = PyramidEncoder(
        width=width, levels=levels, depth=depth, fusion_depth=fusion_depth
    )
    options = dict(levels=tuple(encoder.feature_spec), retain_statistics=True)
    base = Perception(
        encoder,
        dict(
            rgb=DenseHead(
                encoder.feature_spec, channels=3, activation="sigmoid", **options
            ),
            mask=DenseHead(encoder.feature_spec, **options),
        ),
    )
    if weights is not None:
        base.load_state_dict(
            torch.load(weights, map_location="cpu", weights_only=True)["model"],
            strict=True,
        )
    return Perception(
        PatchDetailEncoder(base.encoder), dict(rgb=PatchDetailHead(base.heads["rgb"]))
    ).requires_grad_(False)


def build_model(codec, width=32):
    agent = MultimodalAgent(
        width=width,
        encoders={"image": codec.encoder.base.encoder},
        decoders={
            "image": StateFeatureDecoder(
                width, codec.encoder.feature_spec, codec.heads["rgb"]
            )
        },
        updater=ObservationUpdate(width),
        dynamics=LatentDynamics(width),
        thinker=Thinker(width),
        memory=EpisodicMemory(capacity=4, retrieve_count=2),
        action_head=ActionHead(width),
        monitor=ErrorMonitor(width),
    )
    return MemoryOutput(agent, codec.encoder)


def load_model(path, device="cpu"):
    record = torch.load(path, map_location="cpu", weights_only=True)
    settings = record["settings"]
    codec = make_codec(
        **{k: settings[k] for k in ("width", "levels", "depth", "fusion_depth")}
    )
    model = build_model(codec, settings["width"])
    model.load_state_dict(record["model"], strict=True)
    return model.to(device).eval()


def fact_labels(logits):
    return torch.stack([x.argmax(1) for x in logits.split((4, 2, 2), -1)], -1)


def factual_loss(logits, target):
    return (
        sum(
            F.cross_entropy(x, target[:, i])
            for i, x in enumerate(logits.split((4, 2, 2), -1))
        )
        / 3
    )


def weighted_error(image, target):
    weights = 1 + 9 * (target - BACKGROUND / 255).abs().amax(1, keepdim=True).gt(0.01)
    return ((image - target).square() * weights).sum((1, 2, 3)) / (
        3 * weights.sum((1, 2, 3))
    )


def objective(model, batch, reset):
    history = model.observe_history(batch["images"])
    state = model.query(
        history["final"], batch["images"][:, -1], "reset" if reset else "ordinary"
    )
    output = model.output(state)
    with torch.no_grad():
        targets = model.teacher(batch["target"])
    stored = model.facts(model.working(history["stored"]))
    facts = factual_loss(output["facts"], batch["labels"])
    write = factual_loss(stored, batch["labels"])
    pixels = weighted_error(output["image"], batch["target"]).mean()
    latent = model.agent.decoders["image"].latent_loss(output["features"], targets)
    loss = facts + 0.5 * write + pixels + 0.1 * latent
    return loss, dict(
        factual_loss=float(facts.detach()),
        write_loss=float(write.detach()),
        rgb_loss=float(pixels.detach()),
        latent_loss=float(latent.detach()),
    )


def default_settings(seed=7801):
    return dict(
        seed=seed,
        steps=1536,
        batch_size=16,
        wall_seconds=300,
        width=32,
        levels=3,
        depth=2,
        fusion_depth=2,
        objective="factor CE + 0.5 write CE + foreground-weighted RGB MSE + 0.1 standardized feature MSE; independent direct-reader CE",
        example_labels={
            "input": "Held-out targets (never query inputs)",
            "rgb": "Reset-recall image outputs",
        },
    )


@torch.no_grad()
def evaluate(model, data, device, modes, batch_size=16):
    arrays = {
        "target": [],
        "labels": [],
        "history": [],
        "direct_logits": [],
        "stored_logits": [],
        "teacher_image": [],
    }
    for mode in modes:
        arrays[mode + "_logits"], arrays[mode + "_image"] = [], []
    for start in range(0, len(data), batch_size):
        batch = data.batch(range(start, min(start + batch_size, len(data))), device)
        images = batch["images"]
        history = model.observe_history(images)
        for key, value in dict(
            target=batch["target"],
            labels=batch["labels"],
            history=images,
            direct_logits=model.direct(model.direct_tokens(images)),
            stored_logits=model.facts(model.working(history["stored"])),
            teacher_image=model.agent.decoders["image"].head(
                model.teacher(batch["target"])
            ),
        ).items():
            arrays[key].append(value.cpu())
        for mode in modes:
            if mode == "erased_history":
                output = model(images, mode)
            else:
                output = model.output(
                    model.query(history["final"], images[:, -1], mode)
                )
            arrays[mode + "_logits"].append(output["facts"].cpu())
            arrays[mode + "_image"].append(output["image"].cpu())
    tensors = {key: torch.cat(value) for key, value in arrays.items()}
    return score(tensors, modes), {k: v.numpy() for k, v in tensors.items()}


def score(arrays, modes):
    target, labels = arrays["target"], arrays["labels"]
    paired = torch.arange(len(labels)) ^ 1
    result = {}
    for name in ("direct", "stored"):
        result[name + "_joint_accuracy"] = float(
            (fact_labels(arrays[name + "_logits"]) == labels).all(1).float().mean()
        )
    result["teacher_mse"] = float((arrays["teacher_image"] - target).square().mean())
    result["teacher_image_accuracy"] = float(
        (image_labels(arrays["teacher_image"]) == labels).all(1).float().mean()
    )
    result["background_weighted_mse"] = float(
        weighted_error(torch.full_like(target, BACKGROUND / 255), target).mean()
    )
    result["pair_mean_weighted_mse"] = float(
        weighted_error((target + target[paired]) / 2, target).mean()
    )
    for mode in modes:
        pixels, facts = arrays[mode + "_image"], fact_labels(arrays[mode + "_logits"])
        visual = image_labels(pixels)
        fc, vc = (facts == labels).all(1), (visual == labels).all(1)
        result[mode] = dict(
            factual_accuracy=float(fc.float().mean()),
            image_accuracy=float(vc.float().mean()),
            factual_pair_accuracy=float(fc.reshape(-1, 2).all(1).float().mean()),
            image_pair_accuracy=float(vc.reshape(-1, 2).all(1).float().mean()),
            agreement=float((facts == visual).all(1).float().mean()),
            rgb_mse=float((pixels - target).square().mean()),
            weighted_mse=float(weighted_error(pixels, target).mean()),
            alternate_factual_accuracy=float(
                (facts == labels[paired]).all(1).float().mean()
            ),
            alternate_image_accuracy=float(
                (visual == labels[paired]).all(1).float().mean()
            ),
        )
    return result


def passes(metrics):
    for mode in ("ordinary", "reset"):
        m = metrics[mode]
        if not (
            m["factual_accuracy"] >= 0.9
            and m["image_accuracy"] >= 0.9
            and m["factual_pair_accuracy"] >= 0.8
            and m["image_pair_accuracy"] >= 0.8
        ):
            return False
        for key in ("factual_accuracy", "image_accuracy"):
            if m[key] - metrics["erased_history"][key] < 0.3:
                return False
        if m["weighted_mse"] >= min(
            metrics["background_weighted_mse"], metrics["pair_mean_weighted_mse"]
        ):
            return False
    return all(
        metrics["reset"][key] - metrics["reset_erased"][key] >= 0.3
        for key in ("factual_accuracy", "image_accuracy")
    ) and all(
        metrics["reset_swapped"][key] >= 0.8
        for key in ("alternate_factual_accuracy", "alternate_image_accuracy")
    )


def comparison_panel(path, arrays):
    from PIL import Image, ImageDraw

    columns = [
        "Selection",
        "Last seen",
        "Hidden",
        "Target",
        "Ordinary",
        "Recall",
        "Erase bank",
        "Swap bank",
    ]
    count, cell = min(8, len(arrays["target"])), 96
    canvas = Image.new("RGB", (len(columns) * cell, 26 + count * (cell + 18)), "white")
    draw = ImageDraw.Draw(canvas)
    for j, label in enumerate(columns):
        draw.text((j * cell + 3, 5), label, fill="black")
    for i in range(count):
        row = [
            *arrays["history"][i],
            arrays["target"][i],
            *(
                arrays[k + "_image"][i]
                for k in ("ordinary", "reset", "reset_erased", "reset_swapped")
            ),
        ]
        for j, value in enumerate(row):
            pixels = np.rint(np.clip(value.transpose(1, 2, 0), 0, 1) * 255).astype(
                "uint8"
            )
            canvas.paste(
                Image.fromarray(pixels).resize((cell, cell), Image.Resampling.NEAREST),
                (j * cell, 26 + i * (cell + 18)),
            )
        draw.text(
            (3, 26 + i * (cell + 18) + cell),
            f"Example {i + 1}; color/shape/side {arrays['labels'][i].tolist()}",
            fill="black",
        )
    canvas.save(path)


def train(
    model,
    training,
    validation,
    test,
    *,
    output,
    settings,
    device="cpu",
    resume=False,
    stop_after=None,
):
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=0.001,
        weight_decay=0.0001,
    )
    identity = dict(
        train=training.identity, validation=validation.identity, test=test.identity
    )
    run = Run(
        output,
        settings=settings,
        data=identity,
        recipe=__file__,
        model=model,
        optimizer=optimizer,
        device=device,
        resume=resume,
    )
    frozen = dict(
        encoder=state_hash(model.teacher),
        decoder=state_hash(model.agent.decoders["image"].head),
    )
    prior = (
        json.loads((run.path / "runtime.json").read_text())["training_seconds"]
        if resume
        else 0.0
    )
    start = perf_counter()
    try:
        end = (
            settings["steps"]
            if stop_after is None
            else min(settings["steps"], run.step + stop_after)
        )
        for step in range(run.step + 1, end + 1):
            if prior + perf_counter() - start >= settings["wall_seconds"]:
                break
            batch = training.batch(
                run.sample(len(training), settings["batch_size"]), device
            )
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = objective(model, batch, reset=bool(step % 2))
            direct = factual_loss(
                model.direct(model.direct_tokens(batch["images"])), batch["labels"]
            )
            if not torch.isfinite(loss + direct):
                raise ValueError("Nonfinite training loss")
            (loss + direct).backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, error_if_nonfinite=True
            )
            optimizer.step()
            run.step = step
            run.log(
                dict(
                    step=step,
                    split="train",
                    loss=float(loss.detach()),
                    direct_loss=float(direct.detach()),
                    **metrics,
                )
            )
            if step % 128 == 0 or step == end:
                scores, values = evaluate(
                    model, validation, device, ["ordinary", "reset"]
                )
                run.log(
                    dict(
                        step=step,
                        split="diagnostic_development",
                        nll=float(
                            factual_loss(
                                torch.from_numpy(values["reset_logits"]),
                                torch.from_numpy(values["labels"]),
                            )
                        ),
                        factual_accuracy=scores["reset"]["factual_accuracy"],
                        image_accuracy=scores["reset"]["image_accuracy"],
                    )
                )
                print(
                    f"step {step}: validation reset facts {scores['reset']['factual_accuracy']:.3f}, image {scores['reset']['image_accuracy']:.3f}, stored {scores['stored_joint_accuracy']:.3f}",
                    flush=True,
                )
                run.save()
                atomic_json(
                    run.path / "runtime.json",
                    dict(training_seconds=prior + perf_counter() - start),
                )
        run.save()
        elapsed = prior + perf_counter() - start
        atomic_json(run.path / "runtime.json", dict(training_seconds=elapsed))
        completed = run.step == settings["steps"]
        # No held-out test exposure during an interrupted training chunk.
        modes = [
            "ordinary",
            "ordinary_no_bank",
            "reset",
            "reset_erased",
            "reset_swapped",
            "erased_history",
        ]
        scores, arrays = evaluate(
            model, test if completed else validation, device, modes
        )
        unchanged = frozen == dict(
            encoder=state_hash(model.teacher),
            decoder=state_hash(model.agent.decoders["image"].head),
        )
        if not unchanged or not all(np.isfinite(v).all() for v in arrays.values()):
            raise RuntimeError("Frozen mutation or nonfinite evaluation output")
        atomic_json(
            run.path / "result.json",
            dict(
                completed=completed,
                step=run.step,
                gate=completed and passes(scores),
                evaluation_split="held-out combinations"
                if completed
                else "validation only; training incomplete",
                metrics=scores,
                frozen_unchanged=unchanged,
                frozen_hashes=frozen,
                limitations=[
                    "synthetic familiar factors, unseen combinations",
                    "fixed task; no general language",
                    "two supplied snapshots both retrieved; no search or learned write policy",
                    "supervised write readout differs from independent direct encoder diagnostic",
                ],
            ),
        )
        np.savez_compressed(run.path / "predictions.npz", **arrays)
        torch.save(
            dict(model=model.state_dict(), settings=settings), run.path / "weights.pt"
        )
        run.status("completed" if completed else "interrupted", "pending")
        comparison_panel(run.path / "comparison.png", arrays)
        write_report(
            run.path,
            {"rgb": torch.from_numpy(arrays["target"][:8])},
            {"rgb": torch.from_numpy(arrays["reset_image"][:8])},
        )
        run.status("completed" if completed else "interrupted", "structural-only")
        return model
    except BaseException as exc:
        status = json.loads((run.path / "status.json").read_text())
        result_done = status["result"] in ("completed", "interrupted")
        run.status(
            status["result"] if result_done else "failed",
            "failed" if result_done else "pending",
            f"{type(exc).__name__}: {exc}",
        )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=7801)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    seed_everything(args.seed)
    if torch.device(args.device).type == "cuda":
        free, total = torch.cuda.mem_get_info(args.device)
        limit = min(4 * 1024**3, free - 1024**3)
        if limit <= 0:
            raise RuntimeError("Insufficient GPU headroom")
        index = torch.device(args.device).index or 0
        torch.cuda.set_per_process_memory_fraction(limit / total, index)
        torch.cuda.reset_peak_memory_stats(args.device)
    codec = make_codec(weights=args.weights).to(args.device).eval()
    model = build_model(codec).to(args.device).eval()
    training = MemoryOutputEpisodes(128, seed=7701)
    validation = MemoryOutputEpisodes(32, seed=7702)
    test = MemoryOutputEpisodes(64, seed=7703, split="test")
    with torch.no_grad():
        # Eight unique training targets suffice; no validation/test target calibration.
        targets = training.batch(range(8), args.device)["target"]
        model.agent.decoders["image"].calibrate(model.teacher(targets))
    settings = default_settings(args.seed)
    settings["donor_sha256"] = file_hash(args.weights)
    if args.check:
        batch = MemoryOutputEpisodes(4, seed=17701).batch(range(4), args.device)
        loss, metrics = objective(model, batch, reset=True)
        loss.backward()
        args.output.mkdir(parents=True, exist_ok=False)
        atomic_json(
            args.output / "check.json", dict(loss=float(loss.detach()), **metrics)
        )
        print(json.dumps(metrics), flush=True)
        return
    train(
        model,
        training,
        validation,
        test,
        output=args.output,
        settings=settings,
        device=args.device,
        resume=args.resume,
        stop_after=args.stop_after,
    )
    atomic_json(
        args.output / "resources.json",
        dict(
            peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(args.device)
            if torch.device(args.device).type == "cuda"
            else 0
        ),
    )


if __name__ == "__main__":
    main()
