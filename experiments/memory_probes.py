"""Cache frozen agent stages, then fit independent diagnostic token readers."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from pathwm.data.memory_output import MemoryOutputEpisodes
from pathwm.evaluation.report import write_report
from pathwm.io import (
    Run,
    atomic_json,
    environment,
    file_hash,
    seed_everything,
    source_record,
    state_hash,
)
from pathwm.models.memory_output import TokenProbe, load_model


@torch.no_grad()
def probe_tokens(model, images):
    history = model.observe_history(images)
    recall = model.query(history["final"], images[:, -1], "reset")
    values = dict(
        encoder=model.direct_tokens(images),
        initial_working=model.working(history["initial"]),
        stored_working=model.working(history["stored"]),
        stored_all=history["stored"].tokens,
        recall_working=model.working(recall),
        recall_all=recall.tokens,
    )
    native = dict(
        encoder=model.direct(values["encoder"]),
        stored=model.facts(values["stored_working"]),
        recall=model.facts(values["recall_working"]),
    )
    return {k: v.detach().clone() for k, v in values.items()}, native


@torch.no_grad()
def cache_states(model, data, device):
    before = state_hash(model)
    chunks, native = {}, {}
    for start in range(0, len(data), 16):
        images = data.batch(range(start, min(start + 16, len(data))), device)["images"]
        values, heads = probe_tokens(model, images)
        for target, source in ((chunks, values), (native, heads)):
            for key, value in source.items():
                target.setdefault(key, []).append(value.cpu())
    if state_hash(model) != before:
        raise RuntimeError("Frozen agent changed while caching")
    return dict(
        features={k: torch.cat(v) for k, v in chunks.items()},
        native={k: torch.cat(v) for k, v in native.items()},
        labels=torch.from_numpy(data.labels.copy()),
        history=torch.from_numpy(data.images.copy()),
    )


def prepare(checkpoint, directory, device):
    directory.mkdir(parents=True, exist_ok=False)
    start = perf_counter()
    model = load_model(checkpoint, device).requires_grad_(False).eval()
    before = state_hash(model)
    populations, hashes = {}, {}
    for split, pairs, seed in (
        ("train", 128, 7701),
        ("validation", 32, 7712),
        ("test", 64, 7713),
    ):
        data = MemoryOutputEpisodes(
            pairs,
            seed=seed,
            curriculum="relocation",
            split="test" if split == "test" else "train",
        )
        values = cache_states(model, data, device)
        torch.save(values, directory / f"{split}.pt")
        hashes[split] = file_hash(directory / f"{split}.pt")
        populations[split] = data.identity
    atomic_json(
        directory / "cache.json",
        dict(
            checkpoint=str(checkpoint.resolve()),
            checkpoint_sha256=file_hash(checkpoint),
            frozen_model_sha256=before,
            frozen_unchanged=before == state_hash(model),
            populations=populations,
            hashes=hashes,
            source=source_record(__file__, model),
            environment=environment(device),
            preparation_seconds=perf_counter() - start,
        ),
    )


def make_probes(training, seed):
    result = nn.ModuleDict()
    for name, values in training.items():
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            result[name] = TokenProbe(values)
    return result


def load_probes(path, device="cpu"):
    record = torch.load(path, map_location="cpu", weights_only=True)
    samples = {k: torch.zeros(1, 1, record["width"]) for k in record["stages"]}
    probes = make_probes(samples, seed=0)
    probes.load_state_dict(record["model"], strict=True)
    return probes.to(device).eval()


def labels_from_logits(logits):
    return torch.stack([x.argmax(1) for x in logits.split((4, 2, 2), -1)], -1)


def loss_for(logits, labels):
    return (
        sum(
            F.cross_entropy(x, labels[:, i])
            for i, x in enumerate(logits.split((4, 2, 2), -1))
        )
        / 3
    )


def probe_score(logits, labels):
    correct = labels_from_logits(logits) == labels
    joint = correct.all(1)
    return dict(
        joint_accuracy=float(joint.float().mean()),
        color_accuracy=float(correct[:, 0].float().mean()),
        shape_accuracy=float(correct[:, 1].float().mean()),
        side_accuracy=float(correct[:, 2].float().mean()),
        side_nll=float(F.cross_entropy(logits[:, 6:], labels[:, 2])),
        side_relocation_pair_accuracy=float(
            correct[:, 2].reshape(-1, 2, 2).all(1).float().mean()
        ),
        joint_relocation_pair_accuracy=float(
            joint.reshape(-1, 2, 2).all(1).float().mean()
        ),
        joint_selection_pair_accuracy=float(joint.reshape(-1, 2).all(1).float().mean()),
    )


@torch.no_grad()
def evaluate(probes, cache, device):
    arrays = {"labels": cache["labels"].cpu().numpy()}
    metrics = {}
    for name, probe in probes.items():
        features = cache["features"][
            "recall_working" if name == "random_labels" else name
        ]
        outputs = [
            probe(features[i : i + 16].to(device)).cpu()
            for i in range(0, len(features), 16)
        ]
        logits = torch.cat(outputs)
        arrays[name] = logits.numpy()
        metrics[name] = probe_score(logits, cache["labels"].cpu())
    for name, logits in cache["native"].items():
        arrays["native_" + name] = logits.cpu().numpy()
        metrics["native_" + name] = probe_score(logits.cpu(), cache["labels"].cpu())
    return metrics, arrays


def default_settings(seed=7901):
    return dict(
        seed=seed,
        steps=1536,
        batch_size=16,
        wall_seconds=120,
        purpose="frozen-state diagnostic",
        objective="independent mean factor CE; fixed training-only channel statistics",
        random_label_seed=17901,
    )


def train_probes(
    caches, *, output, settings, identity, device="cpu", resume=False, stop_after=None
):
    training = dict(caches["train"]["features"])
    training["random_labels"] = training["recall_working"]
    probes = make_probes(training, settings["seed"]).to(device)
    labels = caches["train"]["labels"].to(device)
    permutation = torch.randperm(
        len(labels),
        generator=torch.Generator().manual_seed(settings["random_label_seed"]),
    )
    randomized = labels[permutation.to(device)]
    optimizer = torch.optim.AdamW(probes.parameters(), lr=0.001, weight_decay=0.0001)
    run = Run(
        output,
        settings=settings,
        data=identity,
        recipe=__file__,
        model=probes,
        optimizer=optimizer,
        device=device,
        resume=resume,
    )
    calibration = {k: v.clone() for k, v in probes.named_buffers()}
    torch.save(permutation, run.path / "random_label_permutation.pt")
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
            ids = run.sample(len(labels), settings["batch_size"])
            optimizer.zero_grad(set_to_none=True)
            losses = {
                name: loss_for(
                    probe(training[name][ids].to(device)),
                    (randomized if name == "random_labels" else labels)[ids],
                )
                for name, probe in probes.items()
            }
            total = sum(losses.values())
            if not torch.isfinite(total):
                raise ValueError("Nonfinite probe loss")
            total.backward()
            for probe in probes.values():
                torch.nn.utils.clip_grad_norm_(
                    probe.parameters(), 1.0, error_if_nonfinite=True
                )
            optimizer.step()
            run.step = step
            run.log(
                dict(
                    step=step,
                    split="train",
                    loss=float(total.detach()) / len(probes),
                    **{k: float(v.detach()) for k, v in losses.items()},
                )
            )
            if step % 256 == 0 or step == end:
                scores, _ = evaluate(probes, caches["validation"], device)
                run.log(
                    dict(
                        step=step,
                        split="diagnostic_development",
                        nll=scores["recall_working"]["side_nll"],
                    )
                )
                print(
                    f"step {step}: validation side "
                    + str(
                        {
                            k: round(v["side_accuracy"], 3)
                            for k, v in scores.items()
                            if not k.startswith("native_")
                        }
                    ),
                    flush=True,
                )
                run.save()
                atomic_json(
                    run.path / "runtime.json",
                    dict(training_seconds=prior + perf_counter() - start),
                )
        run.save()
        atomic_json(
            run.path / "runtime.json",
            dict(training_seconds=prior + perf_counter() - start),
        )
        completed = run.step == settings["steps"]
        target = caches["test"] if completed else caches["validation"]
        metrics, arrays = evaluate(probes, target, device)
        training_metrics, training_arrays = evaluate(probes, caches["train"], device)
        # Report the selectivity control's fit to its randomized TRAIN labels too.
        training_metrics["random_labels_fit"] = probe_score(
            torch.from_numpy(training_arrays["random_labels"]), randomized.cpu()
        )
        if not all(torch.equal(v, calibration[k]) for k, v in probes.named_buffers()):
            raise RuntimeError("Probe calibration changed")
        if not all(np.isfinite(v).all() for v in arrays.values()):
            raise RuntimeError("Nonfinite probe output")
        controls = (
            metrics["encoder"]["joint_accuracy"] >= 0.9
            and metrics["native_encoder"]["joint_accuracy"] >= 0.9
            and metrics["initial_working"]["side_accuracy"] == 0.5
            and metrics["random_labels"]["side_accuracy"] <= 0.6
        )
        screens = {
            k: v["joint_accuracy"] >= 0.9
            and v["side_accuracy"] >= 0.9
            and v["side_relocation_pair_accuracy"] >= 0.8
            for k, v in metrics.items()
            if k not in ("initial_working", "random_labels")
            and not k.startswith("native_")
        }
        atomic_json(
            run.path / "result.json",
            dict(
                completed=completed,
                step=run.step,
                evaluation_split="fresh-background histories with familiar tuples/motion"
                if completed
                else "validation only; interrupted",
                metrics=metrics,
                training_metrics=training_metrics,
                controls_passed=completed and controls,
                stage_screens=screens,
                frozen_calibration_unchanged=True,
                limitations=[
                    "probe accessibility only; no agent repair",
                    "failure does not prove absent information",
                    "same capacity/steps do not imply equal extraction difficulty",
                    "different token counts at different interfaces",
                    "formal tensors cached on GPU; source CPU/GPU numerical limit retained",
                ],
            ),
        )
        np.savez_compressed(run.path / "predictions.npz", **arrays)
        np.savez_compressed(run.path / "training_predictions.npz", **training_arrays)
        torch.save(
            dict(
                model=probes.state_dict(),
                stages=list(probes),
                width=labels.new_tensor(training["encoder"].shape[-1]).item(),
                settings=settings,
            ),
            run.path / "weights.pt",
        )
        run.status("completed" if completed else "interrupted", "pending")
        write_report(run.path)
        run.status("completed" if completed else "interrupted", "structural-only")
        return probes
    except BaseException as exc:
        status = json.loads((run.path / "status.json").read_text())
        done = status["result"] in ("completed", "interrupted")
        run.status(
            status["result"] if done else "failed",
            "failed" if done else "pending",
            f"{type(exc).__name__}: {exc}",
        )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=7901)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after", type=int)
    args = parser.parse_args()
    seed_everything(args.seed)
    if torch.device(args.device).type == "cuda":
        free, total = torch.cuda.mem_get_info(args.device)
        limit = min(4 * 1024**3, free - 1024**3)
        if limit <= 0:
            raise RuntimeError("Insufficient GPU headroom")
        torch.cuda.set_per_process_memory_fraction(
            limit / total, torch.device(args.device).index or 0
        )
        torch.cuda.reset_peak_memory_stats(args.device)
    if args.prepare:
        if args.checkpoint is None:
            parser.error("--prepare needs --checkpoint")
        prepare(args.checkpoint, args.cache, args.device)
        return
    if args.output is None:
        parser.error("Probe training needs --output")
    metadata = json.loads((args.cache / "cache.json").read_text())
    if not metadata["frozen_unchanged"]:
        raise ValueError("Cache source changed")
    caches = {}
    for split, expected in metadata["hashes"].items():
        path = args.cache / f"{split}.pt"
        if file_hash(path) != expected:
            raise ValueError("Cache hash mismatch")
        caches[split] = torch.load(path, map_location="cpu", weights_only=True)
    train_probes(
        caches,
        output=args.output,
        settings=default_settings(args.seed),
        identity=metadata,
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
