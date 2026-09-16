"""Paired direction diagnosis on real-image-derived controlled pans.

python -m experiments.video_order --output runs/my_order --mode train
Use existing Run/report infrastructure; no natural-motion or agent capability claim.
"""

import argparse
from collections import Counter
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from pathwm.data.video_order import pan_pairs, cyclic_pan_pairs
from pathwm.models.spatial_vae import SpatialVAE
from pathwm.models.video_vae import CausalLatentMixer, local_correlation
from pathwm.io import Run, atomic_json, file_hash, seed_everything, evaluation_mode
from pathwm.evaluation.report import write_report

SOURCES = dict(
    train=["0EJAG", "0GFE8", "0JQ26", "0LDP7"],
    validation=["12VVC"],
    evaluation=["1KKYX"],
)
MODES = ("frozen", "train", "current", "previous", "correlation", "correlation_only")


def input_control(x, mode):
    if mode == "current":
        return x[:, -1:].expand_as(x)
    if mode == "previous":
        return x[:, -2:-1].expand_as(x)
    if mode == "unordered":
        return x.mean(1, keepdim=True).expand_as(x)
    if mode == "swap":
        return x[:, [1, 0, 2]]
    if mode != "normal":
        raise ValueError("Unknown input control")
    return x


class OrderReadout(nn.Module):
    """Diagnostic consumer of spatial AND separate temporal features.

    The image decoder continues to consume the untouched spatial code. All modes
    reserve the same five correlation channels, zero when disabled.
    """

    def __init__(self, channels, mode):
        super().__init__()
        if mode not in MODES:
            raise ValueError("Unknown direction arm")
        self.mode = mode
        self.temporal = CausalLatentMixer(channels)
        self.head = nn.Sequential(
            nn.Conv2d(2 * channels + 5, 16, 1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(16, 2),
        )
        if mode in ("frozen", "correlation_only"):
            self.temporal.requires_grad_(False)

    def forward(self, x, *, trace=None):
        if self.mode in ("current", "previous"):
            x = input_control(x, self.mode)
        times = (
            torch.arange(x.shape[1], dtype=torch.float64, device=x.device)[None].expand(
                len(x), -1
            )
            / 4
        )
        valid = torch.ones_like(times, dtype=torch.bool)
        temporal = self.temporal.features(x, times, valid)[:, -1]
        appearance = x[:, -1]
        correlation = local_correlation(x[:, -2], x[:, -1])
        learned = torch.cat((appearance, temporal), 1)[..., 2:-2, 2:-2]
        if self.mode == "correlation_only":
            learned = torch.zeros_like(learned)
        if self.mode not in ("correlation", "correlation_only"):
            correlation = torch.zeros_like(correlation)
        if trace is not None:
            for name, value in dict(
                appearance=appearance, temporal=temporal, correlation=correlation
            ).items():
                trace[name] = value.detach().cpu().clone()
        return self.head(torch.cat((learned, correlation), 1))


def tensor_hash(x):
    return hashlib.sha256(x.contiguous().numpy().tobytes()).hexdigest()


@torch.no_grad()
def pixel_oracle(frames):
    x = frames.flatten(0, 1)
    previous, current = x[:, -2], x[:, -1]
    shifts = [-8, -6, -4, -2, 2, 4, 6, 8]
    errors = torch.stack(
        [
            (
                (previous[..., 8:-8, 8 + s : 48 - 8 + s] - current[..., 8:-8, 8:-8])
                ** 2
            ).mean((1, 2, 3))
            for s in shifts
        ],
        1,
    )
    best = errors.argmin(1)
    pred = torch.tensor(shifts)[best] < 0
    tied = errors <= errors.min(1, keepdim=True).values + 1e-10
    ambiguous = tied[:, :4].any(1) & tied[:, 4:].any(1)
    return pred.long().reshape(frames.shape[:2]), ambiguous.reshape(frames.shape[:2])


@torch.no_grad()
def prepare(args, *, balanced=False, evaluation_only=False):
    seed_everything(7500)
    image = SpatialVAE.load(args.source).requires_grad_(False).eval()
    sets, identity = {}, {}
    for split, names in SOURCES.items():
        if evaluation_only and split != "evaluation":
            continue
        images, records = [], []
        for name in names:
            path = args.data / name / "video.mp4"
            command = [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                str(path),
                "-t",
                "2",
                "-an",
                "-vf",
                "fps=2,scale=80:64",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "pipe:1",
            ]
            raw = subprocess.run(command, check=True, capture_output=True).stdout
            decoded = np.frombuffer(raw, dtype=np.uint8).copy().reshape(-1, 64, 80, 3)
            if len(decoded) != 4:
                raise ValueError("Expected four decoded source frames")
            images.append(torch.from_numpy(decoded).permute(0, 3, 1, 2).float() / 255)
            records.append(
                dict(
                    path=str(path.resolve()),
                    sha256=file_hash(path),
                    decoded_sha256=hashlib.sha256(raw).hexdigest(),
                    command=command,
                )
            )
        images = torch.cat(images)
        populations = [(split if split != "evaluation" else "known", (2, 4))]
        if split == "evaluation":
            populations.append(("wide", (6, 8)))
        for name, shifts in populations:
            if balanced:
                d = cyclic_pan_pairs(images[..., 4:52, 12:60], shifts=shifts)
                bank = torch.cat([image.encode(v).mu for v in d["views"].split(32)])
                mu = bank[d["indices"]]
            else:
                d = pan_pairs(images, shifts=shifts)
                flat = d["frames"].flatten(0, 2)
                mu = torch.cat([image.encode(v).mu for v in flat.split(32)])
                mu = mu.reshape(*d["frames"].shape[:3], *mu.shape[1:])
            assert torch.equal(mu[:, 0, -1], mu[:, 1, -1])
            assert torch.equal(mu[:, 0, :2], mu[:, 1, :2].flip(1))
            oracle, ambiguous = pixel_oracle(d["frames"])
            balance_checks = 0
            if balanced:
                for i in d["metadata"][:, 0].unique():
                    for displacement in shifts:
                        chosen = (d["metadata"][:, 0] == i) & (
                            d["metadata"][:, -1] == displacement
                        )
                        assert int(chosen.sum()) == 48
                        for values in (d["frames"][chosen], mu[chosen]):
                            flat = values.flatten(0, 1)
                            labels = d["labels"][chosen].flatten()
                            for t in range(3):
                                a = Counter(
                                    tensor_hash(v) for v in flat[labels == 0, t]
                                )
                                b = Counter(
                                    tensor_hash(v) for v in flat[labels == 1, t]
                                )
                                assert a == b
                                balance_checks += 1
            sets[name] = dict(
                features=mu,
                labels=d["labels"],
                metadata=d["metadata"],
                examples=d["frames"][:2].clone(),
                oracle=oracle,
                ambiguous=ambiguous,
            )
            identity[name] = dict(
                sources=records,
                pairs=len(mu),
                shifts=list(shifts),
                feature_sha256=tensor_hash(mu),
                labels_sha256=tensor_hash(d["labels"]),
                metadata_sha256=tensor_hash(d["metadata"]),
                construction=(
                    "all48 circular phases, paired(p-d,p+d,p)/(p+d,p-d,p)"
                    if balanced
                    else "paired crops(-d,+d,0)/(+d,-d,0), no wrapping"
                ),
                exact_marginal_balance_checks=balance_checks,
            )
    return sets, identity


def metrics(logits, labels):
    pred = logits.argmax(-1)
    correct = pred == labels
    matrix = torch.bincount((2 * labels + pred).flatten(), minlength=4).reshape(2, 2)
    return dict(
        accuracy=float(correct.float().mean()),
        pair_accuracy=float(correct.all(1).float().mean()),
        flip_rate=float((pred[:, 0] != pred[:, 1]).float().mean()),
        confusion=matrix.tolist(),
        pairs=len(labels),
        cross_entropy=float(F.cross_entropy(logits.flatten(0, 1), labels.flatten())),
    )


@torch.no_grad()
def evaluate(model, sets):
    scores, arrays = {}, {}
    with evaluation_mode(model):
        for split, data in sets.items():
            scores[split] = {}
            x, labels = data["features"], data["labels"]
            controls = (
                ["normal"]
                if split in ("train", "validation")
                else ["normal", "current", "previous", "unordered", "swap"]
            )
            for control in controls:
                logits = torch.cat(
                    [
                        model(input_control(chunk, control))
                        for chunk in x.flatten(0, 1).split(64)
                    ]
                ).reshape(*labels.shape, 2)
                truth = 1 - labels if control == "swap" else labels
                score = metrics(logits, truth)
                score["by_displacement"] = {
                    str(int(d)): metrics(
                        logits[data["metadata"][:, -1] == d],
                        truth[data["metadata"][:, -1] == d],
                    )
                    for d in data["metadata"][:, -1].unique()
                }
                scores[split][control] = score
                arrays[f"{split}_{control}_logits"] = logits.cpu().numpy()
            arrays[f"{split}_labels"] = labels.numpy()
            arrays[f"{split}_metadata"] = data["metadata"].numpy()
            scores[split]["pixel_oracle_accuracy"] = float(
                (data["oracle"] == labels).float().mean()
            )
            scores[split]["pixel_oracle_ambiguous"] = int(data["ambiguous"].sum())
    return scores, arrays


def train(args, prepared=None):
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output.parent).free < 302 * 1024**2:
        raise RuntimeError("Preserve300MiB disk reserve")
    balanced = getattr(args, "balanced_training", False)
    sets, identity = prepare(args, balanced=balanced) if prepared is None else prepared
    assert identity["train"]["construction"].startswith("all48") == balanced
    seed_everything(args.seed)
    model = OrderReadout(sets["train"]["features"].shape[3], args.mode)
    source = torch.load(args.temporal_source, map_location="cpu", weights_only=True)
    model.temporal.load_state_dict(source["model"], strict=True)
    cfg = dict(
        seed=args.seed,
        mode=args.mode,
        balanced_training=balanced,
        steps=args.steps,
        batch_pairs=8,
        lr=0.003,
        weight_decay=0.0001,
        wall_seconds=45,
        source=str(args.source.resolve()),
        source_sha256=file_hash(args.source),
        temporal_source=str(args.temporal_source.resolve()),
        temporal_sha256=file_hash(args.temporal_source),
        purpose="paired last-step direction on constructed pans of real image contents",
        objective="balanced-pair direction cross entropy; frozen spatial encoder and unchanged image decoder",
        radius=2,
        example_labels={
            "input": "Pair members: all three frames, shared current frame"
        },
    )
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )
    run = Run(
        output,
        settings=cfg,
        data=identity,
        recipe=__file__,
        model=model,
        optimizer=optimizer,
        device="cpu",
        resume=args.resume,
    )
    initial_temporal = {
        k: v.detach().clone() for k, v in model.temporal.state_dict().items()
    }
    if not args.resume:
        atomic_json(
            output / "initial.json",
            dict(
                parameters=sum(p.numel() for p in model.parameters()),
                trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),
            ),
        )
    start = perf_counter()
    prior = sum(r.get("elapsed_seconds", 0) for r in run.rows if r["split"] == "timing")
    end = min(args.steps, run.step + args.stop_after) if args.stop_after else args.steps
    try:
        for step in range(run.step + 1, end + 1):
            if prior + perf_counter() - start > cfg["wall_seconds"]:
                break
            index = run.sample(len(sets["train"]["labels"]), cfg["batch_pairs"])
            x = sets["train"]["features"][index].flatten(0, 1)
            y = sets["train"]["labels"][index].flatten()
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1, error_if_nonfinite=True
            )
            optimizer.step()
            run.step = step
            run.log(
                dict(
                    step=step,
                    split="train",
                    loss=float(loss.detach()),
                    accuracy=float((logits.argmax(1) == y).float().mean()),
                    gradient=float(grad),
                )
            )
            if step % 128 == 0 or step == end:
                val, _ = evaluate(model, {"validation": sets["validation"]})
                run.log(
                    dict(
                        step=step,
                        split="validation",
                        loss=val["validation"]["normal"]["cross_entropy"],
                        accuracy=val["validation"]["normal"]["accuracy"],
                    )
                )
                run.save()
        run.log(
            dict(step=run.step, split="timing", elapsed_seconds=perf_counter() - start)
        )
        run.save()
        if args.mode in ("frozen", "correlation_only"):
            assert all(
                torch.equal(v, initial_temporal[k])
                for k, v in model.temporal.state_dict().items()
            )
        assert all(not d["features"].requires_grad for d in sets.values())
        scores, arrays = evaluate(model, sets)
        trace = {}
        model(sets["known"]["features"][:2].flatten(0, 1), trace=trace)
        arrays.update({f"trace_{k}": v.numpy() for k, v in trace.items()})
        np.savez_compressed(output / "evaluation.npz", **arrays)
        state = (
            "completed"
            if run.step == args.steps
            else "paused"
            if run.step == end
            else "budget-stopped"
        )
        atomic_json(
            output / "result.json",
            dict(
                completed=state == "completed",
                step=run.step,
                scores=scores,
                metrics={
                    f"{split}_{control}": {
                        k: v for k, v in row.items() if k != "by_displacement"
                    }
                    for split, items in scores.items()
                    for control, row in items.items()
                    if isinstance(row, dict)
                },
                parameters=sum(p.numel() for p in model.parameters()),
                trainable_parameters=sum(
                    p.numel() for p in model.parameters() if p.requires_grad
                ),
                training_seconds=sum(
                    r.get("elapsed_seconds", 0)
                    for r in run.rows
                    if r["split"] == "timing"
                ),
                evaluation_scope="Controlled crop-pan directions from real image contents. Same current frame and unordered multiset per pair; source-disjoint previously inspected development clips. Not natural motion, forecasting, speed estimation, or agent integration. Correlation is an explicit matching primitive; readout learned.",
            ),
        )
        run.status(state, "pending")
        write_report(output, batch={"rgb": sets["known"]["examples"][:1].flatten(0, 2)})
    except BaseException as error:
        run.save()
        status = json.loads((output / "status.json").read_text())
        run.status(
            status["result"] if (output / "result.json").exists() else "failed",
            "failed",
            str(error),
        )
        raise
    print(
        f"{output}: {state}; known={scores['known']['normal']['accuracy']:.3f}, wide={scores['wide']['normal']['accuracy']:.3f}",
        flush=True,
    )
    return scores


def challenge(args):
    """Evaluation only on all circular phases; preserve original failed gates."""
    start = perf_counter()
    sets, identity = prepare(args, balanced=True, evaluation_only=True)
    args.output.mkdir(parents=True, exist_ok=False)
    model_records, results, arrays = {}, {}, {}
    for seed in (7501, 7502):
        for mode in MODES:
            if perf_counter() - start > 120:
                raise RuntimeError("Balanced challenge evaluation budget exceeded")
            path = args.challenge_models / f"seed{seed}" / mode
            record = json.loads((path / "run.json").read_text())
            cfg = record["identity"]["settings"]
            assert file_hash(args.source) == cfg["source_sha256"]
            assert (
                identity["known"]["sources"]
                == record["identity"]["data"]["known"]["sources"]
            )
            model = OrderReadout(sets["known"]["features"].shape[3], mode)
            state = torch.load(path / "last.pt", map_location="cpu", weights_only=True)
            model.load_state_dict(state["model"], strict=True)
            model.requires_grad_(False).eval()
            scores, values = evaluate(model, sets)
            name = f"{seed}_{mode}"
            results[name] = scores
            arrays.update({name + "__" + k: v for k, v in values.items()})
            model_records[name] = dict(
                path=str((path / "last.pt").resolve()),
                sha256=file_hash(path / "last.pt"),
                training_recipe_sha256=record["source"]["files"][
                    record["source"]["recipe"]
                ],
            )
    gates = {}
    for mode in ("train", "correlation", "correlation_only"):
        criteria = {}
        for seed in (7501, 7502):
            for split in ("known", "wide"):
                row = results[f"{seed}_{mode}"][split]
                a = row["normal"]
                criteria[f"{seed}_{split}"] = dict(
                    accuracy=a["accuracy"] >= 0.9,
                    pair=a["pair_accuracy"] >= 0.8,
                    flip=a["flip_rate"] >= 0.9,
                    current=row["current"]["accuracy"] == 0.5
                    and row["current"]["pair_accuracy"] == 0,
                    previous=row["previous"]["accuracy"] == 0.5,
                    unordered=row["unordered"]["accuracy"] == 0.5
                    and row["unordered"]["pair_accuracy"] == 0,
                )
        gates[mode] = dict(
            passed=all(all(v.values()) for v in criteria.values()), criteria=criteria
        )
    for mode in ("current", "previous"):
        assert all(
            results[f"{seed}_{mode}"][s]["normal"]["accuracy"] == 0.5
            for seed in (7501, 7502)
            for s in ("known", "wide")
        )
    np.savez_compressed(args.output / "evaluation.npz", **arrays)
    atomic_json(
        args.output / "run.json",
        dict(
            schema="pathwm-evaluation-v1",
            identity=dict(
                settings=dict(
                    purpose="frozen-model balanced periodic-motion challenge",
                    new_training_steps=0,
                    original_gate_preserved="failed: previous-only static cue",
                    example_labels={
                        "input": "Paired periodic pans: first three frames, then reverse-prefix partner"
                    },
                ),
                data=identity,
                models=model_records,
            ),
            source=dict(recipe=__file__, sha256=file_hash(__file__)),
        ),
    )
    atomic_json(
        args.output / "result.json",
        dict(
            completed=True,
            gate=gates["train"]["passed"],
            candidate_gates=gates,
            scores=results,
            metrics={
                f"{name}_{split}": rows[split]["normal"]
                for name, rows in results.items()
                for split in ("known", "wide")
            },
            evaluation_seconds=perf_counter() - start,
            training_steps=0,
            evaluation_scope="Separate evaluation-only challenge; original crop-control gate remains failed. Exhaustive48 circular phases, real-image contents with periodic synthetic shifts, four images from one previously inspected reserved source. Exact single-frame label marginals verified on RGB and frozen encoded grids. No new fitting, natural-motion, forecasting or agent-integration claim.",
        ),
    )
    atomic_json(
        args.output / "status.json",
        dict(result="completed", report="pending", step=0, error=None),
    )
    (args.output / "metrics.jsonl").write_text("")
    write_report(
        args.output, batch={"rgb": sets["known"]["examples"][:1].flatten(0, 2)}
    )
    print(
        json.dumps(dict(gates=gates, seconds=perf_counter() - start), indent=2),
        flush=True,
    )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(
            "runs/spatial_vae_repair_v1/formal/seed57301/color/training/weights.pt"
        ),
    )
    parser.add_argument(
        "--temporal-source",
        type=Path,
        default=Path("runs/video_context_v1/seed7401/k3/history/last.pt"),
    )
    parser.add_argument(
        "--data", type=Path, default=Path("data/memory_media_v1/episodes")
    )
    parser.add_argument("--mode", choices=MODES, default="train")
    parser.add_argument("--steps", type=int, default=512)
    parser.add_argument("--seed", type=int, default=7501)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after", type=int)
    parser.add_argument(
        "--balanced-training",
        action="store_true",
        help="Use exhaustive circular phases for all training/evaluation populations",
    )
    parser.add_argument(
        "--challenge-models",
        type=Path,
        help="Evaluation only: root containing the12 fixed comparison arms",
    )
    args = parser.parse_args()
    if args.steps < 1 or (args.stop_after is not None and args.stop_after < 1):
        parser.error("Positive step budgets required")
    (challenge if args.challenge_models else train)(args)
