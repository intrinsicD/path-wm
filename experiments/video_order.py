"""Paired direction diagnosis on real-image-derived controlled pans.

python -m experiments.video_order --output runs/my_order --mode train
Use existing Run/report infrastructure; no natural-motion or agent capability claim.
"""

import argparse
from collections import Counter
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from pathwm.data.video_order import pan_pairs, cyclic_pan_pairs, reflect_pairs
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
def pixel_oracle(frames, *, shifts=(2, 4, 6, 8)):
    """Exhaustive signed alignment on a common valid interior; no wrap shortcut."""
    x = frames.flatten(0, 1)
    previous, current = x[:, -2], x[:, -1]
    magnitudes = sorted(set(shifts))
    width, height = x.shape[-1], x.shape[-2]
    if not magnitudes or any(
        type(d) is not int or not 0 < d < width / 2 for d in magnitudes
    ):
        raise ValueError("Oracle shifts must be positive integers below half the width")
    margin = max(magnitudes)
    vertical = min(margin, (height - 1) // 2)
    shifts = [-d for d in reversed(magnitudes)] + magnitudes
    errors = torch.stack(
        [
            (
                (
                    previous[
                        ...,
                        vertical : height - vertical,
                        margin + s : width - margin + s,
                    ]
                    - current[
                        ..., vertical : height - vertical, margin : width - margin
                    ]
                )
                ** 2
            ).mean((1, 2, 3))
            for s in shifts
        ],
        1,
    )
    best = errors.argmin(1)
    pred = torch.tensor(shifts)[best] < 0
    tied = errors <= errors.min(1, keepdim=True).values + 1e-10
    ambiguous = tied[:, : len(magnitudes)].any(1) & tied[:, len(magnitudes) :].any(1)
    return pred.long().reshape(frames.shape[:2]), ambiguous.reshape(frames.shape[:2])


def load_displacements(path=None):
    """Optional periodic-task support, separate from source content and labels."""
    spec = (
        dict(schema=1, train=[2, 4], evaluation=dict(known=[2, 4], wide=[6, 8]))
        if path is None
        else json.loads(Path(path).read_text())
    )
    if set(spec) != {"schema", "train", "evaluation"} or spec["schema"] != 1:
        raise ValueError("Expected schema, train and evaluation displacement fields")
    groups = spec["evaluation"]
    if not isinstance(groups, dict) or not {"known", "wide"} <= groups.keys():
        raise ValueError("Evaluation requires known and wide groups")
    if any(
        not re.fullmatch(r"[a-z][a-z_]*", name)
        or name in ("train", "validation")
        or name.startswith("confirm_")
        for name in groups
    ):
        raise ValueError("Invalid/reserved displacement group name")
    for shifts in [spec["train"], *groups.values()]:
        if (
            not isinstance(shifts, list)
            or not shifts
            or any(type(d) is not int or not 0 < d < 24 for d in shifts)
            or shifts != sorted(set(shifts))
        ):
            raise ValueError("Displacements must be sorted unique integers in [1,23]")
    return spec


def phase_pair_groups(metadata):
    """Index complete image/phase groups; all groups have identical shift slots."""
    if metadata.ndim != 2 or metadata.shape[1] != 3 or not len(metadata):
        raise ValueError("Expected periodic image, phase, displacement metadata")
    shifts = metadata[:, -1].unique(sorted=True)
    if len(metadata) % len(shifts):
        raise ValueError("Incomplete phase groups")
    grouped = metadata.reshape(-1, len(shifts), 3)
    if (
        not torch.equal(
            grouped[:, :, :2], grouped[:, :1, :2].expand(-1, len(shifts), -1)
        )
        or not torch.equal(grouped[:, :, -1], shifts.expand(len(grouped), -1))
        or len(grouped[:, 0, :2].unique(dim=0)) != len(grouped)
    ):
        raise ValueError("Phase groups require consistent unique displacement slots")
    return torch.arange(len(metadata)).reshape(-1, len(shifts))


def sample_phase_pairs(sample, groups, count):
    # The same two RNG draws per update preserve matched content across supports.
    rows = sample(len(groups), count)
    columns = sample(groups.shape[1], count)
    return groups[rows, columns]


def verify_marginals(data, features, shifts):
    """Exact class-conditional counts per source image, displacement and time."""
    checks = 0
    for image_id in data["metadata"][:, 0].unique():
        for displacement in shifts:
            chosen = (data["metadata"][:, 0] == image_id) & (
                data["metadata"][:, -1] == displacement
            )
            assert int(chosen.sum()) == data["frames"].shape[-1]
            labels = data["labels"][chosen].flatten()
            for values in (data["frames"][chosen], features[chosen]):
                flat = values.flatten(0, 1)
                for t in range(3):
                    assert Counter(
                        tensor_hash(v) for v in flat[labels == 0, t]
                    ) == Counter(tensor_hash(v) for v in flat[labels == 1, t])
                    checks += 1
    return checks


def load_sources(manifest, directory):
    """Resolve a small source list and reject exact-file/confirmation-subject leaks."""
    if manifest is None:
        groups = {
            split: [
                dict(
                    id=name,
                    subject=None,
                    path=str(directory / name / "video.mp4"),
                    frames=4,
                )
                for name in names
            ]
            for split, names in SOURCES.items()
        }
    else:
        record = json.loads(Path(manifest).read_text())
        if record.get("schema") != 1:
            raise ValueError("Unsupported video source manifest")
        groups = record["splits"]
    required = {"train", "validation", "evaluation"}
    if not required <= groups.keys() or groups.keys() - (required | {"confirmation"}):
        raise ValueError(
            "Expected train, validation, evaluation and optional confirmation"
        )
    seen, seen_ids, result = {}, set(), {}
    for split, records in groups.items():
        if not records:
            raise ValueError("Empty source split")
        result[split] = []
        for record in records:
            if not isinstance(record.get("id"), str) or not record["id"]:
                raise ValueError("Nonempty source id required")
            if record["id"] in seen_ids:
                raise ValueError("Exact duplicate source id in manifest")
            seen_ids.add(record["id"])
            if type(record["frames"]) is not int or record["frames"] < 1:
                raise ValueError("frames must be a positive integer")
            path = Path(record["path"])
            if manifest is not None and not path.is_absolute():
                path = Path(manifest).parent / path
            path = path.resolve()
            digest = file_hash(path)
            if digest in seen:
                raise ValueError("Exact duplicate video in source manifest")
            if record.get("sha256", digest) != digest:
                raise ValueError("Source hash changed")
            seen[digest] = split
            result[split].append(dict(record, path=str(path), sha256=digest))
    if "confirmation" in result:
        subjects = [r.get("subject") for r in result["confirmation"]]
        others = {
            r.get("subject")
            for split, records in result.items()
            if split != "confirmation"
            for r in records
        }
        if (
            not all(subjects)
            or not all(others)
            or len(set(subjects)) != len(subjects)
            or set(subjects) & others
        ):
            raise ValueError("Confirmation subject overlap or missing subject")
    return result


@torch.no_grad()
def prepare(args, *, balanced=False, evaluation_only=False, include_reflection=False):
    displacement_path = getattr(args, "displacement_spec", None)
    displacements = load_displacements(displacement_path)
    if displacement_path is not None and not balanced:
        raise ValueError("Custom displacements require balanced periodic phases")
    oracle_shifts = sorted(
        {2, 4}
        | {
            d
            for values in [
                displacements["train"],
                *displacements["evaluation"].values(),
            ]
            for d in values
        }
    )
    seed_everything(7500)
    image = SpatialVAE.load(args.source).requires_grad_(False).eval()
    sets, identity = {}, {}
    groups = load_sources(getattr(args, "source_manifest", None), args.data)
    crop_owners = {}
    for split, specifications in groups.items():
        if evaluation_only and split != "evaluation":
            continue
        images, records, image_sources = [], [], []
        for source_number, specification in enumerate(specifications):
            path = Path(specification["path"])
            frame_count = specification["frames"]
            command = [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                str(path),
                "-t",
                str(frame_count / 2),
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
            if len(decoded) != frame_count:
                raise ValueError("Decoded frame count differs from source manifest")
            for crop in decoded[:, 4:52, 12:60]:
                digest = hashlib.sha256(crop.tobytes()).hexdigest()
                if digest in crop_owners and crop_owners[digest] != split:
                    raise ValueError("Exact duplicate RGB crop across source splits")
                crop_owners[digest] = split
            image_sources.extend([source_number] * frame_count)
            images.append(torch.from_numpy(decoded).permute(0, 3, 1, 2).float() / 255)
            records.append(
                dict(
                    id=specification["id"],
                    subject=specification.get("subject"),
                    frames=frame_count,
                    path=str(path.resolve()),
                    sha256=specification["sha256"],
                    decoded_sha256=hashlib.sha256(raw).hexdigest(),
                    command=command,
                )
            )
        images = torch.cat(images)
        populations = [(split, displacements["train"] if split == "train" else [2, 4])]
        if split in ("evaluation", "confirmation"):
            prefix = "confirm_" if split == "confirmation" else ""
            populations = [
                (prefix + name, shifts)
                for name, shifts in displacements["evaluation"].items()
            ]
        bank = None
        for name, shifts in populations:
            if balanced:
                d = cyclic_pan_pairs(images[..., 4:52, 12:60], shifts=shifts)
                if bank is None:
                    bank = torch.cat([image.encode(v).mu for v in d["views"].split(32)])
                mu = bank[d["indices"]]
            else:
                d = pan_pairs(images, shifts=shifts)
                flat = d["frames"].flatten(0, 2)
                mu = torch.cat([image.encode(v).mu for v in flat.split(32)])
                mu = mu.reshape(*d["frames"].shape[:3], *mu.shape[1:])
            assert torch.equal(mu[:, 0, -1], mu[:, 1, -1])
            assert torch.equal(mu[:, 0, :2], mu[:, 1, :2].flip(1))
            oracle, ambiguous = pixel_oracle(d["frames"], shifts=oracle_shifts)
            balance_checks = verify_marginals(d, mu, shifts) if balanced else 0
            sets[name] = dict(
                features=mu,
                labels=d["labels"],
                metadata=d["metadata"],
                examples=d["frames"][:2].clone(),
                oracle=oracle,
                ambiguous=ambiguous,
                source_index=torch.tensor(image_sources)[d["metadata"][:, 0]],
                source_ids=[r["id"] for r in records],
            )
            identity[name] = dict(
                sources=records,
                image_sources=image_sources,
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
                oracle_shifts=oracle_shifts,
            )
            if include_reflection:
                if not balanced:
                    raise ValueError("Reflection comparison requires balanced phases")
                reflected = reflect_pairs(d)
                reflected_bank = torch.cat(
                    [image.encode(v).mu for v in reflected["views"].split(32)]
                )
                reflected_mu = reflected_bank[reflected["indices"]]
                reflected_oracle, reflected_ambiguous = pixel_oracle(
                    reflected["frames"], shifts=oracle_shifts
                )
                reflection_checks = verify_marginals(reflected, reflected_mu, shifts)
                sets[name].update(
                    reflected_features=reflected_mu,
                    reflected_labels=reflected["labels"],
                )
                identity[name]["reflection"] = dict(
                    feature_sha256=tensor_hash(reflected_mu),
                    labels_sha256=tensor_hash(reflected["labels"]),
                    rgb_sha256=tensor_hash(reflected["views"]),
                    marginal_checks=reflection_checks,
                    encoded_vs_latent_flip_mse=float(
                        (reflected_mu - mu.flip(-1)).square().mean()
                    ),
                    pixel_oracle_accuracy=float(
                        (reflected_oracle == reflected["labels"]).float().mean()
                    ),
                    pixel_oracle_ambiguous=int(reflected_ambiguous.sum()),
                    method="exact RGB horizontal reflection before frozen encoding; original pair indices",
                )
    return sets, identity


def metrics(logits, labels):
    pred = logits.argmax(-1)
    correct = pred == labels
    matrix = torch.bincount((2 * labels + pred).flatten(), minlength=4).reshape(2, 2)
    margin = (
        logits.gather(-1, labels[..., None])
        - logits.gather(-1, (1 - labels)[..., None])
    ).squeeze(-1)
    return dict(
        accuracy=float(correct.float().mean()),
        pair_accuracy=float(correct.all(1).float().mean()),
        flip_rate=float((pred[:, 0] != pred[:, 1]).float().mean()),
        confusion=matrix.tolist(),
        pairs=len(labels),
        cross_entropy=float(F.cross_entropy(logits.flatten(0, 1), labels.flatten())),
        margin_mean=float(margin.mean()),
        margin_p10=float(torch.quantile(margin.flatten(), 0.1)),
        confident_wrong_fraction=float(
            ((~correct) & (logits.softmax(-1).amax(-1) >= 0.95)).float().mean()
        ),
    )


def source_metrics(logits, labels, data):
    """Weight source clips equally, independently of their frame/pair counts."""
    rows = {
        name: metrics(
            logits[data["source_index"] == i], labels[data["source_index"] == i]
        )
        for i, name in enumerate(data["source_ids"])
    }
    return dict(
        by_source=rows,
        **{
            "source_macro_" + key: sum(row[key] for row in rows.values()) / len(rows)
            for key in ("accuracy", "pair_accuracy", "flip_rate")
        },
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
            if "reflected_features" in data:
                controls = [*controls, "reflection"]
            for control in controls:
                values = data["reflected_features"] if control == "reflection" else x
                logits = torch.cat(
                    [
                        model(
                            input_control(
                                chunk, "normal" if control == "reflection" else control
                            )
                        )
                        for chunk in values.flatten(0, 1).split(64)
                    ]
                ).reshape(*labels.shape, 2)
                truth = 1 - labels if control in ("swap", "reflection") else labels
                score = metrics(logits, truth)
                score["by_displacement"] = {
                    str(int(d)): metrics(
                        logits[data["metadata"][:, -1] == d],
                        truth[data["metadata"][:, -1] == d],
                    )
                    for d in data["metadata"][:, -1].unique()
                }
                if "source_index" in data:
                    score.update(source_metrics(logits, truth, data))
                scores[split][control] = score
                arrays[f"{split}_{control}_logits"] = logits.cpu().numpy()
            arrays[f"{split}_labels"] = labels.numpy()
            arrays[f"{split}_metadata"] = data["metadata"].numpy()
            if "source_index" in data:
                arrays[f"{split}_source_index"] = data["source_index"].numpy()
            scores[split]["pixel_oracle_accuracy"] = float(
                (data["oracle"] == labels).float().mean()
            )
            scores[split]["pixel_oracle_ambiguous"] = int(data["ambiguous"].sum())
    return scores, arrays


def training_batch(data, index, step, reflect=False):
    """Select one orientation per update without changing sampled source pairs."""
    prefix = "reflected_" if reflect and step % 2 else ""
    return data[prefix + "features"][index].flatten(0, 1), data[prefix + "labels"][
        index
    ].flatten()


def train(args, prepared=None):
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output.parent).free < 302 * 1024**2:
        raise RuntimeError("Preserve300MiB disk reserve")
    balanced = getattr(args, "balanced_training", False)
    matched_phases = getattr(args, "matched_phase_sampling", False)
    if matched_phases and not balanced:
        raise ValueError("Matched phase sampling requires balanced periodic data")
    reflect = getattr(args, "reflect_training", False)
    head_seed = getattr(args, "head_seed", None)
    head_seed = args.seed if head_seed is None else head_seed
    sets, identity = (
        prepare(args, balanced=balanced, include_reflection=reflect)
        if prepared is None
        else prepared
    )
    assert identity["train"]["construction"].startswith("all48") == balanced
    phase_groups = (
        phase_pair_groups(sets["train"]["metadata"]) if matched_phases else None
    )
    if reflect and "reflected_features" not in sets["train"]:
        raise ValueError("Missing reflected RGB encoding for training")
    seed_everything(head_seed)
    model = OrderReadout(sets["train"]["features"].shape[3], args.mode)
    source = torch.load(args.temporal_source, map_location="cpu", weights_only=True)
    model.temporal.load_state_dict(source["model"], strict=True)
    cfg = dict(
        seed=args.seed,
        head_seed=head_seed,
        matched_phase_sampling=matched_phases,
        displacement_spec_sha256=file_hash(args.displacement_spec)
        if getattr(args, "displacement_spec", None)
        else None,
        source_manifest_sha256=file_hash(args.source_manifest)
        if getattr(args, "source_manifest", None)
        else None,
        reflect_training=reflect,
        reflection_schedule="odd absolute steps mirrored, even original"
        if reflect
        else "original only",
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
            index = (
                sample_phase_pairs(run.sample, phase_groups, cfg["batch_pairs"])
                if matched_phases
                else run.sample(len(sets["train"]["labels"]), cfg["batch_pairs"])
            )
            x, y = training_batch(sets["train"], index, step, reflect)
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
                    reflected=bool(reflect and step % 2),
                    pair_indices=index.tolist(),
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
        sampled = np.array(
            [row["pair_indices"] for row in run.rows if row["split"] == "train"],
            dtype=np.int64,
        ).ravel()
        pair_counts = np.bincount(sampled, minlength=len(sets["train"]["labels"]))
        image_indices = sets["train"]["metadata"][:, 0].numpy()
        image_counts = np.bincount(image_indices, weights=pair_counts).astype(np.int64)
        source_counts = np.bincount(
            sets["train"]["source_index"].numpy(), weights=pair_counts
        ).astype(np.int64)
        magnitudes, displacement_indices = np.unique(
            sets["train"]["metadata"][:, -1].numpy(), return_inverse=True
        )
        displacement_counts = np.bincount(
            displacement_indices, weights=pair_counts
        ).astype(np.int64)
        np.savez_compressed(
            output / "exposure.npz",
            sampled_pairs=sampled,
            pair_counts=pair_counts,
            image_counts=image_counts,
            source_counts=source_counts,
            displacements=magnitudes,
            displacement_counts=displacement_counts,
        )
        exposure = dict(
            sampled_pairs=int(len(sampled)),
            unique_pairs=int((pair_counts > 0).sum()),
            unique_images=int((image_counts > 0).sum()),
            source_ids=sets["train"]["source_ids"],
            source_counts=source_counts.tolist(),
            image_counts=image_counts.tolist(),
            displacement_counts=dict(
                zip(map(str, magnitudes), map(int, displacement_counts))
            ),
            unit="paired examples; each has two labeled clips",
        )
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
                exposure=exposure,
                metrics={
                    f"{split}_{control}": {
                        k: v
                        for k, v in row.items()
                        if k not in ("by_displacement", "by_source")
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
                evaluation_scope="Controlled crop-pan directions from real image contents. Same current frame and unordered multiset per pair; source-disjoint clips, with optional reserved confirmation sources listed in data identity. Not natural motion, forecasting, speed estimation, or agent integration. Correlation is an explicit matching primitive; readout learned.",
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
            old_sources = record["identity"]["data"]["known"]["sources"]
            new_sources = identity["known"]["sources"]
            assert len(old_sources) == len(new_sources)
            assert all(
                a[k] == b[k]
                for a, b in zip(old_sources, new_sources)
                for k in ("path", "sha256", "decoded_sha256")
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
    parser.add_argument(
        "--source-manifest",
        type=Path,
        help="Optional explicit video paths, subjects and frame counts by split",
    )
    parser.add_argument("--mode", choices=MODES, default="train")
    parser.add_argument(
        "--displacement-spec",
        type=Path,
        help="Optional JSON train/evaluation magnitudes; balanced phases only",
    )
    parser.add_argument(
        "--matched-phase-sampling",
        action="store_true",
        help="Sample image/phase then displacement to match content across supports",
    )
    parser.add_argument("--steps", type=int, default=512)
    parser.add_argument("--seed", type=int, default=7501)
    parser.add_argument(
        "--head-seed",
        type=int,
        help="Independent model/head initialization; --seed controls batch order",
    )
    parser.add_argument(
        "--reflect-training",
        action="store_true",
        help="Alternate exact RGB reflections with inverted directions; requires --balanced-training",
    )
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
