"""Controlled output metrics and panels, consumed by the ordinary run report."""

import base64
from html import escape
import itertools
import json
from pathlib import Path
import time
import wave
import shutil

import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image

from pathwm.data.modality_readout import (
    KINDS,
    MODES,
    COLORS,
    PLACES,
    DIRECTIONS,
    canonical,
)
from pathwm.models.modalities import bytes_text
from pathwm.models.photo_probe import RidgeReader
from pathwm.io import atomic_json, file_hash


def capture_readout_stages(core, inputs, *, encoder_token_cap=64):
    """Detached measurements of the real path; no replacement or second forward.

    Encoder flattening preserves spatial/temporal locations but gives different
    probe capacities across stages. A failed linear probe is not information loss.
    The core's optional return_state is its post-observation, pre-Thinker state.
    """
    encoded, hooks = {}, []

    def capture(name):
        def hook(module, arguments, result):
            # Give each scale fixed diagnostic coordinates. Right-padding the
            # flattened pyramid would shift coarser scales when text grows.
            padded = []
            for item in result.scales:
                if item.values.shape[1] > encoder_token_cap:
                    raise ValueError("Encoder exceeds diagnostic per-scale token cap")
                values = item.values.masked_fill(~item.valid[..., None], 0)
                padded.append(
                    F.pad(values, (0, 0, 0, encoder_token_cap - values.shape[1]))
                    .detach()
                    .cpu()
                    .flatten(1)
                )
            encoded[name] = torch.cat(padded, 1).clone()

        return hook

    try:
        for name in inputs:
            hooks.append(core.agent.encoders[name].register_forward_hook(capture(name)))
        tokens, state = core(inputs, return_state=True)
    finally:
        for hook in hooks:
            hook.remove()
    stages = {"encoder": torch.cat([encoded[k] for k in sorted(encoded)], 1)}
    for name, value in (
        ("posterior", state.logits.softmax(-1)),
        ("codes", state.stochastic),
        ("observed", state.tokens),
        ("thought", tokens),
    ):
        stages[name] = value.detach().cpu().flatten(1).clone()
    return tokens, stages


def predict_factor_probe(reader, features):
    logits = (np.asarray(features, dtype=np.float64) - reader["mean"]) / reader[
        "scale"
    ] @ reader["weights"] + reader["bias"]
    return np.stack([v.argmax(1) for v in np.split(logits, (3, 6), axis=1)], 1)


def fit_factor_probe(
    train,
    labels,
    validation,
    validation_labels,
    *,
    alphas=(0.01, 0.1, 1.0, 10.0, 100.0),
):
    """Train-only standardized linear ridge; selection sees validation only."""
    train = np.asarray(train, dtype=np.float64)
    validation = np.asarray(validation, dtype=np.float64)
    labels, validation_labels = np.asarray(labels), np.asarray(validation_labels)
    if train.ndim != 2 or validation.ndim != 2 or train.shape[1] != validation.shape[1]:
        raise ValueError("Probe inputs need matched feature dimensions")
    if not np.isfinite(train).all() or not np.isfinite(validation).all():
        raise ValueError("Probe features must be finite")
    y = np.concatenate([np.eye(n)[labels[:, i]] for i, n in enumerate((3, 3, 2))], 1)
    factor = RidgeReader.factor(torch.from_numpy(train), torch.from_numpy(y))
    readers, choices = [], []
    for alpha in alphas:
        if alpha <= 0:
            raise ValueError("Ridge must be positive")
        # Existing reader normalizes the Gram matrix by feature width. Convert
        # raw ridge alpha accordingly, retaining comparable declared units.
        solved = RidgeReader.solve(factor, ridge=alpha / train.shape[1])
        reader = dict(
            mean=solved["mean"].numpy(),
            scale=solved["std"].numpy(),
            weights=(solved["training"].T @ solved["alpha"] / train.shape[1]).numpy(),
            bias=solved["target_mean"].numpy(),
            alpha=alpha,
        )
        acc = float(
            (predict_factor_probe(reader, validation) == validation_labels).mean()
        )
        readers.append(reader)
        choices.append(dict(alpha=alpha, validation_accuracy=acc))
    return readers[int(np.argmax([c["validation_accuracy"] for c in choices]))], choices


def save_stage_panel(directory, rows):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    stages = ("encoder", "posterior", "codes", "observed", "thought")
    fig = Figure(figsize=(14, 8), layout="constrained")
    FigureCanvasAgg(fig)
    axes = fig.subplots(2, 6)
    for si, split in enumerate(("seen", "heldout")):
        for mi, mode in enumerate(MODES):
            values = np.array(
                [
                    next(
                        r["factor_accuracy"]
                        for r in rows
                        if r["split"] == split
                        and r["input_mode"] == mode
                        and r["stage"] == stage
                        and r["probe"] == "ridge"
                    )
                    for stage in stages
                ]
            )
            ax = axes[si, mi]
            ax.imshow(values, vmin=0, vmax=1, cmap="viridis", aspect="auto")
            ax.set_xticks(range(3), ["color", "place", "direction"], rotation=70)
            ax.set_yticks(range(5), stages if mi == 0 else [""] * 5)
            ax.set_title(split + " / " + mode, fontsize=9)
            for i in range(5):
                for j in range(3):
                    ax.text(
                        j,
                        i,
                        f"{values[i, j]:.0%}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="white" if values[i, j] < 0.5 else "black",
                    )
    fig.suptitle(
        "Frozen linear accessibility by stage · validation-selected ridge · not an information-loss proof"
    )
    fig.savefig(Path(directory) / "comparison.png", dpi=140)


def text_factors(ids):
    result = []
    for row in ids:
        parts = bytes_text(row).split()
        try:
            if len(parts) != 3:
                raise ValueError("Three words required")
            result.append(
                [
                    COLORS.index(parts[0]),
                    PLACES.index(parts[1]),
                    DIRECTIONS.index(parts[2]),
                ]
            )
        except ValueError:
            result.append([-1, -1, -1])
    return torch.tensor(result, device=ids.device)


@torch.no_grad()
def video_motion_metrics(output, target, *, background=0.04):
    """Small bright-object diagnostic, not general optical flow or tracking.

    Foreground mass is RGB maximum above the known synthetic background. Motion
    requires nonempty predicted endpoint frames and >=0.5px displacement; static
    or blank clips cannot pass because a sign comparison happened to match.
    """
    if output.shape != target.shape or output.ndim != 5 or output.shape[1] < 2:
        raise ValueError("Motion metrics need matched [B,T,C,H,W] clips with T>=2")
    h, w = output.shape[-2:]
    y, x = torch.meshgrid(
        torch.arange(h, device=output.device, dtype=output.dtype),
        torch.arange(w, device=output.device, dtype=output.dtype),
        indexing="ij",
    )

    def centroids(video):
        mass = (video.amax(2) - background).clamp_min(0)
        total = mass.sum((-2, -1))
        center = torch.stack(
            [(mass * axis).sum((-2, -1)) / total.clamp_min(1e-8) for axis in (x, y)], -1
        )
        return center, total

    pred, mass = centroids(output)
    truth, _ = centroids(target)
    error = (pred - truth).norm(dim=-1)
    displacement = pred[:, -1, 0] - pred[:, 0, 0]
    expected = truth[:, -1, 0] - truth[:, 0, 0]
    moving = expected.abs() >= 1
    correct = (
        (displacement.sign() == expected.sign())
        & (displacement.abs() >= 0.5)
        & (mass[:, 0] > 1e-8)
        & (mass[:, -1] > 1e-8)
        & moving
    )
    return dict(
        centroid_error_pixels=float(error.mean()),
        displacement_mae_pixels=float((displacement - expected).abs().mean()),
        motion_direction_accuracy=float(
            correct.float().sum() / moving.sum().clamp_min(1)
        ),
        moving_clips=int(moving.sum()),
        per_frame_mse=(output - target).square().mean((0, 2, 3, 4)).tolist(),
    )


def scores(kind, output, target, templates, normalizer=None, generated=None):
    if kind == "text":
        loss = float(
            F.cross_entropy(
                output.flatten(0, 1), target["text"][:, 1:].flatten(), ignore_index=0
            )
        )
        pred = text_factors(generated)
        exact = float(
            np.mean(
                [
                    bytes_text(a) == bytes_text(b)
                    for a, b in zip(generated, target["text"])
                ]
            )
        )
        result = dict(teacher_ce=loss, free_exact=exact)
    else:
        error = (output - target[kind]).square()
        loss = float(error.mean())
        distances = (
            (output[:, None] - templates[kind][None]).square().flatten(2).mean(2)
        )
        pred = templates["factors"][distances.argmin(1)]
        result = dict(mse=loss, normalized_mse=loss / normalizer)
        if kind in ("image", "video"):
            axis = 1 if kind == "image" else 2
            fg = (target[kind].amax(axis, keepdim=True) > 0.1).expand_as(error)
            result["foreground_mse"] = float(error[fg].mean())
            result["background_mse"] = float(error[~fg].mean())
        if kind == "video":
            result.update(video_motion_metrics(output, target[kind]))
    correct = pred == target["factors"]
    result["factor_accuracy"] = correct.float().mean(0).tolist()
    result["all_accuracy"] = float(correct.all(1).float().mean())
    result["gate"] = result["all_accuracy"] >= 0.8 and (
        result["free_exact"] >= 0.8
        if kind == "text"
        else result["normalized_mse"] <= 0.2
        if kind == "audio"
        else result["mse"] <= 0.02 and result["foreground_mse"] <= 0.05
    )
    return result, pred


@torch.no_grad()
def evaluate_outputs(outputs, cache, populations, kinds, normalizers, device):
    reference = canonical(list(itertools.product(range(3), range(3), range(2))))
    templates = {k: v.to(device) for k, v in reference.items()}
    rows, arrays = [], {}
    for split in ("seen", "heldout"):
        target = {k: v.to(device) for k, v in populations[split]["targets"].items()}
        for k, v in target.items():
            arrays[f"{split}.target.{k}"] = v.cpu().numpy()
        for mode, cached in cache[split].items():
            tokens = cached.to(device)
            arrays[f"{split}.{mode}.context"] = cached.numpy()
            wrong = tokens.roll(len(tokens) // 2, 0)
            assert (
                (target["factors"] != target["factors"].roll(len(tokens) // 2, 0))
                .any(1)
                .all()
            )
            for kind in kinds:
                key = f"{split}.{mode}.{kind}"
                prefix = target["text"][:, :-1] if kind == "text" else None
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                start = time.perf_counter()
                output = outputs(kind, tokens, prefix)
                generated = outputs.generate(tokens) if kind == "text" else None
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                latency = time.perf_counter() - start
                result, pred = scores(
                    kind, output, target, templates, normalizers.get(kind), generated
                )
                if kind == "video":
                    # These are interventions on the prediction, not replacement
                    # targets or new decoder inputs. Save arrays for independent QA.
                    result["temporal_controls"] = {}
                    for name, changed in (
                        ("static_first", output[:, :1].expand_as(output)),
                        ("reversed", output.flip(1)),
                    ):
                        measured, _ = scores(
                            kind, changed, target, templates, normalizers[kind]
                        )
                        result["temporal_controls"][name] = measured
                        arrays[key + f".{name}_output"] = changed.cpu().numpy()
                arrays[key + ".output"] = output.cpu().numpy()
                arrays[key + ".factors"] = pred.cpu().numpy()
                if generated is not None:
                    arrays[key + ".generated"] = generated.cpu().numpy()
                controls = {}
                for name, context in (
                    ("zero", torch.zeros_like(tokens)),
                    ("wrong", wrong),
                ):
                    alternative = outputs(kind, context, prefix)
                    altgen = outputs.generate(context) if kind == "text" else None
                    alt, altpred = scores(
                        kind,
                        alternative,
                        target,
                        templates,
                        normalizers.get(kind),
                        altgen,
                    )
                    controls[name] = alt
                    arrays[key + f".{name}_output"] = alternative.cpu().numpy()
                    arrays[key + f".{name}_factors"] = altpred.cpu().numpy()
                    if altgen is not None:
                        arrays[key + f".{name}_generated"] = altgen.cpu().numpy()
                rows.append(
                    dict(
                        split=split,
                        input_mode=mode,
                        output=kind,
                        **result,
                        controls=controls,
                        decode_batch_seconds=latency,
                        batch=len(tokens),
                        expected_missing_information=mode.startswith("without_"),
                    )
                )
    return rows, arrays


def save_panels(directory, arrays, rows):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    directory = Path(directory)
    select = np.arange(0, 48, 8)
    kinds = {r["output"] for r in rows}
    if "image" in kinds or "video" in kinds:
        kind = "image" if "image" in kinds else "video"
        target = arrays[f"heldout.target.{kind}"][select]
        output = arrays[f"heldout.all.{kind}.output"][select]
        if kind == "video":
            target, output = target[:, 0], output[:, 0]
        fig = Figure(figsize=(10, 5), layout="constrained")
        FigureCanvasAgg(fig)
        axes = fig.subplots(3, len(select))
        for row, (label, values) in enumerate(
            (
                ("Target", target),
                ("Output", output),
                ("Absolute error", np.abs(target - output)),
            )
        ):
            for i, v in enumerate(values):
                axes[row, i].imshow(
                    v.transpose(1, 2, 0).clip(0, 1),
                    vmin=0,
                    vmax=1,
                    interpolation="nearest",
                )
                axes[row, i].set_xticks([])
                axes[row, i].set_yticks([])
                if i == 0:
                    axes[row, i].set_ylabel(label)
        fig.suptitle(
            "Held-out combinations · all complete inputs · one view per combination"
        )
        fig.savefig(directory / "comparison.png", dpi=140)
    if "video" in kinds:
        for name, key in [
            ("target", "heldout.target.video"),
            ("output", "heldout.all.video.output"),
        ]:
            frames = [
                Image.fromarray(
                    np.rint(x.transpose(1, 2, 0).clip(0, 1) * 255).astype("uint8")
                )
                for x in arrays[key][0]
            ]
            frames[0].save(
                directory / f"readout_video_{name}.gif",
                save_all=True,
                append_images=frames[1:],
                duration=250,
                loop=0,
            )
    if "audio" in kinds:
        for name, key in [
            ("target", "heldout.target.audio"),
            ("output", "heldout.all.audio.output"),
        ]:
            samples = np.rint(
                arrays[key][select].reshape(-1).clip(-1, 1) * 32767
            ).astype("<i2")
            with wave.open(str(directory / f"readout_audio_{name}.wav"), "wb") as f:
                f.setnchannels(1)
                f.setsampwidth(2)
                f.setframerate(8000)
                f.writeframes(samples.tobytes())
    if "text" in kinds:
        text = [
            dict(
                target=bytes_text(arrays["heldout.target.text"][i]),
                generated=bytes_text(arrays["heldout.all.text.generated"][i]),
            )
            for i in select
        ]
        atomic_json(directory / "text_examples.json", text)


def readout_inspection(directory):
    repair = directory / "repair.json"
    if repair.exists():
        data = json.loads(repair.read_text())
        parts = [
            "<section><h2>Separate diagnoses and controlled repairs</h2><p>Core factor accuracy averages six input modes. Oracle decoder controls receive explicit true facts and are not agent performance. See child reports for full cells, quality gates, omission controls and raw outputs.</p>"
        ]
        for name in ("repair_scores.png", "repair_learning.png"):
            url = (
                "data:image/png;base64,"
                + base64.b64encode((directory / name).read_bytes()).decode()
            )
            parts.append(f'<img class="chart" alt="{escape(name)}" src="{url}">')
        parts.append("<h3>Individual reports</h3><ul>")
        for path in data["reports"]:
            parts.append(
                f'<li><a href="{escape(path)}/report.html">{escape(path)}</a></li>'
            )
        parts.append(
            "</ul><details><summary>Exact comparison data</summary><pre>"
            + escape(json.dumps(data, indent=2))
            + "</pre></details></section>"
        )
        return parts
    diagnostic = directory / "stage_probe.json"
    if diagnostic.exists():
        data = json.loads(diagnostic.read_text())
        return [
            "<section><h2>Frozen stage accessibility</h2><p>Train-only standardized linear readers; validation-only ridge selection. Encoder feature widths differ from posterior/code/state widths. Failures do not establish irrecoverable information loss. Source model frozen; all generated factors below are diagnostic predictions.</p><details><summary>Exact stage scores, reader selection and oracle control</summary><pre>"
            + escape(json.dumps(data, indent=2))
            + "</pre></details></section>"
        ]
    path = directory / "readout.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    parts = [
        "<section><h2>Multimodal latent readout</h2><p>Controlled arrows, symbolic tones and short descriptions. Audio is not speech; video is observed-clip reconstruction, not a dynamics forecast. Frozen readout and joint core training are different interventions. Latent factor supervision is supplied, not discovered.</p>"
    ]
    if data.get("aggregate"):
        parts.append(
            f"<p><strong>Registered joint screens passed: {sum(r['gate'] for r in data['summary'])}/{len(data['summary'])}.</strong> These screens require both factual correctness and output quality.</p>"
        )
        parts.append(
            "<p>Every frozen row combines four independently trained output branches reading the same fixed core. Joint rows train core and all outputs together. Agreement can be wrong; all-correct requires the target factors in every output. Recurrent variants add parameters versus native; iteration counts share parameters but not compute.</p>"
        )
        parts.append(
            '<div class="table"><table><tr><th>Seed</th><th>Training</th><th>Readout</th><th>Split</th><th>Input</th><th>Text</th><th>Image</th><th>Audio</th><th>Video</th><th>All correct</th><th>Agreement</th><th>Screen</th></tr>'
        )
        for r in data["summary"]:
            vals = [
                r["seed"],
                r["stage"],
                r["variant"],
                r["split"],
                r["input_mode"],
                *[f"{r['accuracy'][k]:.1%}" for k in KINDS],
                f"{r['all_correct']:.1%}",
                f"{r['agreement']:.1%}",
                r["gate"],
            ]
            parts.append(
                "<tr>"
                + "".join("<td>" + escape(str(v)) + "</td>" for v in vals)
                + "</tr>"
            )
        parts.append("</table></div>")
        if data.get("oracle_controls"):
            parts.append(
                '<h3>Positive output control: explicit target facts</h3><p>Same native decoder initialization and256 updates, with complete ground-truth factor vectors. This is not learned agent performance. Text accuracy requires a complete freely generated answer.</p><div class="table"><table><tr><th>Seed</th><th>Output</th><th>Known combinations</th><th>Held-out combinations</th><th>Held-out quality screen</th></tr>'
            )
            for r in data["oracle_controls"]:
                vals = [
                    r["seed"],
                    r["output"],
                    f"{r['seen']['all_accuracy']:.1%}",
                    f"{r['heldout']['all_accuracy']:.1%}",
                    r["heldout"]["gate"],
                ]
                parts.append(
                    "<tr>"
                    + "".join("<td>" + escape(str(v)) + "</td>" for v in vals)
                    + "</tr>"
                )
            parts.append("</table></div>")
        for name, label in [
            (
                "readout_summary.png",
                "Complete factor correctness; averages across two seeds and six input modes",
            ),
            (
                "readout_examples.png",
                "Representative outputs: first seed, joint native readout; held-out combinations",
            ),
        ]:
            if (directory / name).exists():
                url = (
                    "data:image/png;base64,"
                    + base64.b64encode((directory / name).read_bytes()).decode()
                )
                parts.append(
                    f'<h3>{escape(label)}</h3><img class="chart" alt="{escape(label)}" src="{url}">'
                )
        if data.get("example_source"):
            parts.append(
                "<p>Text/audio/video examples below come from "
                + escape(data["example_source"])
                + ". They show a fixed representative run, not a selected winner.</p>"
            )
    else:
        if data["stage"] == "oracle":
            parts.append(
                "<p><strong>Oracle positive control:</strong> output receives explicit ground-truth factor tokens. This measures decoder/task feasibility, not learned agent performance. The input-mode label all denotes the complete supplied facts.</p>"
            )
        if data.get("core_scores") is not None:
            parts.append(
                "<p>Training objective: "
                + escape(data.get("factor_task", "all"))
                + ". Working-context read: "
                + escape(data.get("belief_readout", "sampled"))
                + ". Continuous access is not recovery from sampled codes. "
                + "The task screen requires90% in each registered known input condition ("
                + escape(", ".join(data.get("task_input_modes", MODES)))
                + "); other conditions are transfer diagnostics. Passed: "
                + escape(
                    str(data.get("task_gate", "not recorded for this historical run"))
                )
                + ".</p>"
            )
            parts.append(
                "<h3>Core factor readout</h3><pre>"
                + escape(json.dumps(data["core_scores"], indent=2))
                + "</pre>"
            )
        parts.append(
            '<div class="table"><table><tr><th>Split</th><th>Input</th><th>Output</th><th>All factors correct</th><th>CE/MSE</th><th>Zero context</th><th>Wrong context</th><th>Screen</th></tr>'
        )
        for r in data["metrics"]:
            vals = [
                r["split"],
                r["input_mode"],
                r["output"],
                f"{r['all_accuracy']:.1%}",
                f"{r.get('teacher_ce', r.get('mse')):.5g}",
                f"{r['controls']['zero']['all_accuracy']:.1%}",
                f"{r['controls']['wrong']['all_accuracy']:.1%}",
                "missing evidence" if r["expected_missing_information"] else r["gate"],
            ]
            parts.append(
                "<tr>"
                + "".join("<td>" + escape(str(v)) + "</td>" for v in vals)
                + "</tr>"
            )
        parts.append(
            "</table></div><p>Image/audio/video factor labels are nearest canonical-template diagnostics, not evidence of realistic output. Pixel/waveform errors are separate gates. Text labels use strict free-generation parsing; CE uses teacher forcing. Without-source rows deliberately lack one fact.</p>"
        )
    if (directory / "text_examples.json").exists():
        parts.append(
            "<h3>Freely generated text · held-out combinations</h3><pre>"
            + escape((directory / "text_examples.json").read_text())
            + "</pre>"
        )
    for kind, ext, mime in [
        ("audio", "wav", "audio/wav"),
        ("video", "gif", "image/gif"),
    ]:
        for name in ("target", "output"):
            p = directory / f"readout_{kind}_{name}.{ext}"
            if not p.exists():
                continue
            url = f"data:{mime};base64," + base64.b64encode(p.read_bytes()).decode()
            tag = (
                f'<audio controls src="{url}"></audio>'
                if kind == "audio"
                else f'<img alt="Video {name}" width="160" src="{url}">'
            )
            parts.append(
                f"<p>{kind} {name} · "
                + (
                    "six separate 24-ms symbolic sequences at8kHz"
                    if kind == "audio"
                    else "first held-out example; display4fps"
                )
                + "</p>"
                + tag
            )
    parts.append(
        "<details><summary>Exact scores, resources and controls</summary><pre>"
        + escape(json.dumps(data, indent=2))
        + "</pre></details></section>"
    )
    return parts


def aggregate(root, *, oracle_root=None, probe_root=None):
    root = Path(root)
    summary = []
    children = []
    resources = []
    for seedpath in sorted(root.glob("seed*")):
        seed = int(seedpath.name[4:])
        for stage in ("frozen", "joint"):
            for variant in ("native", "adapter1", "adapter2", "adapter4"):
                files = (
                    [seedpath / f"frozen_{variant}_{k}" for k in KINDS]
                    if stage == "frozen"
                    else [seedpath / f"joint_{variant}"]
                )
                loaded = []
                for path in files:
                    d = json.loads((path / "readout.json").read_text())
                    with np.load(path / "outputs.npz") as z:
                        arr = {
                            k: z[k]
                            for k in z.files
                            if k.endswith(".factors") or k.endswith(".target.factors")
                        }
                    loaded.append((d, arr))
                    children.append(str(path.relative_to(root)))
                    resources.append(
                        dict(run=str(path.relative_to(root)), **d["resources"])
                    )
                if stage == "frozen":
                    assert len({d["resources"]["core_sha256"] for d, _ in loaded}) == 1
                for split in ("seen", "heldout"):
                    for mode in MODES:
                        preds = {}
                        metrics = {}
                        for d, arr in loaded:
                            for k in KINDS:
                                key = f"{split}.{mode}.{k}.factors"
                                if key in arr:
                                    preds[k] = arr[key]
                                    metrics[k] = next(
                                        r
                                        for r in d["metrics"]
                                        if r["split"] == split
                                        and r["input_mode"] == mode
                                        and r["output"] == k
                                    )
                            target = arr[f"{split}.target.factors"]
                        stacked = np.stack([preds[k] for k in KINDS], 1)
                        all_correct = float(
                            (stacked == target[:, None]).all((1, 2)).mean()
                        )
                        agreement = float(
                            (stacked == stacked[:, :1]).all((1, 2)).mean()
                        )
                        summary.append(
                            dict(
                                seed=seed,
                                stage=stage,
                                variant=variant,
                                split=split,
                                input_mode=mode,
                                accuracy={k: metrics[k]["all_accuracy"] for k in KINDS},
                                all_correct=all_correct,
                                agreement=agreement,
                                gate=all(metrics[k]["gate"] for k in KINDS)
                                and all_correct >= 0.7,
                            )
                        )
    benefits = []
    for stage in ("frozen", "joint"):
        for variant in ("adapter1", "adapter2", "adapter4"):
            per_seed = []
            for seed in sorted({r["seed"] for r in summary}):

                def avg(v):
                    selected = [
                        r
                        for r in summary
                        if r["seed"] == seed
                        and r["stage"] == stage
                        and r["variant"] == v
                        and r["split"] == "heldout"
                    ]
                    return np.mean([r["all_correct"] for r in selected]), {
                        k: np.mean([r["accuracy"][k] for r in selected]) for k in KINDS
                    }

                new, aa = avg(variant)
                old, bb = avg("native")
                per_seed.append(
                    dict(
                        seed=seed,
                        all_correct_delta=float(new - old),
                        per_output_delta={k: float(aa[k] - bb[k]) for k in KINDS},
                        gate=bool(
                            new - old >= 0.05
                            and min(aa[k] - bb[k] for k in KINDS) >= -0.05
                        ),
                    )
                )
            benefits.append(
                dict(
                    stage=stage,
                    variant=variant,
                    seeds=per_seed,
                    gate=all(r["gate"] for r in per_seed),
                )
            )
    data = dict(
        aggregate=True,
        summary=summary,
        benefits=benefits,
        resources=resources,
        child_reports=sorted(set(children)),
        scope="Controlled learned factors; two seeds, unequal compute; no general modality capability or architectural superiority claim.",
    )
    if oracle_root is not None:
        data["oracle_controls"] = []
        for path in sorted(Path(oracle_root).glob("seed*/*/readout.json")):
            d = json.loads(path.read_text())
            data["oracle_controls"].append(
                dict(
                    seed=d["seed"],
                    output=d["modality"],
                    seen=next(r for r in d["metrics"] if r["split"] == "seen"),
                    heldout=next(r for r in d["metrics"] if r["split"] == "heldout"),
                    source=str(path),
                    source_sha256=file_hash(path),
                )
            )
    if probe_root is not None:
        data["linear_probes"] = [
            dict(source=str(p), source_sha256=file_hash(p), **json.loads(p.read_text()))
            for p in sorted(Path(probe_root).glob("seed*/results.json"))
        ]
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    fig = Figure(figsize=(11, 7), layout="constrained")
    FigureCanvasAgg(fig)
    for i, stage in enumerate(("frozen", "joint")):
        for j, split in enumerate(("seen", "heldout")):
            ax = fig.add_subplot(2, 2, i * 2 + j + 1)
            matrix = []
            for variant in ("native", "adapter1", "adapter2", "adapter4"):
                rr = [
                    r
                    for r in summary
                    if r["stage"] == stage
                    and r["split"] == split
                    and r["variant"] == variant
                ]
                matrix.append(
                    [np.mean([r["accuracy"][k] for r in rr]) for k in KINDS]
                    + [np.mean([r["all_correct"] for r in rr])]
                )
            ax.imshow(matrix, vmin=0, vmax=1, cmap="viridis")
            ax.set_xticks(range(5), ["Text", "Image", "Audio", "Video", "All four"])
            ax.set_yticks(range(4), ["Native", "Adapter1", "Adapter2", "Adapter4"])
            ax.set_title(stage + " / " + split)
            for row in range(4):
                for col in range(5):
                    ax.text(
                        col,
                        row,
                        f"{matrix[row][col]:.0%}",
                        ha="center",
                        va="center",
                        color="white" if matrix[row][col] < 0.5 else "black",
                    )
    fig.suptitle(
        "Correct color + location + direction · averages, not independent repetitions"
    )
    fig.savefig(root / "readout_summary.png", dpi=140)
    example = sorted(root.glob("seed*/joint_native"))[0]
    data["example_source"] = str(example.relative_to(root))
    for filename in (
        "text_examples.json",
        "readout_audio_target.wav",
        "readout_audio_output.wav",
        "readout_video_target.gif",
        "readout_video_output.gif",
    ):
        if (example / filename).exists():
            shutil.copyfile(example / filename, root / filename)
    shutil.copyfile(example / "comparison.png", root / "readout_examples.png")
    atomic_json(root / "readout.json", data)
    # Analysis-only report in the same renderer, grounded in exact child records.
    identity = {
        str(p.relative_to(root)): file_hash(p)
        for p in root.glob("seed*/*/readout.json")
    }
    atomic_json(
        root / "run.json",
        dict(
            identity=dict(
                settings=dict(
                    analysis_only=True,
                    experiment="multimodal readout comparison",
                    children=identity,
                )
            )
        ),
    )
    (root / "metrics.jsonl").write_text("")
    atomic_json(
        root / "status.json",
        dict(result="complete", report="pending", step=0, error=None),
    )
    from pathwm.evaluation.report import write_report

    write_report(root)
    return data


def summarize_repair(root, reference):
    """Rebuild a read-only comparison from completed, source-bound child runs."""
    from pathwm.evaluation.report import write_report
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    root, reference = Path(root), Path(reference)
    variants = ("original", "continued", "posterior_aux", "raw_aux", "warmup")
    core, oracle, sources, reports, learning = [], [], [], [], {}
    for seed in (7201, 7202):
        for variant in variants:
            p = (
                reference / "formal" / f"seed{seed}" / "core"
                if variant == "original"
                else root / "core" / f"seed{seed}" / variant
            )
            d = json.loads((p / "readout.json").read_text())
            if json.loads((p / "status.json").read_text())["result"] != "complete":
                raise ValueError(f"Incomplete comparison source: {p}")
            sources.append(
                dict(
                    path=str(p),
                    scores_sha256=file_hash(p / "readout.json"),
                    checkpoint_sha256=file_hash(p / "last.pt"),
                )
            )
            if variant != "original":
                reports.append(str(p.relative_to(root)))
                learning[(seed, variant)] = [
                    json.loads(line)["factor_loss"]
                    for line in (p / "metrics.jsonl").read_text().splitlines()
                ]
            for split in ("seen", "heldout"):
                rows = [r for r in d["core_scores"] if r["split"] == split]
                core.append(
                    dict(
                        seed=seed,
                        variant=variant,
                        split=split,
                        factor_accuracy=np.mean(
                            [r["factor_accuracy"] for r in rows], 0
                        ).tolist(),
                        all_correct=float(np.mean([r["all_correct"] for r in rows])),
                        core_gate=d["core_gate"],
                        resources=d["resources"],
                    )
                )
        for steps, base in ((256, reference / "oracle"), (1024, root / "oracle_long")):
            for kind in KINDS:
                p = base / f"seed{seed}" / kind
                d = json.loads((p / "readout.json").read_text())
                if json.loads((p / "status.json").read_text())["result"] != "complete":
                    raise ValueError(f"Incomplete comparison source: {p}")
                sources.append(
                    dict(
                        path=str(p),
                        scores_sha256=file_hash(p / "readout.json"),
                        checkpoint_sha256=file_hash(p / "last.pt"),
                    )
                )
                if steps == 1024:
                    reports.append(str(p.relative_to(root)))
                for row in d["metrics"]:
                    oracle.append(dict(seed=seed, steps=steps, **row))
    benefits = []
    for variant in variants[2:]:
        per_seed = []
        for seed in (7201, 7202):

            def select(name):
                return next(
                    r
                    for r in core
                    if r["seed"] == seed
                    and r["variant"] == name
                    and r["split"] == "seen"
                )

            delta = (
                np.array(select(variant)["factor_accuracy"])
                - select("continued")["factor_accuracy"]
            )
            per_seed.append(
                dict(
                    seed=seed,
                    location_direction_gain=float(delta[1:].mean()),
                    color_delta=float(delta[0]),
                    gate=bool(delta[1:].mean() >= 0.05 and delta[0] >= -0.05),
                )
            )
        benefits.append(
            dict(variant=variant, seeds=per_seed, gate=all(r["gate"] for r in per_seed))
        )
    reports += [
        str(p.relative_to(root))
        for p in sorted((root / "diagnosis_fixed").glob("seed*"))
    ]
    payload = dict(
        core=core,
        oracle=oracle,
        benefits=benefits,
        reports=reports,
        sources=sources,
        scope="Exploratory controlled combinations; means across six modes and two seeds are not independent trials. No general language/media claim. Browser interaction unavailable.",
    )
    atomic_json(root / "repair.json", payload)
    atomic_json(
        root / "run.json",
        dict(
            identity=dict(
                settings=dict(
                    purpose="diagnostic",
                    source=str(reference.resolve()),
                    renderer_sha256=file_hash(__file__),
                    comparison="Stage diagnosis, duration control and three training-only interventions",
                )
            )
        ),
    )
    atomic_json(
        root / "result.json",
        dict(
            evaluation_scope="Kontrollierte kurze Texte, Pfeilbilder/-videos und symbolische Töne. Die Auswertung trennt Verbesserungen einzelner Merkmale vom vollständigen Lernziel und Oracle-Ausgaben von gelerntem Agentenverhalten. Ein positives Teilergebnis ist keine vollständige Reparatur.",
            metrics={
                "Paired known-factor benefit screens passed": [
                    r["variant"] for r in benefits if r["gate"]
                ],
                "Core capability screens passed": sum(
                    r["core_gate"]
                    for r in core
                    if r["split"] == "seen" and r["variant"] != "original"
                ),
                "Oracle1024 known-output screens passed": sum(
                    r["gate"]
                    for r in oracle
                    if r["steps"] == 1024 and r["split"] == "seen"
                ),
                "Oracle1024 new-output screens passed": sum(
                    r["gate"]
                    for r in oracle
                    if r["steps"] == 1024 and r["split"] == "heldout"
                ),
            },
            limitations=payload["scope"],
        ),
    )
    (root / "metrics.jsonl").write_text(
        "".join(
            json.dumps(
                dict(
                    step=1,
                    split="diagnostic",
                    **{k: v for k, v in r.items() if k not in ("split", "resources")},
                    population=r["split"],
                )
            )
            + "\n"
            for r in core
        )
    )
    atomic_json(
        root / "status.json",
        dict(result="complete", report="pending", step=1, error=None),
    )

    fig = Figure(figsize=(12, 9), layout="constrained")
    FigureCanvasAgg(fig)
    axes = fig.subplots(2, 2)
    for j, split in enumerate(("seen", "heldout")):
        values = np.array(
            [
                np.mean(
                    [
                        r["factor_accuracy"]
                        for r in core
                        if r["variant"] == v and r["split"] == split
                    ],
                    0,
                )
                for v in variants
            ]
        )
        ax = axes[0, j]
        ax.imshow(values, vmin=0, vmax=1, cmap="viridis", aspect="auto")
        ax.set_yticks(range(5), variants)
        ax.set_xticks(range(3), ["color", "location", "direction"])
        ax.set_title("Core / " + split + " combinations")
        for i in range(5):
            for k in range(3):
                ax.text(
                    k,
                    i,
                    f"{values[i, k]:.0%}",
                    ha="center",
                    va="center",
                    color="white" if values[i, k] < 0.5 else "black",
                )
        values = np.array(
            [
                [
                    np.mean(
                        [
                            r["all_accuracy"]
                            for r in oracle
                            if r["steps"] == steps
                            and r["output"] == kind
                            and r["split"] == split
                        ]
                    )
                    for steps in (256, 1024)
                ]
                for kind in KINDS
            ]
        )
        ax = axes[1, j]
        ax.imshow(values, vmin=0, vmax=1, cmap="viridis", aspect="auto")
        ax.set_yticks(range(4), KINDS)
        ax.set_xticks(range(2), ["256 updates", "1024 updates"])
        ax.set_title("Oracle complete-fact accuracy / " + split)
        for i in range(4):
            for k in range(2):
                ax.text(
                    k,
                    i,
                    f"{values[i, k]:.0%}",
                    ha="center",
                    va="center",
                    color="white" if values[i, k] < 0.5 else "black",
                )
    fig.suptitle(
        "Two seeds · oracle gives true facts, not learned agent state · output quality is scored separately"
    )
    fig.savefig(root / "repair_scores.png", dpi=140)
    fig = Figure(figsize=(12, 4), layout="constrained")
    FigureCanvasAgg(fig)
    axes = fig.subplots(1, 2)
    for variant in variants[1:]:
        v = np.mean([learning[(s, variant)] for s in (7201, 7202)], 0)
        axes[0].plot(
            np.arange(32, len(v) + 1),
            np.convolve(v, np.ones(32) / 32, mode="valid"),
            label=variant,
        )
    axes[0].set(
        title="Core downstream factor CE (auxiliary loss excluded)",
        xlabel="Additional updates",
        ylabel="32-update moving mean",
    )
    axes[0].legend(fontsize=8)
    for kind in KINDS:
        curves = []
        for seed in (7201, 7202):
            p = root / "oracle_long" / f"seed{seed}" / kind / "metrics.jsonl"
            curves.append(
                [json.loads(line)["loss"] for line in p.read_text().splitlines()]
            )
        v = np.mean(curves, 0)
        axes[1].plot(
            np.arange(32, len(v) + 1),
            np.convolve(v, np.ones(32) / 32, mode="valid"),
            label=kind,
        )
    axes[1].axvline(256, color="black", linestyle="--", linewidth=1)
    axes[1].set(
        title="Oracle decoder training (normalized per modality)",
        xlabel="Updates",
        ylabel="32-update moving mean",
    )
    axes[1].legend(fontsize=8)
    fig.savefig(root / "repair_learning.png", dpi=140)
    # Representative first-seed image errors, same six held-out tuples as v1.
    old = np.load(reference / "oracle/seed7201/image/outputs.npz")
    new = np.load(root / "oracle_long/seed7201/image/outputs.npz")
    select = np.arange(0, 48, 8)
    fig = Figure(figsize=(11, 7), layout="constrained")
    FigureCanvasAgg(fig)
    axes = fig.subplots(4, 6)
    target = new["heldout.target.image"][select]
    output = new["heldout.all.image.output"][select]
    for row, (label, images) in enumerate(
        (
            ("Target", target),
            ("Oracle256", old["heldout.all.image.output"][select]),
            ("Oracle1024", output),
            ("Absolute error", np.abs(output - target)),
        )
    ):
        for col, image in enumerate(images):
            axes[row, col].imshow(
                image.transpose(1, 2, 0).clip(0, 1), interpolation="nearest"
            )
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            if col == 0:
                axes[row, col].set_ylabel(label)
    fig.suptitle(
        "Held-out combinations · supplied oracle facts · targets and reconstruction errors"
    )
    fig.savefig(root / "comparison.png", dpi=140)
    return write_report(root)
