"""Controlled output metrics and panels, consumed by the ordinary run report."""

import base64
from html import escape
import itertools
import json
from pathlib import Path
import time
import wave

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
from pathwm.io import atomic_json, file_hash


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
    else:
        if data["stage"] == "oracle":
            parts.append(
                "<p><strong>Oracle positive control:</strong> output receives explicit ground-truth factor tokens. This measures decoder/task feasibility, not learned agent performance. The input-mode label all denotes the complete supplied facts.</p>"
            )
        if data.get("core_scores") is not None:
            parts.append(
                "<h3>Auxiliary core-factor readout</h3><pre>"
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


def aggregate(root):
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
