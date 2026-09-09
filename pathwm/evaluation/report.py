"""Portable run report from raw rows and saved examples; no app/plugin or network.

Chart contract: training/validation objective versus optimizer update, line with
markers (table-only when no points), explicit split legend, neutral grid, blue/
gold plus solid/dashed distinctions. RGB panels share [0,1]. Feature PCA fits one
basis per named level across displayed examples, clips at 2/98 percentiles and is
labelled as a projection, not semantic evidence. PNGs are standalone exports;
HTML embeds their exact bytes and exposes settings and raw rows.
"""

import base64
from html import escape
import io
import os
import tempfile
import json
from pathlib import Path
import numpy as np
from PIL import Image
from pathwm.io import atomic_json, file_hash


STYLE = """body{font:16px system-ui,sans-serif;color:#202c39;background:#f6f7f9;margin:0}
main{max-width:1080px;margin:auto;padding:28px}h1{font-size:30px;margin-bottom:8px}h2{font-size:21px;margin-top:32px}
p{line-height:1.55}section{background:white;padding:20px;margin:20px 0;border:1px solid #dfe4ea;border-radius:8px}
.tag{display:inline-block;padding:6px 12px;background:#e6edf6;font-weight:600}img.chart{width:100%;height:auto}
.gallery{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:12px}figure{margin:0}
figure img{width:100%;image-rendering:pixelated;border:1px solid #ddd}figcaption{font-size:13px;line-height:1.4}
pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}.table{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px}
th,td{text-align:left;border-bottom:1px solid #e4e7eb;padding:8px}td{font-variant-numeric:tabular-nums}
summary{cursor:pointer;font-weight:600}code{overflow-wrap:anywhere}@media(max-width:600px){main{padding:14px}section{padding:12px}h1{font-size:24px}}"""


def image_url(array):
    x = np.asarray(array)
    if x.ndim == 3 and x.shape[0] in (1, 3):
        x = x.transpose(1, 2, 0)
    if x.ndim == 3 and x.shape[-1] == 1:
        x = x[..., 0]
    image = Image.fromarray(np.rint(np.clip(x, 0, 1) * 255).astype("uint8"))
    out = io.BytesIO()
    image.save(out, format="PNG")
    return "data:image/png;base64," + base64.b64encode(out.getvalue()).decode()


def curves(rows, path, mobile=False):
    os.environ.setdefault(
        "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "pathwm-matplotlib")
    )
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    fig = Figure(figsize=(4.2 if mobile else 10, 3.5), layout="constrained")
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    for split, color, style, marker in [
        ("train", "#2763a4", "-", "."),
        ("validation", "#b08418", "--", "o"),
    ]:
        selected = [r for r in rows if r["split"] == split and "loss" in r]
        if selected:
            ax.plot(
                [r["step"] for r in selected],
                [r["loss"] for r in selected],
                label=split,
                color=color,
                linestyle=style,
                marker=marker,
                markersize=4,
            )
    ax.set(
        xlabel="Optimizer update",
        ylabel="Objective (recipe-defined)",
        title="Training and validation objective",
    )
    if mobile:
        ax.set_title("Training and validation objective", fontsize=11)
        ax.tick_params(labelsize=9)
    ax.grid(alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    if rows:
        ax.legend(frameon=False)
    fig.savefig(path, dpi=140)


def pca_maps(array):
    b, c, h, w = array.shape
    values = array.transpose(0, 2, 3, 1).reshape(-1, c).astype("float64")
    mean = values.mean(0)
    values -= mean
    _, singular, vectors = np.linalg.svd(values, full_matrices=False)
    basis = vectors[: min(3, c)]
    # Deterministic component sign for visual consistency on rerender.
    for v in basis:
        if v[np.abs(v).argmax()] < 0:
            v *= -1
    color = values @ basis.T
    if color.shape[1] < 3:
        color = np.pad(color, ((0, 0), (0, 3 - color.shape[1])))
    low, high = np.percentile(color, [2, 98], axis=0)
    color = np.clip((color - low) / np.maximum(high - low, 1e-10), 0, 1)
    return color.reshape(b, h, w, 3), {
        "mean": mean,
        "basis": basis,
        "low": low,
        "high": high,
        "explained_variance_ratio": singular[:3] ** 2
        / max(float((singular**2).sum()), 1e-15),
    }


def model_inspection(directory):
    """Optional multimodal diagnostics supplied by the same recipe/run, no inference."""
    meta_path, arrays_path = directory / "inspection.json", directory / "inspection.npz"
    if not meta_path.exists() or not arrays_path.exists():
        return []
    meta = json.loads(meta_path.read_text())
    with np.load(arrays_path, allow_pickle=False) as archive:
        attention, activity = archive["attention"], archive["token_activity"]
    parts = [
        "<section><h2>Inside the multimodal model</h2>",
        "<p>Observed inputs → modality encoders → latent state → memory and thinking → imagined states → decoders and action scoring.</p>",
        "<p>These are development measurements. Attention and activation magnitude do not establish what a feature means.</p>",
        '<div class="table"><table><tr><th>Latent group</th><th>Tokens</th><th>Activation RMS</th><th>Image change after zeroing group (MSE)</th></tr>',
    ]
    for name, count in meta["latent_groups"].items():
        change = meta["zero_group_image_change_mse"].get(name)
        change_text = "not applied" if change is None else f"{change:.6g}"
        parts.append(
            f"<tr><td>{escape(name)}</td><td>{count}</td><td>{meta['group_activity_rms'][name]:.6g}</td><td>{change_text}</td></tr>"
        )
    parts.append(
        "</table></div><p>The intervention replaces one group with zero and measures the change in the current image decoder. This is a local sensitivity test, not proof that the group has its intended semantic role.</p>"
    )
    for value, title, description in [
        (
            attention,
            "Observation attention",
            "Rows: latent queries. Columns: the recorded observation inputs and optional context/null keys. Heads are averaged for example 1.",
        ),
        (
            activity,
            "Latent activation magnitude",
            "Rows: latent tokens in the model layout recorded by the recipe. Columns: feature dimensions. Absolute values for example 1.",
        ),
    ]:
        high = max(float(np.max(value)), 1e-12)
        parts.append(
            f'<h3>{title}</h3><p>{description} Grayscale runs from 0 (black) to {high:.6g} (white).</p><img class="chart" style="max-width:620px;image-rendering:pixelated" alt="{title}" src="{image_url(value / high)}">'
        )
    parts.append(
        "</section><section><h2>Generated modalities and imagined actions</h2><p>Untrained or briefly trained outputs are shown as produced, including incorrect text and noisy media.</p>"
    )
    gif = directory / "imagined.gif"
    if gif.exists():
        url = "data:image/gif;base64," + base64.b64encode(gif.read_bytes()).decode()
        parts.append(
            f'<figure><img style="width:160px" alt="Imagined future sequence preview" src="{url}"><figcaption>Example 1: imagined video preview. Playback timing is illustrative; state times are recorded below.</figcaption></figure>'
        )
    wav = directory / "imagined.wav"
    if wav.exists():
        url = "data:audio/wav;base64," + base64.b64encode(wav.read_bytes()).decode()
        parts.append(
            f'<p>Predicted waveform chunks, concatenated at {meta["audio_sample_rate"]} Hz. Each state produces {meta["audio_samples_per_state"]} samples.</p><audio style="max-width:100%" controls preload="none" src="{url}">Audio playback is unavailable in this browser.</audio>'
        )
    parts.append(
        "<p>Generated text for the displayed examples:</p><pre>"
        + escape("\n".join(meta["generated_text"]))
        + "</pre>"
    )
    for label, value in [
        ("Environmental time after observation", meta["time"]),
        ("Imagined future times", meta["future_times"]),
        ("Memory provenance", meta["memory_sources"]),
        ("Candidate scores (lower is better)", meta["plan_scores"]),
        ("Selected candidate per example", meta["selected_candidates"]),
    ]:
        parts.append(
            f"<p><strong>{label}</strong></p><pre>{escape(json.dumps(value, indent=2))}</pre>"
        )
    parts.append(
        "<details><summary>Inspection metadata and limits</summary><pre>"
        + escape(json.dumps(meta, indent=2))
        + "</pre></details></section>"
    )
    return parts


def render_report(directory):
    directory = Path(directory)
    record = json.loads((directory / "run.json").read_text())
    status = json.loads((directory / "status.json").read_text())
    rows = [
        json.loads(s)
        for s in (directory / "metrics.jsonl").read_text().splitlines()
        if s
    ]
    curve_path = directory / "learning_curve.png"
    curves(rows, curve_path)
    mobile_curve = directory / "learning_curve_mobile.png"
    curves(rows, mobile_curve, mobile=True)
    mobile_picture = (
        "data:image/png;base64," + base64.b64encode(mobile_curve.read_bytes()).decode()
    )
    picture = (
        "data:image/png;base64," + base64.b64encode(curve_path.read_bytes()).decode()
    )
    title = escape(directory.name)
    parts = [
        f'<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{title} — PATH-WM</title><style>{STYLE}</style><main>',
        f'<h1>{title}</h1><p class="tag">{escape(status["result"])} · update {status["step"]}</p>',
        (
            "<p>Development experiment. These results check the training workflow; they do not establish reliable world-model control.</p>"
            if record["identity"]["settings"].get("purpose", "development")
            == "development"
            else "<p>Recorded metrics follow the declared recipe, population and budget. Prediction and control require their own evidence.</p>"
        ),
    ]
    if status.get("error"):
        parts.append(f"<p><strong>Failure:</strong> {escape(status['error'])}</p>")
    parts.append(
        f'<section><picture><source media="(max-width: 600px)" srcset="{mobile_picture}"><img class="chart" alt="Training and validation objective by optimizer update" src="{picture}"></picture><p>Source: metrics.jsonl. Validation population and weighted loss terms are fixed by this run’s recipe.</p></section>'
    )
    validation = [r for r in rows if r["split"] == "validation"]
    if validation:
        parts.append(
            '<section><h2>Latest validation values</h2><div class="table"><table><tr><th>Metric</th><th>Value</th></tr>'
        )
        for k, v in validation[-1].items():
            parts.append(f"<tr><td>{escape(k)}</td><td>{escape(str(v))}</td></tr>")
        parts.append("</table></div></section>")
    example_path = directory / "examples.npz"
    if example_path.exists():
        meta_path = directory / "examples.meta.json"
        example_step = (
            json.loads(meta_path.read_text())["step"]
            if meta_path.exists()
            else "unrecorded"
        )
        parts.append(
            f"<p>Image and feature examples from update {escape(str(example_step))}.</p>"
        )
        with np.load(example_path, allow_pickle=False) as z:
            examples = {k: z[k] for k in z.files}
        for key, label in [
            (
                "input",
                "Target images at the final horizon"
                if "horizon" in record["identity"]["settings"]
                else "Input images",
            ),
            (
                "rgb",
                "Rendered predicted state"
                if "horizon" in record["identity"]["settings"]
                else "Reconstructed images",
            ),
            ("mask", "Predicted foreground probability"),
            ("target_mask", "Foreground labels"),
        ]:
            if key not in examples:
                continue
            parts.append(f'<section><h2>{label}</h2><div class="gallery">')
            for i, x in enumerate(examples[key]):
                parts.append(
                    f'<figure><img alt="{label}, example {i + 1}" src="{image_url(x)}"><figcaption>Example {i + 1}</figcaption></figure>'
                )
            parts.append("</div></section>")
        axes = {}
        for key, value in examples.items():
            if not key.startswith("feature_"):
                continue
            name = key.removeprefix("feature_")
            maps, axis = pca_maps(value)
            axes.update({name + "_" + k: v for k, v in axis.items()})
            parts.append(
                f'<section><h2>{escape(name)} features · PCA</h2><p>One basis across these examples, independently fitted for this level. Colors are projection coordinates, not semantic labels; each channel clips the 2nd–98th percentile. Variance fractions are saved in pca_axes.npz. Shape: {escape(str(value.shape))}.</p><div class="gallery">'
            )
            for i, x in enumerate(maps):
                parts.append(
                    f'<figure><img alt="{escape(name)} PCA example {i + 1}" src="{image_url(x)}"><figcaption>Example {i + 1}</figcaption></figure>'
                )
            parts.append("</div></section>")
        if axes:
            np.savez_compressed(directory / "pca_axes.npz", **axes)
    parts.extend(model_inspection(directory))
    for title, data in [
        ("Resolved settings and source identities", record),
        ("Exact metric rows", rows),
    ]:
        parts.append(
            f"<section><details><summary>{title}</summary><pre>{escape(json.dumps(data, indent=2))}</pre></details></section>"
        )
    parts.append("</main></html>")
    html = "".join(parts)
    if "<script" in html or 'src="http' in html:
        raise ValueError("Report must be self-contained and script-free")
    temporary = directory / "report.partial.html"
    temporary.write_text(html)
    temporary.replace(directory / "report.html")
    atomic_json(
        directory / "report.qa.json",
        {
            "status": "structural_verified",
            "report_sha256": file_hash(directory / "report.html"),
            "rows": len(rows),
            "self_contained": True,
            "browser_checked": False,
            "renderer_sha256": file_hash(__file__),
        },
    )
    status["report"] = "structural_verified"
    atomic_json(directory / "status.json", status)
    return directory / "report.html"


def write_report(directory, batch=None, outputs=None, features=None):
    directory = Path(directory)
    if batch is not None:
        arrays = {"input": batch["rgb"].detach().cpu().numpy()}
        for k, v in (outputs or {}).items():
            arrays[k] = (v.sigmoid() if k == "mask" else v).detach().cpu().numpy()
        if "mask" in batch:
            arrays["target_mask"] = batch["mask"].detach().cpu().numpy()
        arrays.update(
            {
                "feature_" + k: v.detach().cpu().numpy()
                for k, v in (features or {}).items()
            }
        )
        np.savez_compressed(directory / "examples.npz", **arrays)
        atomic_json(
            directory / "examples.meta.json",
            {
                "step": json.loads((directory / "status.json").read_text())["step"],
                "sha256": file_hash(directory / "examples.npz"),
            },
        )
    return render_report(directory)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Rebuild an offline report from saved results"
    )
    parser.add_argument("run", type=Path)
    print(render_report(parser.parse_args().run))
