"""Optional isolated-modality diagnostics in the existing standalone report."""

import base64
from html import escape
import json


def modality_inspection(directory):
    path = directory / "modality_audit.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    # Separate curves: text CE and waveform/pixel MSE must not form one zigzag line.
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    rows = [
        json.loads(line)
        for line in (directory / "metrics.jsonl").read_text().splitlines()
    ]
    fig = Figure(figsize=(9, 5), layout="constrained")
    FigureCanvasAgg(fig)
    for ax, kind in zip(fig.subplots(2, 2).flat, ("text", "audio", "video", "image")):
        selected = [r for r in rows if r.get("modality") == kind]
        if selected:
            ax.plot(
                range(1, len(selected) + 1),
                [r["loss"] for r in selected],
                color="#2763a4",
            )
        else:
            ax.text(
                0.5,
                0.5,
                "Frozen in this continuation",
                ha="center",
                transform=ax.transAxes,
            )
        ax.set(title=kind, xlabel="Updates to this branch", ylabel="Training objective")
        ax.grid(alpha=0.2)
    fig.savefig(directory / "modality_curves.png", dpi=130)
    chart = (
        "data:image/png;base64,"
        + base64.b64encode((directory / "modality_curves.png").read_bytes()).decode()
    )
    parts = [
        "<section><h2>Each modality tested independently</h2>",
        "<p>Four fixed examples per branch. Direct codec fitting is separate from the untrained persistent state path. These are not held-out speech, language or video capability tests.</p>",
        '<div class="table"><table><tr><th>Modality</th><th>Initial loss</th><th>Final loss</th><th>Zero context</th><th>Wrong example</th><th>Untrained state path</th><th>Learning check</th></tr>',
    ]
    for kind, m in data["metrics"].items():
        values = [
            kind,
            m.get("initial_loss"),
            m["loss"],
            m["zero_context_loss"],
            m["shuffled_context_loss"],
            m["untrained_state_path_loss"],
            m.get("learning_gate"),
        ]
        parts.append(
            "<tr>"
            + "".join(
                "<td>"
                + escape(f"{v:.6g}" if isinstance(v, float) else str(v))
                + "</td>"
                for v in values
            )
            + "</tr>"
        )
    parts.append(
        "</table></div><p>Text loss is cross-entropy; other losses are normalized pixel/sample MSE. Absolute values across modalities are not comparable. Video reconstructs observed frames; its state-path score uses the last observed frame.</p>"
    )
    parts.append(
        f'<img class="chart" alt="Separate training curves for each modality" src="{chart}">'
    )
    text = data["metrics"]["text"]
    parts.append(
        '<h3>Free text generation</h3><div class="table"><table><tr><th>Target</th><th>Generated</th><th>Zero-context generation</th></tr>'
    )
    for row in zip(text["target"], text["generated"], text["zero_context_generated"]):
        parts.append(
            "<tr>"
            + "".join("<td>" + escape(value) + "</td>" for value in row)
            + "</tr>"
        )
    parts.append(
        "</table></div><h3>Audio</h3><p>8 kHz, four 256-sample chunks concatenated (128 ms total). Synthetic tones, not speech. Chunk boundaries are not modeled as a continuous recording.</p>"
    )
    for stem in ["audio_target", "audio_output"]:
        file = directory / (stem + ".wav")
        if file.exists():
            url = (
                "data:audio/wav;base64," + base64.b64encode(file.read_bytes()).decode()
            )
            parts.append(
                f'<p>{escape(stem)} <audio controls preload="none" src="{url}"></audio></p>'
            )
    parts.append('<h3>Observed video reconstruction</h3><div class="gallery">')
    for stem in ["video_target", "video_output"]:
        file = directory / (stem + ".gif")
        if file.exists():
            url = (
                "data:image/gif;base64," + base64.b64encode(file.read_bytes()).decode()
            )
            parts.append(
                f'<figure><img alt="{escape(stem)}" src="{url}"><figcaption>{escape(stem)} · first example, 4 fps</figcaption></figure>'
            )
    parts.append(
        "</div><details><summary>Real input shapes, exact scores and limitations</summary><pre>"
        + escape(json.dumps(data, indent=2))
        + "</pre></details></section>"
    )
    return parts
