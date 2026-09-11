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
    title = (
        "Training objective and factual NLL"
        if any(r["split"].startswith("diagnostic_") for r in rows)
        else "Training and validation objective"
    )
    for split, color, style, marker in [
        ("train", "#2763a4", "-", "."),
        ("validation", "#b08418", "--", "o"),
        ("diagnostic_train", "#247b59", "--", "o"),
        ("diagnostic_development", "#b08418", ":", "s"),
    ]:
        metric = "nll" if split.startswith("diagnostic_") else "loss"
        selected = [r for r in rows if r["split"] == split and metric in r]
        if selected:
            ax.plot(
                [r["step"] for r in selected],
                [r[metric] for r in selected],
                label=split.replace("diagnostic_", "factual NLL / "),
                color=color,
                linestyle=style,
                marker=marker,
                markersize=4,
            )
    ax.set(
        xlabel="Optimizer update",
        ylabel="Objective (recipe-defined)",
        title=title,
    )
    if mobile:
        ax.set_title(title, fontsize=11)
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


def entity_inspection(directory):
    source = directory / "entity_source.json"
    if source.exists():
        data = json.loads(source.read_text())
        parts = [
            "<section><h2>Frozen source retrieval</h2>",
            f"<p>Declared gates: {'pass' if data['passed'] else 'fail'}. No training.</p>",
            "<p>A learned descriptor matcher selects the source among three records. The learned copy update remains frozen. Wrong-query controls deliberately point at the opposing-state distractor; their low accuracy is expected. Unknown queries must roll back the entire event. This does not establish learned graph discovery.</p>",
            '<div class="table"><table><tr><th>Condition</th><th>Episodes</th><th>State accuracy</th><th>Source accuracy</th><th>NLL/entity</th><th>Rejection</th><th>Integrity</th><th>Gate</th></tr>',
        ]
        for name, c in data["cohorts"].items():
            parts.append(
                f"<tr><td>{name}</td><td>{c['examples']}</td><td>{c['accuracy']:.2%}</td><td>{c['source_accuracy']:.2%}</td><td>{c['nll']:.6f}</td><td>{c['rejection_rate']:.2%}</td><td>{c['integrity']}</td><td>{c['passed']}</td></tr>"
            )
        parts.append(
            "</table></div><p>Source: entity_source.json. Lookup gates require95% state/source accuracy and NLL≤0.15; wrong-query accuracy≤5%; unknown rejection100%. Unknown-state accuracy measures preservation. Integrity covers retries, restoration, non-target preservation and batch/runtime agreement. Variants share descriptor families.</p>"
        )
        parts.append(
            f"<details><summary>Source retrieval examples</summary><pre>{escape(json.dumps(data['cohorts']['lookup']['episodes'][:2], indent=2))}</pre></details></section>"
        )
        return parts
    interaction = directory / "entity_interaction.json"
    if interaction.exists():
        data = json.loads(interaction.read_text())
        parts = [
            "<section><h2>Directed entity interaction</h2>",
            f"<p>Source latent: {'zeroed control' if data['blind'] else 'available'}. Capability gates: {'pass' if data['passed'] else 'fail'}.</p>",
            "<p>Only the interaction update trains. Recognition and ordinary state dynamics remain frozen. Directed endpoints and copy type are supplied; graph structure is not learned. Swapped roles test symmetry, not novel IDs. Families share correlated variants.</p>",
            '<div class="table"><table><tr><th>Condition</th><th>Episodes</th><th>Pair accuracy</th><th>NLL/entity</th><th>Runtime</th><th>Transactions / latents</th><th>Gate</th></tr>',
        ]
        for name, row in data["cohorts"].items():
            scores, runtime = row["scores"], row["runtime"]
            parts.append(
                f"<tr><td>{name}</td><td>{scores['examples']}</td><td>{scores['pair_accuracy']:.2%}</td><td>{scores['nll']:.6f}</td><td>{runtime['pair_accuracy']:.2%}</td><td>{runtime['transactions']} / {runtime['latent_agreement']}</td><td>{row['passed']}</td></tr>"
            )
        parts.append(
            "</table></div><p>Source: entity_interaction.json. Capability gates require95% pair accuracy and NLL≤0.15, plus runtime checks. The source-zero control is expected to fail capability gates and remain at most60% on reference histories.</p>"
        )
        parts.append(
            f"<details><summary>Interaction examples</summary><pre>{escape(json.dumps(data['cohorts']['reference']['runtime']['episodes'][:2], indent=2))}</pre></details></section>"
        )
        return parts
    temporal = directory / "entity_temporal.json"
    if temporal.exists():
        data = json.loads(temporal.read_text())
        parts = [
            "<section><h2>Frozen temporal generalization</h2>",
            (
                "<p>No-information updates: explicit identity rule for deterministic dynamics; preservation is imposed, not learned.</p>"
                if data.get("preserve_no_information")
                else "<p>No-information updates: learned recurrent transition.</p>"
            ),
            f"<p>Declared gates: {'pass' if data['passed'] else 'fail'}. No training.</p>",
            "<p>Fresh descriptor families shared across conditions. Reset order changes event order; longer-toggle and no-information conditions test different length effects. These controlled scores do not establish general belief or graph learning.</p>",
            '<div class="table"><table><tr><th>Condition</th><th>Episodes</th><th>Pair accuracy</th><th>NLL/entity</th><th>Runtime accuracy</th><th>Transactions / latents</th><th>Gate</th></tr>',
        ]
        for name, row in data["cohorts"].items():
            scores, runtime = row["scores"], row["runtime"]
            parts.append(
                f"<tr><td>{name}</td><td>{scores['examples']}</td><td>{scores['pair_accuracy']:.2%}</td><td>{scores['nll']:.6f}</td><td>{runtime['pair_accuracy']:.2%}</td><td>{runtime['transactions']} / {runtime['latent_agreement']}</td><td>{row['passed']}</td></tr>"
            )
        parts.append(
            "</table></div><p>Source: entity_temporal.json. Each condition requires95% pair accuracy, NLL≤0.15 and runtime agreement. Shared families imply correlated scores.</p></section>"
        )
        return parts
    state = directory / "entity_state.json"
    if state.exists():
        data = json.loads(state.read_text())
        parts = [
            "<section><h2>Learned persistent entity state</h2>",
            (
                "<p>No-information updates: explicit identity rule for deterministic dynamics; preservation is imposed, not learned.</p>"
                if data.get("preserve_no_information")
                else "<p>No-information updates: learned recurrent transition.</p>"
            ),
            f"<p>Declared gates: {'pass' if data['passed'] else 'fail'}.</p>",
            "<p>Frozen recognition; shared learned binary-state updates. Paired histories have identical final observations. This is controlled state tracking, not general belief or graph learning.</p>",
            '<div class="table"><table><tr><th>Population</th><th>Episodes</th><th>State-pair accuracy</th><th>NLL per entity</th></tr>',
        ]
        for name in ("train", "development"):
            row = data[name]
            parts.append(
                f"<tr><td>{name}</td><td>{row['examples']}</td><td>{row['pair_accuracy']:.2%}</td><td>{row['nll']:.6f}</td></tr>"
            )
        runtime = data["runtime"]
        parts.append(
            f"</table></div><p>Persistent runtime accuracy: {runtime['pair_accuracy']:.2%}. Retry/restore equality: {runtime['transactions']}. Runtime/training latent agreement: {runtime['latent_agreement']}.</p>"
        )
        parts.append(
            f"<details><summary>Example histories and persisted state</summary><pre>{escape(json.dumps(dict(inputs=data['manifest'][:2], outputs=runtime['episodes'][:2]), indent=2))}</pre></details></section>"
        )
        return parts
    growth = directory / "entity_growth.json"
    if growth.exists():
        data = json.loads(growth.read_text())
        parts = [
            "<section><h2>Frozen growing entity memory</h2>",
            f"<p>Declared gates: {'pass' if data['passed'] else 'fail'}. No optimizer updates.</p>",
            "<p>32 descriptor families shared across capacities; synthetic separated unit descriptors. Confidence is not calibrated. Allocation and revisit accuracy include uncertain deferrals as misses.</p>",
            '<div class="table"><table><tr><th>Capacity</th><th>Event</th><th>Correct / total</th><th>Accuracy</th><th>Uncertain</th></tr>',
        ]
        for capacity, scores in data["scores"].items():
            for kind in ("create", "revisit", "overflow"):
                row = scores[kind]
                parts.append(
                    f"<tr><td>{capacity}</td><td>{kind}</td><td>{row['correct']} / {row['count']}</td><td>{row['accuracy']:.2%}</td><td>{row['uncertain']}</td></tr>"
                )
        parts.append(
            "</table></div><p>Source: entity_growth.json. Gates require 95% per event and exact retry/restore equality. The first allocation is automatic.</p>"
        )
        parts.append(
            f"<details><summary>Exact scores and transaction checks</summary><pre>{escape(json.dumps(data['scores'], indent=2))}</pre></details></section>"
        )
        return parts
    path = directory / "entity_results.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    if data.get("task") == "matching":
        parts = [
            "<section><h2>Known versus new entity matching</h2>",
            f"<p>{escape(data['scope'])}</p>",
            f"<p>Development gates: {'pass' if data['scores']['development']['gates']['passed'] else 'fail'}. Update {data['final_step']}.</p>",
            '<div class="table"><table><tr><th>Population</th><th>N</th><th>Accuracy</th><th>NLL</th><th>Coverage</th><th>Selected error</th><th>False merge</th><th>False split</th></tr>',
        ]
        for split, scores in data["scores"].items():
            for cohort, row in scores["views"].items():
                values = [
                    split + " / " + cohort,
                    row["examples"],
                    f"{row['accuracy']:.1%}",
                    f"{row['nll']:.6f}",
                ]
                values += [
                    "—" if row[k] is None else f"{row[k]:.1%}"
                    for k in (
                        "coverage",
                        "selected_error",
                        "false_merge",
                        "false_split",
                    )
                ]
                parts.append(
                    "<tr>"
                    + "".join(f"<td>{escape(str(v))}</td>" for v in values)
                    + "</tr>"
                )
        parts.append(
            "</table></div><p>New is a classification answer; no memory record is allocated. Candidate counts and shared descriptor groups follow the recorded dataset. The margins deliberately separate known and new queries.</p>"
        )
        parts.append(
            f"<details><summary>Exact novelty gates and scores</summary><pre>{escape(json.dumps(data['scores'], indent=2))}</pre></details>"
        )
        parts.append(
            f"<details><summary>Matching inputs and predictions</summary><p>Answer order: candidate slots followed by new; padded slots are masked.</p><pre>{escape(json.dumps(data['examples'], indent=2))}</pre></details></section>"
        )
        return parts
    passed = data["scores"]["development"]["gates"]["passed"]
    parts = [
        "<section><h2>Two-object entity memory</h2>",
        f"<p><strong>Development gates: {'pass' if passed else 'fail'}.</strong> Final update {data['final_step']}.</p>",
        f"<p>{escape(data['scope'])}</p>",
        '<div class="table"><table><tr><th>Population</th><th>N</th><th>Identity</th><th>State pair</th><th>Effect pair</th><th>Mean NLL</th><th>Selection coverage</th></tr>',
    ]
    for split, measured in data["scores"].items():
        row = measured["views"]["identifiable"]
        cells = [
            split + " / identifiable",
            row["examples"],
            *[f"{row[n + '_accuracy']:.1%}" for n in ("identity", "state", "effect")],
            f"{row['nll']:.6f}",
            f"{row['coverage']:.1%}",
        ]
        parts.append(
            "<tr>" + "".join(f"<td>{escape(str(c))}</td>" for c in cells) + "</tr>"
        )
    parts.append(
        "</table></div><p>Final-view-only optimal accuracy: identity 50%; each state pair 25%. Simulator identities are not model inputs.</p>"
    )
    parts.append(
        '<h3>Ambiguous development episodes</h3><div class="table"><table><tr><th>Answer</th><th>Excess NLL above oracle</th></tr>'
    )
    row = data["scores"]["development"]["views"]["ambiguous"]
    for name in ("identity", "state", "effect"):
        parts.append(
            f"<tr><td>{name}</td><td>{row[name + '_excess_nll']:.6f}</td></tr>"
        )
    parts.append(
        "</table></div><p>These episodes hide final recognition features. Excess log loss uses the exact conditional target distribution; it is not a general calibration guarantee.</p>"
    )
    parts.append(
        f"<details><summary>Exact gates, paired controls and decision costs</summary><pre>{escape(json.dumps(data['scores'], indent=2))}</pre></details></section>"
    )
    parts.append(
        "<section><h2>Inspectable development cases</h2><p>Target probabilities and model probabilities are ordered as identity candidate 0/1 and state pairs 00/01/10/11.</p>"
    )
    for i in (0, 1, 2, 3, 16, 17, 18, 19):
        if i >= len(data["examples"]):
            continue
        example = data["examples"][i]
        parts.append(
            f"<details><summary>Case {i}: {escape(example['cohort'])}</summary><pre>{escape(json.dumps(example, indent=2))}</pre></details>"
        )
    parts.append(
        f"<details><summary>All fixed development examples</summary><pre>{escape(json.dumps(data['examples'], indent=2))}</pre></details></section>"
    )
    return parts


def fact_inspection(directory):
    path = directory / "fact_results.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    title = (
        "Facts through the agent reader"
        if data.get("reader", "direct") == "event"
        else "Direct fact extraction"
    )
    parts = [
        f"<section><h2>{title}</h2>",
        f"<p><strong>Extraction gates: {'pass' if data['gates']['extraction'] else 'fail'}.</strong> "
        f"Final update {data['final_step']}. Entity, location and joint accuracy are scored separately.</p>",
        f"<p>{escape(data['scope'])}</p>",
    ]
    if data.get("encoder_initialization"):
        parts.append(
            "<details><summary>Transferred encoder identity</summary><pre>"
            + escape(json.dumps(data["encoder_initialization"], indent=2))
            + "</pre></details>"
        )
    parts.append(
        '<div class="table"><table><tr><th>Population</th><th>N</th><th>Entity accuracy</th><th>Location accuracy</th><th>Joint accuracy</th><th>Entity NLL</th><th>Location NLL</th></tr>'
    )
    for name, r in data["views"].items():
        cells = [
            name,
            r["examples"],
            *[
                f"{r[k]:.1%}"
                for k in ("entity_accuracy", "location_accuracy", "factual_accuracy")
            ],
            f"{r['entity_nll']:.6g}",
            f"{r['location_nll']:.6g}",
        ]
        parts.append(
            "<tr>" + "".join(f"<td>{escape(str(c))}</td>" for c in cells) + "</tr>"
        )
    parts.append(
        "</table></div><p>Development holds out entity/location combinations, while every entity and location appears in training. Canonical text only.</p>"
    )
    parts.append(
        f"<details><summary>Per-entity metrics and counts</summary><pre>{escape(json.dumps(data['per_entity'], indent=2))}</pre></details></section>"
    )
    binding = data["binding"]
    parts.append("<section><h2>Two-record binding reference</h2>")
    if binding["status"] == "evaluated":
        parts.append(
            f"<p><strong>Binding gate: {'pass' if binding['gate'] else 'fail'}.</strong> "
            f"Correct before and after location swaps: {binding['coherent_swap_success']:.1%}. "
            "Each pair is queried for both entities; paired success requires both answers.</p>"
        )
        parts.append(
            '<div class="table"><table><tr><th>Constituents</th><th>Pairs</th><th>Queries</th><th>Query accuracy</th><th>Paired success</th><th>Location NLL</th><th>Single-fact joint accuracy</th></tr>'
        )
        for name in ("seen", "mixed", "held_out"):
            r = binding["groups"][name]
            cells = [
                {
                    "seen": "Both seen",
                    "mixed": "One held out",
                    "held_out": "Both held out",
                }[name],
                r["pairs"],
                r["queries"],
                f"{r['accuracy']:.1%}",
                f"{r['paired_success']:.1%}",
                f"{r['nll']:.6g}",
                f"{r['constituent_baseline']['factual_accuracy']:.1%}",
            ]
            parts.append(
                "<tr>" + "".join(f"<td>{escape(str(c))}</td>" for c in cells) + "</tr>"
            )
        parts.append("</table></div>")
    else:
        parts.append(
            "<p>Binding evaluation skipped because the extraction gates failed.</p>"
        )
    parts.append(
        f"<p>The selector is explicit and parameter-free. Pair-order invariance checks implementation, not learned binding. Swapped facts are grouped by their own training/development membership. Pair counts reuse the same 128 canonical facts and are not independent samples.</p><details><summary>Exact binding metrics, constituent baselines and swaps</summary><pre>{escape(json.dumps(binding, indent=2))}</pre></details></section>"
    )
    parts.append(
        f"<section><h2>Held-out factual examples</h2><details><summary>All fixed development examples</summary><pre>{escape(json.dumps(data['examples'], indent=2))}</pre></details></section>"
    )
    return parts


def recall_diagnostic_inspection(directory):
    path = directory / "recall_diagnostic.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    gates = data["gates"]
    parts = [
        "<section><h2>Current/recent recall diagnostic</h2>",
        f"<p><strong>{'Advance to a declared memory comparison' if gates['advance_to_memory_comparison'] else 'Stop and inspect factual learning'}.</strong> "
        f"Tiny-set fit: {'pass' if gates['tiny_set_fit'] else 'fail'}; fresh examples: {'pass' if gates['fresh_examples'] else 'fail'}.</p>",
        f"<p>Final update {data['final_step']}; raw factual probabilities. "
        "One terminal development evaluation; no checkpoint selection, calibration fitting or final-test access. "
        "Factual accuracy includes abstentions; absence is a separate factual class.</p>",
        '<div class="table"><table><tr><th>Population / group</th><th>N</th><th>Factual accuracy</th><th>NLL</th><th>Coverage</th><th>Task loss</th></tr>',
    ]
    for population, values in data["views"].items():
        for group, row in [("overall", values["overall"]), *values["groups"].items()]:
            cells = [
                f"{population} / {group}",
                row["examples"],
                f"{row['factual_accuracy']:.1%}",
                f"{row['nll']:.6g}",
                f"{row['coverage']:.1%}",
                f"{row['task_loss']:.6g}",
            ]
            parts.append(
                "<tr>" + "".join(f"<td>{escape(str(x))}</td>" for x in cells) + "</tr>"
            )
    parts.append(
        "</table></div><p>Current/recent rows contain seen entities only; absence is reported separately. Seen combines current and recent.</p>"
    )
    parts.append(
        f"<details><summary>Declared routing thresholds</summary><pre>{escape(json.dumps(gates['thresholds'], indent=2))}</pre></details>"
    )
    parts.append(
        f"<details><summary>Per-class metrics and counts</summary><pre>{escape(json.dumps({k: v['classes'] for k, v in data['views'].items()}, indent=2))}</pre></details></section>"
    )
    parts.append(
        "<section><h2>Auditable diagnostic examples</h2><p>First example per group, chosen independently of correctness. Delivered history is evaluator evidence.</p>"
    )
    shown = set()
    for example in data["examples"]:
        if example["group"] not in shown:
            shown.add(example["group"])
            parts.append(
                f"<details><summary>{escape(example['group'])}: {escape(example['query']['session_id'])}</summary><pre>{escape(json.dumps(example, indent=2))}</pre></details>"
            )
    parts.append("<p>" + " ".join(escape(x) for x in data["limits"]) + "</p></section>")
    return parts


def recall_inspection(directory):
    path = directory / "recall_results.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    fit = data["calibration_fit"]
    headline = data["views"]["calibrated"]["overall"]
    parts = [
        "<section><h2>Selective historical recall</h2>",
        f"<p><strong>{headline['answered']} answers from {headline['examples']} test episodes; "
        f"task loss {headline['task_loss']:.6g}; factual accuracy {headline['factual_accuracy']:.1%}.</strong> "
        "Factual accuracy scores the top class even when the decision abstains.</p>",
        f"<p>Selected update {data['selected_step']} by development factual NLL. "
        f"Calibration: {escape(fit['status'])}; temperature {fit['temperature']:.6g}. "
        "Raw and calibrated views reuse the same frozen test logits. Correct answers cost 0, "
        "wrong answers cost 1; abstention cost is specified in this run’s settings.</p>",
        "<p>Validation curves show factual NLL; training also includes weighted grounding and KL terms. "
        "A null answered error means no answers were emitted.</p>",
        '<div class="table"><table><tr><th>View / population</th><th>N</th><th>Coverage</th>'
        "<th>Answered error</th><th>Task loss</th><th>Factual accuracy</th><th>NLL</th></tr>",
    ]
    for view, values in data["views"].items():
        for group, row in [("overall", values["overall"]), *values["groups"].items()]:
            cells = [
                f"{view} / {group}",
                row["examples"],
                row["coverage"],
                row["answered_error"],
                row["task_loss"],
                row["factual_accuracy"],
                row["nll"],
            ]
            parts.append(
                "<tr>"
                + "".join(
                    f"<td>{escape(f'{x:.6g}' if isinstance(x, float) else str(x))}</td>"
                    for x in cells
                )
                + "</tr>"
            )
    parts.append(
        "</table></div><p>Populations overlap: seen_old combines compressed and consolidated source positions. "
        "These are reference retention groups, not claims that the model retrieved from a particular store.</p></section>"
    )
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    fig = Figure(figsize=(9, 7), layout="constrained")
    FigureCanvasAgg(fig)
    axes = fig.subplots(2, 1)
    axes[0].plot(
        [0, 1], [0, 1], color="#999999", linestyle=":", label="perfect agreement"
    )
    for view, color, style in (
        ("raw", "#2763a4", "-"),
        ("calibrated", "#b08418", "--"),
    ):
        diagnostics = data["views"][view]["diagnostics"]
        bins = [x for x in diagnostics["reliability"] if x["count"]]
        axes[0].plot(
            [x["confidence"] for x in bins],
            [x["accuracy"] for x in bins],
            color=color,
            linestyle=style,
            marker="o",
            label=view,
        )
        curve = diagnostics["risk_coverage"]
        axes[1].plot(
            [x["coverage"] for x in curve],
            [x["risk"] for x in curve],
            color=color,
            linestyle=style,
            marker=".",
            label=view,
        )
    axes[0].set(
        xlabel="Mean confidence in nonempty bin",
        ylabel="Factual accuracy",
        title="Reliability · ten fixed bins",
    )
    axes[1].set(
        xlabel="Fraction of test episodes answered",
        ylabel="Error among answers",
        title="Risk and coverage · attainable confidence thresholds",
    )
    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.2)
        ax.legend(frameon=False)
    chart = directory / "recall_confidence.png"
    fig.savefig(chart, dpi=140)
    url = "data:image/png;base64," + base64.b64encode(chart.read_bytes()).decode()
    parts.append(
        f'<section><h2>Confidence diagnostics</h2><img class="chart" alt="Reliability and risk versus coverage for raw and calibrated test probabilities" src="{url}"><p>Small held-out sample. Lines join measured points; they provide no risk guarantee. Calibration can change confidence ranking across examples.</p></section>'
    )
    parts.append(
        "<section><h2>Confidence sample counts</h2><p>Fixed confidence intervals, followed by episode count; empty bins are omitted.</p>"
    )
    for view, values in data["views"].items():
        counts = "; ".join(
            f"{b['low']:.1f}–{b['high']:.1f}: {b['count']}"
            for b in values["diagnostics"]["reliability"]
            if b["count"]
        )
        parts.append(f"<p>{escape(view)}: {counts}</p>")
    parts.append("</section>")
    parts.append(
        "<section><h2>Reference losses and cost sensitivity</h2><p>The absence-only oracle recognizes unseen entities perfectly and abstains on every seen entity. It can improve overall loss without any location recall.</p>"
    )
    for title, record in (
        ("Baselines at the declared cost", data["baselines"]),
        (
            "Costs from cached test logits",
            {k: v["costs"] for k, v in data["views"].items()},
        ),
    ):
        parts.append(
            f"<details><summary>{title}</summary><pre>{escape(json.dumps(record, indent=2))}</pre></details>"
        )
    parts.append(
        "</section><section><h2>Auditable examples</h2><p>These logs are evaluator evidence. They are not a model-accessible lookup table.</p>"
    )
    # First example of each reference group, fixed independently of correctness.
    shown = set()
    for example in data["examples"]:
        if example["group"] in shown:
            continue
        shown.add(example["group"])
        parts.append(
            f"<details><summary>{escape(example['group'])}: {escape(example['query']['session_id'])}</summary><pre>{escape(json.dumps(example, indent=2))}</pre></details>"
        )
    parts.append("<p>" + " ".join(escape(x) for x in data["limits"]) + "</p></section>")
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
    if not any(
        (directory / name).exists()
        for name in ("entity_growth.json", "entity_temporal.json", "entity_source.json")
    ):
        curve_path = directory / "learning_curve.png"
        curves(rows, curve_path)
        mobile_curve = directory / "learning_curve_mobile.png"
        curves(rows, mobile_curve, mobile=True)
        mobile_picture = (
            "data:image/png;base64,"
            + base64.b64encode(mobile_curve.read_bytes()).decode()
        )
        picture = (
            "data:image/png;base64,"
            + base64.b64encode(curve_path.read_bytes()).decode()
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
    if not any(
        (directory / name).exists()
        for name in ("entity_growth.json", "entity_temporal.json", "entity_source.json")
    ):
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
    parts.extend(recall_inspection(directory))
    parts.extend(recall_diagnostic_inspection(directory))
    parts.extend(entity_inspection(directory))
    parts.extend(fact_inspection(directory))
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
