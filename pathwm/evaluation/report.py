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
    key_box = directory / "key_box.json"
    if key_box.exists():
        data = json.loads(key_box.read_text())
        parts = [
            "<section><h2>Integrated key-and-box task</h2>",
            f"<p>Declared screen: {'pass' if data['passed'] else 'fail'}. Known initial content accuracy: {data['known_accuracy']:.2%}.</p>",
            "<p>Entity latents enter the actual belief-agent thinking workspace. Planner uses supplied task mechanics, not learned imagined dynamics. Inspect/open/retrieve feedback is committed after execution. No-history control removes historical store and agent state before the action loop.</p>",
            '<div class="table"><table><tr><th>Policy</th><th>Reachable success</th><th>Absent correct stop</th><th>Utility</th><th>Mean cost</th></tr>',
        ]
        for policy, scores in data["summary"].items():
            parts.append(
                f"<tr><td>{policy}</td><td>{scores['success']:.2%}</td><td>{scores['absent_stop']:.2%}</td><td>{scores['utility']:.6f}</td><td>{scores['mean_cost']:.4f}</td></tr>"
            )
        parts.append(
            "</table></div><p>Acceptance: known accuracy≥95%, reachable success and absent correct stop≥90%, utility advantage over no-history≥0.01; supplied-state control100%. All required. Four-action budget. Utility=verified success−0.05×external cost.</p>"
        )
        if "reference" in data:
            parts.append(
                "<p>Frozen checkpoint comparison on the same fresh cases:</p><pre>"
                + escape(
                    json.dumps(
                        dict(
                            known_accuracy=data["reference"]["known_accuracy"],
                            summary=data["reference"]["summary"],
                        ),
                        indent=2,
                    )
                )
                + "</pre>"
            )
        if isinstance(data.get("stress"), dict):
            stress = data["stress"]
            parts.append(
                "<h3>Long history and mid-episode relocation</h3>"
                + f"<p>Combined screen: {'pass' if data['robustness_passed'] else 'fail'}. "
                + "Stress requires known and correction read accuracy≥95%, reachable/absent≥90%, oracle100%. Utility is descriptive. One correction before decision1; not arbitrary change timing.</p><pre>"
                + escape(
                    json.dumps(
                        dict(
                            passed=stress["passed"],
                            correction_accuracy=stress["correction_accuracy"],
                            correction_reads=stress["correction_reads"],
                            summary=stress["summary"],
                            reference=data.get("reference", {})
                            .get("stress", {})
                            .get("summary"),
                        ),
                        indent=2,
                    )
                )
                + "</pre>"
                + "<details><summary>State-change example</summary><pre>"
                + escape(
                    json.dumps(
                        next(
                            r
                            for r in stress["episodes"]
                            if r["policy"] == "integrated"
                            and r["condition"] == "remembered"
                        ),
                        indent=2,
                    )
                )
                + "</pre></details>"
            )
        for condition in ("remembered", "moved", "uncertain", "absent"):
            row = next(
                r
                for r in data["episodes"]
                if r["condition"] == condition and r["policy"] == "integrated"
            )
            parts.append(
                "<details><summary>"
                + escape(condition)
                + " example</summary><pre>"
                + escape(json.dumps(row, indent=2))
                + "</pre></details>"
            )
        parts.append(
            "<p>Same task templates with fresh descriptors; not broad transfer. Raw results: key_box.json.</p></section>"
        )
        return parts
    diagnosis = directory / "entity_source_diagnosis.json"
    if diagnosis.exists():
        data = json.loads(diagnosis.read_text())
        parts = [
            "<section><h2>Source-change trace diagnosis</h2><p>Saved variance-aware trajectories only. No policy change or new evaluation. Categories describe check availability, not causal explanations. Pure means all rewards in the recent block were acquired after the evaluation boundary; older values may still include calibration.</p>"
        ]
        parts.append("<pre>" + escape(json.dumps(data["summary"], indent=2)) + "</pre>")
        parts.append(
            '<div class="table"><table><tr><th>World</th><th>Condition</th><th>Source</th><th>Acquired</th><th>Mixed checks</th><th>Eligible pure checks</th><th>First pure check</th><th>First reset</th><th>Category</th></tr>'
        )
        for row in data["sources"]:
            values = [
                row["world"],
                "drift" if row["swapped"] else "static",
                row["source"],
                row["acquired"],
                row["mixed_checks"],
                row["pure_checks"],
                row["first_pure_check_case"],
                row["first_reset_case"],
                row["category"],
            ]
            parts.append(
                "<tr>"
                + "".join("<td>" + escape(str(v)) + "</td>" for v in values)
                + "</tr>"
            )
        parts.append(
            "</table></div><p>Case indices are zero-based after calibration. None means no event. Every threshold crossing agrees with the recorded reset. Raw gaps and thresholds: entity_source_diagnosis.json. This is integrity verification, not a capability pass.</p></section>"
        )
        return parts
    drift = directory / "entity_source_drift.json"
    if drift.exists():
        data = json.loads(drift.read_text())
        if "coverage" in data:
            parts = [
                "<section><h2>Matched-budget source coverage</h2>",
                f"<p>Declared screen: {'pass' if data['passed'] else 'fail'}.</p>",
                "<p>Both policies acquire on every deferred case, cost0.055 including feedback. Same epsilon0.5 coins; exploratory source is random versus lower lifetime acquisition count. Exploitation and detector (z2,block32) are shared. This is conditional on forced acquisition, not the original stopping policy.</p>",
                '<div class="table"><table><tr><th>Condition</th><th>Policy</th><th>Early utility</th><th>Late utility</th><th>Whole utility</th></tr>',
            ]
            for condition, policies in data["summary"].items():
                for name, values in policies.items():
                    parts.append(
                        f"<tr><td>{condition}</td><td>{name}</td><td>{values['early_utility']:.6f}</td><td>{values['late_utility']:.6f}</td><td>{values['utility']:.6f}</td></tr>"
                    )
            parts.append("</table></div>")
            parts.append(
                f"<p>Drift sources reaching an eligible pure-feedback check: {escape(str(data['coverage']['pure_source_counts']))}. Coverage static reset episodes: {data['detector']['static_reset_episode_rate']:.2%}.</p>"
            )
            parts.append(
                "<p>Acceptance: late drift gain≥0.01; whole drift loss≤0.01; static loss≤0.02; static reset episodes≤25%; strictly more sources reaching pure checks. All criteria required. Raw action counts and diagnostic check records are in entity_source_drift.json. Coverage alone is not utility improvement.</p></section>"
            )
            return parts
        triggered = "detector" in data
        candidate = data.get("detector", {}).get("candidate", "triggered")
        parts = [
            "<section><h2>Online source drift</h2>",
            f"<p>Declared adaptation screen: {'pass' if data['passed'] else 'fail'}.</p>",
            (
                "<p>Opaque sources silently swap quality after calibration; matched unchanged-source control. Online means update only after acquired outcomes, with epsilon0.2 exploration. Triggered forgetting compares disjoint32-outcome blocks with older retained rewards (absolute mean difference≥0.15). Both it and window32 are engineered, not learned change detection. Acquisition costs0.05 and online outcome feedback0.005.</p>"
                if triggered
                else "<p>Unconditional window32 versus cumulative/frozen/no-feedback controls; acquisition0.05 and feedback0.005.</p>"
            ),
            '<div class="table"><table><tr><th>Condition</th><th>Policy</th><th>Early utility</th><th>Late utility</th><th>Post-calibration utility</th><th>Combined utility</th></tr>',
        ]
        for condition, policies in data["summary"].items():
            for name, v in policies.items():
                parts.append(
                    f"<tr><td>{condition}</td><td>{name}</td><td>{v['early_utility']:.6f}</td><td>{v['late_utility']:.6f}</td><td>{v['utility']:.6f}</td><td>{v['combined_utility']:.6f}</td></tr>"
                )
        parts.append("</table></div>")
        if "selection" in data:
            parts.append(
                f"<p>Variance-aware policy: z={data['selection']['selected_z']}; development qualified={data['selection']['qualified']}. Threshold=max(0.15,z√(s²_old/n_old+s²_recent/32+0.0001)). This heuristic has no sequential confidence guarantee. Select smallest z with stationary reset rate≤12.5% on8 development worlds; freeze before16 held-out worlds.</p>"
            )
            for trial in data["development"]:
                parts.append(
                    f"<p>Development z={trial['detector']['change_z']}: reset episodes {trial['detector']['static_reset_episode_rate']:.2%}.</p>"
                )
        if triggered:
            parts.append(
                f"<p>Candidate: {candidate}. Static episodes with post-calibration resets: {data['detector']['static_reset_episode_rate']:.2%} (limit25%).</p>"
            )
            parts.append(
                '<div class="table"><table><tr><th>World</th><th>Condition</th><th>Calibration resets</th><th>Post resets</th><th>First post reset (case, zero-based)</th></tr>'
            )
            for row in data["episodes"]:
                if row["policy"] == candidate:
                    when = row["first_reset_case"]
                    parts.append(
                        f"<tr><td>{row['world']}</td><td>{'drift' if row['swapped'] else 'static'}</td><td>{row['calibration_resets']}</td><td>{row['post_resets']}</td><td>{when if when is not None else 'none'}</td></tr>"
                    )
            parts.append(
                "</table></div><p>Reset timing is not proof of correct change identification; unselected sources supply no feedback.</p>"
            )
        parts.append(
            (
                "<p>Late is the last128 of256 post-calibration cases. Acceptance: candidate late drift utility≥frozen+0.01 and cumulative+0.01, whole drift≥frozen−0.01, static≥frozen−0.02; static episodes with resets≤25%; development selection must qualify when present. Same exploration coins and candidate actions across online policies. Feedback availability and fixed change timing are supplied assumptions. Source: entity_source_drift.json.</p></section>"
                if triggered
                else "<p>Acceptance: window late drift≥frozen+0.01 and cumulative+0.01; whole drift≥frozen−0.01; static≥frozen−0.02.</p></section>"
            )
        )
        return parts
    choice = directory / "entity_source_choice.json"
    if choice.exists():
        data = json.loads(choice.read_text())
        parts = [
            "<section><h2>Outcome-trained source choice</h2>",
            f"<p>Declared screen: {'pass' if data['passed'] else 'fail'}.</p>",
            "<p>A two-action value table learns from selected calibration outcomes. Perception is frozen; both sources cost0.05. Hidden source properties swap between16 worlds. No learned reliability head or persistent-memory integration.</p>",
            '<div class="table"><table><tr><th>Policy</th><th>Evaluation utility</th><th>Accuracy</th><th>Accept recall</th><th>Ignore recall</th><th>Acquisition rate</th></tr>',
        ]
        for name, v in data["summary"].items():
            parts.append(
                f"<tr><td>{name}</td><td>{v['utility']:.6f}</td><td>{v['accuracy']:.2%}</td><td>{v['positive_recall']:.2%}</td><td>{v['negative_recall']:.2%}</td><td>{v['acquisition_rate']:.2%}</td></tr>"
            )
        parts.append(
            f"</table></div><p>Useful source selected: {data['useful_source_rate']:.2%}. Mean exploration cost: {data['calibration_cost_mean']:.4f} accuracy units per world. Combined calibration/evaluation utility: {data['combined_utility']:.6f}; stop baseline: {data['combined_stop']:.6f}.</p>"
        )
        parts.append(
            "<p>Acceptance: evaluation utility≥best fixed source+0.01 and above stop, useful source≥75%, combined utility≥stop. Each world has512 calibration and256 evaluation cases. Feedback requires known task outcomes; source quality stays static. Source: entity_source_choice.json.</p>"
        )
        examples = [
            {
                k: r[k]
                for k in (
                    "world",
                    "choice",
                    "useful_source",
                    "values",
                    "permutation",
                    "scores",
                )
            }
            for r in data["worlds"][:2]
        ]
        parts.append(
            f"<details><summary>Source learning examples</summary><pre>{escape(json.dumps(examples, indent=2))}</pre></details></section>"
        )
        return parts
    evidence = directory / "entity_evidence_sources.json"
    if evidence.exists():
        data = json.loads(evidence.read_text())
        parts = [
            "<section><h2>Alternate evidence acquisition</h2>",
            f"<p>Declared comparison: {'pass' if data['passed'] else 'fail'}.</p>",
            "<p>Fixed source policies; same-source rho0.9 costs0.02, alternate rho0 costs0.05. Utility is accuracy minus cost per extra observation. Correlation is an environment assumption, never a policy input. No learned source choice or reliability estimation.</p>",
            '<div class="table"><table><tr><th>Source</th><th>Noise</th><th>Strategy</th><th>Accuracy</th><th>Accept recall</th><th>Ignore recall</th><th>Reread rate</th><th>Utility</th></tr>',
        ]
        for source, result in data["sources"].items():
            for noise, c in result["cohorts"].items():
                for strategy, v in c["strategies"].items():
                    parts.append(
                        f"<tr><td>{source}</td><td>{noise}</td><td>{strategy}</td><td>{v['accuracy']:.2%}</td><td>{v['positive_recall']:.2%}</td><td>{v['negative_recall']:.2%}</td><td>{v['reread_rate']:.2%}</td><td>{v['utility']:.6f}</td></tr>"
                    )
        parts.append(
            f"</table></div><p>High-noise paired utility delta: {data['paired_utility_delta']:.6f}; descriptive paired bootstrap95% interval: {data['paired_bootstrap_95']}. Break-even alternate cost against selective same-source: {data['break_even_alternate_cost']}. Bootstrap resamples128 underlying pairs, not severity rows.</p>"
        )
        parts.append(
            "<p>Acceptance requires ≥2-point high-noise accuracy gain, positive utility gains over same-source and first-only, ignore loss≤2 points and low-noise loss≤1 point. Source: entity_evidence_sources.json. Marginal noise matched; both labels balanced. Results are conditional on supplied static sensor properties.</p></section>"
        )
        return parts
    shift = directory / "entity_gate_shift.json"
    if shift.exists():
        data = json.loads(shift.read_text())
        parts = [
            "<section><h2>Frozen gate noise sensitivity</h2>",
            f"<p>{'Reobservation' if data.get('reobserve') else 'Robustness'} gate: {'pass' if data['passed'] else 'fail'}. No training.</p>",
            "<p>Labels follow underlying context identity; noise affects observations. All severities share128 prototype pairs. High-noise errors can reflect ambiguity; this does not diagnose a unique model defect. Probability scores are not calibrated beliefs.</p>",
            '<div class="table"><table><tr><th>Noise</th><th>Threshold</th><th>Accuracy</th><th>Accept recall</th><th>Ignore recall</th><th>Brier</th></tr>',
        ]
        for name, c in data["cohorts"].items():
            for threshold, v in c["thresholds"].items():
                parts.append(
                    f"<tr><td>{name}</td><td>{threshold}</td><td>{v['accuracy']:.2%}</td><td>{v['positive_recall']:.2%}</td><td>{v['negative_recall']:.2%}</td><td>{c['brier']:.6f}</td></tr>"
                )
        parts.append(
            "</table></div><p>Source: entity_gate_shift.json. Primary gate requires both class recalls≥95% at threshold0.5 in every severity. Other thresholds are descriptive. Distance baseline accepts below0.5 Euclidean distance; it is not an oracle.</p>"
        )
        examples = {
            name: dict(
                distance_baseline=c["distance_baseline"],
                noise_to_separation=c["noise_to_separation"][:4],
                probabilities=c["probability"][:4],
            )
            for name, c in data["cohorts"].items()
        }
        parts.append(
            f"<details><summary>Baseline and paired examples</summary><pre>{escape(json.dumps(examples, indent=2))}</pre></details></section>"
        )
        if "correlation" in data:
            parts.append(
                f"<p>Raw-noise correlation parameter: {data['correlation']}. Rho1 is an unchanged-decision endpoint check, not a gain requirement. Matched first observations and innovations; static context. Diagnostics: {escape(json.dumps(data['noise_diagnostics']))}.</p>"
            )
        if data.get("reobserve"):
            parts.append(
                "<section><h2>Defer and reobserve</h2><p>Supplied policy: reread once at probability0.2–0.8, then average unit cues. Unchanged context is assumed; noise independence applies only when correlation is zero. Cost0.02 per extra observation; no calibration or learned sensing claim.</p><div class=table><table><tr><th>Noise</th><th>Strategy</th><th>Accuracy</th><th>Accept recall</th><th>Ignore recall</th><th>Reread rate</th><th>Utility</th></tr>"
            )
            for noise, c in data["cohorts"].items():
                for name, v in c["strategies"].items():
                    parts.append(
                        f"<tr><td>{noise}</td><td>{name}</td><td>{v['accuracy']:.2%}</td><td>{v['positive_recall']:.2%}</td><td>{v['negative_recall']:.2%}</td><td>{v['reread_rate']:.2%}</td><td>{v['utility']:.6f}</td></tr>"
                    )
            parts.append(
                "</table></div><p>Reobservation pass requires ≥2-point high-noise gain, positive net utility gain, ignore loss≤2 points and low-noise accuracy loss≤1 point. Duplicate decisions must be identical. The robustness criterion above applies to the first-observation baseline only.</p></section>"
            )
        return parts
    gate = directory / "entity_gate.json"
    if gate.exists():
        data = json.loads(gate.read_text())
        parts = [
            "<section><h2>Context write gate</h2>",
            f"<p>Declared gates: {'pass' if data['passed'] else 'fail'}.</p>",
            "<p>Only context comparison trains through frozen source-selection loss. Near-matching versus separated contexts define relevance; ambiguous contexts and semantic discovery are not established. Correlated cases share descriptor families.</p>",
            '<div class="table"><table><tr><th>Condition</th><th>State accuracy</th><th>Source accuracy</th><th>Source NLL</th><th>Soft/hard agreement</th><th>Integrity</th><th>Pass</th></tr>',
        ]
        for name, c in data["cohorts"].items():
            parts.append(
                f"<tr><td>{name}</td><td>{c['accuracy']:.2%}</td><td>{c['source_accuracy']:.2%}</td><td>{c['nll']:.6f}</td><td>{c['soft_hard_agreement']:.2%}</td><td>{c['integrity']}</td><td>{c['passed']}</td></tr>"
            )
        parts.append(
            "</table></div><p>Source: entity_gate.json. Learned gates require ≥95% source/state accuracy and soft/hard agreement, NLL≤0.15; controls require50% accuracy. All require integrity.</p>"
        )
        parts.append(
            f"<details><summary>Gate examples and unlabelled distance sweep</summary><pre>{escape(json.dumps(dict(examples=data['cohorts']['reference']['episodes'][:2], sweep=data['distance_sweep']), indent=2))}</pre></details></section>"
        )
        if "shift_after" in data:
            parts.append(
                f"<p>Clean-retention KL weight: {data.get('retention_weight', 0)}. Teacher consistency is not ground-truth correctness; all acceptance checks use held-out labels.</p>"
            )
            parts.append(
                f"<section><h2>Matched continuation: {'noise mixture' if data['augmented'] else 'low-noise control'}</h2><p>Adaptation gate: {data['adaptation_passed']}. Compare each class at fixed threshold0.5. Exposure adaptation is not denoising or calibrated semantic relevance.</p>"
            )
            parts.append(
                '<div class="table"><table><tr><th>Model</th><th>Noise</th><th>Accuracy</th><th>Accept recall</th><th>Ignore recall</th></tr>'
            )
            for label in ("shift_before", "shift_after"):
                for noise, cohort in data[label]["cohorts"].items():
                    v = cohort["thresholds"]["0.5"]
                    parts.append(
                        f"<tr><td>{label}</td><td>{noise}</td><td>{v['accuracy']:.2%}</td><td>{v['positive_recall']:.2%}</td><td>{v['negative_recall']:.2%}</td></tr>"
                    )
            parts.append(
                "</table></div><p>Adaptation requires both low-noise recalls≥95%, high-noise accuracy gain≥2 points over frozen and ignore recall loss≤5 points. Benefit over the matched control is audited separately. All contexts are synthetic; high noise can be ambiguous.</p></section>"
            )
        return parts
    relation = directory / "entity_relations.json"
    if relation.exists():
        data = json.loads(relation.read_text())
        parts = [
            "<section><h2>Remembered relation keys</h2>",
            f"<p>Declared gates: {'pass' if data['passed'] else 'fail'}.</p>",
            "<p>An earlier cue writes a learned key. A later destination-only action retrieves the current source state. Only key encoding/decoding trains; relation slots, latest-write replacement and persistence are explicit. This is not general graph discovery.</p>",
            '<div class="table"><table><tr><th>Condition</th><th>Episodes</th><th>State accuracy</th><th>Source accuracy</th><th>NLL/entity</th><th>Rejection</th><th>Integrity</th><th>Gate</th></tr>',
        ]
        for name, c in data["cohorts"].items():
            parts.append(
                f"<tr><td>{name}</td><td>{c['examples']}</td><td>{c['accuracy']:.2%}</td><td>{c['source_accuracy']:.2%}</td><td>{c['nll']:.6f}</td><td>{c['rejection_rate']:.2%}</td><td>{c['integrity']}</td><td>{c['passed']}</td></tr>"
            )
        parts.append(
            "</table></div><p>Source: entity_relations.json. Normal gates require95% state/source accuracy and NLL≤0.15; erased-key source accuracy must stay≤60%. Key erasure is an ablation, not a supported forgetting operation. Correlated variants share families. Integrity checks replay/restore and unchanged non-target states/relation keys.</p>"
        )
        parts.append(
            f"<details><summary>Relation memory examples</summary><pre>{escape(json.dumps(data['cohorts']['reference']['episodes'][:2], indent=2))}</pre></details></section>"
        )
        return parts
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


def visual_memory_inspection(directory):
    path = Path(directory) / "visual_memory.json"
    if not path.exists():
        return []
    result = json.loads(path.read_text())
    parts = [
        "<section><h2>Direct numerical weights</h2><p>Zero optimizer updates. "
        "Constructed weights use handwritten routing; the adjusted head is fitted from labeled examples. "
        "Selection used development scenes before final evaluation.</p>",
        f"<p>Selected: {escape(result['selected'])}. All declared gates passed: {result['passed']}.</p>",
        '<div class="table"><table><tr><th>Version / population</th><th>Accuracy</th><th>Both in pair</th><th>NLL</th></tr>',
    ]
    metrics = [
        (c["name"] + " / development", c["development"]) for c in result["candidates"]
    ]
    metrics += [
        ("Ordinary initialization / test", result["baseline"]),
        ("Selected / test", result["evaluation"]),
        ("Earlier history erased / test", result["erased"]),
    ]
    for name, values in metrics:
        parts.append(
            f"<tr><td>{escape(name)}</td><td>{values['accuracy']:.4f}</td><td>{values['pair_both']:.4f}</td><td>{values['nll']:.6f}</td></tr>"
        )
    parts.append(
        "</table></div><p>Left/right labels denote last visible association. "
        "Paired histories end with identical pixels; reversal cases begin at the opposite location. "
        "This is a synthetic, task-specific test. Real webcam transfer is untested.</p></section>"
    )
    with np.load(Path(directory) / "visual_examples.npz", allow_pickle=False) as data:
        parts.append(
            '<section><h2>Observed histories, in time order</h2><div class="gallery" '
            'style="grid-template-columns:repeat(auto-fit,minmax(min(100%,420px),1fr))">'
        )
        for i, (frames, label) in enumerate(zip(data["images"], data["labels"])):
            strip = np.concatenate(list(frames), axis=1).astype("float32") / 255
            prediction = int(np.argmax(result["evaluation"]["logits"][i]))
            parts.append(
                f'<figure><img alt="Four observed frames for episode {i}" src="{image_url(strip)}"><figcaption>Episode {i}: answer {("left", "right")[int(label)]}; predicted {("left", "right")[prediction]}</figcaption></figure>'
            )
        parts.append("</div></section>")
    parts.append(
        f"<section><details><summary>Exact results, resource use and weight provenance</summary><pre>{escape(json.dumps(result, indent=2))}</pre></details></section>"
    )
    return parts


def capability_inspection(directory):
    path = directory / "capabilities.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    parts = [
        "<section><h2>Current capability baseline</h2>",
        "<p><strong>These are separate saved checkpoints, not one agent with all these abilities.</strong> No neural weights were trained or selected during this evaluation. Pass/fail refers to the declared small screen, not general competence.</p>",
        "<p>Frozen protocol: <code>"
        + data["protocol_sha256"]
        + "</code>. Future repeats measure regression; they are no longer fresh generalization tests. Specific input perturbations do not establish general robustness.</p>",
        '<div class="table"><table><tr><th>Capability / checkpoint role</th><th>Screen</th><th>Measurements and limits</th></tr>',
    ]
    for case in data["cases"]:
        roles = ", ".join(case["checkpoint"])
        visible = list(case["metrics"].items())[:8]
        metrics = "; ".join(
            f"{k}: {v:.6g}" if isinstance(v, (int, float)) else f"{k}: {v}"
            for k, v in visible
        )
        detail = dict(
            metrics=case["metrics"],
            gates=case["gates"],
            input_sha256=case["input_sha256"],
            raw_file=case["raw_file"],
        )
        parts.append(
            f"<tr><td><strong>{escape(case['id'])}</strong><br>{escape(roles)}</td><td>{escape(case['status'])}</td><td>{escape(metrics)}<br><small>{escape(case['scope'])}</small><details><summary>Exact scores and gates</summary><pre>{escape(json.dumps(detail, indent=2))}</pre></details></td></tr>"
        )
    parts.append(
        "</table></div></section><section><h2>Coverage gaps</h2><p>These have no valid capability score yet. Missing evaluation or implementation is different from failing a measured task.</p><ul>"
    )
    for gap in data["gaps"]:
        parts.append(
            f"<li><strong>{escape(gap['id'])}:</strong> {escape(gap['reason'])}</li>"
        )
    parts.append("</ul></section>")
    parts.append(
        "<section><h2>What the model actually saw and produced</h2><p>Targets and predictions are displayed separately. Reconstructions are model outputs, not interpretations of everything retained in memory.</p>"
    )
    for condition in ("original", "swap_red_blue", "noise"):
        with np.load(directory / f"visual_{condition}.npz", allow_pickle=False) as z:
            film = np.concatenate(list(z["images"][1]), axis=1).astype("float32") / 255
        parts.append(
            f'<figure><img style="max-width:650px" alt="Four observation frames: {condition}" src="{image_url(film)}"><figcaption>Visual memory: {condition}, episode 1. Query concerns the last visible location; the final view hides the target.</figcaption></figure>'
        )
    for name in ("prediction_synthetic", "prediction_pusht", "constructed_modalities"):
        with np.load(directory / f"{name}.npz", allow_pickle=False) as z:
            if name == "constructed_modalities":
                panels = [z["input"][0], z["image"][0]]
                label = "Real photo input | constructed-checkpoint reconstruction"
            else:
                panels = [z["current"][0], z["target"][0, -1], z["predicted"][0, -1]]
                label = name + ": last observation | true future | predicted future"
            film = np.concatenate([x.transpose(1, 2, 0) for x in panels], axis=1)
        parts.append(
            f'<figure><img style="max-width:420px" alt="{escape(label)}" src="{image_url(film)}"><figcaption>{escape(label)}</figcaption></figure>'
        )
    parts.append("</section>")
    software = directory / "software.json"
    if software.exists():
        evidence = json.loads(software.read_text())
        parts.append(
            "<section><h2>Software contracts, separately tested</h2><p>Unit and integration checks cover implementation behavior, not learned task quality.</p><pre>"
            + escape(json.dumps(evidence, indent=2))
            + "</pre></section>"
        )
    parts.append(
        "<section><h2>Checkpoint identities and resources</h2><pre>"
        + escape(
            json.dumps(
                dict(
                    checkpoints=data["checkpoints"],
                    resources=data["resources"],
                    limits=data["limits"],
                ),
                indent=2,
            )
        )
        + "</pre></section>"
    )
    comparison = directory / "comparison.json"
    if comparison.exists():
        parts.append(
            "<section><h2>Change from the specified baseline</h2><p>Signed raw metric differences: new minus previous. Check the metric direction before interpreting improvement.</p><pre>"
            + escape(comparison.read_text())
            + "</pre></section>"
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
    show_objective_curve = any(
        (r["split"] in ("train", "validation") and "loss" in r)
        or (r["split"].startswith("diagnostic_") and "nll" in r)
        for r in rows
    ) and not any(
        (directory / name).exists()
        for name in (
            "modality_audit.json",
            "capabilities.json",
            "visual_memory.json",
            "entity_growth.json",
            "entity_temporal.json",
            "entity_source.json",
            "entity_gate_shift.json",
            "entity_evidence_sources.json",
            "entity_source_choice.json",
            "entity_source_diagnosis.json",
            "entity_source_drift.json",
        )
    )
    if show_objective_curve:
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
    result_path = directory / "result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text())
        parts.append("<section><h2>Recorded result</h2>")
        scope = result.get("evaluation_split", result.get("evaluation_scope", ""))
        parts.append(f"<p>{escape(str(scope))}</p>")
        if "gate" in result:
            label = "passed" if result["gate"] is True else "not passed"
            parts.append(
                f"<p><strong>Declared capability screen: {label}.</strong></p>"
            )
        parts.append(
            "<p>Accuracy values are fractions from 0 to 1. Error metrics retain the recipe's scale.</p>"
        )
        metrics = result.get("metrics", {})
        for name, value in metrics.items() if isinstance(metrics, dict) else []:
            if isinstance(value, dict):
                parts.append(
                    f"<details><summary>{escape(name.replace('_', ' '))}</summary><table><tr><th>Metric</th><th>Value</th></tr>"
                )
                for metric, number in value.items():
                    parts.append(
                        f"<tr><td>{escape(metric.replace('_', ' '))}</td><td>{escape(str(number))}</td></tr>"
                    )
                parts.append("</table></details>")
            else:
                parts.append(
                    f"<p>{escape(name.replace('_', ' '))}: <strong>{escape(str(value))}</strong></p>"
                )
        parts.append(
            "<details><summary>Exact result and limitations</summary><pre>"
            + escape(result_path.read_text())
            + "</pre></details></section>"
        )
    scene_fit = directory / "training_scene_fit.json"
    if scene_fit.exists():
        training = json.loads(scene_fit.read_text())
        parts.append(
            "<section><h2>Training fit by scene</h2><p>Training diagnostics only; "
            "not a held-out gate. Duplicate conditions retain their separate block "
            "indices. Accuracy is a fraction from 0 to 1.</p>"
            '<div class="table"><table><tr><th>Block / scene parameters</th>'
            "<th>Histories</th><th>Query</th><th>Facts</th><th>Image</th>"
            "<th>Factual shape</th><th>Image shape</th></tr>"
        )
        for block in training["blocks"]:
            label = escape(
                str(block["block"]) + " / " + json.dumps(block["scene"], sort_keys=True)
            )
            for mode, scores in block["metrics"].items():
                cells = "".join(
                    f"<td>{escape(str(scores[k]))}</td>"
                    for k in (
                        "factual_accuracy",
                        "image_accuracy",
                        "factual_shape_accuracy",
                        "image_shape_accuracy",
                    )
                )
                parts.append(
                    f"<tr><td>{label}</td><td>{escape(str(block['examples']))}</td><td>{escape(mode)}</td>{cells}</tr>"
                )
        parts.append(
            "</table></div><details><summary>Exact training scene scores</summary><pre>"
            + escape(scene_fit.read_text())
            + "</pre></details></section>"
        )
    comparison_panel = directory / "comparison.png"
    if comparison_panel.exists():
        panel = (
            "data:image/png;base64,"
            + base64.b64encode(comparison_panel.read_bytes()).decode()
        )
        parts.append(
            f'<section><h2>Examples in context</h2><img class="chart" alt="Labeled observation, target and output comparison" src="{panel}"></section>'
        )
    if show_objective_curve:
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
            label = escape(
                record["identity"]["settings"].get("example_labels", {}).get(key, label)
            )
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
    parts.extend(capability_inspection(directory))
    from pathwm.evaluation.modality_suite import suite_inspection

    parts.extend(suite_inspection(directory))
    parts.extend(visual_memory_inspection(directory))
    parts.extend(model_inspection(directory))
    from pathwm.evaluation.world_state import world_state_inspection

    parts.extend(world_state_inspection(directory))
    from pathwm.evaluation.modality_audit import modality_inspection

    parts.extend(modality_inspection(directory))
    from pathwm.evaluation.modality_readout import readout_inspection

    parts.extend(readout_inspection(directory))
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
