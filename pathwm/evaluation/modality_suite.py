"""Scoped capability records for the existing symbolic multimodal recipe.

Ordinary evaluation functions, not a trainer or task registry. Adding a capability
requires an actual measurement path and a versioned protocol, not only a new row.
"""

from collections import Counter
from html import escape
import json

import numpy as np
import torch

from pathwm.data.modality_readout import KINDS, MODES, observations
from pathwm.io import digest, evaluation_mode


PROTOCOL = dict(
    version="symbolic-modality-capabilities-v1",
    data_version="factor-combinations-v1",
    categorical_draws=3,
    factor_minimum=0.8,
    joint_minimum=0.8,
    omitted_chance_margin=0.1,
    source_drop_minimum=0.2,
    source_population="18 Cartesian combinations x 4 views; intervention only",
    scope="Symbolic factor access, not general language, speech or natural video",
)

FACTORS = ("color", "location", "direction")
SIZES = (3, 3, 2)
STAGES = ("encoder", "posterior", "codes", "observed", "thought")


def factor_diagnostics(predictions, targets, ids):
    """Three draws of the same examples are never counted as independent data."""
    p, y = np.asarray(predictions), np.asarray(targets)
    if (
        y.ndim != 2
        or y.shape[1] != 3
        or not len(y)
        or p.ndim != 3
        or p.shape[0] != PROTOCOL["categorical_draws"]
        or p.shape[1:] != y.shape
        or len(ids) != len(y)
        or len(set(ids)) != len(ids)
    ):
        raise ValueError(
            "Expected aligned [3,N,3] predictions, [N,3] targets and unique IDs"
        )
    for values in (p, y):
        if (
            not np.issubdtype(values.dtype, np.number)
            or not np.isfinite(values).all()
            or not np.equal(values, np.floor(values)).all()
            or (values < 0).any()
            or (values >= np.array(SIZES)).any()
        ):
            raise ValueError(
                "Finite integer factor classes within declared ranges required"
            )
    correct = p == y[None]
    factor = correct.mean(1)
    joint = correct.all(-1).mean(1)
    failures = []
    for draw in range(len(p)):
        for i in np.flatnonzero(~correct[draw].all(1))[:6]:
            failures.append(
                dict(
                    draw=draw,
                    example_id=ids[i],
                    index=int(i),
                    expected=y[i].tolist(),
                    predicted=p[draw, i].tolist(),
                )
            )
    groups = []
    for key in np.unique(y, axis=0):
        take = (y == key).all(1)
        groups.append(
            dict(
                factors=key.tolist(),
                examples=int(take.sum()),
                factor_accuracy=correct[:, take].mean((0, 1)).tolist(),
                joint_accuracy=float(correct[:, take].all(-1).mean()),
            )
        )
    confusion = []
    for col, size in enumerate(SIZES):
        matrix = np.zeros((len(p), size, size), dtype=int)
        for draw in range(len(p)):
            np.add.at(
                matrix[draw], (y[:, col].astype(int), p[draw, :, col].astype(int)), 1
            )
        confusion.append(matrix.tolist())
    return dict(
        examples=len(y),
        draws=len(p),
        factor_accuracy_by_draw=factor.tolist(),
        joint_accuracy_by_draw=joint.tolist(),
        min_factor_accuracy=factor.min(0).tolist(),
        min_joint_accuracy=float(joint.min()),
        mean_factor_accuracy=factor.mean(0).tolist(),
        mean_joint_accuracy=float(joint.mean()),
        by_combination=groups,
        confusion_by_factor=confusion,
        failures=failures,
    )


@torch.no_grad()
def measure_factors(core, populations, *, seed, device):
    """Actual deployed factor head, including explicit source-omission controls."""
    measured, arrays, draws = {}, {}, []
    with evaluation_mode(core):
        for si, split in enumerate(("seen", "heldout", "intervention")):
            data = populations[split]
            y = data["targets"]["factors"].numpy()
            arrays[f"{split}.targets"] = y
            arrays[f"{split}.ids"] = np.asarray(data["ids"])
            modes = (
                MODES
                if split != "intervention"
                else (
                    "complementary",
                    "without_image",
                    "without_audio",
                    "without_video",
                )
            )
            for mi, mode in enumerate(modes):
                # Paired stochastic draws for full and omitted-source controls.
                condition = 0 if split == "intervention" else mi
                outputs = []
                for draw in range(PROTOCOL["categorical_draws"]):
                    rng_seed = seed + 50000 + si * 1000 + condition * 10 + draw
                    torch.manual_seed(rng_seed)
                    omit = (
                        mode.removeprefix("without_")
                        if mode.startswith("without_")
                        else None
                    )
                    inputs = observations(
                        data,
                        "complementary" if omit else mode,
                        device=device,
                        omit=omit,
                    )
                    logits = core.factors(core(inputs))
                    packed = torch.cat(logits, -1).detach().cpu().numpy()
                    if not np.isfinite(packed).all():
                        raise ValueError("Nonfinite actual factor logits")
                    arrays[f"{split}.{mode}.logits.{draw}"] = packed
                    outputs.append(
                        np.stack(
                            [v.detach().cpu().numpy().argmax(-1) for v in logits], -1
                        )
                    )
                    draws.append(dict(split=split, mode=mode, draw=draw, seed=rng_seed))
                pred = np.stack(outputs)
                arrays[f"{split}.{mode}.predictions"] = pred
                measured[f"{split}.{mode}"] = factor_diagnostics(pred, y, data["ids"])
    return measured, arrays, draws


def _case(case_id, modality, scope, *, measured=None, gates=None, implemented=True):
    present = measured is not None
    return dict(
        id=case_id,
        modality=modality,
        scope=scope,
        implementation="implemented" if implemented else "not_implemented",
        execution="complete" if present else "not_run",
        assessment=("pass" if all(gates.values()) else "fail")
        if present and gates
        else "not_scored",
        metrics=measured,
        gates=gates or {},
        raw_file="capability_predictions.npz" if present else None,
    )


def _factor_gates(row):
    values = np.array([*row["min_factor_accuracy"], row["min_joint_accuracy"]])
    if (
        values.shape != (4,)
        or not np.isfinite(values).all()
        or ((values < 0) | (values > 1)).any()
    ):
        raise ValueError("Invalid capability metrics")
    return {
        **{
            name: bool(values[i] >= PROTOCOL["factor_minimum"])
            for i, name in enumerate(FACTORS)
        },
        "joint": bool(values[3] >= PROTOCOL["joint_minimum"]),
    }


def diagnostic_hints(measured, probes):
    """Reader-specific transitions to investigate, explicitly not causal findings."""
    hints = []
    for split in ("seen", "heldout"):
        for mode in MODES:
            actual = measured.get(f"{split}.{mode}")
            if actual is None:
                continue
            by_stage = {
                r["stage"]: r["factor_accuracy"]
                for r in probes
                if r.get("split") == split
                and r.get("input_mode") == mode
                and r.get("probe") == "ridge"
            }
            for col, factor in enumerate(FACTORS):
                if actual["min_factor_accuracy"][col] >= PROTOCOL["factor_minimum"]:
                    continue
                readable = {
                    s: by_stage[s][col] >= PROTOCOL["factor_minimum"]
                    for s in STAGES
                    if s in by_stage
                }
                drops = [
                    f"{a} -> {b}"
                    for a, b in zip(STAGES, STAGES[1:])
                    if readable.get(a) and b in readable and not readable[b]
                ]
                if readable.get("thought"):
                    next_check = "Working-state probe succeeds; compare deployed factor readout learning on frozen state."
                elif drops:
                    next_check = (
                        "Probe decline at "
                        + ", ".join(drops)
                        + "; isolate transition with matched readers and a controlled bypass/refit."
                    )
                else:
                    next_check = "No localized passing-to-failing boundary; inspect input reference, probe capacity and learning."
                hints.append(
                    dict(
                        split=split,
                        input_mode=mode,
                        factor=factor,
                        actual_min=actual["min_factor_accuracy"][col],
                        probe_scores={
                            s: by_stage[s][col] for s in STAGES if s in by_stage
                        },
                        next_check=next_check,
                        attribution="Hypothesis for diagnosis; no unique cause or erased-information claim.",
                    )
                )
    return hints


def build_suite(measured, probes, *, source, seed):
    cases = []
    prefixes = dict(
        text="TXT",
        image="IMG",
        audio="AUD",
        video="VID",
        all="CORE.all",
        complementary="CORE.complementary",
    )
    for mode in MODES:
        for split in ("seen", "heldout"):
            row = measured.get(f"{split}.{mode}")
            cases.append(
                _case(
                    f"{prefixes[mode]}.symbolic.{split}",
                    mode if mode in KINDS else "core",
                    f"Actual factor head, {mode}, {split} combinations. "
                    "Symbolic input; video orientation is visible in a single frame.",
                    measured=row,
                    gates=_factor_gates(row) if row else None,
                )
            )
    full = measured.get("intervention.complementary")
    for col, modality in enumerate(("image", "audio", "video")):
        removed = measured.get(f"intervention.without_{modality}")
        metrics, gates = None, None
        if full is not None and removed is not None:
            a = np.asarray(full["factor_accuracy_by_draw"])[:, col]
            b = np.asarray(removed["factor_accuracy_by_draw"])[:, col]
            if (
                a.shape != (3,)
                or b.shape != (3,)
                or not np.isfinite([a, b]).all()
                or (np.asarray([a, b]) < 0).any()
                or (np.asarray([a, b]) > 1).any()
            ):
                raise ValueError("Invalid source control measurements")
            metrics = dict(
                factor=FACTORS[col],
                chance=1 / SIZES[col],
                full_by_draw=a.tolist(),
                omitted_by_draw=b.tolist(),
                drop_by_draw=(a - b).tolist(),
                full=full,
                omitted=removed,
            )
            gates = dict(
                full=bool((a >= PROTOCOL["factor_minimum"]).all()),
                omitted=bool(
                    (b <= 1 / SIZES[col] + PROTOCOL["omitted_chance_margin"]).all()
                ),
                drop=bool((a - b >= PROTOCOL["source_drop_minimum"]).all()),
            )
        cases.append(
            _case(
                f"CORE.source_{modality}",
                "core",
                "Balanced Cartesian intervention; source dependence, not held-out transfer.",
                measured=metrics,
                gates=gates,
            )
        )
    gaps = (
        (
            "TXT.negation_roles",
            "text",
            "Negation, roles, reference resolution and paraphrases",
        ),
        ("TXT.dialogue", "text", "Dialogue history, instructions and correction"),
        (
            "IMG.real_objects",
            "image",
            "Natural objects, identity, relationships and counting",
        ),
        ("IMG.fine_detail", "image", "Small objects, fine detail and text in images"),
        (
            "AUD.speech",
            "audio",
            "Speech content and speaker continuity; tone codes are not speech",
        ),
        ("AUD.real_events", "audio", "Natural sound events and noise robustness"),
        ("VID.natural_motion", "video", "Natural object/camera motion and stillness"),
        (
            "VID.tracking",
            "video",
            "Persistent identity, occlusion and state transitions",
        ),
        ("VID.forecast", "video", "Future events/states, no target-future access"),
        ("CORE.memory", "core", "Long-history storage, retrieval and correction"),
        ("CORE.conflict", "core", "Conflicting evidence and calibrated uncertainty"),
        (
            "CORE.streaming",
            "core",
            "Streaming timing, resets, bounded state and late inputs",
        ),
        (
            "CORE.av_sync",
            "core",
            "Audiovisual synchronization and natural cross-modal binding",
        ),
        (
            "ACT.effects",
            "action",
            "Action-conditioned consequences and control success",
        ),
    )
    cases += [_case(i, m, s, implemented=False) for i, m, s in gaps]
    # Existing decoder evaluators require trained output checkpoints and their own protocol.
    cases += [
        _case(
            f"{prefixes[m]}.output",
            m,
            "Output evaluation exists separately; not run on this core-only source.",
        )
        for m in KINDS
    ]
    counts = Counter(c["assessment"] for c in cases)
    return dict(
        protocol=PROTOCOL,
        protocol_sha256=digest(PROTOCOL),
        source=source,
        seed=seed,
        cases=cases,
        coverage=dict(
            total=len(cases),
            passed=counts["pass"],
            failed=counts["fail"],
            not_implemented=sum(
                c["implementation"] == "not_implemented" for c in cases
            ),
            not_run=sum(
                c["implementation"] == "implemented" and c["execution"] == "not_run"
                for c in cases
            ),
        ),
        diagnostics=dict(
            stage_probes=probes,
            hints=diagnostic_hints(measured, probes),
            attribution="No unique failed layer inferred; probes have unequal widths/capacities.",
            unmeasured=[
                "random encoder comparison",
                "memory/retrieval stage",
                "decoder on this source",
                "natural-input semantics",
            ],
        ),
        limitations=[
            "Previously inspected symbolic populations: development regression, not fresh generalization.",
            "Three categorical RNG draws of one checkpoint, not three trained models or a confidence interval.",
            "Encoder and core probes measure accessibility; random-encoder comparison is absent.",
            "A failed probe does not prove information destruction; independent branch passes do not certify a whole pipeline.",
            "Held-out combinations constrain color/location; source omissions are scored ONLY on the independent Cartesian population.",
            "No new neural training; diagnostic ridge fitting uses training labels and validation-only alpha selection.",
        ],
    )


def suite_inspection(directory):
    path = directory / "capability_suite.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    parts = [
        "<section><h2>Modality capability coverage</h2>",
        "<p>Scoped symbolic tasks. Separate modality paths and shared-core tests; no overall understanding score.</p>",
        "<p>Protocol: <code>" + escape(data["protocol_sha256"]) + "</code></p>",
        "<p><strong>" + escape(json.dumps(data["coverage"])) + "</strong></p>",
        '<div class="table"><table><tr><th>Case / scope</th><th>Implementation</th><th>Execution</th><th>Assessment</th><th>Evidence</th></tr>',
    ]
    for c in data["cases"]:
        parts.append(
            "<tr><td><strong>"
            + escape(c["id"])
            + "</strong><br>"
            + escape(c["scope"])
            + "</td>"
            + "".join(
                "<td>" + escape(c[k]) + "</td>"
                for k in ("implementation", "execution", "assessment")
            )
            + "<td><details><summary>Metrics, gates and failure examples</summary><pre>"
            + escape(
                json.dumps(
                    dict(
                        metrics=c["metrics"], gates=c["gates"], raw_file=c["raw_file"]
                    ),
                    indent=2,
                )
            )
            + "</pre></details></td></tr>"
        )
    parts.append(
        "</table></div></section><section><h2>Diagnostic limits and unmeasured stages</h2><ul>"
    )
    parts.extend("<li>" + escape(s) + "</li>" for s in data["limitations"])
    parts.append(
        "</ul><p>Unmeasured: "
        + escape(", ".join(data["diagnostics"]["unmeasured"]))
        + "</p></section>"
    )
    parts.append(
        '<section><h2>Where to investigate next</h2><p>Actual task failures compared with frozen-stage reader access. These are diagnostic hypotheses, not causal verdicts.</p><div class="table"><table><tr><th>Condition / factor</th><th>Actual head minimum</th><th>Probe access by stage</th><th>Next bounded check</th></tr>'
    )
    for hint in data["diagnostics"]["hints"]:
        parts.append(
            "<tr><td>"
            + escape(f"{hint['split']} / {hint['input_mode']} / {hint['factor']}")
            + f"</td><td>{hint['actual_min']:.1%}</td><td>"
            + escape(
                "; ".join(f"{s}: {v:.1%}" for s, v in hint["probe_scores"].items())
            )
            + "</td><td>"
            + escape(hint["next_check"])
            + "</td></tr>"
        )
    parts.append("</table></div></section>")
    example_path = directory / "capability_examples.json"
    if example_path.exists():
        from pathwm.evaluation.report import image_url

        examples = json.loads(example_path.read_text())
        parts.append(
            "<section><h2>Observed inputs and actual errors</h2><p>Expected and predicted factors are [color, location, direction]; source media are inputs, not generated outputs.</p>"
        )
        for e in examples:
            parts.append(
                "<details><summary>"
                + escape(e["mode"] + " / " + e["id"])
                + "</summary><pre>"
                + escape(
                    json.dumps(
                        {k: v for k, v in e.items() if k not in ("image", "video")},
                        indent=2,
                    )
                )
                + "</pre>"
            )
            for name in ("image", "video"):
                if name in e:
                    x = np.asarray(e[name])
                    if name == "video":
                        x = np.concatenate(list(x), axis=-1)
                    parts.append(
                        f'<img alt="{name} input" style="max-width:100%;width:480px;image-rendering:pixelated" src="{image_url(x)}">'
                    )
            if "audio" in e:
                values = np.asarray(e["audio"]).flatten()
                points = " ".join(
                    f"{i * 600 / max(len(values) - 1, 1):.2f},{50 - 40 * v:.2f}"
                    for i, v in enumerate(values)
                )
                parts.append(
                    "<p>Symbolic tone samples, not speech; chunk timing and validity above.</p>"
                    + '<svg aria-label="Symbolic input waveform" role="img" viewBox="0 0 600 100" style="width:100%;max-width:600px">'
                    + f'<polyline fill="none" stroke="#2763a4" stroke-width="1.5" points="{points}"/></svg>'
                )
            parts.append("</details>")
        parts.append("</section>")
    return parts
