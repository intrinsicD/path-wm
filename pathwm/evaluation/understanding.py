"""Task answers, source controls and separate frozen-state diagnostic readers.

No agent training; answer scoring is an explicit replaceable interface. No single
score combines task correctness with diagnostic accessibility or missing coverage.
"""

from collections import defaultdict
from pathlib import Path
from html import escape
import base64
import io
import json
import wave

import numpy as np
import torch
from torch.nn import functional as F

from pathwm.models.modalities import bytes_batch
from pathwm.models.photo_probe import RidgeReader

PROTOCOL = dict(
    version="understanding-regression-v1",
    draws=3,
    accuracy_min=0.8,
    paired_min=0.8,
    source_gain_min=0.15,
    omission_chance_margin=0.1,
    encoder_bins=16,
    ridge_alphas=[0.1, 1.0, 10.0],
    readout="byte-decoder-choice-mean-loglik-v1",
)


def choice_scores(outputs, tokens, choices):
    """Score every answer independently; do not insert target answers in the core."""
    ids, valid = bytes_batch(choices, device=tokens.device)
    context = tokens.expand(len(choices), -1, -1)
    logits = outputs("text", context, ids[:, :-1])
    logp = logits.log_softmax(-1).gather(-1, ids[:, 1:, None]).squeeze(-1)
    mask = valid[:, 1:]
    return (logp * mask).sum(-1) / mask.sum(-1)


def capture_stages(core, inputs, *, requests=None):
    """Bounded detached encoder summaries from the same actual forward pass."""
    encoders, hooks = {}, []

    def capture(kind):
        def hook(module, arguments, pyramid):
            if kind in encoders:
                return  # Later request re-encoding is not the observation probe.
            values = []
            for scale in pyramid.scales:
                x = scale.values.masked_fill(~scale.valid[..., None], 0)
                # Fixed coordinates; pooling itself can hide fine/temporal detail.
                pooled = F.adaptive_avg_pool1d(
                    x.transpose(1, 2), PROTOCOL["encoder_bins"]
                )
                values.append(pooled.detach().cpu().flatten(1))
            encoders[kind] = torch.cat(values, 1)

        return hook

    try:
        for kind in inputs:
            hooks.append(core.agent.encoders[kind].register_forward_hook(capture(kind)))
        kwargs = {} if requests is None else dict(requests=requests)
        tokens, state = core(inputs, return_state=True, **kwargs)
    finally:
        for h in hooks:
            h.remove()
    # Always the same feature coordinates, including zero for absent modalities.
    width = next(iter(encoders.values())).shape[-1]
    zero = torch.zeros(len(tokens), width)
    return tokens, dict(
        encoder=torch.cat(
            [encoders.get(k, zero) for k in ("text", "image", "audio", "video")], -1
        ),
        posterior=state.logits.softmax(-1).detach().cpu().flatten(1),
        working=tokens.detach().cpu().flatten(1),
    )


def task_metrics(full, omitted, empty, labels, pairs, classes):
    full, omitted, empty = map(np.asarray, (full, omitted, empty))
    labels = np.asarray(labels)
    if (
        labels.ndim != 1
        or len(labels) < 2
        or full.shape != (3, len(labels))
        or omitted.shape != full.shape
        or empty.shape != full.shape
        or len(pairs) != len(labels)
    ):
        raise ValueError("Expected three aligned draws and targets")
    for a in (full, omitted, empty, labels):
        if (
            not np.isfinite(a).all()
            or not np.equal(a, np.floor(a)).all()
            or ((a < 0) | (a >= classes)).any()
        ):
            raise ValueError("Invalid answer indices")
    correct = full == labels[None]
    acc = correct.mean(1)
    omit = (omitted == labels[None]).mean(1)
    blank = (empty == labels[None]).mean(1)
    groups = defaultdict(list)
    for i, p in enumerate(pairs):
        if p is not None:
            groups[p].append(i)
    paired = None
    if groups:
        if len(groups) * 2 != len(labels) or any(
            len(v) != 2 or len(set(labels[v])) != 2 for v in groups.values()
        ):
            raise ValueError("Incomplete/opposite-label pair groups")
        paired = np.stack([correct[:, ids].all(1) for ids in groups.values()], 1).mean(
            1
        )
    gates = dict(
        accuracy=bool((acc >= PROTOCOL["accuracy_min"]).all()),
        context_gain=bool((acc - omit >= PROTOCOL["source_gain_min"]).all()),
        no_evidence_gain=bool((acc - blank >= PROTOCOL["source_gain_min"]).all()),
        omission=bool((omit <= 1 / classes + PROTOCOL["omission_chance_margin"]).all()),
    )
    if paired is not None:
        gates["paired"] = bool((paired >= PROTOCOL["paired_min"]).all())
    return dict(
        accuracy_by_draw=acc.tolist(),
        accuracy_min=float(acc.min()),
        accuracy_mean=float(acc.mean()),
        paired_by_draw=None if paired is None else paired.tolist(),
        paired_min=None if paired is None else float(paired.min()),
        omitted_by_draw=omit.tolist(),
        empty_by_draw=blank.tolist(),
        source_gain_min=float((acc - omit).min()),
        examples=len(labels),
        pairs=len(groups),
        chance=1 / classes,
        gates=gates,
        passed=all(gates.values()),
    )


def compare_understanding(before, after):
    if before["contract"] != after["contract"]:
        raise ValueError(
            "Understanding comparison contract differs (fixtures/profile/readout/protocol)"
        )
    a, b = ({r["id"]: r for r in x["cases"]} for x in (before, after))
    if len(a) != len(before["cases"]) or len(b) != len(after["cases"]):
        raise ValueError("Duplicate understanding cases")
    if a.keys() != b.keys():
        raise ValueError("Understanding case coverage differs")
    rows = []
    for key, x in a.items():
        y = b[key]
        deltas = {}
        for metric in ("accuracy_min", "paired_min", "source_gain_min"):
            old, new = x["metrics"].get(metric), y["metrics"].get(metric)
            if (old is None) != (new is None):
                raise ValueError("Metric contract differs")
            if old is not None:
                if not np.isfinite([old, new]).all():
                    raise ValueError("Nonfinite comparison metric")
                deltas[metric] = new - old
        old_gates = x["metrics"].get("gates", {})
        new_gates = y["metrics"].get("gates", {})
        if old_gates.keys() != new_gates.keys():
            raise ValueError("Acceptance gate contract differs")
        lost = sorted(k for k in old_gates if old_gates[k] and not new_gates[k])
        gained = sorted(k for k in old_gates if not old_gates[k] and new_gates[k])
        values = list(deltas.values()) + [-1] * len(lost) + [1] * len(gained)
        values.append(int(y["passed"]) - int(x["passed"]))
        status = (
            "unchanged"
            if all(v == 0 for v in values)
            else "improved"
            if all(v >= 0 for v in values)
            else "regressed"
            if all(v <= 0 for v in values)
            else "mixed"
        )
        rows.append(
            dict(
                id=key,
                status=status,
                delta=deltas,
                before_passed=x["passed"],
                after_passed=y["passed"],
                lost_gates=lost,
                gained_gates=gained,
            )
        )
    return rows


def _probe(features, labels, splits, classes):
    train = splits == "calibration"
    validation = splits == "validation"
    test = splits == "test"
    x = torch.as_tensor(features, dtype=torch.float64)
    y = F.one_hot(torch.as_tensor(labels[train]), classes).double()
    factor = RidgeReader.factor(x[train], y)
    candidates = []
    for alpha in PROTOCOL["ridge_alphas"]:
        solved = RidgeReader.solve(factor, ridge=alpha / x.shape[1])
        weights = solved["training"].T @ solved["alpha"] / x.shape[1]
        prediction = (
            ((x - solved["mean"]) / solved["std"] @ weights + solved["target_mean"])
            .argmax(-1)
            .numpy()
        )
        candidates.append(
            (
                float((prediction[validation] == labels[validation]).mean()),
                alpha,
                prediction,
            )
        )
    _, alpha, pred = max(candidates, key=lambda row: row[0])
    return dict(
        test_accuracy=float((pred[test] == labels[test]).mean()),
        alpha=alpha,
        validation=[dict(alpha=a, accuracy=v) for v, a, _ in candidates],
        features=x.shape[1],
    ), pred


def understanding_inspection(directory):
    directory = Path(directory)
    path = directory / "understanding.json"
    if not path.exists():
        return []
    d = json.loads(path.read_text())
    comparison = {r["id"]: r for r in d.get("comparison") or []}
    parts = [
        "<section><h2>Understanding regression by task</h2><p>Answer accuracy, source dependence and paired correctness. A software pass or a good probe cannot pass these tasks. No combined understanding score.</p>",
        "<p>Evidence subsets exclude the shared textual question. "
        + escape(str(d["coverage"]))
        + "</p>",
        '<div class="table"><table><tr><th>Task / evidence</th><th>Accuracy / pair</th><th>Source gain</th><th>Assessment / change</th></tr>',
    ]
    for c in d["cases"]:
        m = c["metrics"]
        pair = "n/a" if m["paired_min"] is None else f"{m['paired_min']:.1%}"
        delta = comparison.get(c["id"])
        change = "" if delta is None else f"{delta['status']}: {delta['delta']}"
        if delta is not None:
            change += f"; lost gates: {delta.get('lost_gates', [])}; gained: {delta.get('gained_gates', [])}"
        parts.append(
            f"<tr><td>{escape(c['id'])}<br>{escape(c['domain'])}; {m['examples']} examples / {c['sources']} groups</td><td>{m['accuracy_min']:.1%} / {pair}</td><td>{m['source_gain_min']:+.1%}</td><td>{'pass' if c['passed'] else 'fail'}<br>{escape(change)}</td></tr>"
        )
    parts.append(
        "</table></div></section><section><h2>Separate diagnostic accessibility</h2><p>These fitted readers are not deployed task answers. Pooling and reader capacity can hide information.</p>"
    )
    for c in d["cases"]:
        parts.append(
            "<details><summary>"
            + escape(c["id"])
            + " — "
            + escape(c["hint"])
            + "</summary><pre>"
            + escape(
                json.dumps(dict(probes=c["diagnostics"], task=c["metrics"]), indent=2)
            )
            + "</pre></details>"
        )
    parts.append(
        "</section><section><h2>Inputs, expected answers and actual predictions</h2>"
    )
    from pathwm.evaluation.report import image_url

    examples = json.loads((directory / "understanding_examples.json").read_text())
    with np.load(directory / "understanding_examples.npz", allow_pickle=False) as z:
        for r in examples:
            parts.append(
                "<details><summary>"
                + escape(r["id"])
                + "</summary><p>"
                + escape(r["question"])
                + "</p><pre>"
                + escape(
                    json.dumps(
                        dict(
                            choices=r["choices"],
                            expected=r["choices"][r["answer"]],
                            predictions={
                                k: [r["choices"][i] for i in v]
                                for k, v in r["predictions"].items()
                            },
                            removed=r["required"],
                        ),
                        ensure_ascii=False,
                        indent=2,
                    )
                )
                + "</pre>"
            )
            for kind, value in r["evidence"].items():
                if kind == "text":
                    parts.append("<p>" + escape(value) + "</p>")
                elif kind == "audio":
                    buffer = io.BytesIO()
                    with wave.open(buffer, "wb") as f:
                        f.setparams((1, 2, 8000, 0, "NONE", "not compressed"))
                        f.writeframes(
                            (np.clip(z[value].flatten(), -1, 1) * 32767)
                            .astype("<i2")
                            .tobytes()
                        )
                    url = (
                        "data:audio/wav;base64,"
                        + base64.b64encode(buffer.getvalue()).decode()
                    )
                    parts.append(
                        '<audio controls preload="none" src="' + url + '"></audio>'
                    )
                else:
                    x = z[value]
                    x = np.concatenate(list(x), axis=-1) if kind == "video" else x
                    parts.append(
                        '<img alt="'
                        + escape(kind + " evidence")
                        + '" style="max-width:100%;width:512px" src="'
                        + image_url(x)
                        + '">'
                    )
            parts.append("</details>")
    parts.append("</section><section><h2>Still untested / unsupported domains</h2><ul>")
    parts.extend("<li>" + escape(x) + "</li>" for x in d["gaps"])
    parts.append(
        "</ul><h2>Readout lineage, resources and limits</h2><pre>"
        + escape(
            json.dumps(
                {k: d[k] for k in ("source", "resources", "limits", "contract")},
                indent=2,
            )
        )
        + "</pre></section>"
    )
    return parts
