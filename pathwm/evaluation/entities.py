"""Proper answer scores, paired history controls and explicit finite-screen gates."""

import torch
from pathwm.data.entities import NAMES


def entity_metrics(logits, targets, cohorts, groups):
    if len(logits) == 1:
        return matching_metrics(logits, targets, cohorts)
    out = {}
    for cohort in ("identifiable", "ambiguous"):
        mask = torch.tensor([c == cohort for c in cohorts])
        row = dict(examples=int(mask.sum()))
        for name, score, target in zip(NAMES, logits, targets):
            if not torch.isfinite(score).all():
                raise ValueError("Nonfinite entity predictions")
            logp = score.double().log_softmax(-1)
            entropy = -(target.double() * target.double().clamp_min(1e-30).log()).sum(
                -1
            )
            nll = -(target.double() * logp).sum(-1)
            row[name + "_nll"] = float(nll[mask].mean())
            row[name + "_excess_nll"] = float((nll - entropy)[mask].mean())
            correct = score.argmax(-1) == target.argmax(-1)
            if cohort == "identifiable":
                row[name + "_accuracy"] = float(correct[mask].double().mean())
                paired = []
                for group in sorted(set(groups)):
                    indices = [
                        i
                        for i, g in enumerate(groups)
                        if g == group and cohorts[i] == cohort
                    ]
                    buckets = [
                        [i for i in indices if int(target[i].argmax()) == k]
                        for k in range(score.shape[-1])
                    ]
                    for k in range(score.shape[-1] // 2):
                        a, b = buckets[k], buckets[score.shape[-1] - 1 - k]
                        if len(a) != len(b):
                            raise ValueError("Unbalanced entity counterfactual pairs")
                        paired.extend(
                            bool(correct[i] and correct[j]) for i, j in zip(a, b)
                        )
                row["paired_" + name] = sum(paired) / len(paired)
        p = logits[0].double().softmax(-1)
        confidence, chosen = p.max(-1)
        select = confidence > 0.75
        error = 1 - targets[0].double().gather(1, chosen[:, None]).squeeze(1)
        row["coverage"] = float(select[mask].double().mean())
        covered = mask & select
        row["selected_error"] = float(error[covered].mean()) if covered.any() else None
        row["decision_cost"] = float(torch.where(select, error, 0.25)[mask].mean())
        row["nll"] = sum(row[n + "_nll"] for n in NAMES) / 3
        out[cohort] = row
    a, b = out["identifiable"], out["ambiguous"]
    gates = dict(
        identity=a["identity_accuracy"] >= 0.95,
        state=a["state_accuracy"] >= 0.9,
        effect=a["effect_accuracy"] >= 0.9,
        paired=all(a["paired_" + n] >= 0.85 for n in NAMES),
        ambiguity=all(b[n + "_excess_nll"] <= 0.10 for n in NAMES),
        decisions=a["coverage"] >= 0.9
        and a["selected_error"] is not None
        and a["selected_error"] <= 0.05,
    )
    gates["passed"] = all(gates.values())
    return dict(
        views=out,
        gates=gates,
        overall=dict(
            nll=(a["nll"] + b["nll"]) / 2,
            factual_accuracy=sum(a[n + "_accuracy"] for n in NAMES) / 3,
            examples=len(cohorts),
        ),
    )


@torch.no_grad()
def association_diagnostics(model, data):
    """Evaluator-only matching labels; one row per independent descriptor group."""
    x = data.inputs[::32]
    weights = model.assignment_weights(x)
    # Evaluator-only nearest matching is valid within the generator's strict margin.
    labels = (x[:, :, 0, None, :8] - x[:, None, 0, :, :8]).square().sum(-1).argmin(-1)
    result = {"descriptor_groups": len(x)}
    for time, name in ((1, "action"), (2, "final")):
        p = weights[:, time].double()
        result[name + "_accuracy"] = float(
            (p.argmax(-1) == labels[:, time]).double().mean()
        )
        result[name + "_nll"] = float(
            -p.gather(1, labels[:, time, None]).clamp_min(1e-30).log().mean()
        )
    return result


def matching_metrics(logits, targets, cohorts):
    scores, target = logits[0].double(), targets[0].argmax(-1)
    if not torch.isfinite(scores).all():
        raise ValueError("Nonfinite matching predictions")
    logp = scores.log_softmax(-1)
    confidence, chosen = logp.exp().max(-1)
    correct, select = chosen == target, confidence > 0.75
    nll = -logp.gather(1, target[:, None]).squeeze(1)
    views = {}
    for cohort in ("known", "novel"):
        mask = torch.tensor([c == cohort for c in cohorts])
        selected = mask & select
        views[cohort] = dict(
            examples=int(mask.sum()),
            accuracy=float(correct[mask].double().mean()),
            nll=float(nll[mask].mean()),
            coverage=float(select[mask].double().mean()),
            selected_error=float((~correct[selected]).double().mean())
            if selected.any()
            else None,
            false_merge=float((chosen[mask] < scores.shape[-1] - 1).double().mean())
            if cohort == "novel"
            else None,
            false_split=float((chosen[mask] == scores.shape[-1] - 1).double().mean())
            if cohort == "known"
            else None,
        )
    gates = dict(
        known=views["known"]["accuracy"] >= 0.95,
        novel=views["novel"]["accuracy"] >= 0.95,
        false_merge=views["novel"]["false_merge"] <= 0.05,
        false_split=views["known"]["false_split"] <= 0.05,
        nll=float(nll.mean()) <= 0.15,
        decisions=all(
            v["coverage"] >= 0.9
            and v["selected_error"] is not None
            and v["selected_error"] <= 0.05
            for v in views.values()
        ),
    )
    gates["passed"] = all(gates.values())
    return dict(
        views=views,
        gates=gates,
        overall=dict(
            nll=float(nll.mean()),
            factual_accuracy=float(correct.double().mean()),
            examples=len(target),
        ),
    )
