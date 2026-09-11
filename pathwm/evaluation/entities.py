"""Proper answer scores, paired history controls and explicit finite-screen gates."""

import torch
from pathwm.data.entities import NAMES


def entity_metrics(logits, targets, cohorts, groups):
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
