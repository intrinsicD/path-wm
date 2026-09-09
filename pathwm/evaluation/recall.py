"""Held-out scalar calibration and all-episode selective-recall measurements."""

import math
import torch
from torch.nn import functional as F


def fit_temperature(logits, labels):
    x, y = logits.detach().double().cpu(), labels.detach().long().cpu()
    if (
        x.ndim != 2
        or x.shape != (len(y), 5)
        or not len(y)
        or not torch.isfinite(x).all()
        or ((y < 0) | (y > 4)).any()
    ):
        return dict(
            status="failed", temperature=1.0, error="Invalid calibration logits/labels"
        )

    # Convex in inverse temperature; fixed bounded golden-section search, no test input.
    def loss(beta):
        return float(F.cross_entropy(x * beta, y))

    low, high, ratio = 0.05, 20.0, (math.sqrt(5) - 1) / 2
    a, b = high - ratio * (high - low), low + ratio * (high - low)
    fa, fb = loss(a), loss(b)
    for _ in range(80):
        if fa < fb:
            high, b, fb = b, a, fa
            a = high - ratio * (high - low)
            fa = loss(a)
        else:
            low, a, fa = a, b, fb
            b = low + ratio * (high - low)
            fb = loss(b)
    choices = [(loss(v), v) for v in (0.05, 20.0, 1.0, (low + high) / 2)]
    after, beta = min(choices)
    return dict(
        status="fitted",
        temperature=1 / beta,
        nll_before=loss(1.0),
        nll_after=after,
        examples=len(y),
        temperature_bounds=[0.05, 20.0],
        iterations=80,
        boundary=beta in (0.05, 20.0),
    )


def recall_metrics(logits, labels, *, temperature=1.0, abstain_cost=0.25):
    x, y = logits.detach().double().cpu(), labels.detach().long().cpu()
    if (
        x.ndim != 2
        or x.shape != (len(y), 5)
        or not len(y)
        or ((y < 0) | (y > 4)).any()
        or not math.isfinite(temperature)
        or temperature <= 0
        or not math.isfinite(abstain_cost)
        or abstain_cost < 0
    ):
        raise ValueError("Invalid recall metric inputs")
    valid = torch.isfinite(x).all(-1)
    p = (x / temperature).softmax(-1)
    best = p.argmax(-1)
    selected = valid & ((1 - p.max(-1).values) < abstain_cost)
    correct = best == y
    wrong = selected & ~correct
    n, answered = len(y), int(selected.sum())
    seen, unseen = y < 4, y == 4
    result = dict(
        examples=n,
        answered=answered,
        abstained=n - answered,
        invalid=int((~valid).sum()),
        coverage=answered / n,
        answered_error=int(wrong.sum()) / answered if answered else None,
        task_loss=(int(wrong.sum()) + abstain_cost * (n - answered)) / n,
        factual_accuracy=float((correct & valid).double().mean()),
        nll=float(F.cross_entropy(x / temperature, y)) if valid.all() else None,
        brier=float((p - F.one_hot(y, 5)).square().sum(-1).mean())
        if valid.all()
        else None,
        seen_examples=int(seen.sum()),
        unseen_examples=int(unseen.sum()),
        false_not_observed=int((selected & seen & (best == 4)).sum()),
        invented_location=int((selected & unseen & (best < 4)).sum()),
        wrong_location=int((selected & seen & (best < 4) & ~correct).sum()),
        mean_answer_confidence=float(p.max(-1).values[selected].mean())
        if answered
        else None,
    )
    return result


def confidence_diagnostics(logits, labels, temperature=1.0):
    p = (logits.detach().double().cpu() / temperature).softmax(-1)
    valid = torch.isfinite(p).all(-1)
    confidence, predicted = p.max(-1)
    correct = predicted == labels.cpu()
    bins = []
    for i in range(10):
        mask = (
            valid
            & (confidence >= i / 10)
            & ((confidence < (i + 1) / 10) if i < 9 else (confidence <= 1))
        )
        bins.append(
            dict(
                low=i / 10,
                high=(i + 1) / 10,
                count=int(mask.sum()),
                confidence=float(confidence[mask].mean()) if mask.any() else None,
                accuracy=float(correct[mask].double().mean()) if mask.any() else None,
            )
        )
    order = confidence[valid].argsort(descending=True, stable=True)
    errors = (~correct[valid][order]).double()
    ranked = confidence[valid][order]
    # A threshold cannot split exactly tied confidences.
    curve = [
        dict(coverage=(i + 1) / len(labels), risk=float(errors[: i + 1].mean()))
        for i in range(len(order))
        if i == len(order) - 1 or ranked[i] != ranked[i + 1]
    ]
    return dict(reliability=bins, risk_coverage=curve)
