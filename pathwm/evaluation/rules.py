"""Rule-induction metrics shared by the shared-core recipe and older evaluation code.

ν per family: balanced accuracy per rule group, normalised by the family's shortcut
floor (`rule_world.floors()`); transition families are scored on Δ = y xor s.
"""

import math

import numpy as np
import torch

from pathwm.data import rule_world as rw

TRANSITION_FAMILIES = ("open", "close", "toggle")


def nu_from_rows(rows, floors):
    """rows: dicts with family, group, truth, s, p. BA per group -> ν per family."""
    groups = {}
    for r in rows:
        groups.setdefault((r["family"], r["group"]), []).append(r)
    per_family, skipped = {}, 0
    for (family, _), items in groups.items():
        truth = torch.tensor([r["truth"] for r in items])
        s = torch.tensor([r["s"] for r in items])
        predicted = torch.tensor([int(r["p"] >= 0.5) for r in items])
        if family in TRANSITION_FAMILIES:
            truth, predicted = truth ^ s, predicted ^ s
        ba = rw.balanced_accuracy(predicted, truth)
        if math.isnan(ba):
            skipped += 1
            continue
        floor = floors[family]["floor"]
        per_family.setdefault(family, []).append((ba - floor) / (1 - floor))
    result = {f: float(np.mean(v)) for f, v in per_family.items()}
    # A group whose queries contain only one scored class has undefined BA. It is
    # counted and makes the formal C2 screen incomplete; it is never a silent drop.
    result["groups_without_both_classes"] = skipped
    return result


def calibration(rows, bins=10):
    if not rows:
        return {}
    p = torch.tensor([r["p"] for r in rows])
    y = torch.tensor([float(r["truth"]) for r in rows])
    edges = torch.linspace(0, 1, bins + 1)
    ece = 0.0
    for i in range(bins):
        mask = (p >= edges[i]) & ((p < edges[i + 1]) if i < bins - 1 else (p <= 1))
        if mask.any():
            ece += float(mask.float().mean() * (p[mask].mean() - y[mask].mean()).abs())
    eps = 1e-6
    return dict(
        ece=ece,
        brier=float((p - y).square().mean()),
        nll=float(-(y * (p + eps).log() + (1 - y) * (1 - p + eps).log()).mean()),
        count=len(rows),
    )
