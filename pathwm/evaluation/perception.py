"""Qualification of Rule World slot perception (docs/shared-core-plan.md, S4).

Rebuilds the earlier C1 measurement for the retrained perception: for fresh scenes,
the slot that wins at an entity's scene coordinate must belong to that entity, and its
readouts must give the right attributes and lamp.
"""

import torch

from pathwm.data import rule_world as rw
from pathwm.models.slots import pointer

THRESHOLDS = dict(attribute=0.95, lamp=0.99, machine_pointer=0.99, object_pointer=0.97)


@torch.no_grad()
def perception_metrics(perception, generator, kinds, count, device, chunk=64):
    totals = dict(attributes=torch.zeros(4), lamp=0.0, machine_pointer=0.0, object_pointer=0.0,
                  machine_detection=0.0, reconstruction=0.0)
    seen = 0
    for start in range(0, count, chunk):
        n = min(chunk, count - start)
        scenes, lamps, rgb, entity = rw.sample_frames(generator, kinds, n)
        p = perception(rgb.to(device))
        entity = entity.to(device)
        owner = p.alpha.argmax(1)  # [B,H,W] winning slot
        rows = torch.arange(n, device=device)

        def dominant(slot):
            """Entity with most pixels among those the slot wins."""
            counts = torch.stack([((owner == slot[:, None, None]) & (entity == e)).flatten(1).sum(-1)
                                  for e in range(rw.ENTITIES)], -1)
            return counts.argmax(-1)

        for m in range(2):
            slot = pointer(p.alpha, scenes.machine_xy[:, m].to(device))
            totals["machine_pointer"] += float((dominant(slot) == 1 + m).float().sum())
            totals["lamp"] += float(((p.lamp[rows, slot] > 0).long().cpu() == lamps[:, m]).float().sum())
        for i in range(4):
            slot = pointer(p.alpha, scenes.object_xy[:, i].to(device))
            totals["object_pointer"] += float((dominant(slot) == 3 + i).float().sum())
            totals["attributes"] += (p.attributes[rows, slot].argmax(-1).cpu() == scenes.attrs[:, i]).float().sum(0)
        machine_slots = p.kind.softmax(-1)[..., 1].topk(2, dim=-1).indices
        found = torch.stack([dominant(machine_slots[:, k]) for k in range(2)], -1).sort(-1).values
        totals["machine_detection"] += float((found == torch.tensor([1, 2], device=device)).all(-1).float().sum())
        totals["reconstruction"] += float((p.recon - rgb.to(device)).square().mean((1, 2, 3)).sum())
        seen += n
    return dict(attribute_accuracy=(totals["attributes"] / (4 * seen)).tolist(),
                lamp_accuracy=totals["lamp"] / (2 * seen),
                machine_pointer_accuracy=totals["machine_pointer"] / (2 * seen),
                object_pointer_accuracy=totals["object_pointer"] / (4 * seen),
                machine_detection=totals["machine_detection"] / seen,
                reconstruction_mse=totals["reconstruction"] / seen, scenes=seen)


def qualification(metrics, thresholds=THRESHOLDS):
    """Per-criterion pass flags; `passed` needs all of them."""
    gates = dict(attribute=min(metrics["attribute_accuracy"]) >= thresholds["attribute"],
                 lamp=metrics["lamp_accuracy"] >= thresholds["lamp"],
                 machine_pointer=metrics["machine_pointer_accuracy"] >= thresholds["machine_pointer"],
                 object_pointer=metrics["object_pointer_accuracy"] >= thresholds["object_pointer"])
    return dict(gates, passed=all(gates.values()))
