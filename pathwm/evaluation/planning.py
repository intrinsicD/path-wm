"""Score explicit candidate action sequences with a caller-supplied cost."""

import torch
from pathwm.models.temporal import imagine


@torch.no_grad()
def choose_sequence(features, memory, candidates, predictor, updater, cost):
    """Single observed state, candidates [N,K,A], cost(states, actions) -> [N].

    Costs, action bounds and terminal semantics belong to the application. This
    generic scorer does not reproduce the retired task-specific controller policy.
    """
    if len(memory) != 1 or candidates.ndim != 3 or len(candidates) == 0:
        raise ValueError("Expected one state and nonempty [N,K,A] candidates")
    n = len(candidates)
    branch = {k: v.expand(n, -1, -1, -1).clone() for k, v in features.items()}
    states = imagine(
        branch, memory.expand(n, -1).clone(), candidates, predictor, updater
    )
    scores = cost(states, candidates)
    if scores.shape != (n,):
        raise ValueError("Cost must return one scalar per candidate")
    valid = torch.isfinite(candidates).flatten(1).all(1) & torch.isfinite(scores)
    for obs, mem in states:
        valid &= torch.isfinite(mem).all(1)
        for x in obs.values():
            valid &= torch.isfinite(x).flatten(1).all(1)
    if not valid.any():
        raise ValueError("All candidate predictions/costs are nonfinite")
    best = int(scores.masked_fill(~valid, float("inf")).argmin())
    return candidates[best].clone(), {
        "index": best,
        "cost": float(scores[best]),
        "invalid_candidates": int((~valid).sum()),
    }
