"""Explicit finite fact and two-record binding diagnostics."""

import torch
from torch.nn import functional as F

from pathwm.models.facts import bind_facts


def fact_metrics(entity_logits, location_logits, entities, locations):
    if (
        not torch.isfinite(entity_logits).all()
        or not torch.isfinite(location_logits).all()
    ):
        raise ValueError("Nonfinite fact predictions")
    entity_ok = entity_logits.argmax(-1) == entities
    location_ok = location_logits.argmax(-1) == locations
    entity_nll = float(F.cross_entropy(entity_logits.double(), entities))
    location_nll = float(F.cross_entropy(location_logits.double(), locations))
    return dict(
        examples=len(entities),
        entity_accuracy=float(entity_ok.double().mean()),
        location_accuracy=float(location_ok.double().mean()),
        factual_accuracy=float((entity_ok & location_ok).double().mean()),
        entity_nll=entity_nll,
        location_nll=location_nll,
        nll=(entity_nll + location_nll) / 2,
    )


def extraction_gates(train, development):
    fit = (
        min(train["entity_accuracy"], train["location_accuracy"]) >= 0.95
        and train["factual_accuracy"] >= 0.9
        and train["nll"] <= 0.35
    )
    fresh = (
        min(development["entity_accuracy"], development["location_accuracy"]) >= 0.9
        and development["factual_accuracy"] >= 0.8
        and development["nll"] <= 0.5
    )
    return dict(training_fit=fit, fresh_combinations=fresh, extraction=fit and fresh)


def binding_reference(entity_logits, location_logits, entities, locations):
    """Exhaustive unordered record pairs; both exact queries, cached facts only."""
    if len(entities) != 128 or set(zip(entities.tolist(), locations.tolist())) != {
        (e, loc) for e in range(32) for loc in range(4)
    }:
        raise ValueError(
            "Binding reference requires every entity/location combination once"
        )
    lookup = torch.empty(32, 4, dtype=torch.long)
    lookup[entities, locations] = torch.arange(128)
    cases = torch.tensor(
        [
            (a, x, b, y)
            for a in range(32)
            for b in range(a + 1, 32)
            for x in range(4)
            for y in range(4)
            if x != y
        ]
    )
    a, x, b, y = cases.T
    pairs = torch.stack((lookup[a, x], lookup[b, y]), -1)
    swaps = torch.stack((lookup[a, y], lookup[b, x]), -1)
    queries = torch.stack((a, b), -1).flatten()

    def predict(indices):
        return bind_facts(
            entity_logits[indices].repeat_interleave(2, 0),
            location_logits[indices].repeat_interleave(2, 0),
            queries,
        )

    probabilities, changed = predict(pairs), predict(swaps)
    targets = locations[pairs].flatten()
    swap_targets = locations[swaps].flatten()
    correct = probabilities.argmax(-1) == targets
    swap_correct = changed.argmax(-1) == swap_targets

    def groups(indices, logp, target):
        held = (entities[indices] + locations[indices]) % 4 == 0
        result = {}
        for name, mask in [
            ("seen", ~held.any(-1)),
            ("mixed", held.sum(-1) == 1),
            ("held_out", held.all(-1)),
        ]:
            m = mask.repeat_interleave(2)
            good = (logp.argmax(-1) == target).reshape(-1, 2)
            result[name] = dict(
                pairs=int(mask.sum()),
                queries=int(m.sum()),
                accuracy=float(good[mask].double().mean()),
                paired_success=float(good[mask].all(-1).double().mean()),
                nll=float(F.nll_loss(logp[m], target[m])),
                constituent_baseline=fact_metrics(
                    entity_logits[indices[mask]].flatten(0, 1),
                    location_logits[indices[mask]].flatten(0, 1),
                    entities[indices[mask]].flatten(),
                    locations[indices[mask]].flatten(),
                ),
            )
        return result

    scored = groups(pairs, probabilities, targets)
    return dict(
        status="evaluated",
        groups=scored,
        swapped_groups=groups(swaps, changed, swap_targets),
        coherent_swap_success=float((correct & swap_correct).double().mean()),
        order_max_log_probability_delta=float(
            (probabilities - predict(pairs.flip(1))).abs().max()
        ),
        gate=scored["held_out"]["accuracy"] >= 0.9
        and scored["held_out"]["paired_success"] >= 0.8,
        scope="Explicit selector with exact entity queries; not the world-model task reader. Swapped facts use their own constituent split groups.",
    ), dict(
        cases=cases,
        queries=queries,
        targets=targets,
        log_probabilities=probabilities,
        swap_targets=swap_targets,
        swapped_log_probabilities=changed,
    )
