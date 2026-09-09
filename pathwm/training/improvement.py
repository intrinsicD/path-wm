"""A bounded, measured learning proposal with complete rejection rollback."""

import copy
import math
import random

import numpy as np
import torch

from pathwm.io import evaluation_mode


def replay_probabilities(errors, uniform_fraction=0.25):
    if (
        errors.ndim != 1
        or errors.numel() == 0
        or not torch.isfinite(errors).all()
        or (errors < 0).any()
    ):
        raise ValueError(
            "Replay priorities must be a nonempty finite nonnegative vector"
        )
    if not 0 < uniform_fraction <= 1:
        raise ValueError("Uniform replay fraction must be in (0,1]")
    priority = errors.detach().clamp_min(1e-8)
    return (1 - uniform_fraction) * priority / priority.sum() + uniform_fraction / len(
        errors
    )


def try_improvement(
    model,
    optimizer,
    propose,
    evaluate,
    *,
    primary,
    min_improvement,
    tolerances,
    generators=(),
):
    """Lower metric values are better. All evaluated metrics need a tolerance.

    evaluate() must read a fixed held-out population and return finite scalars.
    propose() performs a bounded update. Include every mutable training module
    (e.g. EMA teacher and replay buffers) in model, and pass any extra RNG generators.
    This function cannot roll back file writes or arbitrary callback side effects.
    """
    if (
        primary not in tolerances
        or not math.isfinite(min_improvement)
        or min_improvement <= 0
        or any(not math.isfinite(v) or v < 0 for v in tolerances.values())
    ):
        raise ValueError(
            "Declare a positive minimum improvement and finite nonnegative tolerances"
        )
    weights, opt = (
        copy.deepcopy(model.state_dict()),
        copy.deepcopy(optimizer.state_dict()),
    )
    gradients = [
        None if p.grad is None else p.grad.detach().clone() for p in model.parameters()
    ]
    modes = {module: module.training for module in model.modules()}
    rng = (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state(),
        torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None,
        [g.get_state() for g in generators],
    )

    def rollback():
        model.load_state_dict(weights)
        optimizer.load_state_dict(opt)
        for p, grad in zip(model.parameters(), gradients):
            p.grad = grad
        random.setstate(rng[0])
        np.random.set_state(rng[1])
        torch.set_rng_state(rng[2])
        if rng[3] is not None:
            torch.cuda.set_rng_state_all(rng[3])
        for generator, saved in zip(generators, rng[4]):
            generator.set_state(saved)

    def measure():
        if any(
            v.is_floating_point() and not torch.isfinite(v).all()
            for v in model.state_dict().values()
        ):
            raise ValueError("Learning proposal has nonfinite model or training state")
        with torch.no_grad(), evaluation_mode(model):
            result = {k: float(v) for k, v in evaluate().items()}
        if set(result) != set(tolerances) or not all(
            math.isfinite(v) for v in result.values()
        ):
            raise ValueError(
                "Evaluation must return finite values for exactly the declared metrics"
            )
        return result

    try:
        before = measure()
        propose()
        after = measure()
        gain = before[primary] - after[primary]
        accepted = gain >= min_improvement and all(
            after[k] <= before[k] + tol for k, tol in tolerances.items()
        )
        if not accepted:
            rollback()
        return dict(
            accepted=accepted,
            before=before,
            after=after,
            improvement=gain,
            primary=primary,
            min_improvement=min_improvement,
            tolerances=dict(tolerances),
        )
    except BaseException:
        rollback()
        raise
    finally:
        for module, mode in modes.items():
            module.training = mode
