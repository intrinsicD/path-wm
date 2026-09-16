"""Request-path diagnostics: frozen accessibility is distinct from generated use."""

from collections import defaultdict
import numpy as np
import torch
from torch.nn import functional as F

from pathwm.io import evaluation_mode
from pathwm.models.tasks import Actor, OutputControl, TaskRequest, TaskSession
from pathwm.models.photo_probe import RidgeReader


def validate_request_route(model, question_mode):
    """Removing observed questions requires the real request in the task path."""
    if question_mode not in ("full", "neutral", "masked"):
        raise ValueError("Unknown observation question mode")
    if question_mode != "full" and (
        getattr(model.core, "request_readout", "none") != "instruction"
        or model.core.agent.task_interpreter is None
    ):
        raise ValueError("Question routing controls require an instruction source")


def capture_request_stages(model, state, question):
    """Use the deployed task path with a fixed physical context and constant metadata."""
    captured, hooks = {}, []

    def encoder(module, args, pyramid):
        values = []
        for scale in pyramid.scales:
            x = scale.values.masked_fill(~scale.valid[..., None], 0)
            values.append(F.adaptive_avg_pool1d(x.transpose(1, 2), 16).flatten(1))
        captured["encoder"] = torch.cat(values, 1).detach().cpu()

    def instruction(module, args, output):
        captured["instruction"] = output.detach().cpu().flatten(1)

    agent = model.core.agent
    if (
        agent.task_interpreter is None
        or model.core.request_readout != "instruction"
        or model.core.belief_readout != "sampled"
    ):
        raise ValueError("Request diagnosis requires a sampled instruction source")
    caller = Actor("user", "caller")
    task = TaskSession(
        TaskRequest(
            "readout-request",
            question,
            caller,
            (OutputControl("text", "required", caller),),
        )
    )
    with evaluation_mode(model), torch.no_grad():
        try:
            hooks.append(agent.encoders["text"].register_forward_hook(encoder))
            hooks.append(
                agent.task_interpreter.read_instruction.register_forward_hook(
                    instruction
                )
            )
            goal = agent.task_tokens(state, [task])
            tokens = agent.think(state, steps=2, goal=goal).tokens
        finally:
            for hook in hooks:
                hook.remove()
        captured["task"] = goal.detach().cpu().flatten(1)
        captured["working"] = tokens.detach().cpu().flatten(1)
    return captured, tokens


def request_metrics(predictions, rows):
    predictions = np.asarray(predictions)
    if predictions.shape != (len(rows),):
        raise ValueError("One prediction per request required")
    result = {}
    for split in ("calibration", "validation", "test"):
        ids = [i for i, r in enumerate(rows) if r["split"] == split]
        groups = defaultdict(list)
        for i in ids:
            groups[rows[i].get("context", "0"), rows[i]["pair"]].append(i)
        if not ids or any(
            len(v) != 2 or {rows[i]["label"] for i in v} != {0, 1}
            for v in groups.values()
        ):
            raise ValueError("Need complete opposite-label pairs in every split")
        correct = predictions == np.array([r["label"] for r in rows])
        result[split] = dict(
            accuracy=float(correct[ids].mean()),
            paired=float(np.mean([correct[v].all() for v in groups.values()])),
            requests=len(ids),
            pairs=len(groups),
        )
    return result


def request_probe(features, rows, *, family_flip_seed=None):
    """Small shared ridge method, calibration statistics and validation selection only."""
    x = torch.as_tensor(features).double().cpu()
    if x.ndim != 2 or len(x) != len(rows) or not torch.isfinite(x).all():
        raise ValueError("Finite aligned feature matrix required")
    labels = np.array([r["label"] for r in rows])
    splits = np.array([r["split"] for r in rows])
    train, validation = splits == "calibration", splits == "validation"
    if family_flip_seed is not None:
        rng = np.random.default_rng(family_flip_seed)
        # Flip whole pairs together by prefix family; preserve nuisance balance.
        families = dict.fromkeys(
            r["family"] for r in rows if r["split"] == "calibration"
        )
        flips = {family: int(rng.integers(2)) for family in families}
        # Same family is shared across contexts; one final flip per family.
        labels = np.array([y ^ flips.get(r["family"], 0) for y, r in zip(labels, rows)])
    y = F.one_hot(torch.as_tensor(labels[train]), 2).double()
    factor = RidgeReader.factor(x[train], y)
    candidates = []
    for alpha in (0.1, 1.0, 10.0):
        solved = RidgeReader.solve(factor, ridge=alpha / x.shape[1])
        weights = solved["training"].T @ solved["alpha"] / x.shape[1]
        saved = {k: solved[k] for k in ("mean", "std", "target_mean")} | dict(
            weights=weights
        )
        scores = ((x - saved["mean"]) / saved["std"]) @ weights + saved["target_mean"]
        pred = scores.argmax(-1).numpy()
        candidates.append(
            (float((pred[validation] == labels[validation]).mean()), alpha, pred, saved)
        )
    _, alpha, pred, saved = max(candidates, key=lambda c: c[0])
    return dict(
        alpha=alpha,
        features=x.shape[1],
        predictions=pred.tolist(),
        metrics=request_metrics(pred, rows),
        regularizers=[
            dict(alpha=a, metrics=request_metrics(p, rows)) for _, a, p, _ in candidates
        ],
        family_flip_seed=family_flip_seed,
    ), saved
