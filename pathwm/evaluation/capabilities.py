"""Small observable-output checks for the fixed capability baseline recipe."""

import copy
import math

import numpy as np
import torch

from pathwm.data.visual_memory import pixels_hash
from pathwm.io import digest


def score(value, threshold, *, lower=False, strict=False):
    if value is None:
        return dict(status="not_measured", value=None, threshold=threshold)
    if not math.isfinite(value) or not math.isfinite(threshold):
        raise ValueError("Scores and thresholds must be finite")
    passed = value < threshold if lower else value > threshold
    passed |= not strict and value == threshold
    return dict(
        status="pass" if passed else "fail",
        value=value,
        threshold=threshold,
        direction="lower" if lower else "higher",
        strict=strict,
    )


def transform_visual(data, condition, seed=6401):
    result = copy.deepcopy(data)
    x = data.images.astype("float32")
    if condition == "mirror":
        x = x[:, :, :, ::-1]
        result.labels ^= 1
        for record in result.records:
            record["side"] ^= 1
            record["answer"] ^= 1
            record["visible_centers"] = [31 - v for v in record["visible_centers"]]
    elif condition == "swap_red_blue":
        x = x[..., [2, 1, 0]]
    elif condition == "grayscale":
        x = x.mean(-1, keepdims=True).repeat(3, axis=-1)
    elif condition == "dim":
        x *= 0.5
    elif condition == "noise":
        rng = np.random.default_rng(seed)
        noise = rng.normal(0, 25.5, (len(x) // 2, *x.shape[1:]))
        x += noise.repeat(2, axis=0)
    elif condition != "original":
        raise ValueError("Unknown visual condition")
    result.images = np.rint(x).clip(0, 255).astype("uint8")
    result.identity = dict(
        schema="visual-capability-v1",
        condition=condition,
        source=data.identity,
        noise_seed=seed,
        pixels_sha256=pixels_hash(result.images),
        labels_sha256=pixels_hash(result.labels),
    )
    return result


def prediction_metrics(predicted, target, current):
    if predicted.shape != target.shape or predicted.shape[0] != current.shape[0]:
        raise ValueError("Prediction/target alignment mismatch")
    if not all(torch.isfinite(x).all() for x in (predicted, target, current)):
        raise ValueError("Nonfinite observable output")
    return dict(
        image_mse=float((predicted - target).square().mean()),
        copy_image_mse=float((current[:, None] - target).square().mean()),
        gray_image_mse=float((0.5 - target).square().mean()),
    )


def compare_baselines(before, after):
    if before["protocol_sha256"] != after["protocol_sha256"]:
        raise ValueError("Different baseline protocols cannot be directly compared")
    old, new = ({r["id"]: r for r in x["cases"]} for x in (before, after))
    if old.keys() != new.keys():
        raise ValueError("Capability coverage changed")
    rows = []
    for name, a in old.items():
        b = new[name]
        if a["input_sha256"] != b["input_sha256"]:
            raise ValueError(f"Population changed for {name}")
        if a["metrics"].keys() != b["metrics"].keys():
            raise ValueError(f"Metric definitions changed for {name}")
        delta = {}
        for k, value in a["metrics"].items():
            other = b["metrics"][k]
            if isinstance(value, (float, int)) and isinstance(other, (float, int)):
                if not math.isfinite(value) or not math.isfinite(other):
                    raise ValueError("Nonfinite comparison metric")
                delta[k] = other - value
        rows.append(
            dict(
                id=name,
                before_checkpoint=a["checkpoint"],
                after_checkpoint=b["checkpoint"],
                delta=delta,
            )
        )
    return rows


def input_identity(value):
    """Hash actual tensors as well as metadata, without serializing giant JSON arrays."""
    if isinstance(value, torch.Tensor):
        x = value.detach().cpu().contiguous().numpy()
        return dict(shape=list(x.shape), dtype=str(x.dtype), sha256=pixels_hash(x))
    if isinstance(value, np.ndarray):
        return dict(
            shape=list(value.shape), dtype=str(value.dtype), sha256=pixels_hash(value)
        )
    if isinstance(value, dict):
        return {k: input_identity(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [input_identity(v) for v in value]
    return value


def case_record(name, checkpoint, inputs, metrics, gates, scope, raw_file):
    return dict(
        id=name,
        checkpoint=checkpoint,
        input_sha256=digest(input_identity(inputs)),
        metrics=metrics,
        gates=gates,
        scope=scope,
        raw_file=raw_file,
        status=(
            "fail"
            if any(g["status"] == "fail" for g in gates.values())
            else "pass"
            if gates
            else "measured"
        ),
    )
