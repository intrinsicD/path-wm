"""Grouped real photographs and the fixed blank-ended recall episode."""

import json
from pathlib import Path
import numpy as np
import torch
from .images import CocoFrames, Frames


def photo_data(root, counts=(1024, 128, 256), seed=45001):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    splits = ("train", "validation", "test")
    records = manifest["records"]
    groups = {s: {records[i]["group"] for i in manifest["splits"][s]} for s in splits}
    if any(
        groups[a] & groups[b]
        for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]
    ):
        raise ValueError("Photo duplicate group overlap across splits")
    if len(counts) != 3 or min(counts) < 1:
        raise ValueError("Positive photo counts required")
    rng = np.random.default_rng(seed)
    result = {}
    # Verify the immutable prepared frame store once, not separately per subset.
    base = CocoFrames(root, "train")
    for split, count in zip(splits, counts):
        rows = []
        used = set()
        for i in rng.permutation(manifest["splits"][split]):
            group = records[i]["group"]
            if group not in used:
                rows.append(int(i))
                used.add(group)
                if len(rows) == count:
                    break
        if len(rows) != count:
            raise ValueError("Not enough distinct photo groups")
        result[split] = Frames(
            base.frames,
            rows,
            identity=dict(
                base.identity,
                split=split,
                selection_seed=seed,
                selected=[dict(row=i, **records[i]) for i in rows],
            ),
        )
    return result


def photo_history(rgb):
    if rgb.ndim != 4 or rgb.shape[1:] != (3, 64, 64):
        raise ValueError("Photo history requires RGB64")
    return torch.stack([rgb, rgb, torch.full_like(rgb, 40 / 255)], 1)
