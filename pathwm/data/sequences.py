"""Consecutive within-episode windows; action[t] connects image[t] to image[t+1]."""

import numpy as np
import torch
from pathwm.io import digest
from .images import pusht_records


class PushTSequences:
    def __init__(self, root, split="train", history=2, horizon=3, limit=None):
        if history < 1 or horizon < 1:
            raise ValueError("History and prediction horizon must be positive")
        self.frames, all_episodes, identity = pusht_records(root)
        self.episodes = [e for e in all_episodes if e["split"] == split]
        self.history, self.horizon = history, horizon
        self.windows = [
            (i, start)
            for i, e in enumerate(self.episodes)
            for start in range(len(e["source_rows"]) - history - horizon + 1)
        ]
        if limit is not None:
            self.windows = self.windows[:limit]
        if not self.windows:
            raise ValueError("No complete windows for this split/history/horizon")
        self.identity = dict(
            identity,
            split=split,
            history=history,
            horizon=horizon,
            windows_sha256=digest(self.windows),
            count=len(self.windows),
        )

    def __len__(self):
        return len(self.windows)

    def batch(self, indices, device="cpu"):
        result = {
            k: []
            for k in (
                "history",
                "history_actions",
                "actions",
                "initial_previous_action",
                "future",
            )
        }
        for index in indices:
            if not 0 <= index < len(self):
                raise ValueError("Sequence index out of bounds")
            ep, start = self.windows[int(index)]
            e = self.episodes[ep]
            end = start + self.history - 1
            result["history"].append(self.frames[e["source_rows"][start : end + 1]])
            result["history_actions"].append(e["actions"][start:end])
            result["actions"].append(e["actions"][end : end + self.horizon])
            result["initial_previous_action"].append(
                e["actions"][start - 1] if start else np.full(2, -1, dtype="float32")
            )
            result["future"].append(
                self.frames[e["source_rows"][end + 1 : end + 1 + self.horizon]]
            )
        out = {
            k: torch.from_numpy(np.stack(v).copy()).float().to(device)
            for k, v in result.items()
        }
        for k in ("history", "future"):
            out[k] = out[k].permute(0, 1, 4, 2, 3) / 255
        return out
