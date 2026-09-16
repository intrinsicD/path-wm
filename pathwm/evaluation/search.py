"""Bounded branch scheduling with the same prefix-only interface in replay/live.

An action (branch, depth) opens the next root branch or advances a known frontier.
The executor owns all training/state. These helpers cannot predict unseen outcomes.
"""

import math
from statistics import mean

from pathwm.io import digest

POLICIES = ("round_robin12", "round_robin8", "greedy", "plateau1", "plateau2")


class UnsupportedReplay(Exception):
    pass


def _node(row):
    """Expose numerical outcomes only, never checkpoints or private executor state."""
    b, d = row["branch"], row["depth"]
    if type(b) is not int or type(d) is not int or b < 0 or d < 1:
        raise ValueError("Invalid branch/depth")
    score, seconds = float(row["score"]), float(row["seconds"])
    if not math.isfinite(score) or not math.isfinite(seconds) or seconds < 0:
        raise ValueError("Nonfinite outcome or invalid cost")
    return dict(branch=b, depth=d, score=score, seconds=seconds)


def _action(policy, prefix, branches, depth):
    rows = [[n for n in prefix if n["branch"] == b] for b in range(branches)]
    unopened = next((b for b, r in enumerate(rows) if not r), None)
    if unopened is not None:
        return unopened, 1
    available = [b for b, r in enumerate(rows) if len(r) < depth]
    if policy.startswith("round_robin"):
        return min(
            ((b, len(rows[b]) + 1) for b in available),
            key=lambda a: (a[1], a[0]),
            default=None,
        )
    if policy.startswith("plateau"):
        patience = int(policy[-1])
        available = [
            b
            for b in available
            if len(rows[b]) <= patience
            or any(
                rows[b][i]["score"] > max(n["score"] for n in rows[b][:i]) + 0.002
                for i in range(len(rows[b]) - patience, len(rows[b]))
            )
        ]
    if not available:
        return None
    b = max(available, key=lambda b: (rows[b][-1]["score"], -b))
    return b, len(rows[b]) + 1


def run_search(policy, execute, *, branches=4, depth=4):
    """execute(branch, depth) performs exactly one real/recorded continuation.

    Policies only see the observed prefix. A missing replay successor is unsupported,
    not a successful early stop. Utility is a fixed development selection objective.
    """
    if (
        policy not in POLICIES
        or type(branches) is not int
        or type(depth) is not int
        or min(branches, depth) < 1
    ):
        raise ValueError("Unknown policy or invalid horizon")
    limit = 8 if policy == "round_robin8" else 12
    prefix, decisions = [], []
    status = "complete"
    for _ in range(limit):
        action = _action(policy, prefix, branches, depth)
        decisions.append(
            dict(visible=[(r["branch"], r["depth"]) for r in prefix], action=action)
        )
        if action is None:
            break
        try:
            row = _node(execute(*action))
        except UnsupportedReplay:
            status = "unsupported"
            break
        if (row["branch"], row["depth"]) != action:
            raise ValueError("Executor returned a different action")
        prefix.append(row)
    best = max(prefix, key=lambda r: r["score"], default=None)
    return dict(
        policy=policy,
        status=status,
        nodes=prefix,
        decisions=decisions,
        steps=len(prefix),
        seconds=sum(r["seconds"] for r in prefix),
        best_score=None if best is None else best["score"],
        best=None if best is None else (best["branch"], best["depth"]),
        utility=None
        if status != "complete" or best is None
        else best["score"] - 0.001 * len(prefix),
    )


def replay_search(policy, history):
    rows = {}
    branches, depth = (history["contract"][k] for k in ("branches", "depth"))
    for value in history["nodes"]:
        row = _node(value)
        key = row["branch"], row["depth"]
        if key in rows:
            raise ValueError("Duplicate replay node")
        if key[0] >= branches or key[1] > depth:
            raise ValueError("Replay node outside contract")
        if key[1] > 1 and (key[0], key[1] - 1) not in rows:
            raise ValueError("Replay node lacks recorded parent")
        if key[1] == 1 and any((b, 1) not in rows for b in range(key[0])):
            raise ValueError("Replay root branches are out of order")
        rows[key] = row

    def execute(b, d):
        if (b, d) not in rows:
            raise UnsupportedReplay((b, d))
        return rows[b, d]

    return run_search(policy, execute, branches=branches, depth=depth)


def select_policy(histories):
    """Finite predeclared policy set; incumbent is always the first candidate."""
    if not histories or len({h["world"] for h in histories}) != len(histories):
        raise ValueError("Need distinct development worlds")
    if any(h["contract"] != histories[0]["contract"] for h in histories):
        raise ValueError("Incompatible replay contract")
    candidates = []
    for policy in POLICIES:
        replay = [replay_search(policy, h) for h in histories]
        if any(r["utility"] is None for r in replay):
            raise ValueError("Incomplete replay support for policy selection")
        candidates.append(
            dict(
                policy=policy,
                mean_utility=mean(r["utility"] for r in replay),
                replay=replay,
            )
        )
    best = max(candidates, key=lambda r: r["mean_utility"])
    return dict(
        selected=best["policy"],
        mean_utility=best["mean_utility"],
        contract=histories[0]["contract"],
        candidates=candidates,
        history_digests=[digest(h) for h in histories],
        worlds=[h["world"] for h in histories],
        scope="Fixed historical replay only; fresh execution is required.",
    )
