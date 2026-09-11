"""Controlled two-object episodes; simulator identities never enter model inputs."""

from collections import Counter, defaultdict
import itertools
import json
import numpy as np
import torch
from pathwm.io import digest

FEATURES = 15
NAMES = ("identity", "state", "effect")


def entity_oracle(inputs):
    """Read visible history with a nearest-descriptor oracle within the declared margin."""
    x = inputs.cpu()
    keys = [tuple(v[:8].tolist()) for v in x[0]]
    state = {k: int(x[0, i, 9]) for i, k in enumerate(keys)}
    target = keys[int(x[0, :, 11].argmax())]
    acting = int(x[1, :, 13].argmax())

    def nearest(feature):
        return keys[int((x[0, :, :8] - feature).square().sum(-1).argmin())]

    state[nearest(x[1, acting, :8])] = int(x[1, acting, 12])
    orders = (
        ([nearest(v[:8]) for v in x[2]],)
        if x[2, :, 8].all()
        else tuple(itertools.permutations(keys))
    )
    future = int(x[2, :, 14].argmax())
    out = [torch.zeros(n) for n in (2, 4, 4)]
    for order in orders:
        values = [state[k] for k in order]
        out[0][order.index(target)] += 1 / len(orders)
        out[1][2 * values[0] + values[1]] += 1 / len(orders)
        values[future] = 1 - values[future]
        out[2][2 * values[0] + values[1]] += 1 / len(orders)
    return tuple(out)


class EntityEpisodes:
    def __init__(self, split, count, noise=0.0):
        if not np.isfinite(noise) or not 0 <= noise < 0.25:
            raise ValueError("Entity noise must be finite in [0, 0.25)")
        self.noise = float(noise)
        if split not in ("train", "validation", "test") or count <= 0 or count % 32:
            raise ValueError(
                "Entity episodes require a known split and a positive multiple of 32"
            )
        self.split = split
        rng = np.random.default_rng(
            {"train": 8131, "validation": 19231, "test": 38131}[split]
        )
        noise_rng = np.random.default_rng(
            {"train": 9351, "validation": 17391, "test": 33591}[split]
        )
        rows, labels = [], [[], [], []]
        self.manifest, self.descriptor_ids, self.cohorts, self.groups = [], [], [], []
        for group in range(count // 32):
            desc = rng.normal(size=(2, 8)).astype(np.float32)
            desc /= np.linalg.norm(desc, axis=1, keepdims=True)
            self.descriptor_ids.extend(digest(d.tolist()) for d in desc)
            orders = [rng.permutation(2) for _ in range(3)]
            actor, future = rng.integers(2, size=2).tolist()
            drift = noise_rng.normal(size=(3, 2, 8)).astype(np.float32)
            drift /= np.linalg.norm(drift, axis=-1, keepdims=True)
            views = desc[None] + drift * (noise * np.linalg.norm(desc[0] - desc[1]))

            for ambiguous, bits, cue, value in itertools.product(
                (False, True), itertools.product(range(2), repeat=2), range(2), range(2)
            ):
                x = np.zeros((3, 2, FEATURES), dtype=np.float32)
                for t, order in enumerate(orders):
                    if t != 2 or not ambiguous:
                        x[t, :, :8] = views[t, order] if noise else desc[order]
                        x[t, :, 8] = 1
                x[0, :, 9] = np.array(bits)[orders[0]]
                x[0, :, 10] = 1
                x[0, list(orders[0]).index(cue), 11] = 1
                act_slot = list(orders[1]).index(actor)
                x[1, act_slot, 12:14] = (value, 1)
                x[2, future, 14] = 1
                state = list(bits)
                state[actor] = value
                target = [np.zeros(n, dtype=np.float32) for n in (2, 4, 4)]
                possible = (orders[2], orders[2][::-1]) if ambiguous else (orders[2],)
                for order in possible:
                    current = [state[e] for e in order]
                    target[0][list(order).index(cue)] += 1 / len(possible)
                    target[1][2 * current[0] + current[1]] += 1 / len(possible)
                    current[future] = 1 - current[future]
                    target[2][2 * current[0] + current[1]] += 1 / len(possible)
                cohort = "ambiguous" if ambiguous else "identifiable"
                self.manifest.append(
                    dict(
                        inputs=x.tolist(),
                        targets=[p.tolist() for p in target],
                        cohort=cohort,
                        group=group,
                    )
                )
                self.cohorts.append(cohort)
                self.groups.append(group)
                rows.append(x)
                for dest, p in zip(labels, target):
                    dest.append(p)
        self.inputs = torch.from_numpy(np.stack(rows))
        self.targets = tuple(torch.from_numpy(np.stack(y)) for y in labels)
        self.identity = dict(
            dataset="entity_episodes_v1",
            split=split,
            count=count,
            sha256=digest(self.manifest),
            episode_sha256=[digest(r) for r in self.manifest],
            descriptor_sha256=self.descriptor_ids,
        )

    def __len__(self):
        return len(self.manifest)

    def batch(self, indices, device="cpu"):
        ids = torch.as_tensor(list(indices), dtype=torch.long)
        if ((ids < 0) | (ids >= len(self))).any():
            raise ValueError("Entity episode index out of bounds")
        return dict(
            entity_inputs=self.inputs[ids].to(device),
            entity_targets=tuple(t[ids].to(device) for t in self.targets),
        )

    def final_view_bounds(self):
        result = {}
        for name, labels in zip(NAMES, self.targets):
            groups = defaultdict(Counter)
            for i, cohort in enumerate(self.cohorts):
                if cohort != "identifiable":
                    continue
                key = json.dumps(self.manifest[i]["inputs"][-1])
                groups[key][int(labels[i].argmax())] += 1
            result[name] = sum(max(c.values()) for c in groups.values()) / sum(
                sum(c.values()) for c in groups.values()
            )
        return result


class EntityMatches(EntityEpisodes):
    """Two stored unit descriptors and a query; explicit known-versus-new labels."""

    def __init__(self, split, count):
        if split not in ("train", "validation", "test") or count <= 0 or count % 4:
            raise ValueError(
                "Matching episodes require a known split and multiple of four"
            )
        self.split = split
        rng = np.random.default_rng(
            {"train": 41631, "validation": 51731, "test": 61831}[split]
        )
        self.manifest, self.descriptor_ids, self.cohorts, self.groups = [], [], [], []
        rows, labels = [], []

        def unit():
            v = rng.normal(size=8).astype(np.float32)
            return v / np.linalg.norm(v)

        for group in range(count // 4):
            memory = np.stack([unit(), unit()])
            separation = np.linalg.norm(memory[0] - memory[1])
            self.descriptor_ids.extend(digest(v.tolist()) for v in memory)
            for target in (0, 1, 2, 2):
                for _ in range(10000):
                    q = (
                        memory[target] + 0.15 * separation * unit()
                        if target < 2
                        else unit()
                    )
                    q /= np.linalg.norm(q)
                    distances = np.linalg.norm(memory - q, axis=-1)
                    if (target < 2 and distances[target] < 0.35 * separation) or (
                        target == 2 and distances.min() > 0.65 * separation
                    ):
                        break
                else:
                    raise RuntimeError("Unable to sample novelty margin")
                x = np.zeros((3, 2, FEATURES), dtype=np.float32)
                x[0, :, :8] = memory
                x[-1, 0, :8] = q
                target_prob = np.eye(3, dtype=np.float32)[target]
                cohort = "known" if target < 2 else "novel"
                self.manifest.append(
                    dict(
                        inputs=x.tolist(),
                        targets=[target_prob.tolist()],
                        cohort=cohort,
                        group=group,
                    )
                )
                self.cohorts.append(cohort)
                self.groups.append(group)
                rows.append(x)
                labels.append(target_prob)
                self.descriptor_ids.append(digest(q.tolist()))
        self.inputs = torch.from_numpy(np.stack(rows))
        self.targets = (torch.from_numpy(np.stack(labels)),)
        self.identity = dict(
            dataset="entity_matches_v1",
            split=split,
            count=count,
            sha256=digest(self.manifest),
            episode_sha256=[digest(r) for r in self.manifest],
            descriptor_sha256=self.descriptor_ids,
        )

    def final_view_bounds(self):
        return {}
