"""Small rendered observation histories; labels and renderer metadata stay separate."""

import hashlib
import numpy as np
import torch


def pixels_hash(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


class VisualMemoryEpisodes:
    def __init__(self, pairs=32, *, seed=3101):
        if type(pairs) is not int or pairs < 1:
            raise ValueError("Positive scene-pair count required")
        rng = np.random.default_rng(seed)
        scenes, records = [], []
        for group in range(pairs):
            grey = rng.integers(40, 110) + rng.integers(0, 5, (32, 32, 1))
            base = np.repeat(grey, 3, axis=2).astype("uint8")
            centers = [int(8 + rng.integers(-1, 2)), int(24 + rng.integers(-1, 2))]
            blue = [
                int(rng.integers(30, 65)),
                int(rng.integers(80, 130)),
                int(rng.integers(150, 200)),
            ]
            for x in centers:
                base[19:28, x - 5 : x + 5] = blue
                base[18:20, x - 5 : x + 5] = [150, 150, 150]
            red = [
                int(rng.integers(210, 246)),
                int(rng.integers(25, 61)),
                int(rng.integers(20, 61)),
            ]
            radius = int(rng.integers(2, 4))

            def visible(x):
                image = base.copy()
                image[11:17, x - radius : x + radius] = red
                image[12:16, x + radius : x + radius + 2] = red
                image[13:15, x + radius : x + radius + 1] = base[
                    13:15, x + radius : x + radius + 1
                ]
                return image

            reversal = bool(group % 2)
            pair = []
            for side in (0, 1):
                first = visible(centers[1 - side]) if reversal else visible(16)
                frames = np.stack((first, visible(centers[side]), base, base))
                pair.append(frames)
                records.append(
                    dict(
                        scene_group=group,
                        side=side,
                        reversal=reversal,
                        visible_centers=[
                            centers[1 - side] if reversal else 16,
                            centers[side],
                        ],
                        last_visible_frame=1,
                        query_cutoff_frame=3,
                        answer=side,
                        question="Which container was the mug last visibly at?",
                        hidden_current_state="not independently observed",
                        red=red,
                        radius=radius,
                    )
                )
            scenes.append(pair)
        self.images = np.asarray(scenes, dtype="uint8").reshape(-1, 4, 32, 32, 3)
        self.labels = np.tile(np.array([0, 1], dtype="int64"), pairs)
        self.records = records
        self.identity = dict(
            schema="visual-container-v1",
            seed=seed,
            pairs=pairs,
            scene_sha256=[
                pixels_hash(self.images[2 * i : 2 * i + 2]) for i in range(pairs)
            ],
            final_sha256=[pixels_hash(self.images[2 * i, -1]) for i in range(pairs)],
        )

    def __len__(self):
        return len(self.labels)

    def batch(self, indices, device="cpu"):
        indices = np.asarray(list(indices), dtype="int64")
        if ((indices < 0) | (indices >= len(self))).any():
            raise IndexError("Episode outside dataset")
        return dict(
            images=torch.from_numpy(self.images[indices].copy())
            .permute(0, 1, 4, 2, 3)
            .to(device)
            .float()
            / 255,
            labels=torch.from_numpy(self.labels[indices].copy()).to(device),
        )
