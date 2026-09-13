"""Paired rendered histories with explicit composition or relocation populations."""

from collections import Counter, defaultdict
from itertools import product
import hashlib

import numpy as np
import torch


COLORS = np.array(
    [[220, 50, 45], [50, 180, 75], [55, 85, 225], [225, 190, 40]], dtype=np.uint8
)
BACKGROUND = 40


def draw(image, color, shape, side, selected=False):
    x, y = 16 + 32 * side, 32
    if selected:
        image[y - 12 : y + 12, x - 12 : x + 12] = 235
        image[y - 10 : y + 10, x - 10 : x + 10] = BACKGROUND
    if shape == 0:
        image[y - 8 : y + 8, x - 8 : x + 8] = COLORS[color]
    else:
        image[y - 8 : y + 8, x - 4 : x + 4] = COLORS[color]
        image[y - 4 : y + 4, x - 8 : x + 8] = COLORS[color]


def templates(device="cpu"):
    labels = list(product(range(4), range(2), range(2)))
    images = np.full((16, 64, 64, 3), BACKGROUND, dtype=np.uint8)
    for image, label in zip(images, labels):
        draw(image, *label)
    return (
        torch.from_numpy(images).permute(0, 3, 1, 2).to(device).float() / 255,
        torch.tensor(labels, device=device),
    )


def image_labels(images):
    reference, labels = templates(images.device)
    distances = (images[:, None] - reference[None]).square().mean((2, 3, 4))
    return labels[distances.argmin(1)]


class MemoryOutputEpisodes:
    def __init__(
        self,
        pairs=128,
        *,
        seed=7701,
        split="train",
        curriculum="parity",
        input_offset=0,
    ):
        if type(pairs) is not int or pairs < 1 or split not in ("train", "test"):
            raise ValueError("Positive pair count and train/test split required")
        if curriculum not in ("parity", "relocation"):
            raise ValueError("Unknown memory-output curriculum")
        if curriculum == "relocation" and pairs % 16:
            raise ValueError("Relocation requires complete cycles of16 selection pairs")
        if type(input_offset) is not int or not -255 <= input_offset <= 255:
            raise ValueError("Input offset must be an integer in [-255, 255]")
        self.curriculum = curriculum
        rng = np.random.default_rng(seed)
        images, labels, targets = [], [], []
        for group in range(pairs):
            parity, swapped = group % 2, (group // 4) % 2
            left_shape = parity ^ int(split == "test")
            left_color = parity + 2 * ((group // 2) % 2)
            objects = [
                (left_color, left_shape),
                (2 + 2 * parity - left_color, 1 - left_shape),
            ]
            if curriculum == "relocation":
                # A quartet fixes initial identity/appearance; only selection and
                # later movement vary. Both appearance factors vary independently.
                block, swapped = group // 2, group % 2
                parity = block % 2
                left_color = parity + 2 * ((block // 2) % 2)
                left_shape = (block // 4) % 2
                objects = [
                    (left_color, left_shape),
                    (2 + 2 * parity - left_color, 1 - left_shape),
                ]
            if curriculum == "parity" or group % 2 == 0:
                background = np.full((64, 64, 3), rng.integers(35, 46), dtype=np.uint8)
                background[:4] = rng.integers(30, 51, (4, 64, 1))
            initial_swap = swapped if curriculum == "parity" else 0
            final_swap = swapped if curriculum == "relocation" else 0
            for selected in (0, 1):
                first, last = background.copy(), background.copy()
                for side, (color, shape) in enumerate(objects):
                    draw(first, color, shape, side ^ initial_swap, side == selected)
                    draw(last, color, shape, side ^ final_swap)
                target = np.full_like(background, BACKGROUND)
                label = (*objects[selected], selected ^ final_swap)
                draw(target, *label)
                images.append(np.stack((first, last, background)))
                labels.append(label)
                targets.append(target)
        self.images, self.targets = np.stack(images), np.stack(targets)
        self.labels = np.array(labels, dtype=np.int64)
        source_images_hash = hashlib.sha256(self.images.tobytes()).hexdigest()
        if input_offset:
            shifted = self.images.astype(np.int16) + input_offset
            if shifted.min() < 0 or shifted.max() > 255:
                raise ValueError("Input offset would clip observed RGB values")
            self.images = shifted.astype(np.uint8)
        self.identity = dict(
            schema="memory-output-v1"
            if curriculum == "parity"
            else "memory-output-relocation-v1",
            seed=seed,
            pairs=pairs,
            split=split,
            triples=sorted(set(map(tuple, labels))),
            images_sha256=hashlib.sha256(self.images.tobytes()).hexdigest(),
            targets_sha256=hashlib.sha256(self.targets.tobytes()).hexdigest(),
            labels_sha256=hashlib.sha256(self.labels.tobytes()).hexdigest(),
        )

        if input_offset:
            self.identity["input_transform"] = dict(
                kind="additive-rgb-offset-v1",
                offset_uint8=input_offset,
                source_images_sha256=source_images_hash,
                targets="unchanged canonical rendering",
            )

    def shortcut_audit(self):
        """Empirical majority bounds over exact available inputs, not model scores."""

        def majority(keys, answers):
            groups = defaultdict(Counter)
            for key, answer in zip(keys, answers):
                groups[key][answer] += 1
            return sum(max(c.values()) for c in groups.values()) / len(self)

        triples = list(map(tuple, self.labels.tolist()))
        sides = self.labels[:, 2].tolist()
        frames = [[x.tobytes() for x in self.images[:, t]] for t in range(3)]
        return dict(
            appearance_only_side_accuracy=majority([x[:2] for x in triples], sides),
            initial_frame_only_side_accuracy=majority(frames[0], sides),
            final_frame_only_joint_accuracy=majority(frames[1], triples),
            hidden_frame_only_side_accuracy=majority(frames[2], sides),
            hidden_frame_only_joint_accuracy=majority(frames[2], triples),
            tuple_counts={str(k): v for k, v in sorted(Counter(triples).items())},
        )

    def __len__(self):
        return len(self.labels)

    def batch(self, indices, device="cpu"):
        ids = np.asarray(list(indices), dtype=np.int64)
        if ((ids < 0) | (ids >= len(self))).any():
            raise IndexError("Episode outside dataset")
        return dict(
            images=torch.from_numpy(self.images[ids])
            .permute(0, 1, 4, 2, 3)
            .to(device)
            .float()
            / 255,
            target=torch.from_numpy(self.targets[ids])
            .permute(0, 3, 1, 2)
            .to(device)
            .float()
            / 255,
            labels=torch.from_numpy(self.labels[ids]).to(device),
        )
