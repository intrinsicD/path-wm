"""Paired rendered histories with explicit composition or relocation populations."""

from collections import Counter, defaultdict
from copy import copy, deepcopy
from itertools import product
import hashlib

import numpy as np
import torch


COLORS = np.array(
    [[220, 50, 45], [50, 180, 75], [55, 85, 225], [225, 190, 40]], dtype=np.uint8
)
BACKGROUND = 40


def draw(image, color, shape, side, selected=False, *, radius=8):
    x, y = 16 + 32 * side, 32
    if selected:
        outer, inner = radius + 4, radius + 2
        image[y - outer : y + outer, x - outer : x + outer] = 235
        image[y - inner : y + inner, x - inner : x + inner] = BACKGROUND
    if shape == 0:
        image[y - radius : y + radius, x - radius : x + radius] = COLORS[color]
    else:
        half = radius // 2
        image[y - radius : y + radius, x - half : x + half] = COLORS[color]
        image[y - half : y + half, x - radius : x + radius] = COLORS[color]


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

    def with_input_offsets(self, offsets):
        """Whole-history variants; targets stay canonical and ordering is explicit."""
        if (
            not isinstance(offsets, (tuple, list))
            or not offsets
            or any(type(x) is not int or not -255 <= x <= 255 for x in offsets)
        ):
            raise ValueError(
                "Input offsets must be a nonempty list of bounded integers"
            )
        if "input_transform" in self.identity or "input_augmentation" in self.identity:
            raise ValueError("Input offsets require untransformed source histories")
        variants = [self.images.astype(np.int16) + x for x in offsets]
        if any(x.min() < 0 or x.max() > 255 for x in variants):
            raise ValueError("Input offsets would clip observed RGB values")
        result = copy(self)
        result.images = np.concatenate(variants).astype(np.uint8)
        result.targets = np.concatenate([self.targets] * len(offsets))
        result.labels = np.concatenate([self.labels] * len(offsets))
        result.identity = dict(
            deepcopy(self.identity),
            pairs=self.identity["pairs"] * len(offsets),
            images_sha256=hashlib.sha256(result.images.tobytes()).hexdigest(),
            targets_sha256=hashlib.sha256(result.targets.tobytes()).hexdigest(),
            labels_sha256=hashlib.sha256(result.labels.tobytes()).hexdigest(),
            input_augmentation=dict(
                kind="whole-history-rgb-offsets-v1",
                offsets_uint8=list(offsets),
                source_data=deepcopy(self.identity),
            ),
        )
        return result

    def with_scenes(self, scenes):
        """Ordered whole-history scene blocks with unchanged canonical targets."""
        if (
            not isinstance(scenes, (list, tuple))
            or not scenes
            or any(not isinstance(scene, dict) for scene in scenes)
        ):
            raise ValueError("Scenes must be a nonempty list of parameter dictionaries")
        if "input_transform" in self.identity or "input_augmentation" in self.identity:
            raise ValueError("Scenes require untransformed source histories")
        variants = [self.with_scene(**scene) for scene in scenes]
        result = copy(self)
        result.images = np.concatenate([v.images for v in variants])
        result.targets = np.concatenate([v.targets for v in variants])
        result.labels = np.concatenate([v.labels for v in variants])
        result.identity = dict(
            deepcopy(self.identity),
            pairs=self.identity["pairs"] * len(variants),
            images_sha256=hashlib.sha256(result.images.tobytes()).hexdigest(),
            targets_sha256=hashlib.sha256(result.targets.tobytes()).hexdigest(),
            labels_sha256=hashlib.sha256(result.labels.tobytes()).hexdigest(),
            input_augmentation=dict(
                kind="whole-history-scenes-v1",
                scenes=[
                    deepcopy(v.identity["input_transform"]["parameters"])
                    for v in variants
                ],
                source_data=deepcopy(self.identity),
            ),
        )
        return result

    def with_scene(
        self,
        *,
        background_offset=(0, 0, 0),
        texture=0,
        clutter=False,
        radius=8,
        frame_offsets=(0, 0, 0),
        rgb_offset=(0, 0, 0),
        gain=1.0,
        shadow=0,
    ):
        """Render a controlled relocation challenge; outputs retain canonical size.

        Renderer metadata locates objects for data construction only. Neither the
        mask nor parameters are returned in model batches. Background interventions
        precede whole-image gain/offset/shadow; rounding occurs once, without clipping.
        """
        if self.curriculum != "relocation" or any(
            k in self.identity for k in ("input_transform", "input_augmentation")
        ):
            raise ValueError("Scene changes require original relocation histories")
        for value in (background_offset, frame_offsets, rgb_offset):
            if (
                not isinstance(value, (tuple, list))
                or len(value) != 3
                or any(type(x) is not int or not -255 <= x <= 255 for x in value)
            ):
                raise ValueError("Scene offsets require three bounded integers")
        if (
            type(radius) is not int
            or radius not in (4, 6, 8, 10, 12)
            or type(texture) is not int
            or not 0 <= texture <= 255
            or type(shadow) is not int
            or not 0 <= shadow <= 255
            or type(clutter) is not bool
            or type(gain) not in (int, float)
            or not np.isfinite(gain)
            or gain <= 0
        ):
            raise ValueError("Invalid scene size, texture, clutter, gain or shadow")
        images = self.images.copy()
        if radius != 8:
            for i in range(len(self)):
                images[i, :2] = self.images[i, 2]
                for side, (color, shape, final_side) in enumerate(
                    self.labels[2 * (i // 2) : 2 * (i // 2) + 2]
                ):
                    draw(images[i, 0], color, shape, side, side == i % 2, radius=radius)
                    draw(images[i, 1], color, shape, final_side, radius=radius)
        foreground = np.any(
            np.all(images[..., None, :] == COLORS, axis=-1), axis=-1
        ) | np.all(images == 235, axis=-1)
        y, x = np.indices((64, 64))
        checker = 2 * ((x // 8 + y // 8) % 2) - 1
        background_delta = np.array(background_offset) + texture * checker[..., None]
        changed = (
            images.astype(np.float64) + (~foreground[..., None]) * background_delta
        )
        if clutter:
            bands = ((y >= 8) & (y < 18)) | ((y >= 46) & (y < 56))
            changed[np.broadcast_to(bands, foreground.shape) & ~foreground] = [
                70,
                95,
                110,
            ]
        changed *= gain
        changed += np.array(frame_offsets)[None, :, None, None, None]
        changed += np.array(rgb_offset)
        changed -= shadow * (x < 32)[..., None]
        if not np.isfinite(changed).all() or changed.min() < 0 or changed.max() > 255:
            raise ValueError("Scene transform would clip observed RGB values")
        result = copy(self)
        result.images = np.rint(changed).astype(np.uint8)
        result.labels, result.targets = self.labels.copy(), self.targets.copy()
        result.identity = dict(
            deepcopy(self.identity),
            images_sha256=hashlib.sha256(result.images.tobytes()).hexdigest(),
            input_transform=dict(
                kind="relocation-scene-v1",
                source_data=deepcopy(self.identity),
                parameters=dict(
                    background_offset=list(background_offset),
                    texture=texture,
                    clutter=clutter,
                    radius=radius,
                    frame_offsets=list(frame_offsets),
                    rgb_offset=list(rgb_offset),
                    gain=float(gain),
                    shadow=shadow,
                ),
                targets="unchanged canonical color/shape/final-side rendering",
            ),
        )
        return result

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
