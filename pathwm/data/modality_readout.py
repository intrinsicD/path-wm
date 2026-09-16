"""Paired symbolic modalities for controlled latent-readout experiments.

The tone code and arrows are explicit supervised representations of three factors,
not natural speech/vision semantics. No target values are passed to the core.
"""

import itertools
import torch

from pathwm.models.modalities import Observation, bytes_batch

KINDS = ("text", "image", "audio", "video")
MODES = (*KINDS, "all", "complementary")
COLORS = ("rot", "gruen", "blau")
PLACES = ("links", "mitte", "rechts")
DIRECTIONS = ("west", "ost")


def canonical(factors):
    factors = torch.as_tensor(factors, dtype=torch.long)
    b = len(factors)
    c, p, d = factors.unbind(1)
    rgb = torch.tensor([[0.9, 0.15, 0.1], [0.1, 0.85, 0.2], [0.1, 0.2, 0.9]])[c]
    y, x = torch.meshgrid(torch.arange(16), torch.arange(16), indexing="ij")
    frames = []
    for t in range(4):
        center = torch.tensor([4.0, 7.0, 10.0])[p] + t * (2 * d - 1)
        dx = (x[None] - center[:, None, None]) * (2 * d - 1)[:, None, None]
        dy = y[None] - 7
        mask = ((dx >= 0) & (dx <= 2) & (dy.abs() <= 2 - dx)) | (
            (dx >= -2) & (dx < 0) & (dy.abs() <= 0)
        )
        frames.append(torch.where(mask[:, None], rgb[:, :, None, None], 0.04))
    video = torch.stack(frames, 1)
    freq = torch.stack((3 + 2 * c, 2 + 2 * p, 3 + 4 * d), 1)
    phase = torch.arange(64).float() / 64
    audio = 0.6 * torch.sin(2 * torch.pi * freq[..., None] * phase)
    sentences = [
        f"{COLORS[cc]} {PLACES[pp]} {DIRECTIONS[dd]}" for cc, pp, dd in factors.tolist()
    ]
    text, mask = bytes_batch(sentences)
    return dict(
        text=text,
        text_valid=mask,
        image=video[:, 0],
        video=video,
        audio=audio.reshape(b, -1),
        factors=factors,
    )


def dataset(split, seed=7201):
    splits = ("train", "validation", "seen", "heldout", "intervention")
    if split not in splits:
        raise ValueError("Unknown split")
    combinations = [
        x
        for x in itertools.product(range(3), range(3), range(2))
        if split == "intervention" or (((x[0] + x[1]) % 3 == 0) == (split == "heldout"))
    ]
    views = {"train": 12, "validation": 4, "seen": 4, "heldout": 8, "intervention": 4}[
        split
    ]
    factors = torch.tensor([x for x in combinations for _ in range(views)])
    targets = canonical(factors)
    g = torch.Generator().manual_seed(seed + 10000 * splits.index(split))
    b = len(factors)
    # Small independent observation noise; canonical targets remain unchanged.
    image = (
        targets["image"] + 0.015 * torch.randn(targets["image"].shape, generator=g)
    ).clamp(0, 1)
    video = (
        targets["video"] + 0.015 * torch.randn(targets["video"].shape, generator=g)
    ).clamp(0, 1)
    audio = targets["audio"].reshape(b, 3, 64) + 0.01 * torch.randn(
        b, 3, 64, generator=g
    )
    prefixes = ("", "hier: ", "bitte: ")
    choice = torch.randint(3, (b,), generator=g).tolist()
    strings = [
        prefixes[i] + f"{COLORS[c]} {PLACES[p]} {DIRECTIONS[d]}"
        for i, (c, p, d) in zip(choice, factors.tolist())
    ]
    text, valid = bytes_batch(strings)
    inputs = dict(
        text=Observation(text, torch.full_like(text, 4, dtype=torch.float64), valid),
        image=Observation(image[:, None], torch.full((b, 1), 4.0, dtype=torch.float64)),
        audio=Observation(
            audio, torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64).expand(b, -1)
        ),
        video=Observation(video, torch.arange(1, 5, dtype=torch.float64).expand(b, -1)),
    )
    # Complementary sources: color only / location only / direction only.
    c, p, d = factors.unbind(1)
    color = canonical(torch.stack((c, torch.ones_like(p), torch.ones_like(d)), 1))[
        "image"
    ]
    direction = canonical(torch.stack((torch.zeros_like(c), torch.ones_like(p), d), 1))[
        "video"
    ]
    direction = direction.mean(2, keepdim=True).expand(-1, -1, 3, -1, -1).clone()
    request, request_valid = bytes_batch(["beschreibe"] * b)
    complementary = dict(
        image=Observation(color[:, None], inputs["image"].times),
        video=Observation(direction, inputs["video"].times),
        audio=Observation(
            audio,
            inputs["audio"].times,
            torch.tensor([False, True, False]).expand(b, -1),
        ),
        text=Observation(
            request, torch.full_like(request, 4, dtype=torch.float64), request_valid
        ),
    )
    return dict(
        inputs=inputs,
        complementary=complementary,
        targets=targets,
        split=split,
        seed=seed,
        ids=[f"{split}:{i}" for i in range(b)],
    )


def observations(data, mode, indices=None, device="cpu", *, omit=None):
    if mode not in MODES:
        raise ValueError("Unknown input mode")
    source = data["complementary"] if mode == "complementary" else data["inputs"]
    keys = KINDS if mode in ("all", "complementary") else (mode,)
    idx = slice(None) if indices is None else indices
    return {
        k: Observation(
            o.values[idx].to(device),
            o.times[idx].to(device),
            None if o.valid is None else o.valid[idx].to(device),
        )
        for k in keys
        if k != omit
        for o in (source[k],)
    }
