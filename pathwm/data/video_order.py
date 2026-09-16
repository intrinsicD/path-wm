"""Controlled pans of real image contents; not naturally observed trajectories."""

import torch


def pan_pairs(
    images, *, shifts=(2, 4), size=48, anchors=((4, 12), (4, 20), (12, 12), (12, 20))
):
    """Return [pair, member, time, RGB, H, W], labels and diagnostic metadata.

    Crop starts(-d,+d,0) yield rightward image motion in the final interval;
    (+d,-d,0) yields leftward motion. Metadata never belongs in model inputs.
    """
    if images.ndim != 4 or images.shape[1] != 3 or not len(images):
        raise ValueError("Expected nonempty RGB images [B,3,H,W]")
    if not shifts or not anchors or any(d <= 0 for d in shifts):
        raise ValueError("Positive displacements and nonempty anchors required")
    if any(
        y < 0
        or y + size > images.shape[-2]
        or x - max(shifts) < 0
        or x + max(shifts) + size > images.shape[-1]
        for y, x in anchors
    ):
        raise ValueError(
            "Crop/displacement exceeds source image; no padding or wrapping"
        )
    pairs, metadata = [], []
    for i, image in enumerate(images):
        for y, x in anchors:
            for d in shifts:
                clip = torch.stack(
                    [image[:, y : y + size, x + s : x + s + size] for s in (-d, d, 0)]
                )
                pairs.append(torch.stack((clip, clip[[1, 0, 2]])))
                metadata.append((i, y, x, d))
    frames = torch.stack(pairs)
    return dict(
        frames=frames,
        labels=torch.tensor([1, 0]).expand(len(pairs), -1).clone(),
        metadata=torch.tensor(metadata),
    )


def cyclic_pan_pairs(images, *, shifts=(2, 4)):
    """Exhaustive periodic phases: each single-frame label marginal is identical.

    Wrapped, constructed motion. Return unique views and indices as well as RGB
    pairs, so deterministic image features can be computed once per exact view.
    """
    if images.ndim != 4 or images.shape[1] != 3 or not len(images):
        raise ValueError("Expected nonempty RGB images")
    width = images.shape[-1]
    if not shifts or any(type(d) is not int or not 0 < d < width / 2 for d in shifts):
        raise ValueError(
            "Displacement must be positive and smaller than half the period"
        )
    views = torch.stack(
        [torch.roll(im, p, dims=-1) for im in images for p in range(width)]
    )
    indices, metadata = [], []
    for i in range(len(images)):
        for p in range(width):
            for d in shifts:
                sequence = [i * width + (p + s) % width for s in (-d, d, 0)]
                indices.append([sequence, [sequence[1], sequence[0], sequence[2]]])
                metadata.append((i, p, d))
    indices = torch.tensor(indices)
    return dict(
        frames=views[indices],
        views=views,
        indices=indices,
        labels=torch.tensor([0, 1]).expand(len(indices), -1).clone(),
        metadata=torch.tensor(metadata),
    )
