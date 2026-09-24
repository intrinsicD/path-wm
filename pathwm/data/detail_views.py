"""Controlled continuous instance textures; geometry is confined to the generator.

Four independent parts, each sampled from a 3x3 RGB field and a local colored
Gaussian spot. These are procedural textures, not natural images. All views and
visibility subsets are familiar; only texture instances are held out.
"""
import torch
from torch.nn import functional as F
from pathwm.models.detail_memory import from_parts, to_parts


def sample_tiles(generator, count, *, device="cpu"):
    low = torch.rand(count * 4, 3, 3, 3, generator=generator)
    smooth = F.interpolate(low, size=(8, 8), mode="bilinear", align_corners=False)
    xy = torch.rand(count * 4, 2, generator=generator) * 1.5 - .75
    color = torch.rand(count * 4, 3, 1, 1, generator=generator)
    axis = torch.linspace(-1, 1, 8)
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    blob = torch.exp(-((x[None] - xy[:, 0, None, None]).square() + (y[None] - xy[:, 1, None, None]).square()) / .08)[:, None]
    parts = (.8 * smooth + .2 * color * blob).reshape(count, 4, 192)
    return from_parts(parts).to(device)


def target_view(rgb, pose):
    if pose.shape != (len(rgb),) or pose.dtype != torch.long or ((pose < 0) | (pose > 3)).any():
        raise ValueError("Expected one quarter-turn request per image")
    # This operation is target generation only, never called by the model.
    options = torch.stack([torch.rot90(rgb, k, (-2, -1)) for k in range(4)], 1)
    return options[torch.arange(len(rgb), device=rgb.device), pose]


def region_mask(present, pose):
    parts = present.to(torch.float32)[..., None].expand(-1, -1, 192)
    return target_view(from_parts(parts), pose).bool()
