"""Fixed physical design for E0's thin deterministic 2D causal world (§9 E0, DDR §18).

What: normalized-world geometry, radii, integration constants, spawn slots, and batched circle/AABB
collision helpers. This is the single code location for physical numbers and is frozen with E0.
How: semi-implicit Euler integrates force-controlled circles; boundary and static-wall contacts project
centres out of expanded AABBs and damp the normal velocity; agent/object contacts transfer velocity.
Why: E0 must be reproducible, vectorized over independent worlds, and rich enough for action-sensitive
transitions, object permanence, paired interventions, and two routes around a fixed obstacle.
"""
from __future__ import annotations

import torch

# World and dynamics. These values are part of the E0 engine, not experiment knobs (DDR §13.16).
WORLD_MIN = 0.0
WORLD_MAX = 1.0
DT = 0.05
FORCE_GAIN = 4.0
AGENT_FRICTION = 0.86
OBJECT_FRICTION_RANGE = (0.88, 0.94)
MAX_SPEED = 1.25
CONTACT_RESTITUTION = 0.15
PUSH_TRANSFER = 0.80

AGENT_RADIUS = 0.035
OBJECT_RADIUS = 0.040
MAX_OBJECTS = 4

# One tall central obstacle creates top and bottom homotopy classes from SPAWN_REGION to GOAL_RECT.
WALL_RECTS = ((0.46, 0.24, 0.54, 0.76),)  # xmin, ymin, xmax, ymax
CONTAINER_RECT = (0.72, 0.66, 0.92, 0.88)
CONTAINER_RIM = 0.018
GOAL_RECT = (0.78, 0.08, 0.94, 0.22)
SPAWN_REGION = (0.10, 0.36, 0.22, 0.64)

# Slot 0 begins inside the container and is invisible; a later interaction can push it back out.
OBJECT_SPAWN_SLOTS = (
    (0.82, 0.77),
    (0.29, 0.22),
    (0.28, 0.79),
    (0.70, 0.43),
)
SPAWN_JITTER = 0.018

BACKGROUND_RGB = (22, 26, 32)
GOAL_RGB = (35, 72, 46)
WALL_RGB = (112, 118, 128)
CONTAINER_FLOOR_RGB = (31, 52, 59)
CONTAINER_RIM_RGB = (61, 139, 151)
AGENT_RGB = (245, 201, 74)
OBJECT_RGB = ((222, 76, 76), (82, 142, 230), (86, 190, 113), (190, 104, 214))


def integrate_velocity(velocity: torch.Tensor, force: torch.Tensor, friction: torch.Tensor | float) -> torch.Tensor:
    """Semi-implicit Euler velocity update, clipped per body without coupling worlds."""
    velocity = velocity * friction + force * (FORCE_GAIN * DT)
    speed = velocity.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return velocity * (MAX_SPEED / speed).clamp(max=1.0)


def resolve_bounds(
    position: torch.Tensor,
    velocity: torch.Tensor,
    radius: torch.Tensor | float,
    active: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep circle centres within the unit square and reflect only their normal velocity."""
    low = torch.as_tensor(radius, dtype=position.dtype, device=position.device)
    while low.ndim < position.ndim:
        low = low.unsqueeze(0)
    high = WORLD_MAX - low
    hit = (position < low) | (position > high)
    if active is not None:
        hit = hit & active[..., None]
    projected = position.clamp(min=low, max=high)
    position = torch.where(hit, projected, position)
    velocity = torch.where(hit, -CONTACT_RESTITUTION * velocity, velocity)
    return position, velocity


def resolve_walls(
    position: torch.Tensor,
    velocity: torch.Tensor,
    radius: torch.Tensor | float,
    active: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project circle centres out of each radius-expanded fixed wall AABB."""
    r = torch.as_tensor(radius, dtype=position.dtype, device=position.device)
    while r.ndim < position.ndim - 1:
        r = r.unsqueeze(0)
    for xmin, ymin, xmax, ymax in WALL_RECTS:
        left, bottom, right, top = xmin - r, ymin - r, xmax + r, ymax + r
        x, y = position[..., 0], position[..., 1]
        inside = (x > left) & (x < right) & (y > bottom) & (y < top)
        if active is not None:
            inside = inside & active
        distances = torch.stack((x - left, right - x, y - bottom, top - y), dim=-1)
        side = distances.argmin(dim=-1)
        for index, boundary, axis in ((0, left, 0), (1, right, 0), (2, bottom, 1), (3, top, 1)):
            mask = inside & (side == index)
            position_axis = position[..., axis]
            velocity_axis = velocity[..., axis]
            position[..., axis] = torch.where(mask, boundary, position_axis)
            velocity[..., axis] = torch.where(mask, -CONTACT_RESTITUTION * velocity_axis, velocity_axis)
    return position, velocity


def inside_container(position: torch.Tensor) -> torch.Tensor:
    """True when a circle centre is behind the container's opaque rim."""
    xmin, ymin, xmax, ymax = CONTAINER_RECT
    return (
        (position[..., 0] > xmin + CONTAINER_RIM)
        & (position[..., 0] < xmax - CONTAINER_RIM)
        & (position[..., 1] > ymin + CONTAINER_RIM)
        & (position[..., 1] < ymax - CONTAINER_RIM)
    )


def update_winding(
    agent_position: torch.Tensor,
    previous_angle: torch.Tensor,
    winding: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Accumulate unwrapped agent angles around fixed obstacle centroids, in turns."""
    centres = torch.tensor(
        [[(xmin + xmax) * 0.5, (ymin + ymax) * 0.5] for xmin, ymin, xmax, ymax in WALL_RECTS],
        dtype=agent_position.dtype,
        device=agent_position.device,
    )
    relative = agent_position[:, None, :] - centres[None, :, :]
    angle = torch.atan2(relative[..., 1], relative[..., 0])
    delta = torch.atan2(torch.sin(angle - previous_angle), torch.cos(angle - previous_angle))
    return angle, winding + delta / (2.0 * torch.pi)
