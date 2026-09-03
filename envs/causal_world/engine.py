"""Batched E0 environment: deterministic force-driven circles, fixed walls, rendering and snapshots.

What: N independent top-down worlds with one agent, 2-4 movable objects, a central obstacle, a goal
region and an opaque container. reset/step/render emit owned RGB uint8 tensors; ground_truth exposes
physical state, segmentation and obstacle winding only to probes and diagnostics.
How: tensorized semi-implicit Euler and collision projection run across N; save clones every state
tensor plus the private torch.Generator state, and restore clones it again for repeatable forks.
Why: the first slice needs reproducible action-sensitive transitions for E1, while E0 interventions
and later planner branches require bit-exact save/restore (DDR §18; Invariant 11).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from envs.causal_world import physics
from world_state.abi import ABI


@dataclass(frozen=True)
class Snapshot:
    """Opaque, reusable fork state. restore() clones fields so replay never mutates this object."""

    agent_position: torch.Tensor
    agent_velocity: torch.Tensor
    object_position: torch.Tensor
    object_velocity: torch.Tensor
    object_mass: torch.Tensor
    object_friction: torch.Tensor
    object_active: torch.Tensor
    previous_angle: torch.Tensor
    winding: torch.Tensor
    rng_state: torch.Tensor
    step_count: int


class CausalWorld:
    """The deterministic RGB variant used by the first slice and E1_reference."""

    def __init__(self, cfg: dict, abi: ABI) -> None:
        if cfg.get("variant", "deterministic") != "deterministic":
            raise ValueError("the thin E0 engine currently implements only env.variant='deterministic'")
        if "rgb" not in cfg.get("modalities", ["rgb"]):
            raise ValueError("the first-slice causal world requires the rgb modality")
        self.n_worlds = int(cfg["n_worlds"])
        self.resolution = int(cfg["resolution"])
        if self.n_worlds < 1 or self.resolution < 8:
            raise ValueError("env.n_worlds must be positive and env.resolution must be at least 8")
        if abi.action_dims != 2:
            raise ValueError(f"causal_world requires two force dimensions, ABI declares {abi.action_dims}")
        self.abi = abi
        self.generator = torch.Generator(device="cpu")
        coordinates = (torch.arange(self.resolution, dtype=torch.float32) + 0.5) / self.resolution
        self.y_pixels, self.x_pixels = torch.meshgrid(coordinates, coordinates, indexing="ij")
        self._initialized = False

    def _require_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError("call reset(seed) before using the environment")

    def reset(self, seed: int) -> torch.Tensor:
        self.generator.manual_seed(int(seed))
        n = self.n_worlds
        xmin, ymin, xmax, ymax = physics.SPAWN_REGION
        unit = torch.rand(n, 2, generator=self.generator)
        self.agent_position = torch.stack(
            (xmin + (xmax - xmin) * unit[:, 0], ymin + (ymax - ymin) * unit[:, 1]), dim=-1
        )
        self.agent_velocity = torch.zeros(n, 2)

        slots = torch.tensor(physics.OBJECT_SPAWN_SLOTS, dtype=torch.float32)
        jitter = (torch.rand(n, physics.MAX_OBJECTS, 2, generator=self.generator) * 2.0 - 1.0) * physics.SPAWN_JITTER
        self.object_position = slots[None, :, :].expand(n, -1, -1).clone() + jitter
        self.object_velocity = torch.zeros(n, physics.MAX_OBJECTS, 2)
        counts = torch.randint(2, physics.MAX_OBJECTS + 1, (n,), generator=self.generator)
        self.object_active = torch.arange(physics.MAX_OBJECTS)[None, :] < counts[:, None]
        self.object_mass = 0.75 + 0.50 * torch.rand(n, physics.MAX_OBJECTS, generator=self.generator)
        friction_low, friction_high = physics.OBJECT_FRICTION_RANGE
        self.object_friction = friction_low + (friction_high - friction_low) * torch.rand(
            n, physics.MAX_OBJECTS, generator=self.generator
        )

        obstacle_count = len(physics.WALL_RECTS)
        zero_angle = torch.zeros(n, obstacle_count)
        self.previous_angle, _ = physics.update_winding(self.agent_position, zero_angle, zero_angle)
        self.winding = torch.zeros(n, obstacle_count)
        self.step_count = 0
        self._initialized = True
        return self.render()

    def _resolve_agent_object_contacts(
        self,
        agent_position: torch.Tensor,
        agent_velocity: torch.Tensor,
        object_position: torch.Tensor,
        object_velocity: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        offset = object_position - agent_position[:, None, :]
        distance = offset.norm(dim=-1)
        contact = self.object_active & (distance < physics.AGENT_RADIUS + physics.OBJECT_RADIUS)
        normal = offset / distance.clamp_min(1e-7)[..., None]
        fallback = torch.zeros_like(normal)
        fallback[..., 0] = 1.0
        normal = torch.where((distance > 1e-7)[..., None], normal, fallback)
        overlap = (physics.AGENT_RADIUS + physics.OBJECT_RADIUS - distance).clamp_min(0.0) * contact

        correction = normal * overlap[..., None]
        object_position = object_position + 0.75 * correction
        agent_position = agent_position - 0.25 * correction.sum(dim=1)
        approach = ((agent_velocity[:, None, :] - object_velocity) * normal).sum(dim=-1).clamp_min(0.0)
        impulse = physics.PUSH_TRANSFER * approach * contact / self.object_mass
        object_velocity = object_velocity + normal * impulse[..., None]
        agent_velocity = agent_velocity - 0.20 * (normal * impulse[..., None]).sum(dim=1)
        return agent_position, agent_velocity, object_position, object_velocity

    def _resolve_object_contacts(
        self, position: torch.Tensor, velocity: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        for i in range(physics.MAX_OBJECTS):
            for j in range(i + 1, physics.MAX_OBJECTS):
                offset = position[:, j] - position[:, i]
                distance = offset.norm(dim=-1)
                contact = self.object_active[:, i] & self.object_active[:, j] & (
                    distance < 2.0 * physics.OBJECT_RADIUS
                )
                normal = offset / distance.clamp_min(1e-7)[:, None]
                fallback = torch.zeros_like(normal)
                fallback[:, 0] = 1.0
                normal = torch.where((distance > 1e-7)[:, None], normal, fallback)
                overlap = (2.0 * physics.OBJECT_RADIUS - distance).clamp_min(0.0) * contact
                correction = 0.5 * normal * overlap[:, None]
                position[:, i] = position[:, i] - correction
                position[:, j] = position[:, j] + correction

                approach = ((velocity[:, i] - velocity[:, j]) * normal).sum(dim=-1).clamp_min(0.0) * contact
                impulse = 0.5 * approach[:, None] * normal
                velocity[:, i] = velocity[:, i] - impulse
                velocity[:, j] = velocity[:, j] + impulse
        return position, velocity

    def step(self, action: torch.Tensor) -> torch.Tensor:
        self._require_initialized()
        if action.ndim != 2 or tuple(action.shape) != (self.n_worlds, self.abi.action_dims):
            raise ValueError(
                f"action shape {tuple(action.shape)} != ({self.n_worlds}, {self.abi.action_dims})"
            )
        if action.dtype != torch.float32:
            raise ValueError(f"action dtype {action.dtype} != torch.float32")
        self.abi.check_actions(action)
        if not torch.isfinite(action).all():
            raise ValueError("actions must be finite")

        agent_velocity = physics.integrate_velocity(self.agent_velocity, action, physics.AGENT_FRICTION)
        agent_position = self.agent_position + physics.DT * agent_velocity
        agent_position, agent_velocity = physics.resolve_bounds(
            agent_position, agent_velocity, physics.AGENT_RADIUS
        )
        agent_position, agent_velocity = physics.resolve_walls(
            agent_position, agent_velocity, physics.AGENT_RADIUS
        )

        object_velocity = physics.integrate_velocity(
            self.object_velocity, torch.zeros_like(self.object_velocity), self.object_friction[..., None]
        )
        object_velocity = torch.where(self.object_active[..., None], object_velocity, self.object_velocity)
        object_position = self.object_position + physics.DT * object_velocity
        object_position, object_velocity = physics.resolve_bounds(
            object_position, object_velocity, physics.OBJECT_RADIUS, self.object_active
        )
        object_position, object_velocity = physics.resolve_walls(
            object_position, object_velocity, physics.OBJECT_RADIUS, self.object_active
        )
        agent_position, agent_velocity, object_position, object_velocity = self._resolve_agent_object_contacts(
            agent_position, agent_velocity, object_position, object_velocity
        )
        object_position, object_velocity = self._resolve_object_contacts(object_position, object_velocity)
        agent_position, agent_velocity = physics.resolve_bounds(
            agent_position, agent_velocity, physics.AGENT_RADIUS
        )
        agent_position, agent_velocity = physics.resolve_walls(
            agent_position, agent_velocity, physics.AGENT_RADIUS
        )
        object_position, object_velocity = physics.resolve_bounds(
            object_position, object_velocity, physics.OBJECT_RADIUS, self.object_active
        )
        object_position, object_velocity = physics.resolve_walls(
            object_position, object_velocity, physics.OBJECT_RADIUS, self.object_active
        )

        self.agent_position = agent_position
        self.agent_velocity = agent_velocity
        self.object_position = object_position
        self.object_velocity = object_velocity
        self.previous_angle, self.winding = physics.update_winding(
            self.agent_position, self.previous_angle, self.winding
        )
        self.step_count += 1
        return self.render()

    def _rectangle_mask(self, rectangle: tuple[float, float, float, float]) -> torch.Tensor:
        xmin, ymin, xmax, ymax = rectangle
        return (self.x_pixels >= xmin) & (self.x_pixels <= xmax) & (self.y_pixels >= ymin) & (self.y_pixels <= ymax)

    def _circle_mask(self, centre: torch.Tensor, radius: float) -> torch.Tensor:
        dx = self.x_pixels[None, :, :] - centre[:, 0, None, None]
        dy = self.y_pixels[None, :, :] - centre[:, 1, None, None]
        return dx.square() + dy.square() <= radius * radius

    def _paint(
        self,
        rgb: torch.Tensor,
        segmentation: torch.Tensor,
        mask: torch.Tensor,
        color: tuple[int, int, int],
        label: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if mask.ndim == 2:
            mask = mask[None, :, :].expand(self.n_worlds, -1, -1)
        paint = torch.tensor(color, dtype=torch.uint8)[:, None, None]
        rgb = torch.where(mask[:, None, :, :], paint[None, :, :, :], rgb)
        segmentation = torch.where(mask, torch.tensor(label, dtype=segmentation.dtype), segmentation)
        return rgb, segmentation

    def _render_with_segmentation(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._require_initialized()
        background = torch.tensor(physics.BACKGROUND_RGB, dtype=torch.uint8)[None, :, None, None]
        rgb = background.expand(self.n_worlds, 3, self.resolution, self.resolution).clone()
        segmentation = torch.zeros(self.n_worlds, self.resolution, self.resolution, dtype=torch.int16)

        rgb, segmentation = self._paint(
            rgb, segmentation, self._rectangle_mask(physics.GOAL_RECT), physics.GOAL_RGB, 20
        )
        container = self._rectangle_mask(physics.CONTAINER_RECT)
        rgb, segmentation = self._paint(rgb, segmentation, container, physics.CONTAINER_FLOOR_RGB, 8)
        for wall in physics.WALL_RECTS:
            rgb, segmentation = self._paint(
                rgb, segmentation, self._rectangle_mask(wall), physics.WALL_RGB, 7
            )

        hidden = physics.inside_container(self.object_position)
        for index in range(physics.MAX_OBJECTS):
            visible = self.object_active[:, index] & ~hidden[:, index]
            mask = self._circle_mask(self.object_position[:, index], physics.OBJECT_RADIUS) & visible[:, None, None]
            rgb, segmentation = self._paint(
                rgb, segmentation, mask, physics.OBJECT_RGB[index], 2 + index
            )

        xmin, ymin, xmax, ymax = physics.CONTAINER_RECT
        inner = (
            xmin + physics.CONTAINER_RIM,
            ymin + physics.CONTAINER_RIM,
            xmax - physics.CONTAINER_RIM,
            ymax - physics.CONTAINER_RIM,
        )
        rim = container & ~self._rectangle_mask(inner)
        rgb, segmentation = self._paint(rgb, segmentation, rim, physics.CONTAINER_RIM_RGB, 9)
        rgb, segmentation = self._paint(
            rgb,
            segmentation,
            self._circle_mask(self.agent_position, physics.AGENT_RADIUS),
            physics.AGENT_RGB,
            1,
        )
        return rgb, segmentation

    def render(self) -> torch.Tensor:
        rgb, _ = self._render_with_segmentation()
        return rgb.clone()  # caller ownership is explicit even though rendering already allocates

    def ground_truth(self) -> dict[str, torch.Tensor]:
        _, segmentation = self._render_with_segmentation()
        hidden = physics.inside_container(self.object_position) & self.object_active
        step = torch.full((self.n_worlds, 1), float(self.step_count))
        full_state = torch.cat(
            (
                self.agent_position,
                self.agent_velocity,
                self.object_position.flatten(1),
                self.object_velocity.flatten(1),
                self.object_mass,
                self.object_friction,
                self.object_active.float(),
                hidden.float(),
                step,
            ),
            dim=-1,
        )
        return {
            "full_state": full_state.clone(),
            "segmentation": segmentation.clone(),
            "homotopy_signature": self.winding.clone(),
        }

    def save(self) -> Snapshot:
        self._require_initialized()
        return Snapshot(
            agent_position=self.agent_position.clone(),
            agent_velocity=self.agent_velocity.clone(),
            object_position=self.object_position.clone(),
            object_velocity=self.object_velocity.clone(),
            object_mass=self.object_mass.clone(),
            object_friction=self.object_friction.clone(),
            object_active=self.object_active.clone(),
            previous_angle=self.previous_angle.clone(),
            winding=self.winding.clone(),
            rng_state=self.generator.get_state().clone(),
            step_count=self.step_count,
        )

    def restore(self, snapshot: Snapshot) -> None:
        if not isinstance(snapshot, Snapshot):
            raise TypeError(f"snapshot must be {Snapshot.__name__}, got {type(snapshot).__name__}")
        self.agent_position = snapshot.agent_position.clone()
        self.agent_velocity = snapshot.agent_velocity.clone()
        self.object_position = snapshot.object_position.clone()
        self.object_velocity = snapshot.object_velocity.clone()
        self.object_mass = snapshot.object_mass.clone()
        self.object_friction = snapshot.object_friction.clone()
        self.object_active = snapshot.object_active.clone()
        self.previous_angle = snapshot.previous_angle.clone()
        self.winding = snapshot.winding.clone()
        self.generator.set_state(snapshot.rng_state.clone())
        self.step_count = snapshot.step_count
        self._initialized = True
