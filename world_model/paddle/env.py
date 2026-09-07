"""Deterministic, event-integrated paddle world, measured in decision intervals.

Labels/events belong to diagnostics and supervised targets. Only ``render()``
enters the visual observer. Rectangle area coverage preserves subpixel motion.
"""

from __future__ import annotations

import copy

import numpy as np


ENVIRONMENT = {
    "schema_version": "paddle-env-v1",
    "arena_size": 64,
    "interval_seconds": 0.05,
    "ball_half_size": 2,
    "paddle_half_width": 6,
    "paddle_top": 56,
    "paddle_bottom": 59,
    "paddle_limits": [6, 58],
    "ball_x_limits": [2, 62],
    "ceiling": 2,
    "contact_plane": 54,
    "loss_line": 61,
    "catch_tolerance": 8,
    "contact_comparison_epsilon": 1e-10,
    "event_time_comparison_epsilon": 1e-10,
    "max_steps": 200,
    "action_velocity": [-4, 0, 4],
    "initial_x_range": [8, 56],
    "initial_y_range": [8, 16],
    "initial_paddle_range": [16, 48],
    "initial_vx_choices": [-6, -4, -2, 2, 4, 6],
    "initial_vy_choices": [2, 3],
    "renderer": "exact-area-ball-precedence-round-even-v1",
}

_TOL = ENVIRONMENT["event_time_comparison_epsilon"]
_PIXELS = np.arange(64, dtype=np.float64)


def rectangle_coverage(x0: float, x1: float, y0: float, y1: float) -> np.ndarray:
    """Area of rectangle within each unit pixel (empty intersections give zero)."""
    horizontal = np.clip(np.minimum(_PIXELS + 1, x1) - np.maximum(_PIXELS, x0), 0, 1)
    vertical = np.clip(np.minimum(_PIXELS + 1, y1) - np.maximum(_PIXELS, y0), 0, 1)
    return vertical[:, None] * horizontal[None, :]


class PaddleEnv:
    def __init__(self, seed: int = 0, state=None):
        self.reset(seed=seed, state=state)

    def reset(self, seed: int | None = None, state=None) -> np.ndarray:
        """Reset to independently sampled physical coordinates, or explicit state."""
        self.seed = int(seed if seed is not None else getattr(self, "seed", 0))
        if state is None:
            rng = np.random.default_rng(self.seed)
            x = rng.uniform(8, 56)
            y = rng.uniform(8, 16)
            paddle_x = rng.uniform(16, 48)
            vx = rng.choice([-6, -4, -2, 2, 4, 6])
            vy = rng.choice([2, 3])
            state = [x, y, vx, vy, paddle_x]
        self.state = np.asarray(state, dtype=np.float64).copy()
        if self.state.shape != (5,) or not np.isfinite(self.state).all():
            raise ValueError("state must contain five finite values: x,y,vx,vy,paddle_x")
        x, y, _, _, paddle_x = self.state
        if not (2 <= x <= 62 and 2 <= y <= 61 and 6 <= paddle_x <= 58):
            raise ValueError("state coordinates lie outside the physical arena")
        self.step_index = 0
        self.hit_count = 0
        self.terminated = bool(y == 61)
        self.truncated = False
        if self.terminated:
            self.state[2:4] = 0
        self.last_events = []
        return self.render()

    def clone(self) -> "PaddleEnv":
        """Copy physical state, counters, flags, and event metadata exactly."""
        return copy.deepcopy(self)

    def render(self) -> np.ndarray:
        x, y, _, _, px = self.state
        ball = rectangle_coverage(x - 2, x + 2, y - 2, y + 2)
        paddle = rectangle_coverage(px - 6, px + 6, 56, 59)
        overlap = rectangle_coverage(
            max(x - 2, px - 6), min(x + 2, px + 6), max(y - 2, 56), min(y + 2, 59)
        )
        frame = np.empty((64, 64, 3), dtype=np.float64)
        frame[:, :, 0] = ball
        frame[:, :, 1] = frame[:, :, 2] = ball + paddle - overlap
        return np.rint(np.clip(frame, 0, 1) * 255).astype(np.uint8)

    def step(self, action: int) -> tuple[np.ndarray, bool, bool, dict]:
        if self.terminated or self.truncated:
            raise RuntimeError("episode is finished; reset before taking another action")
        if isinstance(action, (bool, np.bool_)) or not isinstance(action, (int, np.integer)) or action not in (0, 1, 2):
            raise ValueError("action must be an executable integer ID: 0=left, 1=stay, 2=right")
        paddle_velocity = float((-4, 0, 4)[int(action)])
        if (self.state[4] <= 6 and paddle_velocity < 0) or (self.state[4] >= 58 and paddle_velocity > 0):
            paddle_velocity = 0.0
        elapsed = 0.0
        events = []

        def record(kind: str):
            events.append({
                "type": kind,
                "transition_index": self.step_index,
                "fractional_time": float(elapsed),
                "ball_x": float(self.state[0]),
                "ball_y": float(self.state[1]),
                "paddle_x": float(self.state[4]),
            })

        for _ in range(128):
            x, y, vx, vy, px = self.state
            remaining = 1.0 - elapsed
            candidates = [(remaining, "end")]
            if vx < 0:
                candidates.append(((2 - x) / vx, "left"))
            elif vx > 0:
                candidates.append(((62 - x) / vx, "right"))
            if vy < 0:
                candidates.append(((2 - y) / vy, "ceiling"))
            elif vy > 0:
                # Exactly on/below the plane means that this opportunity passed.
                # A missed crossing cannot reappear as a zero-time event.
                if y < 54:
                    candidates.append(((54 - y) / vy, "contact"))
                candidates.append(((61 - y) / vy, "loss"))
            if paddle_velocity:
                bound = 6 if paddle_velocity < 0 else 58
                candidates.append(((bound - px) / paddle_velocity, "paddle_bound"))
            valid = [(max(0.0, dt), kind) for dt, kind in candidates if -_TOL <= dt <= remaining + _TOL]
            delta = min(dt for dt, _ in valid)
            delta = min(delta, remaining)
            self.state[0] += vx * delta
            self.state[1] += vy * delta
            self.state[4] += paddle_velocity * delta
            elapsed = min(1.0, elapsed + delta)
            simultaneous = {kind for dt, kind in valid if abs(dt - delta) <= _TOL}
            # Independent boundaries share the same coordinates/time. Side and
            # paddle bounds are snapped before contact overlap is evaluated.
            if "left" in simultaneous or "right" in simultaneous:
                self.state[0] = 2 if "left" in simultaneous else 62
                self.state[2] = -vx
                record("side_wall")
            if "ceiling" in simultaneous:
                self.state[1] = 2
                self.state[3] = -vy
                record("ceiling")
            if "paddle_bound" in simultaneous:
                self.state[4] = 6 if paddle_velocity < 0 else 58
                paddle_velocity = 0.0
                record("paddle_bound")
            if "contact" in simultaneous:
                self.state[1] = 54
                # Inclusive edge contact has physical threshold 8. Accumulated
                # float64 motion can produce 8 + one ULP for an exact contact.
                if abs(self.state[0] - self.state[4]) <= 8 + ENVIRONMENT["contact_comparison_epsilon"]:
                    self.state[3] = -vy
                    self.hit_count += 1
                    record("paddle_hit")
                else:
                    record("paddle_miss")
            if "loss" in simultaneous:
                self.state[1] = 61
                self.state[2:4] = 0
                self.terminated = True
                record("loss")
                break  # Ball and paddle stay frozen through the interval end.
            if "end" in simultaneous or elapsed >= 1:
                break
        else:
            raise RuntimeError(f"event integration exceeded 128 iterations: state={self.state.tolist()}, time={elapsed}, events={events}")
        self.step_index += 1
        self.truncated = self.step_index >= 200 and not self.terminated
        self.last_events = events
        info = {
            "state": self.state.tolist(),
            "step_index": self.step_index,
            "timestamp": self.step_index * 0.05,
            "hit_count": self.hit_count,
            "events": copy.deepcopy(events),
        }
        return self.render(), self.terminated, self.truncated, info
