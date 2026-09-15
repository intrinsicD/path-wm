"""Portable world-state records. Labels describe claims, not guaranteed semantics."""

from dataclasses import dataclass, field
import math
import torch


def finite_time(value):
    if (
        isinstance(value, bool)
        or not isinstance(value, (float, int))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError("Time must be finite and nonnegative")
    return float(value)


def identifier(value):
    if not isinstance(value, str) or not value or len(value) > 512:
        raise ValueError("Expected nonempty bounded identifier")
    return value


@dataclass(frozen=True)
class Limits:
    entities: int = 128
    components: int = 2048
    relations: int = 512
    evidence: int = 1024
    events: int = 512
    values: int = 262144
    bytes: int = 16777216

    def __post_init__(self):
        if any(type(v) is not int or v < 1 for v in vars(self).values()):
            raise ValueError("Store limits must be positive integers")


@dataclass(frozen=True)
class Entity:
    id: str
    label: str
    kind: str
    created_at: float
    last_seen: float | None = None
    existence_confidence: float | None = None


@dataclass(frozen=True)
class Evidence:
    id: str
    event_id: str
    source: str
    modality: str
    occurred_at: float
    available_at: float
    content_ref: str = ""
    content_hash: str = ""
    data: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Component:
    id: str
    entity_id: str
    name: str
    values: tuple[float, ...]
    shape: tuple[int, ...]
    space: str
    model_version: str
    evidence: tuple[str, ...]
    valid_from: float
    available_at: float
    revision: int
    role: str = "observed"
    confidence: float | None = None
    data: dict = field(default_factory=dict)
    parents: tuple[str, ...] = ()
    active: bool = True

    def tensor(self, *, device=None, dtype=torch.float32):
        return torch.tensor(self.values, device=device, dtype=dtype).reshape(self.shape)


@dataclass(frozen=True)
class Endpoint:
    ref: str
    kind: str = "entity"  # entity or component; resolved within the pinned snapshot


@dataclass(frozen=True)
class Relation:
    id: str
    source: Endpoint
    target: Endpoint
    type: str
    evidence: tuple[str, ...]
    valid_from: float
    available_at: float
    revision: int
    confidence: float | None = None
    active: bool = True
    data: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Event:
    id: str
    occurred_at: float
    available_at: float
    kind: str
    payload: dict = field(default_factory=dict)

    def __post_init__(self):
        identifier(self.id)
        if not isinstance(self.payload, dict):
            raise ValueError("Event payload must be a mapping")
        finite_time(self.occurred_at)
        finite_time(self.available_at)
        if self.occurred_at > self.available_at:
            raise ValueError("Event cannot be available before it occurred")
        if self.kind not in {
            "observation",
            "correction",
            "internal",
            "prediction",
            "action",
        }:
            raise ValueError("Unknown event kind")
