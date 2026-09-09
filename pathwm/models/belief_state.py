"""Explicit categorical belief snapshots and caller-owned event transactions."""

from dataclasses import dataclass, fields, replace

import torch

from .agent_state import LatentState
from .modalities import Observation
from .tasks import Provenance


def tree_map(value, fn):
    if isinstance(value, torch.Tensor):
        return fn(value)
    if isinstance(value, tuple):
        return tuple(tree_map(x, fn) for x in value)
    if hasattr(value, "__dataclass_fields__"):
        return replace(
            value,
            **{f.name: tree_map(getattr(value, f.name), fn) for f in fields(value)},
        )
    return value


@dataclass(frozen=True)
class MemoryRecord:
    belief: torch.Tensor
    evidence: torch.Tensor
    valid: torch.Tensor
    start_time: torch.Tensor
    end_time: torch.Tensor
    sources: tuple[str, ...]
    event_id: str
    ordinal: int
    h: torch.Tensor | None = None
    logits: torch.Tensor | None = None
    z: torch.Tensor | None = None
    omitted_sources: int = 0
    author: str = ""
    detail: str = ""
    score: float = 0.0
    state_time: torch.Tensor | None = None
    last_observed_time: torch.Tensor | None = None
    source_valid: torch.Tensor | None = None
    source_start: torch.Tensor | None = None
    source_end: torch.Tensor | None = None

    def to_dict(self):
        return {
            f.name: tree_map(getattr(self, f.name), lambda x: x.detach().clone())
            for f in fields(self)
        }


@dataclass(frozen=True)
class SessionMemory:
    session_id: str
    recent: tuple[MemoryRecord, ...] = ()
    staging: tuple[MemoryRecord, ...] = ()
    compressed: tuple[MemoryRecord, ...] = ()
    protected: tuple[MemoryRecord, ...] = ()
    consolidated: MemoryRecord | None = None

    @property
    def sources(self):
        return tuple(s for record in self.recent for s in record.sources)

    def to_dict(self):
        return dict(
            session_id=self.session_id,
            **{
                name: [r.to_dict() for r in getattr(self, name)]
                for name in ("recent", "staging", "compressed", "protected")
            },
            consolidated=None
            if self.consolidated is None
            else self.consolidated.to_dict(),
        )

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        for key in ("recent", "staging", "compressed", "protected"):
            data[key] = tuple(MemoryRecord(**r) for r in data[key])
        if data["consolidated"] is not None:
            data["consolidated"] = MemoryRecord(**data["consolidated"])
        return cls(**data)


@dataclass(frozen=True, kw_only=True)
class BeliefState(LatentState):
    h: torch.Tensor
    logits: torch.Tensor
    z: torch.Tensor
    stochastic: torch.Tensor  # hard forward sample, straight-through backward
    prior_logits: torch.Tensor
    evidence: torch.Tensor
    evidence_valid: torch.Tensor
    evidence_start: torch.Tensor
    evidence_end: torch.Tensor
    source_valid: torch.Tensor
    source_start: torch.Tensor
    source_end: torch.Tensor
    observation_counts: torch.Tensor
    session_id: str
    event_id: str = ""
    ordinal: int = -1
    phase: str = "initial"
    sources: tuple[str, ...] = ()
    time_unit: str = "steps"

    def to_dict(self):
        # log_scale is a zero compatibility field, never categorical uncertainty.
        return dict(
            schema="pathwm-belief-v1",
            **{
                f.name: tree_map(getattr(self, f.name), lambda x: x.detach().clone())
                for f in fields(self)
                if f.name not in ("memory", "generated_ancestry")
            },
            memory=None if self.memory is None else self.memory.to_dict(),
            generated_ancestry=[p.to_dict() for p in self.generated_ancestry],
        )

    @classmethod
    def from_dict(cls, record):
        record = dict(record)
        if record.pop("schema", None) != "pathwm-belief-v1":
            raise ValueError(
                "Unknown categorical belief schema; no implicit Gaussian conversion"
            )
        if "generated_ancestry" not in record:
            raise ValueError("Belief schema requires provenance")
        record["generated_ancestry"] = tuple(
            Provenance.from_dict(p) for p in record["generated_ancestry"]
        )
        if record["memory"] is not None:
            record["memory"] = SessionMemory.from_dict(record["memory"])
        state = cls(**record)
        if state.z.dtype != torch.long or state.z.shape != state.logits.shape[:-1]:
            raise ValueError("Invalid categorical code in belief schema")
        return state

    def to(self, device, dtype=None):
        return tree_map(
            self,
            lambda x: x.to(
                device=device,
                dtype=(dtype or x.dtype)
                if x.is_floating_point() and x.dtype != torch.float64
                else x.dtype,
            ),
        )


@dataclass(frozen=True)
class Packet:
    source: str
    modality: str
    observation: Observation


@dataclass(frozen=True)
class PendingEvent:
    prior: BeliefState
    state: BeliefState
    noise: torch.Tensor
    packets: tuple[Packet, ...] = ()
    replay: bool = False
