"""Exact task/output records and small learned task consumers.

Records are authoritative caller metadata. Neural encodings are learned context,
never a way to reconstruct authority. Callers must retain provenance on derivation;
this interface does not authenticate callers or detect deliberately stripped tags.
"""

from dataclasses import asdict, dataclass, replace
import json

import torch
from torch import nn

from .modalities import Attend, bytes_batch


OPERATIONS = ("think", "recall", "imagine", "act", "emit", "ask", "finish")
MODES = ("required", "disabled", "automatic")


def identifier(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("IDs and modality names must be nonempty strings")


@dataclass(frozen=True)
class Actor:
    role: str
    identity: str

    def __post_init__(self):
        if self.role not in ("user", "agent", "environment", "system"):
            raise ValueError("Unknown actor role")
        identifier(self.identity)


@dataclass(frozen=True)
class OutputControl:
    modality: str
    mode: str
    specified_by: Actor

    def __post_init__(self):
        identifier(self.modality)
        if self.mode not in MODES or not isinstance(self.specified_by, Actor):
            raise ValueError("Output control needs a mode and explicit actor")


@dataclass(frozen=True)
class TaskRequest:
    task_id: str
    instruction: str
    requested_by: Actor
    controls: tuple[OutputControl, ...] = ()
    parents: tuple[str, ...] = ()

    def __post_init__(self):
        identifier(self.task_id)
        identifier(self.instruction)
        if not isinstance(self.requested_by, Actor):
            raise ValueError("Task requester must be explicit")
        if not isinstance(self.controls, tuple) or not isinstance(self.parents, tuple):
            raise ValueError("Controls and parents must be immutable tuples")
        if len({c.modality for c in self.controls}) != len(self.controls):
            raise ValueError("Specify one resolved control per modality")
        for parent in self.parents:
            identifier(parent)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, record):
        record = dict(record)
        record["requested_by"] = Actor(**record["requested_by"])
        record["controls"] = tuple(
            OutputControl(c["modality"], c["mode"], Actor(**c["specified_by"]))
            for c in record["controls"]
        )
        record["parents"] = tuple(record["parents"])
        return cls(**record)


@dataclass(frozen=True)
class OutputRequest:
    output_id: str
    modality: str
    requested_by: Actor
    in_response_to: str
    purpose: str = "answer"
    parents: tuple[str, ...] = ()

    def __post_init__(self):
        for value in (self.output_id, self.modality, self.in_response_to):
            identifier(value)
        if not isinstance(self.requested_by, Actor):
            raise ValueError("Output requester must be explicit")
        if self.purpose not in ("answer", "candidate", "review", "clarification"):
            raise ValueError("Unknown output purpose")
        if not isinstance(self.parents, tuple):
            raise ValueError("Parents must be an immutable tuple")
        for parent in self.parents:
            identifier(parent)

    @classmethod
    def from_dict(cls, record):
        record = dict(record)
        record["requested_by"] = Actor(**record["requested_by"])
        record["parents"] = tuple(record["parents"])
        return cls(**record)


@dataclass(frozen=True)
class Provenance:
    request: OutputRequest
    produced_by: Actor
    origin: str = "generated"
    ancestors: tuple[str, ...] = ()

    def __post_init__(self):
        if self.origin not in ("generated", "derived_generated"):
            raise ValueError("Unknown generated origin")
        if not isinstance(self.request, OutputRequest) or not isinstance(
            self.produced_by, Actor
        ):
            raise ValueError(
                "Provenance requires separate requester and producer records"
            )
        if not isinstance(self.ancestors, tuple):
            raise ValueError("Ancestry must be an immutable tuple")
        for parent in self.ancestors:
            identifier(parent)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, record):
        return cls(
            OutputRequest.from_dict(record["request"]),
            Actor(**record["produced_by"]),
            record["origin"],
            tuple(record["ancestors"]),
        )

    def derived(self):
        return replace(
            self,
            origin="derived_generated",
            ancestors=tuple(dict.fromkeys((*self.ancestors, self.request.output_id))),
        )


@dataclass(frozen=True)
class GeneratedOutput:
    values: torch.Tensor
    provenance: Provenance

    def to_dict(self):
        return {
            "values": self.values.detach().clone(),
            "provenance": self.provenance.to_dict(),
        }

    @classmethod
    def from_dict(cls, record):
        return cls(record["values"], Provenance.from_dict(record["provenance"]))

    def loopback(self, time):
        """Encoder input with exact attribution. Time is reflection availability.

        Video and text already have an item axis; other outputs represent one item.
        A future custom sequence adapter may construct Observation explicitly.
        """
        from .modalities import Observation

        x = self.values.detach()
        if self.provenance.request.modality not in ("text", "video"):
            x = x[:, None]
        times = torch.as_tensor(time, device=x.device, dtype=torch.float64)
        if times.ndim == 0:
            times = times.expand(len(x))
        if times.shape != (len(x),):
            raise ValueError("Loopback time must be scalar or [B]")
        valid = x != 0 if self.provenance.request.modality == "text" else None
        return Observation(
            x, times[:, None].expand(x.shape[:2]), valid, self.provenance
        )


@dataclass(frozen=True)
class TaskSession:
    request: TaskRequest
    outputs: tuple[GeneratedOutput, ...] = ()
    finished: bool = False
    failure: str | None = None

    def __post_init__(self):
        ids = [o.provenance.request.output_id for o in self.outputs]
        if len(set(ids)) != len(ids):
            raise ValueError("Task has duplicate output IDs")
        if any(
            o.provenance.request.in_response_to != self.request.task_id
            for o in self.outputs
        ):
            raise ValueError("Output belongs to a different task")
        if self.finished and self.remaining:
            raise ValueError("Cannot finish with required outputs pending")
        if self.failure is not None:
            identifier(self.failure)
            if self.finished:
                raise ValueError("A task cannot be both fulfilled and failed")
        controls = {c.modality: c for c in self.request.controls}
        for output in self.outputs:
            r = output.provenance.request
            control = controls.get(r.modality)
            if control and control.mode == "disabled":
                raise ValueError("Task snapshot contains a disabled output")
            if (
                control
                and control.mode == "required"
                and r.purpose == "answer"
                and r.requested_by != control.specified_by
            ):
                raise ValueError(
                    "Answer requester does not match the local control author"
                )
            if (
                control is None or control.mode == "automatic"
            ) and r.requested_by.role != "agent":
                raise ValueError("Automatic output requester must be an agent")

    def abort(self, reason):
        """Caller ends an unsuccessful task while retaining unfulfilled requirements."""
        identifier(reason)
        if self.finished:
            raise ValueError("A fulfilled task cannot be aborted")
        return replace(self, failure=reason)

    @property
    def remaining(self):
        answered = {
            o.provenance.request.modality
            for o in self.outputs
            if o.provenance.request.purpose == "answer"
        }
        return tuple(
            c.modality
            for c in self.request.controls
            if c.mode == "required" and c.modality not in answered
        )

    def context_record(self):
        return {
            "request": self.request.to_dict(),
            "finished": self.finished,
            "failure": self.failure,
            "outputs": [o.provenance.to_dict() for o in self.outputs],
        }

    def to_dict(self):
        return {
            "schema": "pathwm-task-v1",
            "request": self.request.to_dict(),
            "outputs": [o.to_dict() for o in self.outputs],
            "finished": self.finished,
            "failure": self.failure,
        }

    @classmethod
    def from_dict(cls, record):
        if record["schema"] != "pathwm-task-v1":
            raise ValueError("Unknown task schema")
        return cls(
            TaskRequest.from_dict(record["request"]),
            tuple(GeneratedOutput.from_dict(o) for o in record["outputs"]),
            record["finished"],
            record["failure"],
        )


@dataclass(frozen=True)
class Selection:
    operation: str
    requests: tuple[OutputRequest, ...]
    raw_operation: str
    raw_modalities: tuple[str, ...]
    completion_probability: float
    clarification_probability: float


@dataclass(frozen=True)
class TaskPrediction:
    operation_logits: torch.Tensor  # [B,7]
    modality_logits: torch.Tensor  # [B,M], independent choices
    completion_logits: torch.Tensor  # [B]
    modalities: tuple[str, ...]

    def select(self, session, agent, index=0):
        """Hard gates are separate from logits and do not rewrite learned scores."""
        if agent.role != "agent":
            raise ValueError("Automatic output requester must be an agent")
        if any(
            not torch.isfinite(x).all()
            for x in (
                self.operation_logits,
                self.modality_logits,
                self.completion_logits,
            )
        ):
            raise ValueError("Task logits must be finite")
        if self.operation_logits.shape[1:] != (
            len(OPERATIONS),
        ) or self.modality_logits.shape != (
            len(self.operation_logits),
            len(self.modalities),
        ):
            raise ValueError("Task logits have incompatible shapes")
        if self.completion_logits.shape != (len(self.operation_logits),):
            raise ValueError("Completion logits must be [B]")
        controls = {c.modality: c for c in session.request.controls}
        if set(controls) - set(self.modalities):
            raise ValueError("Task policy does not support a controlled modality")
        raw_operation = OPERATIONS[int(self.operation_logits[index].argmax())]
        raw_modalities = tuple(
            m
            for m, score in zip(self.modalities, self.modality_logits[index])
            if score >= 0
        )
        operation = raw_operation
        if session.failure is not None:
            operation = "failed"
        elif session.finished:
            operation = "finish"
        elif operation == "finish" and session.remaining:
            operation = "emit"
        names = [
            m
            for m in self.modalities
            if m in session.remaining
            or (
                m in raw_modalities
                and (m not in controls or controls[m].mode == "automatic")
            )
        ]
        if operation == "emit" and not names:
            operation = "ask"
        requests = []
        if operation == "emit":
            for i, name in enumerate(names):
                control = controls.get(name)
                requester = (
                    control.specified_by
                    if control and control.mode == "required"
                    else agent
                )
                requests.append(
                    OutputRequest(
                        f"{session.request.task_id}/{len(session.outputs) + i}/{name}",
                        name,
                        requester,
                        session.request.task_id,
                        parents=session.request.parents,
                    )
                )
        return Selection(
            operation,
            tuple(requests),
            raw_operation,
            raw_modalities,
            float(self.completion_logits[index].detach().sigmoid()),
            float(
                self.operation_logits[index]
                .detach()
                .softmax(-1)[OPERATIONS.index("ask")]
            ),
        )


@dataclass(frozen=True)
class Emission:
    session: TaskSession
    outputs: tuple[GeneratedOutput, ...]
    errors: dict[str, str]


@dataclass(frozen=True)
class TaskStep:
    state: object
    session: TaskSession
    selection: Selection
    emission: Emission | None = None
    action: torch.Tensor | None = None
    imagined: object | None = None


class MetadataEncoder(nn.Module):
    """Order-sensitive byte encoding of exact metadata, without ID lookup tables."""

    def __init__(self, width):
        super().__init__()
        self.embedding = nn.Embedding(259, width, padding_idx=0)
        self.sequence = nn.GRU(width, width, batch_first=True)

    def forward(self, records):
        ids, valid = bytes_batch(
            [
                json.dumps(r, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
                for r in records
            ],
            self.embedding.weight.device,
        )
        values, _ = self.sequence(self.embedding(ids))
        # Pool every valid position so early actor/ancestry fields cannot disappear
        # merely because later fields occupy a long serialized suffix.
        return ((values * valid[..., None]).sum(1) / valid.sum(1)[:, None])[:, None]


class TaskInterpreter(nn.Module):
    def __init__(self, width, slots=4):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(slots, width) * 0.02)
        self.read_instruction, self.read_state = Attend(width), Attend(width)

    def forward(self, state, instruction, metadata, trace=None):
        query = self.queries.expand(len(state), -1, -1)
        query = self.read_instruction(
            query,
            instruction.values,
            valid=instruction.valid,
            trace=trace,
            name="task.instruction_attention",
        )
        return self.read_state(
            query,
            torch.cat((state, metadata), 1),
            trace=trace,
            name="task.state_attention",
        )


class TaskPolicy(nn.Module):
    def __init__(self, width, modalities=("image", "audio", "text", "video")):
        super().__init__()
        if not modalities or len(set(modalities)) != len(modalities):
            raise ValueError("Declare distinct policy output modalities")
        self.modalities = tuple(modalities)
        self.process = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, width), nn.GELU()
        )
        self.operation = nn.Linear(width, len(OPERATIONS))
        self.modality = nn.Linear(width, len(modalities))
        self.completion = nn.Linear(width, 1)

    def forward(self, tokens):
        x = self.process(tokens.mean(1))
        return TaskPrediction(
            self.operation(x),
            self.modality(x),
            self.completion(x).squeeze(-1),
            self.modalities,
        )
