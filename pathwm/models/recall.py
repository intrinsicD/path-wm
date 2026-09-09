"""Historical task records and a learned factual reader; no evaluator in inference."""

from dataclasses import asdict, dataclass
import math
import torch
from torch import nn

from .modalities import Observation, bytes_batch
from .tasks import TaskRequest

LABELS = ("l0", "l1", "l2", "l3", "not_observed_in_session")


@dataclass(frozen=True)
class SeenRecord:
    ordinal: int
    entity: int
    location: int

    def __post_init__(self):
        if (
            type(self.ordinal) is not int
            or self.ordinal < 1
            or type(self.entity) is not int
            or not 0 <= self.entity < 32
            or type(self.location) is not int
            or not 0 <= self.location < 4
        ):
            raise ValueError("Invalid visible record")

    @property
    def text(self):
        return f"saw entity=e{self.entity:02d} at location=l{self.location}"


@dataclass(frozen=True)
class RecallQuery:
    task: TaskRequest
    session_id: str
    entity: int
    cutoff: int
    wrong_cost: float = 1.0
    abstain_cost: float = 0.25

    def __post_init__(self):
        if (
            not isinstance(self.task, TaskRequest)
            or not self.session_id
            or len(self.session_id) > 256
            or type(self.entity) is not int
            or not 0 <= self.entity < 32
            or type(self.cutoff) is not int
            or self.cutoff < 1
            or not math.isfinite(self.wrong_cost)
            or self.wrong_cost <= 0
            or not math.isfinite(self.abstain_cost)
            or self.abstain_cost < 0
        ):
            raise ValueError("Invalid historical recall query")

    def to_dict(self):
        return dict(schema="pathwm-recall-query-v1", **asdict(self))

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        if data.pop("schema") != "pathwm-recall-query-v1":
            raise ValueError("Unknown recall query schema")
        data["task"] = TaskRequest.from_dict(data["task"])
        return cls(**data)


@dataclass(frozen=True)
class RecallDecision:
    task_id: str
    session_id: str
    cutoff: int
    answer: int | None
    confidence: float | None
    reason: str

    def __post_init__(self):
        if (
            not self.task_id
            or not self.session_id
            or type(self.cutoff) is not int
            or self.cutoff < 1
            or self.reason not in ("answer", "cost", "invalid")
            or (
                self.answer is not None
                and (type(self.answer) is not int or not 0 <= self.answer < 5)
            )
            or (
                self.confidence is not None
                and (
                    not math.isfinite(self.confidence) or not 0 <= self.confidence <= 1
                )
            )
            or (self.reason == "answer") != (self.answer is not None)
        ):
            raise ValueError("Invalid recall decision")

    def to_dict(self):
        return dict(schema="pathwm-recall-decision-v1", **asdict(self))

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        if data.pop("schema") != "pathwm-recall-decision-v1":
            raise ValueError("Unknown recall decision schema")
        return cls(**data)


def historical_target(records, query):
    """Evaluator-only inclusive cutoff over the complete delivered log."""
    last, answer = 0, 4
    for record in records:
        if record.ordinal <= last:
            raise ValueError("Delivered records must be strictly ordered")
        last = record.ordinal
        if record.ordinal <= query.cutoff and record.entity == query.entity:
            answer = record.location
    return answer


def select_recall(probabilities, query):
    p = probabilities.detach()
    answer, confidence, reason = None, None, "invalid"
    if (
        p.shape == (5,)
        and torch.isfinite(p).all()
        and (p >= 0).all()
        and torch.isclose(p.sum(), p.new_tensor(1.0), atol=1e-6, rtol=1e-6)
    ):
        confidence = float(p.max())
        reason = "cost"
        if query.wrong_cost * (1 - confidence) < query.abstain_cost:
            answer, reason = int(p.argmax()), "answer"
    return RecallDecision(
        query.task.task_id, query.session_id, query.cutoff, answer, confidence, reason
    )


def verify_recall(query, decision, records, session_id):
    """The caller supplies the actual log; predicted confidence is never evidence."""
    if session_id != query.session_id or decision.session_id != session_id:
        raise ValueError("Verification session mismatch")
    if decision.task_id != query.task.task_id or decision.cutoff != query.cutoff:
        raise ValueError("Verification task/cutoff mismatch")
    if (
        not records
        or records[-1].ordinal < query.cutoff
        or [r.ordinal for r in records] != list(range(1, len(records) + 1))
    ):
        raise ValueError("Verification requires complete history through cutoff")
    target = historical_target(records, query)
    abstained = decision.answer is None
    correct = decision.answer == target
    return dict(
        schema="pathwm-recall-verification-v1",
        task_id=query.task.task_id,
        session_id=session_id,
        cutoff=query.cutoff,
        target=target,
        verification="unknown"
        if abstained
        else "verified"
        if correct
        else "contradicted",
        disposition="abstained" if abstained else "answered",
        loss=query.abstain_cost if abstained else 0.0 if correct else query.wrong_cost,
    )


class RecallHead(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.output = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(), nn.Linear(width, 5)
        )

    def forward(self, working):
        return self.output(working.mean(1))


def recall_logits(model, state, query):
    """One live stream; two memory reads; opaque IDs/labels never enter the encoder."""
    if (
        len(state.tokens) != 1
        or state.session_id != query.session_id
        or state.ordinal != query.cutoff
        or state.imagined
    ):
        raise ValueError("Recall requires the live session at its query cutoff")
    ids, valid = bytes_batch(
        [f"Where was entity=e{query.entity:02d} last observed?"], state.tokens.device
    )
    features = model.encode(
        state,
        {"text": Observation(ids, state.time[:, None].expand_as(ids), valid)},
        time=state.time,
    )["text"]
    metadata = model.metadata_encoder(
        [dict(objective="last_observed", cutoff=query.cutoff)]
    )
    goal = model.task_interpreter(state.tokens, features.as_tokens(), metadata)
    working = model.think(state, steps=2, goal=goal)
    return model.recall_head(working.tokens[:, model.layout["working"]]), working
