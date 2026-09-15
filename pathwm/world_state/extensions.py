"""Small optional clients of the same store; no autonomous exploration or concept discovery."""

from dataclasses import dataclass, asdict, field
import math
import torch
from torch import nn


@dataclass(frozen=True)
class ControlBinding:
    """Owned by the harness. A stored self/owns relation cannot grant authority."""

    entity_id: str
    sensors: tuple[str, ...] = ()
    actions: tuple[str, ...] = ()

    def permits(self, action):
        return action in self.actions


@dataclass(frozen=True)
class Feedback:
    task_progress: float | None = None
    prediction_error: float | None = None
    human_preference: float | None = None
    social_signal: float | None = None
    physical_success: float | None = None
    novelty: float | None = None
    constraint_violation: float | None = None
    uncertainty_reduction: float | None = None

    def features(self):
        values = list(asdict(self).values())
        if any(v is not None and not math.isfinite(v) for v in values):
            raise ValueError("Feedback must be finite or missing")
        return torch.tensor(
            [v or 0.0 for v in values] + [float(v is not None) for v in values]
        )


@dataclass(frozen=True)
class ActionProposal:
    action: str
    cost: float = 0.0
    novelty: float | None = None
    information_gain: float | None = None
    payload: dict = field(default_factory=dict)


def select_action(proposals, control, *, mode="random", budget=1.0, generator=None):
    """Select among supplied legal proposals. Never execute or infer info gain from surprise."""
    if (
        mode not in {"random", "novelty", "information_gain"}
        or not math.isfinite(budget)
        or budget < 0
    ):
        raise ValueError("Invalid selection mode or budget")
    if any(not math.isfinite(p.cost) or p.cost < 0 for p in proposals):
        raise ValueError("Action cost must be finite/nonnegative")
    eligible = [p for p in proposals if control.permits(p.action) and p.cost <= budget]
    if not eligible:
        return None
    if mode == "random":
        if generator is None:
            raise ValueError("Random exploration requires an explicit generator")
        return eligible[int(torch.randint(len(eligible), (), generator=generator))]
    scored = [(getattr(p, mode), p) for p in eligible]
    if any(v is None or not math.isfinite(v) for v, _ in scored):
        raise ValueError("This policy needs supplied finite estimates")
    return max(scored, key=lambda item: item[0])[1]


class RegulatoryModulator(nn.Module):
    """Optional bounded priority gains; no predefined emotion semantics or automatic use."""

    def __init__(self, input_width=16, outputs=4, *, network=None):
        super().__init__()
        self.network = (
            network if network is not None else nn.Linear(input_width, outputs)
        )

    def forward(self, feedback_features):
        return 1 + 0.5 * torch.tanh(self.network(feedback_features))


def create_prototype(transaction, members, *, label="", name="prototype"):
    """Arithmetic exemplar prototype, not learned abstract concept/function induction."""
    if not members:
        raise ValueError("Prototype requires exemplars")
    first = members[0]
    contract = (first.shape, first.space, first.model_version)
    if any((c.shape, c.space, c.model_version) != contract for c in members):
        raise ValueError("Prototype representation mismatch")
    if len({c.entity_id for c in members}) != len(members):
        raise ValueError("Use one exemplar per distinct instance")
    evidence = tuple(sorted({e for c in members for e in c.evidence}))
    concept = transaction.create_entity(label, kind="concept")
    value = torch.stack([c.tensor() for c in members]).mean(0)
    transaction.put_component(
        concept,
        name,
        value,
        space=first.space,
        model_version=first.model_version,
        evidence=evidence,
        role="inferred",
        data={"method": "arithmetic exemplar mean", "members": [c.id for c in members]},
    )
    for c in members:
        transaction.relate(c.entity_id, concept, "instance_of", evidence=c.evidence)
    return concept
