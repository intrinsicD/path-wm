"""Replaceable neural pieces. Functional forwards train; discrete IDs do not backpropagate.

Supplied candidates are not object discovery. Scorer outputs are uncalibrated
similarities. A trained update/readout requires an explicit task objective.
"""

from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
from pathwm.models.modalities import position


@dataclass(frozen=True)
class Candidate:
    id: str
    source: str
    modality: str
    key: torch.Tensor
    value: torch.Tensor
    space: str
    model_version: str
    content_ref: str = ""
    exclusive_group: str | None = None
    provenance: object | None = (
        None  # retain generated ancestry from the source Observation
    )


class CandidateEncoder(nn.Module):
    """Learn keys and values from supplied region/mention features [N,input_width]."""

    def __init__(self, input_width, key_width, value_width, *, backbone=None):
        super().__init__()
        self.backbone = backbone if backbone is not None else nn.Identity()
        self.key = nn.Linear(input_width, key_width)
        self.value = nn.Linear(input_width, value_width)

    def forward(self, features):
        hidden = self.backbone(features)
        return F.normalize(self.key(hidden), dim=-1), self.value(hidden)

    def pool(self, encoded, selection=None):
        """Masked supplied spans/regions [B,K,N] -> keys/values [B,K,D].

        Defaults to one whole-window candidate per sample. This is a lossy adapter,
        not discovery. Keep modality/version/provenance when making Candidate records.
        """
        tokens = encoded.as_tokens() if hasattr(encoded, "as_tokens") else encoded
        x, valid = tokens.values, tokens.valid
        if x.ndim != 3 or valid.shape != x.shape[:2] or valid.dtype != torch.bool:
            raise ValueError(
                "Candidate pooling requires values [B,N,D] and boolean validity"
            )
        if selection is None:
            selection = torch.ones(
                len(x), 1, x.shape[1], device=x.device, dtype=torch.bool
            )
        if (
            selection.ndim != 3
            or selection.shape[0] != len(x)
            or selection.shape[2] != x.shape[1]
            or selection.shape[1] < 1
            or selection.dtype != torch.bool
            or selection.device != x.device
        ):
            raise ValueError(
                "Candidate selections must be boolean [B,K,N] on the feature device"
            )
        mask = selection & valid[:, None]
        if not mask.any(-1).all():
            raise ValueError(
                "Each supplied candidate requires at least one valid token"
            )
        safe = (
            x[:, None]
            .expand(-1, selection.shape[1], -1, -1)
            .masked_fill(~mask[..., None], 0)
        )
        if not torch.isfinite(safe).all():
            raise ValueError("Selected candidate features must be finite")
        pooled = safe.sum(2) / mask.sum(-1, keepdim=True)
        keys, values = self(pooled.flatten(0, 1))
        return keys.reshape(*pooled.shape[:2], -1), values.reshape(
            *pooled.shape[:2], -1
        )


class CosineScorer(nn.Module):
    def forward(self, query, memory):
        return F.normalize(query, dim=-1) @ F.normalize(memory, dim=-1).T


class QueryGenerator(nn.Module):
    """Task/self/observation features -> retrieval key; train with relevance targets."""

    def __init__(self, input_width, key_width, *, network=None):
        super().__init__()
        self.network = (
            network if network is not None else nn.Linear(input_width, key_width)
        )

    def forward(self, context):
        return F.normalize(self.network(context), dim=-1)


class AssociationScorer(nn.Module):
    """Learn a shared metric; positives and confusable negatives are training targets."""

    def __init__(self, width, hidden=None, *, projection=None):
        super().__init__()
        hidden = hidden or width
        self.projection = (
            projection
            if projection is not None
            else nn.Sequential(
                nn.Linear(width, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
            )
        )

    def forward(self, query, memory):
        return (
            F.normalize(self.projection(query), dim=-1)
            @ F.normalize(self.projection(memory), dim=-1).T
        )


@dataclass(frozen=True)
class BindingDecision:
    status: str
    entity_id: str | None
    scores: dict[str, float]
    reason: str


class AssociationBinder:
    """A replaceable discrete policy around a separately trainable scorer."""

    def __init__(
        self, scorer=None, *, match_threshold=0.8, new_threshold=0.3, margin=0.1
    ):
        if not -1 <= new_threshold < match_threshold <= 1 or not 0 <= margin <= 2:
            raise ValueError("Invalid binding thresholds")
        self.scorer = scorer if scorer is not None else CosineScorer()
        self.match_threshold, self.new_threshold, self.margin = (
            match_threshold,
            new_threshold,
            margin,
        )

    def __call__(self, candidate, context, *, trace=None):
        clipped = any(
            context.omitted.get(k, 0)
            for k in ("entities", "components", "incompatible")
        )
        options = [
            c
            for c in context.components
            if c.name == "recognition"
            and c.space == candidate.space
            and c.model_version == candidate.model_version
            and c.shape == tuple(candidate.key.shape)
        ]
        if not options:
            decision = BindingDecision(
                "unresolved" if clipped else "new",
                None,
                {},
                "incomplete candidate retrieval"
                if clipped
                else "no compatible stored candidates",
            )
        else:
            memory = torch.stack(
                [
                    c.tensor(device=candidate.key.device, dtype=candidate.key.dtype)
                    for c in options
                ]
            )
            with torch.no_grad():
                scores = self.scorer(candidate.key[None], memory)
            if scores.shape != (1, len(options)) or not torch.isfinite(scores).all():
                raise ValueError(
                    "Scorer must return finite [queries,candidates] similarities"
                )
            # Compare identity groups; retain the best original writer attribution.
            grouped = {}
            values = {}
            for c, value in zip(options, scores[0].tolist()):
                values[c.entity_id] = value
                group = context.canonical_ids[c.entity_id]
                if group not in grouped or value > grouped[group][0]:
                    grouped[group] = (value, c.entity_id)
            ranked = sorted(grouped.values(), key=lambda x: (-x[0], x[1]))
            best, owner = ranked[0]
            gap = best - ranked[1][0] if len(ranked) > 1 else float("inf")
            if clipped:
                decision = BindingDecision(
                    "unresolved", None, values, "incomplete candidate retrieval"
                )
            elif best >= self.match_threshold and gap >= self.margin:
                decision = BindingDecision(
                    "matched", owner, values, "threshold and group margin"
                )
            elif best <= self.new_threshold and not any(
                context.omitted.get(k, 0)
                for k in ("entities", "components", "incompatible")
            ):
                decision = BindingDecision(
                    "new", None, values, "all candidates dissimilar"
                )
            else:
                decision = BindingDecision(
                    "unresolved", None, values, "ambiguous or incomplete candidates"
                )
        if trace is not None:
            trace.record(
                "binding",
                dict(
                    candidate=candidate.id,
                    status=decision.status,
                    entity=decision.entity_id,
                    scores=decision.scores,
                    reason=decision.reason,
                ),
            )
        return decision


class RecurrentUpdater(nn.Module):
    def __init__(self, input_width, state_width, *, cell=None):
        super().__init__()
        self.input_width, self.state_width = input_width, state_width
        self.cell = (
            cell if cell is not None else nn.GRUCell(input_width + 1, state_width)
        )

    def forward(self, previous, observation, dt):
        elapsed = torch.as_tensor(
            dt, device=observation.device, dtype=observation.dtype
        ).reshape(-1, 1)
        if elapsed.shape[0] == 1:
            elapsed = elapsed.expand(len(observation), 1)
        if (elapsed < 0).any() or not torch.isfinite(elapsed).all():
            raise ValueError("Update duration must be finite and nonnegative")
        return self.cell(torch.cat((observation, torch.log1p(elapsed)), -1), previous)


class ReplaceUpdater(nn.Module):
    """Supplied complete states only; useful transparent control for the learned cell."""

    def __init__(self, width):
        super().__init__()
        self.input_width = self.state_width = width

    def forward(self, previous, observation, dt):
        if previous.shape != observation.shape:
            raise ValueError("Replacement state width mismatch")
        return observation


class TransitionPredictor(nn.Module):
    """Action/time-conditioned latent forecast, separate from observed state writes."""

    def __init__(self, state_width, action_width, *, network=None):
        super().__init__()
        self.state_width = state_width
        self.network = (
            network
            if network is not None
            else nn.Sequential(
                nn.Linear(state_width + action_width + 1, state_width * 2),
                nn.SiLU(),
                nn.Linear(state_width * 2, state_width * 2),
            )
        )

    def forward(self, state, action, dt):
        dt = (
            torch.as_tensor(dt, device=state.device, dtype=state.dtype)
            .reshape(-1, 1)
            .expand(len(state), 1)
        )
        if (dt < 0).any() or not torch.isfinite(dt).all():
            raise ValueError("Forecast duration must be finite and nonnegative")
        mean, raw_scale = self.network(
            torch.cat((state, action, torch.log1p(dt)), -1)
        ).chunk(2, -1)
        return mean, F.softplus(raw_scale) + 1e-4


@dataclass(frozen=True)
class ContextTokens:
    values: torch.Tensor
    component_ids: tuple[str, ...]
    entity_ids: tuple[str, ...]
    revision: int
    omitted: tuple[str, ...]
    relation_ids: tuple[str, ...] = ()


class RelationEncoder(nn.Module):
    """Configured relation labels and directed local endpoints -> one learned token."""

    def __init__(self, width, relation_types):
        super().__init__()
        self.width = width
        self.types = tuple(relation_types)
        if not self.types or len(set(self.types)) != len(self.types):
            raise ValueError("Declare unique relation types for this reader")
        self.kind = nn.Embedding(len(self.types), width)
        self.source = nn.Linear(width, width, bias=False)
        self.target = nn.Linear(width, width, bias=False)
        self.meta = nn.Linear(3, width)

    def extra_repr(self):
        return f"relation_types={self.types}"

    def forward(self, relation, source_group, target_group, *, now):
        weight = self.kind.weight
        source = position(weight.new_tensor([source_group]), self.width).to(
            weight.dtype
        )
        target = position(weight.new_tensor([target_group]), self.width).to(
            weight.dtype
        )
        meta = weight.new_tensor(
            [
                [
                    math_log_signed(float(now) - relation.valid_from),
                    relation.confidence or 0.0,
                    float(relation.confidence is not None),
                ]
            ]
        )
        return (
            self.kind(
                torch.tensor([self.types.index(relation.type)], device=weight.device)
            )
            + self.source(source)
            + self.target(target)
            + self.meta(meta)
        )


class ContextEncoder(nn.Module):
    """Selected components -> reasoner tokens with local entity, role and age cues.

    Plain supplied projections can be MLPs, CNNs or Transformers. Each must return
    one [1,width] token per component here; finer tokenizers can replace this class.
    """

    def __init__(
        self,
        width,
        projections,
        representations,
        *,
        max_tokens=16,
        relation_encoder=None,
    ):
        super().__init__()
        if type(max_tokens) is not int or max_tokens < 1:
            raise ValueError("Context token budget must be positive")
        self.width, self.max_tokens = width, max_tokens
        self.projections = nn.ModuleDict(projections)
        self.representations = dict(representations)
        self.relation_encoder = relation_encoder
        if set(projections) != set(representations):
            raise ValueError(
                "Each projection needs an explicit representation contract"
            )
        self.role = nn.Embedding(3, width)
        self.metadata = nn.Linear(3, width)
        self._device_anchor = nn.Parameter(torch.zeros(0), requires_grad=False)

    def project_value(
        self, name, values, *, role="inferred", age=0.0, confidence=None, group=0
    ):
        """Differentiable training path using exactly the runtime projection/cues."""
        value = self.projections[name](values)
        if value.shape != (len(values), self.width) or not torch.isfinite(value).all():
            raise ValueError("Component projection must produce finite [B,width]")
        role_id = {"observed": 0, "inferred": 1, "predicted": 2}[role]
        meta = value.new_tensor(
            [[math_log_signed(age), confidence or 0.0, float(confidence is not None)]]
        ).expand(len(values), -1)
        group = torch.as_tensor(group, device=value.device, dtype=value.dtype).expand(
            len(values)
        )
        return (
            value
            + self.role(torch.full((len(values),), role_id, device=value.device))
            + self.metadata(meta)
            + position(group, self.width).to(value.dtype)
        )

    def forward(self, context, *, now, trace=None):
        tokens, refs, entities, omitted = [], [], [], []
        relation_refs = []
        groups = sorted(set(context.canonical_ids.values()))
        device, dtype = self._device_anchor.device, self._device_anchor.dtype
        for c in context.components:
            if (
                c.name not in self.projections
                or (c.space, c.model_version) != tuple(self.representations[c.name])
                or len(tokens) >= self.max_tokens
            ):
                omitted.append(c.id)
                continue
            group = groups.index(context.canonical_ids[c.entity_id])
            age = float(now) - c.valid_from
            value = self.project_value(
                c.name,
                c.tensor(device=device, dtype=dtype)[None],
                role=c.role,
                age=age,
                confidence=c.confidence,
                group=group,
            )
            tokens.append(value)
            refs.append(c.id)
            entities.append(c.entity_id)
        component_owners = {c.id: c.entity_id for c in context.components}
        for relation in context.relations:
            endpoints = [
                e.ref if e.kind == "entity" else component_owners.get(e.ref)
                for e in (relation.source, relation.target)
            ]
            if (
                self.relation_encoder is None
                or relation.type not in self.relation_encoder.types
                or any(e not in context.canonical_ids for e in endpoints)
                or len(tokens) >= self.max_tokens
            ):
                omitted.append(relation.id)
                continue
            indices = [groups.index(context.canonical_ids[e]) for e in endpoints]
            value = self.relation_encoder(relation, *indices, now=now)
            if value.shape != (1, self.width) or not torch.isfinite(value).all():
                raise ValueError("Relation projection must return finite [1,width]")
            tokens.append(value)
            relation_refs.append(relation.id)
        values = (
            torch.cat(tokens, 0)[None]
            if tokens
            else torch.empty(1, 0, self.width, device=device, dtype=dtype)
        )
        if trace is not None:
            trace["world.context.tokens"] = values
            trace.record(
                "context",
                dict(
                    revision=context.revision,
                    components=refs,
                    relations=relation_refs,
                    entities=entities,
                    omitted=omitted,
                    intervention=context.intervention,
                ),
            )
        return ContextTokens(
            values,
            tuple(refs),
            tuple(entities),
            context.revision,
            tuple(omitted),
            tuple(relation_refs),
        )


def math_log_signed(value):
    import math

    if not math.isfinite(value):
        raise ValueError("Context time must be finite")
    return math.copysign(math.log1p(abs(value)), value)
