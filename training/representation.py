"""Encoder-first R0/R1 representation learner for video and audio.

What: masked-input denoising and next-span latent prediction for each modality against EMA encoder+
adapter teachers, plus a synchronized audiovisual InfoNCE term in R1 and an across-token variance
guardrail.
How: caller-owned observations are cloned before temporal masking. Small disposable token predictors
map online evidence to frozen teacher targets; only encoders, evidence adapters, and pretext heads are
trainable in R0/R1. EMA targets update explicitly after an optimizer step.
Why: the failed E1-a warm-up showed that SIGReg+inverse can create varied but temporally discontinuous
features. The first phase must directly teach temporal predictability while giving stop-gradient a
stable target. These pretext predictors are never used for action dynamics or planning.
"""
from __future__ import annotations

import math
from collections.abc import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from contracts import EvidenceTokens, TemporalObservation
from encoders.temporal import TokenTransformer
from training.curriculum import EMATeacher, configure_core_trainability


class EvidenceBranch(nn.Module):
    """One encoder+adapter pair, used only to give the EMA teacher a coherent copied module."""

    def __init__(self, encoder: nn.Module, adapter: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder
        self.adapter = adapter

    def forward(self, observation: TemporalObservation) -> EvidenceTokens:
        return self.adapter.adapt_evidence(self.encoder.encode_observation(observation))


class TokenPredictionHead(nn.Module):
    """Disposable same-layout token predictor for masked and future latent targets."""

    def __init__(self, dim: int, layers: int, heads: int) -> None:
        super().__init__()
        self.transformer = TokenTransformer(dim, layers, heads, mlp_ratio=4)
        self.readout = nn.Linear(dim, dim)

    def forward(self, evidence: EvidenceTokens) -> torch.Tensor:
        hidden = self.transformer(evidence.tokens.float(), evidence.valid_mask)
        predicted = F.layer_norm(self.readout(hidden), (hidden.shape[-1],))
        return predicted.masked_fill(~evidence.valid_mask[..., None], 0.0)


def _corruption_mask(valid: torch.Tensor, ratio: float, generator: torch.Generator) -> torch.Tensor:
    if not 0.0 < ratio < 1.0:
        raise ValueError("representation.mask_ratio must lie strictly between 0 and 1")
    mask = (torch.rand(valid.shape, generator=generator, device=valid.device) < ratio) & valid
    # Every row has both a learning target and some visible context whenever it has >=2 valid samples.
    for row in range(valid.shape[0]):
        indices = valid[row].nonzero(as_tuple=False).flatten()
        if indices.numel() == 0:
            raise ValueError("each temporal observation row needs a valid sample")
        if not mask[row].any():
            mask[row, indices[0]] = True
        if mask[row, indices].all() and indices.numel() > 1:
            mask[row, indices[-1]] = False
    return mask


def corrupt_observation(
    observation: TemporalObservation,
    modality: str,
    ratio: float,
    generator: torch.Generator,
) -> TemporalObservation:
    """Mask native temporal samples out-of-place; target teachers always see the original."""
    mask = _corruption_mask(observation.valid_mask, ratio, generator)
    values = observation.values.clone()
    if modality == "video":
        if values.ndim != 5 or values.shape[:2] != mask.shape:
            raise ValueError("video corruption expects values (B,T,C,H,W) and mask (B,T)")
        values[mask] = 128
    elif modality == "audio":
        if values.ndim != 3 or (values.shape[0], values.shape[2]) != mask.shape:
            raise ValueError("audio corruption expects values (B,C,S) and mask (B,S)")
        values = values.masked_fill(mask[:, None, :], 0.0)
    else:
        raise ValueError(f"no R0 masking policy for modality {modality!r}")
    return TemporalObservation(values, observation.timestamps, observation.valid_mask)


def _masked_mse(predicted: torch.Tensor, target: EvidenceTokens, source: EvidenceTokens) -> torch.Tensor:
    if predicted.shape != target.tokens.shape or predicted.shape != source.tokens.shape:
        raise ValueError("representation targets require equal token layouts within each modality")
    valid = target.valid_mask & source.valid_mask
    if not valid.any():
        raise ValueError("representation target has no mutually valid tokens")
    squared = (predicted.float() - target.tokens.float()).square().mean(dim=-1)
    return squared[valid].mean()


def _variance_loss(evidence: EvidenceTokens, target_std: float = 1.0) -> torch.Tensor:
    values = evidence.tokens.float()[evidence.valid_mask]
    if values.shape[0] < 2:
        raise ValueError("variance guardrail needs at least two valid tokens")
    std = values.var(dim=0, unbiased=False).add(1e-4).sqrt()
    return F.relu(target_std - std).mean()


def _pool(evidence: EvidenceTokens) -> torch.Tensor:
    weights = evidence.valid_mask[..., None].to(evidence.tokens.dtype)
    pooled = (evidence.tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
    return pooled.float()


class RepresentationLearner(nn.Module):
    def __init__(
        self,
        core: nn.Module,
        *,
        decay: float,
        mask_ratio: float,
        predictor_layers: int,
        predictor_heads: int,
        audiovisual_temperature: float,
        weights: Mapping[str, float],
    ) -> None:
        super().__init__()
        if not 0.0 < audiovisual_temperature:
            raise ValueError("audiovisual temperature must be positive")
        self.core = core
        self.mask_ratio = float(mask_ratio)
        self.audiovisual_temperature = float(audiovisual_temperature)
        self.weights = {name: float(value) for name, value in weights.items()}
        required = {"masked_latent", "future_latent", "variance", "audiovisual_sync"}
        if set(self.weights) != required or any(not math.isfinite(value) or value < 0 for value in self.weights.values()):
            raise ValueError(f"representation weights must be finite non-negative values for {sorted(required)}")

        self.teachers = nn.ModuleDict()
        self.masked_heads = nn.ModuleDict()
        self.future_heads = nn.ModuleDict()
        self.av_projectors = nn.ModuleDict()
        for modality in core.encoders:
            branch = EvidenceBranch(core.encoders[modality], core.adapters[modality])
            self.teachers[modality] = EMATeacher(branch, decay)
            dim = core.adapters[modality].abi.evidence_dim
            self.masked_heads[modality] = TokenPredictionHead(dim, predictor_layers, predictor_heads)
            self.future_heads[modality] = TokenPredictionHead(dim, predictor_layers, predictor_heads)
            self.av_projectors[modality] = nn.Linear(dim, dim)
        if not {"video", "audio"} <= set(self.teachers):
            raise ValueError("the initial common-base representation learner requires video and audio")

    def set_stage(self, stage: str) -> None:
        configure_core_trainability(self.core, stage)
        representation_active = stage in {"representation_unimodal", "representation_av"}
        for group in (self.masked_heads, self.future_heads, self.av_projectors):
            group.requires_grad_(representation_active)
        # Parent .train() calls cannot unfreeze an EMA teacher, but restate the invariant here.
        self.teachers.requires_grad_(False)

    @torch.no_grad()
    def update_teachers(self) -> None:
        for modality, teacher in self.teachers.items():
            online = EvidenceBranch(self.core.encoders[modality], self.core.adapters[modality])
            teacher.update(online)

    def _teacher_evidence(
        self,
        observations: Mapping[str, TemporalObservation],
    ) -> dict[str, EvidenceTokens]:
        return {modality: self.teachers[modality](observation) for modality, observation in observations.items()}

    def loss(
        self,
        current: Mapping[str, TemporalObservation],
        future: Mapping[str, TemporalObservation],
        *,
        stage: str,
        generator: torch.Generator,
    ) -> dict[str, torch.Tensor]:
        if stage not in {"representation_unimodal", "representation_av"}:
            raise ValueError("representation loss is defined only for R0/R1")
        if set(current) != set(future) or set(current) != set(self.core.encoders):
            raise ValueError("current/future batches must contain every enabled representation modality")

        corrupted = {
            modality: corrupt_observation(observation, modality, self.mask_ratio, generator)
            for modality, observation in current.items()
        }
        online = self.core.encode_observations(corrupted)
        with torch.no_grad():
            teacher_current = self._teacher_evidence(current)
            teacher_future = self._teacher_evidence(future)

        masked_terms, future_terms, variance_terms = [], [], []
        for modality, evidence in online.items():
            masked_terms.append(
                _masked_mse(self.masked_heads[modality](evidence), teacher_current[modality], evidence)
            )
            future_terms.append(
                _masked_mse(self.future_heads[modality](evidence), teacher_future[modality], evidence)
            )
            variance_terms.append(_variance_loss(evidence))
        masked = torch.stack(masked_terms).mean()
        future_value = torch.stack(future_terms).mean()
        variance = torch.stack(variance_terms).mean()

        audiovisual = torch.zeros((), device=masked.device)
        if stage == "representation_av":
            batch = online["video"].tokens.shape[0]
            if batch < 2 or online["audio"].tokens.shape[0] != batch:
                raise ValueError("audiovisual synchrony needs a matched batch of at least two")
            video = F.normalize(self.av_projectors["video"](_pool(online["video"])), dim=-1)
            audio = F.normalize(self.av_projectors["audio"](_pool(online["audio"])), dim=-1)
            logits = video @ audio.transpose(0, 1) / self.audiovisual_temperature
            labels = torch.arange(batch, device=logits.device)
            audiovisual = 0.5 * (
                F.cross_entropy(logits, labels) + F.cross_entropy(logits.transpose(0, 1), labels)
            )

        total = (
            self.weights["masked_latent"] * masked
            + self.weights["future_latent"] * future_value
            + self.weights["variance"] * variance
            + self.weights["audiovisual_sync"] * audiovisual
        )
        return {
            "total": total,
            "masked_latent": masked,
            "future_latent": future_value,
            "variance": variance,
            "audiovisual_sync": audiovisual,
        }


def build_representation_learner(cfg: dict, core: nn.Module) -> RepresentationLearner:
    section = cfg["representation"]
    if section["target"] != "ema":
        raise ValueError("the common-base representation target must be EMA")
    objectives = set(section["objectives"])
    if objectives != {"masked_latent", "future_latent", "audiovisual_sync"}:
        raise ValueError("the initial R0/R1 objective set is fixed by the common-base plan")
    return RepresentationLearner(
        core,
        decay=float(section["ema_decay"]),
        mask_ratio=float(section["mask_ratio"]),
        predictor_layers=int(section["predictor_layers"]),
        predictor_heads=int(section["predictor_heads"]),
        audiovisual_temperature=float(section["audiovisual_temperature"]),
        weights=section["weights"],
    )
