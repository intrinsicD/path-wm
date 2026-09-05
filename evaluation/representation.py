"""Held-out R0/R1 representation panel for the common-base promotion gates.

What: per-modality collapse, prediction, temporal retrieval, cross-modal retrieval and shifted-time
synchrony metrics computed from frozen evaluation views.
How: every batch comes from the manifest's ``eval`` split; fixed roll negatives keep cohort size equal.
Prediction advantages are identity/source MSE minus learned-head MSE, so positive means the head beats
copying the corrupted current evidence.
Why: training loss alone cannot distinguish temporal learning from collapse or recording/scene shortcuts.
These names exactly match the fail-closed curriculum gate (common-base architecture §5; DDR §22).
"""
from __future__ import annotations

import math
from collections.abc import Mapping

import torch
import torch.nn.functional as F

from contracts import EvidenceTokens, RepresentationBatch, RepresentationData, TemporalObservation


def _valid_values(evidence: EvidenceTokens) -> torch.Tensor:
    values = evidence.tokens.float()[evidence.valid_mask]
    if values.shape[0] < 2 or not torch.isfinite(values).all():
        raise ValueError("representation diagnostics need at least two finite valid evidence tokens")
    return values


def _feature_std(evidence: EvidenceTokens) -> float:
    return float(_valid_values(evidence).var(dim=0, unbiased=False).sqrt().mean())


def _effective_rank_fraction(evidence: EvidenceTokens) -> float:
    values = _valid_values(evidence)
    centered = values - values.mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(centered)
    power = singular.square()
    if float(power.sum()) <= 0:
        return 0.0
    probabilities = power / power.sum()
    probabilities = probabilities[probabilities > 0]
    effective_rank = torch.exp(-(probabilities * probabilities.log()).sum())
    maximum_rank = min(values.shape[0] - 1, values.shape[1])
    return float(effective_rank / max(maximum_rank, 1))



def _within_position_variation_fraction(evidence: EvidenceTokens) -> float:
    """Fraction of token variation across windows at matching positions (DDR §30).

    Position-only evidence has zero within-position variation despite potentially high rank.
    This contextual diagnostic is not a semantic-content score or a promotion gate.
    """
    valid = evidence.valid_mask[..., None]
    # masked_fill excludes padding even when padded values are NaN.
    values = evidence.tokens.float().masked_fill(~valid, 0)
    position_mean = values.sum(dim=0) / valid.sum(dim=0).clamp_min(1)
    global_mean = values.sum(dim=(0, 1)) / valid.sum().clamp_min(1)
    within = (values - position_mean).masked_fill(~valid, 0).square().sum()
    total = (values - global_mean).masked_fill(~valid, 0).square().sum()
    return float((within / total.clamp_min(torch.finfo(values.dtype).tiny)).clamp(0, 1))


def _pool(evidence: EvidenceTokens) -> torch.Tensor:
    weights = evidence.valid_mask[..., None].to(evidence.tokens.dtype)
    return ((evidence.tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)).float()


def _paired_retrieval_margin(query: torch.Tensor, target: torch.Tensor) -> float:
    if query.shape != target.shape or query.ndim != 2 or query.shape[0] < 2:
        raise ValueError("retrieval diagnostics need matched (B,D) tensors with B>=2")
    query = F.normalize(query.float(), dim=-1)
    target = F.normalize(target.float(), dim=-1)
    positive = (query * target).sum(dim=-1)
    negative = (query * torch.roll(target, shifts=1, dims=0)).sum(dim=-1)
    return float((positive - negative).mean())


def _prediction_advantage(
    source: EvidenceTokens,
    target: EvidenceTokens,
    predicted: torch.Tensor,
) -> float:
    if predicted.shape != source.tokens.shape or predicted.shape != target.tokens.shape:
        raise ValueError("prediction diagnostics require equal source, target and prediction layouts")
    valid = source.valid_mask & target.valid_mask
    if not valid.any():
        raise ValueError("prediction diagnostics have no mutually valid tokens")
    target_values = target.tokens.float()[valid]
    baseline = (source.tokens.float()[valid] - target_values).square().mean()
    learned = (predicted.float()[valid] - target_values).square().mean()
    return float(baseline - learned)


def _synchrony_above_chance(current: Mapping[str, torch.Tensor], shifted: Mapping[str, torch.Tensor]) -> float:
    if set(current) != {"video", "audio"} or set(shifted) != {"video", "audio"}:
        raise ValueError("synchrony diagnostics require current and shifted video/audio embeddings")
    video = F.normalize(current["video"].float(), dim=-1)
    audio = F.normalize(current["audio"].float(), dim=-1)
    shifted_video = F.normalize(shifted["video"].float(), dim=-1)
    shifted_audio = F.normalize(shifted["audio"].float(), dim=-1)
    positive = (video * audio).sum(dim=-1)
    forward = (positive > (video * shifted_audio).sum(dim=-1)).float()
    reverse = (positive > (audio * shifted_video).sum(dim=-1)).float()
    return float(torch.cat((forward, reverse)).mean() - 0.5)



def _temporal_change_alignment(
    current: Mapping[str, torch.Tensor], shifted: Mapping[str, torch.Tensor],
) -> float:
    """Half the dot product of normalized A/V changes; time-constant embeddings cancel (DDR §33)."""
    if set(current) != {"video", "audio"} or set(shifted) != {"video", "audio"}:
        raise ValueError("temporal-change diagnostics require current and shifted video/audio embeddings")
    video, audio, shifted_video, shifted_audio = [
        F.normalize(group[modality].float(), dim=-1)
        for group in (current, shifted) for modality in ("video", "audio")
    ]
    if video.ndim != 2 or any(value.shape != video.shape for value in (audio, shifted_video, shifted_audio)):
        raise ValueError("temporal-change diagnostics require four matched (B,D) embeddings")
    return float(0.5 * ((video - shifted_video) * (audio - shifted_audio)).sum(dim=-1).mean())


def _batch_metrics(views: Mapping[str, object], stage: str) -> dict[str, float]:
    required = {"online", "masked_source", "future_source", "teacher_current", "teacher_future", "masked_prediction", "future_prediction"}
    missing = required - set(views)
    if missing:
        raise ValueError(f"representation evaluation views missing {sorted(missing)}")
    online = views["online"]
    teacher_current = views["teacher_current"]
    teacher_future = views["teacher_future"]
    masked_prediction = views["masked_prediction"]
    masked_source = views["masked_source"]
    future_source = views["future_source"]
    future_prediction = views["future_prediction"]
    if not all(isinstance(value, Mapping) for value in (online, masked_source, future_source, teacher_current, teacher_future, masked_prediction, future_prediction)):
        raise ValueError("representation evaluation view groups must be modality mappings")
    if not all(set(value) == {"video", "audio"} for value in (online, masked_source, future_source, teacher_current, teacher_future, masked_prediction, future_prediction)):
        raise ValueError("representation evaluation requires video and audio views")

    metrics: dict[str, float] = {}
    for modality in ("video", "audio"):
        metrics[f"{modality}_feature_std"] = _feature_std(online[modality])
        metrics[f"{modality}_effective_rank_fraction"] = _effective_rank_fraction(online[modality])
        metrics[f"{modality}_within_position_variation_fraction"] = _within_position_variation_fraction(online[modality])
        metrics[f"{modality}_masked_prediction_advantage"] = _prediction_advantage(
            masked_source[modality], teacher_current[modality], masked_prediction[modality]
        )
        metrics[f"{modality}_future_prediction_advantage"] = _prediction_advantage(
            future_source[modality], teacher_future[modality], future_prediction[modality]
        )
        # Same-basis copy removes student/EMA alignment gain from the temporal comparison (DDR §26).
        metrics[f"{modality}_future_teacher_copy_advantage"] = _prediction_advantage(
            teacher_current[modality], teacher_future[modality], future_prediction[modality]
        )
        metrics[f"{modality}_temporal_retrieval_margin"] = _paired_retrieval_margin(
            _pool(online[modality]), _pool(teacher_future[modality])
        )

    if stage == "representation_av":
        current, shifted = views.get("av_current"), views.get("av_shifted")
        if not isinstance(current, Mapping) or not isinstance(shifted, Mapping):
            raise ValueError("R1 evaluation requires projected current and shifted A/V embeddings")
        metrics["video_to_audio_retrieval_margin"] = _paired_retrieval_margin(
            current["video"], current["audio"]
        )
        metrics["audio_to_video_retrieval_margin"] = _paired_retrieval_margin(
            current["audio"], current["video"]
        )
        metrics["synchrony_accuracy_above_chance"] = _synchrony_above_chance(current, shifted)
        metrics["audiovisual_temporal_change_alignment"] = _temporal_change_alignment(current, shifted)
    return metrics



def _learner_device(learner: torch.nn.Module) -> torch.device:
    try:
        return next(learner.parameters()).device
    except (AttributeError, StopIteration):
        return torch.device("cpu")


def _move_batch(batch: RepresentationBatch, device: torch.device) -> RepresentationBatch:
    def move_views(views: Mapping[str, TemporalObservation]) -> dict[str, TemporalObservation]:
        return {
            modality: TemporalObservation(
                observation.values.to(device),
                observation.timestamps.to(device),
                observation.valid_mask.to(device),
            )
            for modality, observation in views.items()
        }

    return RepresentationBatch(move_views(batch.current), move_views(batch.future), move_views(batch.shifted))


def evaluate_representation(
    learner: torch.nn.Module,
    data: RepresentationData,
    *,
    stage: str,
    batches: int,
    batch_size: int,
    generator: torch.Generator,
) -> dict[str, float]:
    if stage not in {"representation_unimodal", "representation_av"}:
        raise ValueError(f"no representation panel for stage {stage!r}")
    if batches < 1 or batch_size < 2:
        raise ValueError("representation panel needs batches>=1 and batch_size>=2")
    learner.eval()
    device = _learner_device(learner)
    collected: list[dict[str, float]] = []
    with torch.no_grad():
        for _ in range(batches):
            batch = _move_batch(data.sample("eval", stage, batch_size, generator), device)
            views = learner.evaluation_views(batch, stage=stage, generator=generator)
            collected.append(_batch_metrics(views, stage))
    names = collected[0].keys()
    metrics = {name: sum(values[name] for values in collected) / len(collected) for name in names}
    if any(not math.isfinite(value) for value in metrics.values()):
        raise ValueError("representation panel produced a non-finite metric")
    return metrics
