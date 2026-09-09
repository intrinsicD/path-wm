"""Processed input pyramids with explicit masks, causal support and FiLM control.

Scale ordinals describe content in THIS input window; they are not global stream
IDs or a claim about information inside an external conditioning code. Availability
includes the separately declared code time. Causality comparisons hold code fixed.
"""

from dataclasses import dataclass, replace
import math

import torch
from torch import nn
from torch.nn import functional as F

from .modalities import (
    TokenBatch,
    ImageEncoder,
    TextEncoder,
    observation_values,
    position,
)


@dataclass(frozen=True)
class FeatureScale(TokenBatch):
    ends: torch.Tensor  # [B,N], content support-end ordinal within this input window
    grid: tuple[int, ...]
    content_times: torch.Tensor | None = (
        None  # sensor support bound, before code time floor
    )


@dataclass(frozen=True)
class FeaturePyramid:
    scales: tuple[FeatureScale, ...]  # fine to coarse, all fully processed
    condition_time: torch.Tensor | None = None

    def as_tokens(self):
        return TokenBatch(
            *(
                torch.cat([getattr(s, name) for s in self.scales], 1)
                for name in ("values", "times", "valid")
            )
        )


def pool_scale(fine, factors):
    """Masked non-overlapping local means; return the exact fine-key footprint.

    The dense membership matrix is intentionally simple for tiny development inputs.
    Replace this merge for large inputs; the representation is not a sparse kernel.
    """
    if len(factors) != len(fine.grid) or min(factors) < 1:
        raise ValueError("Pooling factors must match the grid and be positive")
    grid = tuple((n + f - 1) // f for n, f in zip(fine.grid, factors))
    coords = torch.meshgrid(
        *(torch.arange(n, device=fine.values.device) for n in fine.grid), indexing="ij"
    )
    groups = sum(
        (coordinate // factor) * math.prod(grid[i + 1 :])
        for i, (coordinate, factor) in enumerate(zip(coords, factors))
    ).flatten()
    membership = torch.arange(math.prod(grid), device=groups.device)[:, None] == groups
    members = membership[None] & fine.valid[:, None]
    valid = members.any(-1)
    weights = members.to(fine.values.dtype)
    values = torch.bmm(weights, fine.values.masked_fill(~fine.valid[..., None], 0))
    values = values / weights.sum(-1, keepdim=True).clamp_min(1)
    times = (
        fine.times[:, None]
        .expand_as(members)
        .masked_fill(~members, -torch.inf)
        .amax(-1)
    )
    times = times.masked_fill(~valid, 0)
    ends = fine.ends[:, None].expand_as(members).masked_fill(~members, -1).amax(-1)
    content = fine.times if fine.content_times is None else fine.content_times
    content = (
        content[:, None]
        .expand_as(members)
        .masked_fill(~members, -torch.inf)
        .amax(-1)
        .masked_fill(~valid, 0)
    )
    return FeatureScale(values, times, valid, ends, grid, content), membership


class ConditionedBlock(nn.Module):
    """Pre-norm attention and MLP residuals; code zero is ALWAYS neutral.

    The bias-free map has no zero residual gate: code and controller gradients are
    live at initialization. Modulation only changes residual branches, not identity.
    """

    def __init__(self, width, code_width, heads=4):
        super().__init__()
        if width % heads or min(width, code_width, heads) < 1:
            raise ValueError(
                "Positive widths required; attention width must divide by heads"
            )
        self.heads = heads
        self.query_norm, self.key_norm, self.mlp_norm = (
            nn.LayerNorm(width) for _ in range(3)
        )
        self.attention = nn.MultiheadAttention(
            width, heads, batch_first=True, dropout=0.0
        )
        self.mlp = nn.Sequential(
            nn.Linear(width, width * 2), nn.GELU(), nn.Linear(width * 2, width)
        )
        self.modulation = nn.Linear(code_width, 4 * width, bias=False)
        nn.init.normal_(self.modulation.weight, std=0.02)

    def forward(
        self,
        query,
        condition,
        *,
        context=None,
        footprint=None,
        trace=None,
        name="attention",
    ):
        context = query if context is None else context
        scale_a, shift_a, scale_m, shift_m = (
            0.1 * self.modulation(condition).tanh()
        ).chunk(4, -1)
        q = self.query_norm(query.values) * (1 + scale_a[:, None]) + shift_a[:, None]
        key = self.key_norm(context.values)
        allowed = (
            query.valid[:, :, None]
            & context.valid[:, None, :]
            & (context.times[:, None] <= query.times[:, :, None])
            & (context.ends[:, None] <= query.ends[:, :, None])
        )
        if footprint is not None:
            allowed = allowed & footprint[None]
        empty = ~allowed.any(-1)
        # Invalid rows use one temporary fallback solely to keep softmax finite;
        # its entire result and measured weights are removed, with zero gradient.
        safe = allowed.clone()
        safe[:, :, 0] |= empty
        mask = (~safe).repeat_interleave(self.heads, 0)
        read, weights = self.attention(
            q,
            key,
            key,
            attn_mask=mask,
            need_weights=trace is not None,
            average_attn_weights=False,
        )
        read = read.masked_fill(empty[..., None], 0)
        x = query.values + read
        normalized = self.mlp_norm(x) * (1 + scale_m[:, None]) + shift_m[:, None]
        x = (x + self.mlp(normalized)).masked_fill(~query.valid[..., None], 0)
        if trace is not None:
            trace[name] = (
                weights.masked_fill(empty[:, None, :, None], 0).detach().cpu().clone()
            )
        return replace(query, values=x)


class ScaleProcessor(nn.Module):
    def __init__(self, width, code_width, depth=1):
        super().__init__()
        if depth < 1:
            raise ValueError("Every scale needs at least one processing block")
        self.identity = nn.Parameter(torch.randn(width) * 0.02)
        self.blocks = nn.ModuleList(
            [ConditionedBlock(width, code_width) for _ in range(depth)]
        )

    def forward(self, scale, condition, *, trace=None, name="scale"):
        scale = replace(
            scale,
            values=(scale.values + self.identity).masked_fill(
                ~scale.valid[..., None], 0
            ),
        )
        for i, block in enumerate(self.blocks):
            scale = block(scale, condition, trace=trace, name=f"{name}.attention.{i}")
        return scale


class ScaleMerge(nn.Module):
    def __init__(self, width, code_width, factors, cross_scale=True):
        super().__init__()
        self.factors = tuple(factors)
        self.cross_attention = (
            ConditionedBlock(width, code_width) if cross_scale else None
        )

    def forward(self, fine, condition, *, trace=None, name="merge"):
        coarse, footprint = pool_scale(fine, self.factors)
        if self.cross_attention is not None:
            coarse = self.cross_attention(
                coarse,
                condition,
                context=fine,
                footprint=footprint,
                trace=trace,
                name=name + ".attention",
            )
        return coarse


class FeatureHierarchy(nn.Module):
    def __init__(self, width, code_width, factors, levels=3, depth=1, cross_scale=True):
        super().__init__()
        if levels < 1:
            raise ValueError("Feature hierarchy needs at least one scale")
        self.stages = nn.ModuleList(
            [ScaleProcessor(width, code_width, depth) for _ in range(levels)]
        )
        self.merges = nn.ModuleList(
            [
                ScaleMerge(width, code_width, factors, cross_scale)
                for _ in range(levels - 1)
            ]
        )

    def forward(self, fine, condition, *, condition_time=None, trace=None):
        scales = []
        for i, stage in enumerate(self.stages):
            if i:
                fine = self.merges[i - 1](
                    scales[-1], condition, trace=trace, name=f"merge.{i - 1}"
                )
            # This finished value is the sole source for BOTH public output and merge.
            finished = stage(fine, condition, trace=trace, name=f"scale.{i}")
            scales.append(finished)
            if trace is not None:
                for field in ("values", "times", "content_times", "valid", "ends"):
                    trace[f"scale.{i}.{field}"] = (
                        getattr(finished, field).detach().cpu().clone()
                    )
                trace[f"scale.{i}.grid"] = list(finished.grid)
        return FeaturePyramid(tuple(scales), condition_time)


def check_order(observation):
    _, times, valid = observation_values(observation)
    if not times.shape[1]:
        raise ValueError("Input sequence must be nonempty")
    for row, mask in zip(times, valid):
        if (row[mask].diff() < 0).any():
            raise ValueError(
                "Valid input times must be nondecreasing in sequence order"
            )


def condition_inputs(fine, condition, condition_time, code_width):
    # Preserve sensor timing even when newer context raises feature availability.
    # Float64 phase calculation retains sub-second detail at large absolute times.
    fine = replace(
        fine,
        content_times=fine.times,
        values=fine.values
        + position(fine.times.to(torch.float64), fine.values.shape[-1]).to(
            fine.values.dtype
        ),
    )
    if condition is None:
        condition = fine.values.new_zeros(len(fine.values), code_width)
    if (
        condition.shape != (len(fine.values), code_width)
        or condition.device != fine.values.device
        or condition.dtype != fine.values.dtype
        or not torch.isfinite(condition).all()
    ):
        raise ValueError(
            "Feature condition must be finite [B,code_width] with encoder dtype/device"
        )
    if condition_time is not None:
        condition_time = torch.as_tensor(
            condition_time, device=fine.values.device, dtype=torch.float64
        )
        if condition_time.ndim == 0:
            condition_time = condition_time.expand(len(fine.values))
        if (
            condition_time.shape != (len(fine.values),)
            or not torch.isfinite(condition_time).all()
        ):
            raise ValueError("Feature condition time must be finite scalar or [B]")
        times = torch.maximum(fine.times.to(torch.float64), condition_time[:, None])
        fine = replace(fine, times=times.masked_fill(~fine.valid, 0))
    fine = replace(
        fine,
        values=fine.values.masked_fill(~fine.valid[..., None], 0),
        ends=fine.ends.masked_fill(~fine.valid, -1),
    )
    return fine, condition, condition_time


class MultiScaleImageEncoder(nn.Module):
    """Image: spatial pyramid. Video: spatial pyramid plus adjacent-frame pooling."""

    def __init__(
        self,
        width,
        patch_size=4,
        *,
        video=False,
        code_width=16,
        levels=3,
        depth=1,
        cross_scale=True,
    ):
        super().__init__()
        self.width, self.code_width, self.video = width, code_width, video
        self.stem = ImageEncoder(width, patch_size)
        self.pyramid = FeatureHierarchy(
            width, code_width, (2 if video else 1, 2, 2), levels, depth, cross_scale
        )

    def forward(self, observation, *, condition=None, condition_time=None, trace=None):
        check_order(observation)
        tokens = self.stem(observation)
        b, t, _, h, w = observation.values.shape
        grid = (t, h // self.stem.patch_size, w // self.stem.patch_size)
        ends = (
            torch.arange(t, device=tokens.values.device)
            .repeat_interleave(math.prod(grid[1:]))[None]
            .expand(b, -1)
        )
        fine = FeatureScale(tokens.values, tokens.times, tokens.valid, ends, grid)
        fine, condition, condition_time = condition_inputs(
            fine, condition, condition_time, self.code_width
        )
        return self.pyramid(fine, condition, condition_time=condition_time, trace=trace)


class MultiScaleAudioEncoder(nn.Module):
    """Waveform patch pyramid. Chunk-end times are retained, never fabricated."""

    def __init__(
        self,
        samples,
        width,
        patch_size=4,
        *,
        code_width=16,
        levels=3,
        depth=1,
        cross_scale=True,
    ):
        super().__init__()
        if min(samples, patch_size) < 1:
            raise ValueError("Audio sizes must be positive")
        self.width, self.input_width, self.patch_size, self.code_width = (
            width,
            samples,
            patch_size,
            code_width,
        )
        self.stem = nn.Linear(patch_size, width)
        self.modality = nn.Parameter(torch.randn(width) * 0.02)
        self.pyramid = FeatureHierarchy(
            width, code_width, (2,), levels, depth, cross_scale
        )

    def forward(self, observation, *, condition=None, condition_time=None, trace=None):
        check_order(observation)
        x, times, valid = observation_values(observation)
        if x.ndim != 3 or x.shape[-1] != self.input_width:
            raise ValueError(f"Audio values must be [B,T,{self.input_width}]")
        b, t, samples = x.shape
        patches = (samples + self.patch_size - 1) // self.patch_size
        x = F.pad(x, (0, patches * self.patch_size - samples)).reshape(
            b, t * patches, self.patch_size
        )
        ends = torch.arange(t * patches, device=x.device)[None].expand(b, -1)
        values = self.stem(x) + position(ends.to(x.dtype), self.width) + self.modality
        fine = FeatureScale(
            values,
            times.repeat_interleave(patches, 1),
            valid.repeat_interleave(patches, 1),
            ends,
            (t * patches,),
        )
        fine, condition, condition_time = condition_inputs(
            fine, condition, condition_time, self.code_width
        )
        return self.pyramid(fine, condition, condition_time=condition_time, trace=trace)


class MultiScaleTextEncoder(nn.Module):
    """Causal token/span pyramid; spans do not imply learned linguistic units."""

    def __init__(
        self,
        width,
        vocabulary=259,
        *,
        code_width=16,
        levels=3,
        depth=1,
        cross_scale=True,
    ):
        super().__init__()
        self.width, self.code_width = width, code_width
        self.stem = TextEncoder(width, vocabulary)
        self.pyramid = FeatureHierarchy(
            width, code_width, (2,), levels, depth, cross_scale
        )

    def forward(self, observation, *, condition=None, condition_time=None, trace=None):
        check_order(observation)
        tokens = self.stem(observation)
        b, n = tokens.values.shape[:2]
        ends = torch.arange(n, device=tokens.values.device)[None].expand(b, -1)
        fine = FeatureScale(tokens.values, tokens.times, tokens.valid, ends, (n,))
        fine, condition, condition_time = condition_inputs(
            fine, condition, condition_time, self.code_width
        )
        return self.pyramid(fine, condition, condition_time=condition_time, trace=trace)


class FeatureController(nn.Module):
    """The caller chooses pre-observation state tokens; no observation input here."""

    def __init__(self, width, code_width=16):
        super().__init__()
        self.code_width = code_width
        self.output = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, code_width), nn.Tanh()
        )

    def forward(self, tokens):
        return self.output(tokens.mean(1))
