"""Predict-then-correct recurrent belief updater for ABI v2.

What: learned canonical queries initialize belief from any evidence subset; subsequent observations
correct the action-conditioned predictor prior through one gated cross-attention block.
How: variable evidence streams are concatenated in sorted modality order and padding-masked. Rows with
no valid evidence fall back to their prior; a globally empty mapping returns the predictor output bit
exactly.
Why: E1-a's frame state aliases hidden velocity. A persistent posterior carries history while keeping
the planner-facing predictor Markov in the complete state `W`.
"""
from __future__ import annotations

import weakref
from collections.abc import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from contracts import ActionTokens, EvidenceTokens, WorldPredictorV2
from world_state.abi_v2 import ABIv2, ROOT, load_abi_v2


class PredictThenCorrectUpdater(nn.Module):
    def __init__(
        self,
        abi: ABIv2,
        predictor: WorldPredictorV2,
        heads: int,
        mlp_ratio: int,
        correction_gate_bias: float,
    ) -> None:
        super().__init__()
        if abi.dim % heads or heads < 1 or mlp_ratio < 1:
            raise ValueError("updater needs positive heads/mlp_ratio and ABI dim divisible by heads")
        self.abi = abi
        # The composed model owns P. A weak reference avoids registering/saving the same predictor twice.
        object.__setattr__(self, "_predictor_ref", weakref.ref(predictor))
        self.initial_tokens = nn.Parameter(torch.empty(1, abi.n_tokens, abi.dim))
        self.query_norm = nn.LayerNorm(abi.dim)
        self.evidence_norm = nn.LayerNorm(abi.evidence_dim)
        self.cross_attention = nn.MultiheadAttention(
            abi.dim,
            heads,
            dropout=0.0,
            batch_first=True,
            kdim=abi.evidence_dim,
            vdim=abi.evidence_dim,
        )
        self.correction_mlp = nn.Sequential(
            nn.LayerNorm(abi.dim),
            nn.Linear(abi.dim, mlp_ratio * abi.dim),
            nn.GELU(),
            nn.Linear(mlp_ratio * abi.dim, abi.dim),
        )
        self.gate = nn.Linear(2 * abi.dim, abi.dim)
        nn.init.trunc_normal_(self.initial_tokens, std=0.02)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, correction_gate_bias)

    @property
    def predictor(self) -> WorldPredictorV2:
        predictor = self._predictor_ref()
        if predictor is None:
            raise RuntimeError("the updater's predictor no longer exists")
        return predictor

    def _combine(
        self,
        evidence: Mapping[str, EvidenceTokens],
        *,
        batch: int | None = None,
        device: torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not evidence:
            raise ValueError("at least one evidence stream is required")
        token_parts, mask_parts = [], []
        inferred_batch = batch
        for key in sorted(evidence):
            stream = evidence[key]
            if key != stream.modality:
                raise ValueError(f"evidence key {key!r} != stream modality {stream.modality!r}")
            self.abi.check_evidence(stream.tokens, stream.timestamps, stream.valid_mask)
            if inferred_batch is None:
                inferred_batch = stream.tokens.shape[0]
            if stream.tokens.shape[0] != inferred_batch:
                raise ValueError("evidence streams have different batch sizes")
            if device is not None and stream.tokens.device != device:
                raise ValueError("evidence and belief must be on the same device")
            token_parts.append(stream.tokens.float())
            mask_parts.append(stream.valid_mask)
        tokens = torch.cat(token_parts, dim=1)
        valid = torch.cat(mask_parts, dim=1)
        has_observation = valid.any(dim=1)
        # MultiheadAttention cannot consume an all-masked row. Its substitute token is suppressed by
        # has_observation after attention, so this is numerically safe and semantically an exact prior.
        if not has_observation.all():
            tokens = tokens.clone()
            valid = valid.clone()
            missing = ~has_observation
            tokens[missing, 0] = 0.0
            valid[missing, 0] = True
        return tokens, valid, has_observation

    def _correct(
        self,
        prior: torch.Tensor,
        evidence: Mapping[str, EvidenceTokens],
    ) -> torch.Tensor:
        self.abi.check_state(prior)
        tokens, valid, has_observation = self._combine(
            evidence,
            batch=prior.shape[0],
            device=prior.device,
        )
        prior_float = prior.float()
        correction, _ = self.cross_attention(
            self.query_norm(prior_float),
            self.evidence_norm(tokens),
            self.evidence_norm(tokens),
            key_padding_mask=~valid,
            need_weights=False,
        )
        correction = correction + self.correction_mlp(correction)
        gate = torch.sigmoid(self.gate(torch.cat((prior_float, correction), dim=-1)))
        posterior = F.layer_norm(prior_float + gate * correction, (self.abi.dim,)).to(self.abi.dtype)
        return torch.where(has_observation[:, None, None], posterior, prior)

    def initialize(self, evidence: Mapping[str, EvidenceTokens]) -> torch.Tensor:
        if not evidence:
            raise ValueError("belief initialization requires at least one evidence stream")
        first = next(iter(evidence.values()))
        initial = F.layer_norm(
            self.initial_tokens.expand(first.tokens.shape[0], -1, -1),
            (self.abi.dim,),
        ).to(device=first.tokens.device, dtype=self.abi.dtype)
        return self._correct(initial, evidence)

    def update(
        self,
        W_prev: torch.Tensor,
        evidence: Mapping[str, EvidenceTokens],
        actions: ActionTokens,
        delta_t: torch.Tensor,
    ) -> torch.Tensor:
        prior = self.predictor.predict_state(W_prev, actions, delta_t)
        if not evidence:
            return prior
        return self._correct(prior, evidence)


def build_belief_updater(cfg: dict, predictor: WorldPredictorV2) -> nn.Module:
    section = cfg["updater"]
    if section["kind"] != "predict_then_correct_v2":
        raise ValueError("the common-base slice implements updater.kind='predict_then_correct_v2'")
    abi = load_abi_v2(ROOT / cfg["abi"])
    return PredictThenCorrectUpdater(
        abi,
        predictor,
        heads=int(section["heads"]),
        mlp_ratio=int(section.get("mlp_ratio", 2)),
        correction_gate_bias=float(section.get("correction_gate_bias", -1.0)),
    )
