"""ABI-v2 projection from modality-native tokens to shared evidence tokens.

What: a per-token projection, learned modality identity and continuous timestamp embedding.
How: metadata is preserved exactly; projected valid tokens cross a non-affine LayerNorm boundary and
padded tokens are zeroed.
Why: fusion needs a shared numerical width without erasing the different token geometry owned by each
sensor encoder. Timestamp conditioning makes asynchronous evidence explicit rather than positional luck.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from contracts import EvidenceTokens
from encoders.temporal import CoordinateEmbedding
from world_state.abi_v2 import ABIv2


class EvidenceProjection(nn.Module):
    def __init__(self, modality: str, input_dim: int, abi: ABIv2) -> None:
        super().__init__()
        if input_dim < 1:
            raise ValueError("evidence input_dim must be positive")
        self.modality = modality
        self.input_dim = input_dim
        self.abi = abi
        self.projection = nn.Linear(input_dim, abi.evidence_dim)
        self.modality_embedding = nn.Parameter(torch.empty(1, 1, abi.evidence_dim))
        self.time_embedding = CoordinateEmbedding(1, abi.evidence_dim)
        nn.init.trunc_normal_(self.modality_embedding, std=0.02)

    def adapt_evidence(self, evidence: EvidenceTokens) -> EvidenceTokens:
        tokens, timestamps, valid = evidence.tokens, evidence.timestamps, evidence.valid_mask
        if evidence.modality != self.modality:
            raise ValueError(
                f"adapter for {self.modality!r} received {evidence.modality!r} evidence"
            )
        if tokens.ndim != 3 or tokens.shape[-1] != self.input_dim:
            raise ValueError(
                f"native evidence shape {tuple(tokens.shape)} != (B,N,{self.input_dim})"
            )
        if tuple(timestamps.shape) != tuple(tokens.shape[:2]) or not torch.is_floating_point(timestamps):
            raise ValueError("evidence timestamps must be floating (B,N)")
        if tuple(valid.shape) != tuple(tokens.shape[:2]) or valid.dtype != torch.bool:
            raise ValueError("evidence valid_mask must be bool (B,N)")
        projected = self.projection(tokens.float())
        projected = projected + self.modality_embedding + self.time_embedding(timestamps[..., None])
        projected = F.layer_norm(projected, (self.abi.evidence_dim,))
        projected = projected.masked_fill(~valid[..., None], 0.0).to(self.abi.evidence_dtype)
        return EvidenceTokens(projected, timestamps.clone(), valid.clone(), self.modality)

    def forward(self, evidence: EvidenceTokens) -> EvidenceTokens:
        return self.adapt_evidence(evidence)
