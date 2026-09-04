"""Composed video/audio world-model core for the ABI-v2 common base.

What: registered modality encoders/adapters, an embodiment action adapter, a Markov slot predictor and
the predict-then-correct belief updater behind one planner-facing module.
How: raw observations are encoded only in initialize/observe; imagine sees canonical belief and raw
actions but never sensory tensors. All component selection is owned by configs/dev/common_base.yaml.
Why: a single executable graph is the smallest test that the modular boundaries compose. Scientific
training objectives and datasets remain stage-specific and are not hidden inside the runtime model.
"""
from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn as nn

from contracts import ActionSequence, EvidenceTokens, TemporalObservation
from encoders import build_evidence_encoder
from encoders.adapters import build_evidence_adapter
from predictors import build_world_predictor
from world_state.action import build_action_adapter
from world_state.updater import build_belief_updater


class CommonWorldModel(nn.Module):
    def __init__(
        self,
        encoders: Mapping[str, nn.Module],
        adapters: Mapping[str, nn.Module],
        action_adapter: nn.Module,
        predictor: nn.Module,
        updater: nn.Module,
    ) -> None:
        super().__init__()
        if not encoders or set(encoders) != set(adapters):
            raise ValueError("common world model needs matching non-empty encoder/adapter maps")
        self.encoders = nn.ModuleDict(dict(encoders))
        self.adapters = nn.ModuleDict(dict(adapters))
        self.action_adapter = action_adapter
        self.predictor = predictor
        self.updater = updater

    def encode_observations(
        self,
        observations: Mapping[str, TemporalObservation],
    ) -> dict[str, EvidenceTokens]:
        unknown = set(observations) - set(self.encoders)
        if unknown:
            raise ValueError(f"no enabled encoder for modalities {sorted(unknown)}")
        evidence: dict[str, EvidenceTokens] = {}
        for modality, observation in observations.items():
            native = self.encoders[modality].encode_observation(observation)
            evidence[modality] = self.adapters[modality].adapt_evidence(native)
        return evidence

    def initialize(self, observations: Mapping[str, TemporalObservation]) -> torch.Tensor:
        return self.updater.initialize(self.encode_observations(observations))

    def observe(
        self,
        W_prev: torch.Tensor,
        observations: Mapping[str, TemporalObservation],
        actions: ActionSequence,
        delta_t: torch.Tensor,
    ) -> torch.Tensor:
        action_tokens = self.action_adapter.adapt_actions(actions)
        evidence = self.encode_observations(observations)
        return self.updater.update(W_prev, evidence, action_tokens, delta_t)

    def imagine(
        self,
        W: torch.Tensor,
        actions: ActionSequence,
        delta_t: torch.Tensor,
    ) -> torch.Tensor:
        action_tokens = self.action_adapter.adapt_actions(actions)
        return self.predictor.predict_state(W, action_tokens, delta_t)


def build_common_world_model(cfg: dict) -> CommonWorldModel:
    """Build every enabled evidence path and the shared belief/dynamics core on CPU."""
    encoders: dict[str, nn.Module] = {}
    adapters: dict[str, nn.Module] = {}
    for modality, section in cfg.get("modalities", {}).items():
        if not section.get("enabled", False):
            continue
        encoder = build_evidence_encoder(cfg, modality)
        encoders[modality] = encoder
        adapters[modality] = build_evidence_adapter(cfg, modality, encoder.output_dim)
    predictor = build_world_predictor(cfg)
    updater = build_belief_updater(cfg, predictor)
    return CommonWorldModel(
        encoders,
        adapters,
        build_action_adapter(cfg),
        predictor,
        updater,
    )
