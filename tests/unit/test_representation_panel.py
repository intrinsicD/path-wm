"""The R0/R1 gate panel must use held-out views and expose every configured metric."""
from __future__ import annotations

import importlib
from pathlib import Path

import pytest
import torch
import yaml

import contracts
from training.curriculum import CommonBaseCurriculum

ROOT = Path(__file__).resolve().parents[2]


def _module():
    try:
        return importlib.import_module("evaluation.representation")
    except ModuleNotFoundError:
        pytest.fail("no implementation: evaluation.representation.evaluate_representation(...)")


def _evidence(tokens: torch.Tensor, modality: str) -> contracts.EvidenceTokens:
    batch, count = tokens.shape[:2]
    return contracts.EvidenceTokens(
        tokens,
        torch.zeros(batch, count),
        torch.ones(batch, count, dtype=torch.bool),
        modality,
    )


class HeldOutData:
    def __init__(self) -> None:
        self.calls = []

    def sample(self, split, stage, batch_size, generator):
        self.calls.append((split, stage, batch_size))
        return contracts.RepresentationBatch({}, {}, {})


class ExactLearner:
    def eval(self):
        return self

    def evaluation_views(self, batch, *, stage, generator):
        del batch, stage, generator
        basis = torch.eye(4).unsqueeze(1)
        online = {modality: _evidence(basis, modality) for modality in ("video", "audio")}
        targets = {modality: _evidence(2.0 * basis, modality) for modality in ("video", "audio")}
        predictions = {modality: 2.0 * basis for modality in ("video", "audio")}
        embeddings = {modality: basis[:, 0] for modality in ("video", "audio")}
        shifted = {modality: torch.roll(basis[:, 0], 1, 0) for modality in ("video", "audio")}
        return {
            "online": online,
            "masked_source": online,
            "future_source": online,
            "teacher_current": targets,
            "teacher_future": targets,
            "masked_prediction": predictions,
            "future_prediction": predictions,
            "av_current": embeddings,
            "av_shifted": shifted,
        }


def test_panel_is_held_out_per_modality_and_closes_both_gate_schemas():
    data = HeldOutData()
    metrics = _module().evaluate_representation(
        ExactLearner(),
        data,
        stage="representation_av",
        batches=2,
        batch_size=4,
        generator=torch.Generator().manual_seed(0),
    )

    assert data.calls == [("eval", "representation_av", 4)] * 2
    assert metrics["video_temporal_retrieval_margin"] == pytest.approx(1.0)
    assert metrics["audio_temporal_retrieval_margin"] == pytest.approx(1.0)
    assert metrics["video_to_audio_retrieval_margin"] == pytest.approx(1.0)
    assert metrics["audio_to_video_retrieval_margin"] == pytest.approx(1.0)
    assert metrics["synchrony_accuracy_above_chance"] == pytest.approx(0.5)
    for modality in ("video", "audio"):
        assert metrics[f"{modality}_feature_std"] > 0
        assert 0 < metrics[f"{modality}_effective_rank_fraction"] <= 1
        assert metrics[f"{modality}_masked_prediction_advantage"] > 0
        assert metrics[f"{modality}_future_prediction_advantage"] > 0

    cfg = yaml.safe_load((ROOT / "configs/dev/common_base.yaml").read_text())
    curriculum = CommonBaseCurriculum.from_config(cfg["curriculum"])
    for stage in ("representation_unimodal", "representation_av"):
        result = curriculum.evaluate(stage, metrics)
        assert not any("missing metric" in failure for failure in result.failures)
