"""Executable encoder-first R0/R1 training path for video and audio."""
from __future__ import annotations

import importlib
from copy import deepcopy
from pathlib import Path

import pytest
import torch
import yaml

import contracts
from training.base_model import build_common_world_model
from training.representation import _covariance_loss, _variance_loss

ROOT = Path(__file__).resolve().parents[2]


def _build(spec: str = "common_base.yaml"):
    try:
        module = importlib.import_module("training.representation")
    except ModuleNotFoundError:
        pytest.fail("no implementation: training.representation.build_representation_learner(cfg, core)")
    cfg = yaml.safe_load((ROOT / "configs/dev" / spec).read_text())
    core = build_common_world_model(cfg)
    return cfg, core, module.build_representation_learner(cfg, core)


def _observations(seed: int, batch: int = 3) -> dict[str, contracts.TemporalObservation]:
    generator = torch.Generator().manual_seed(seed)
    return {
        "video": contracts.TemporalObservation(
            torch.randint(0, 256, (batch, 4, 3, 64, 64), dtype=torch.uint8, generator=generator),
            torch.linspace(-0.12, 0.0, 4).expand(batch, -1).clone(),
            torch.ones(batch, 4, dtype=torch.bool),
        ),
        "audio": contracts.TemporalObservation(
            torch.randn(batch, 1, 2048, generator=generator).clamp(-1, 1),
            torch.linspace(-0.128, 0.0, 2048).expand(batch, -1).clone(),
            torch.ones(batch, 2048, dtype=torch.bool),
        ),
    }


def _has_gradient(module: torch.nn.Module) -> bool:
    return any(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        and parameter.grad.abs().sum() > 0
        for parameter in module.parameters()
        if parameter.requires_grad
    )


def _evidence(values: torch.Tensor) -> contracts.EvidenceTokens:
    batch, tokens, _ = values.shape
    return contracts.EvidenceTokens(
        values,
        torch.zeros(batch, tokens),
        torch.ones(batch, tokens, dtype=torch.bool),
        "video",
    )


def test_covariance_guardrail_detects_dimensional_collapse_at_matched_variance():
    root_two = 2.0**0.5
    decorrelated = torch.tensor(
        [[[-root_two, 0.0], [root_two, 0.0], [0.0, -root_two], [0.0, root_two]]],
        requires_grad=True,
    )
    correlated = torch.tensor(
        [[[-1.0, -1.0], [1.0, 1.0], [-1.0, -1.0], [1.0, 1.0]]],
        requires_grad=True,
    )

    decorrelated_loss = _covariance_loss(_evidence(decorrelated))
    torch.testing.assert_close(
        _variance_loss(_evidence(decorrelated)),
        _variance_loss(_evidence(correlated)),
    )
    correlated_loss = _covariance_loss(_evidence(correlated))
    correlated_loss.backward()

    torch.testing.assert_close(decorrelated_loss, torch.zeros_like(decorrelated_loss))
    assert correlated_loss > 0
    assert correlated.grad is not None and correlated.grad.abs().sum() > 0


def test_rank_specs_change_only_the_declared_covariance_intervention():
    base = yaml.safe_load((ROOT / "configs/dev/common_base.yaml").read_text())
    rank = yaml.safe_load((ROOT / "configs/dev/common_base_rank.yaml").read_text())
    balanced = yaml.safe_load((ROOT / "configs/dev/common_base_rank_balanced.yaml").read_text())

    assert rank["representation"]["weights"]["covariance"] == pytest.approx(0.01)
    assert balanced["representation"]["weights"]["covariance"] == pytest.approx(0.002)
    without_covariance = deepcopy(rank)
    without_covariance["representation"]["objectives"].remove("covariance")
    without_covariance["representation"]["weights"].pop("covariance")
    assert without_covariance == base
    balanced_at_pilot_weight = deepcopy(balanced)
    balanced_at_pilot_weight["representation"]["weights"]["covariance"] = 0.01
    assert balanced_at_pilot_weight == rank


def test_declared_covariance_objective_is_included_in_r0_total():
    cfg, _, learner = _build("common_base_rank.yaml")
    learner.set_stage("representation_unimodal")
    values = learner.loss(
        contracts.RepresentationBatch(_observations(11), _observations(12), {}),
        stage="representation_unimodal",
        generator=torch.Generator().manual_seed(13),
    )
    weights = cfg["representation"]["weights"]
    expected = (
        weights["masked_latent"] * values["masked_latent"]
        + weights["future_latent"] * values["future_latent"]
        + weights["variance"] * values["variance"]
        + weights["covariance"] * values["covariance"]
    )

    assert values["covariance"] > 0
    torch.testing.assert_close(values["total"], expected)


def test_r0_loss_trains_both_encoders_but_not_belief_or_dynamics():
    _, core, learner = _build()
    learner.set_stage("representation_unimodal")
    current, future = _observations(21), _observations(22)
    originals = {key: value.values.clone() for key, value in current.items()}

    values = learner.loss(
        contracts.RepresentationBatch(current, future, {}),
        stage="representation_unimodal",
        generator=torch.Generator().manual_seed(23),
    )
    values["total"].backward()

    assert torch.isfinite(values["total"]) and values["total"] > 0
    assert values["masked_latent"] > 0 and values["future_latent"] > 0
    assert values["covariance"] == 0
    assert values["audiovisual_sync"] == 0
    assert _has_gradient(core.encoders["video"])
    assert _has_gradient(core.encoders["audio"])
    assert not _has_gradient(core.predictor)
    assert not _has_gradient(core.updater)
    assert all(torch.equal(current[key].values, originals[key]) for key in current)


def test_r1_adds_synchronized_av_objective_and_keeps_teacher_frozen():
    _, core, learner = _build()
    learner.set_stage("representation_av")
    current, future, shifted = _observations(31), _observations(32), _observations(34)
    values = learner.loss(
        contracts.RepresentationBatch(current, future, shifted),
        stage="representation_av",
        generator=torch.Generator().manual_seed(33),
    )
    values["total"].backward()

    assert values["audiovisual_sync"] > 0
    assert all(not parameter.requires_grad for teacher in learner.teachers.values() for parameter in teacher.parameters())
    assert _has_gradient(core.encoders["video"])
    assert _has_gradient(core.encoders["audio"])


def test_optimizer_step_then_ema_update_moves_teacher_toward_online_branch():
    _, core, learner = _build()
    learner.set_stage("representation_unimodal")
    optimizer = torch.optim.AdamW(
        [parameter for parameter in learner.parameters() if parameter.requires_grad],
        lr=0.001,
    )
    teacher_before = learner.teachers["video"].module.encoder.patch_embed.weight.detach().clone()
    values = learner.loss(
        contracts.RepresentationBatch(_observations(41), _observations(42), {}),
        stage="representation_unimodal",
        generator=torch.Generator().manual_seed(43),
    )
    values["total"].backward()
    optimizer.step()
    online_after = core.encoders["video"].patch_embed.weight.detach().clone()
    learner.update_teachers()
    teacher_after = learner.teachers["video"].module.encoder.patch_embed.weight.detach()

    assert not torch.equal(online_after, teacher_before)
    assert not torch.equal(teacher_after, teacher_before)
    assert torch.linalg.vector_norm(teacher_after - online_after) < torch.linalg.vector_norm(
        teacher_before - online_after
    )


def test_switching_to_dynamics_freezes_encoders_and_representation_heads():
    _, core, learner = _build()
    learner.set_stage("dynamics_one_step")

    assert all(not parameter.requires_grad for parameter in core.encoders.parameters())
    assert all(not parameter.requires_grad for parameter in learner.masked_heads.parameters())
    assert any(parameter.requires_grad for parameter in core.predictor.parameters())
    assert any(parameter.requires_grad for parameter in core.updater.parameters())
    assert any(parameter.requires_grad for parameter in core.action_adapter.parameters())
