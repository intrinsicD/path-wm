"""Executable encoder-first R0/R1 training path for video and audio."""
from __future__ import annotations

import importlib
from pathlib import Path

import pytest
import torch
import yaml

import contracts
from training.base_model import build_common_world_model

ROOT = Path(__file__).resolve().parents[2]


def _build():
    try:
        module = importlib.import_module("training.representation")
    except ModuleNotFoundError:
        pytest.fail("no implementation: training.representation.build_representation_learner(cfg, core)")
    cfg = yaml.safe_load((ROOT / "configs/dev/common_base.yaml").read_text())
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


def test_r0_loss_trains_both_encoders_but_not_belief_or_dynamics():
    _, core, learner = _build()
    learner.set_stage("representation_unimodal")
    current, future = _observations(21), _observations(22)
    originals = {key: value.values.clone() for key, value in current.items()}

    values = learner.loss(
        current,
        future,
        stage="representation_unimodal",
        generator=torch.Generator().manual_seed(23),
    )
    values["total"].backward()

    assert torch.isfinite(values["total"]) and values["total"] > 0
    assert values["masked_latent"] > 0 and values["future_latent"] > 0
    assert values["audiovisual_sync"] == 0
    assert _has_gradient(core.encoders["video"])
    assert _has_gradient(core.encoders["audio"])
    assert not _has_gradient(core.predictor)
    assert not _has_gradient(core.updater)
    assert all(torch.equal(current[key].values, originals[key]) for key in current)


def test_r1_adds_synchronized_av_objective_and_keeps_teacher_frozen():
    _, core, learner = _build()
    learner.set_stage("representation_av")
    values = learner.loss(
        _observations(31),
        _observations(32),
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
        _observations(41),
        _observations(42),
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
