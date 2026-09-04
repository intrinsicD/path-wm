"""Gate ordering, module freezing, and EMA target semantics for the common-base curriculum."""
from __future__ import annotations

import importlib
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import yaml

ROOT = Path(__file__).resolve().parents[2]


def _curriculum_module():
    try:
        return importlib.import_module("training.curriculum")
    except ModuleNotFoundError:
        pytest.fail("no implementation: training.curriculum.CommonBaseCurriculum")


def test_every_stage_transition_has_an_explicit_held_out_gate():
    module = _curriculum_module()
    cfg = yaml.safe_load((ROOT / "configs/dev/common_base.yaml").read_text())
    curriculum = module.CommonBaseCurriculum.from_config(cfg["curriculum"])

    assert curriculum.stages == (
        "representation_unimodal",
        "representation_av",
        "belief_bootstrap",
        "dynamics_one_step",
        "dynamics_rollout",
        "planning",
    )
    assert set(curriculum.transition_gates) == set(curriculum.stages[:-1])


def test_gate_fails_closed_on_missing_or_bad_metrics_and_reports_why():
    module = _curriculum_module()
    cfg = yaml.safe_load((ROOT / "configs/dev/common_base.yaml").read_text())
    curriculum = module.CommonBaseCurriculum.from_config(cfg["curriculum"])

    missing = curriculum.evaluate("dynamics_one_step", {})
    bad = curriculum.evaluate(
        "dynamics_one_step",
        {
            "correct_vs_identity_ratio": 0.9,
            "correct_vs_zero_ratio": 1.1,
            "correct_vs_shuffled_ratio": 0.8,
        },
    )
    good = curriculum.evaluate(
        "dynamics_one_step",
        {
            "correct_vs_identity_ratio": 0.9,
            "correct_vs_zero_ratio": 0.95,
            "correct_vs_shuffled_ratio": 0.8,
        },
    )

    assert not missing.passed and "missing" in " ".join(missing.failures)
    assert not bad.passed and "correct_vs_zero_ratio" in " ".join(bad.failures)
    assert good.passed and not good.failures


def test_planning_stage_freezes_the_world_model_core():
    module = _curriculum_module()

    assert module.trainable_groups("representation_unimodal") == frozenset(
        {"encoders", "evidence_adapters", "representation_heads"}
    )
    assert module.trainable_groups("belief_bootstrap") == frozenset(
        {"evidence_adapters", "updater"}
    )
    assert module.trainable_groups("dynamics_one_step") == frozenset(
        {"action_adapter", "updater", "predictor"}
    )
    assert module.trainable_groups("planning") == frozenset({"planner", "critic"})


def test_ema_teacher_is_frozen_and_updates_toward_online_parameters():
    module = _curriculum_module()
    online = nn.Linear(3, 2, bias=False)
    with torch.no_grad():
        online.weight.fill_(1.0)
    teacher = module.EMATeacher(online, decay=0.5)
    before = teacher.module.weight.detach().clone()
    with torch.no_grad():
        online.weight.fill_(3.0)
    teacher.update(online)

    assert all(not parameter.requires_grad for parameter in teacher.module.parameters())
    assert torch.allclose(teacher.module.weight, before * 0.5 + online.weight * 0.5)
