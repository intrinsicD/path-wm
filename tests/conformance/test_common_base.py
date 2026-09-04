"""Structural conformance for the ABI-v2 video/audio world-model core.

These tests intentionally load the implementation inside the test body: at slice step 2 the missing
builder is a named FAILED test, not a collection error.  No learned-quality threshold is asserted here.
"""
from __future__ import annotations

import importlib
from pathlib import Path

import pytest
import torch
import yaml

import contracts
from world_state.abi_v2 import load_abi_v2

ROOT = Path(__file__).resolve().parents[2]
CFG_PATH = ROOT / "configs/dev/common_base.yaml"


def _cfg() -> dict:
    return yaml.safe_load(CFG_PATH.read_text())


def _build():
    try:
        module = importlib.import_module("training.base_model")
    except ModuleNotFoundError:
        pytest.fail("no implementation: training.base_model.build_common_world_model(cfg)")
    builder = getattr(module, "build_common_world_model", None)
    if builder is None:
        pytest.fail("no implementation: training.base_model.build_common_world_model(cfg)")
    torch.manual_seed(0)
    return builder(_cfg())


def _observations(batch: int = 2) -> dict[str, contracts.TemporalObservation]:
    video = torch.randint(
        0,
        256,
        (batch, 4, 3, 64, 64),
        dtype=torch.uint8,
        generator=torch.Generator().manual_seed(11),
    )
    audio = torch.randn(batch, 1, 2048, generator=torch.Generator().manual_seed(12)).clamp(-1, 1)
    return {
        "video": contracts.TemporalObservation(
            video,
            torch.linspace(-0.12, 0.0, 4).expand(batch, -1).clone(),
            torch.ones(batch, 4, dtype=torch.bool),
        ),
        "audio": contracts.TemporalObservation(
            audio,
            torch.linspace(-0.128, 0.0, 2048).expand(batch, -1).clone(),
            torch.ones(batch, 2048, dtype=torch.bool),
        ),
    }


def _actions(batch: int = 2, observed: bool = True) -> contracts.ActionSequence:
    return contracts.ActionSequence(
        values=torch.tensor([[[0.4, -0.2]]]).expand(batch, -1, -1).clone(),
        timestamps=torch.zeros(batch, 1),
        valid_mask=torch.ones(batch, 1, dtype=torch.bool),
        observed_mask=torch.full((batch, 1), observed, dtype=torch.bool),
    )


def test_video_and_audio_encoders_preserve_inputs_and_emit_valid_adapted_evidence():
    model = _build().eval()
    abi = load_abi_v2(ROOT / _cfg()["abi"])
    observations = _observations()

    for modality, observation in observations.items():
        before = observation.values.clone()
        native = model.encoders[modality].encode_observation(observation)
        adapted = model.adapters[modality].adapt_evidence(native)
        assert isinstance(model.encoders[modality], contracts.EvidenceEncoder)
        assert isinstance(model.adapters[modality], contracts.EvidenceAdapter)
        assert native.modality == adapted.modality == modality
        assert native.tokens.shape[:2] == adapted.tokens.shape[:2]
        assert torch.equal(adapted.timestamps, native.timestamps)
        assert torch.equal(adapted.valid_mask, native.valid_mask)
        assert torch.equal(observation.values, before)
        assert torch.isfinite(native.tokens).all()
        abi.check_evidence(adapted.tokens, adapted.timestamps, adapted.valid_mask)


def test_core_accepts_video_audio_or_both_and_keeps_a_fixed_belief_abi():
    model = _build().eval()
    abi = load_abi_v2(ROOT / _cfg()["abi"])
    observations = _observations()

    assert isinstance(model, contracts.WorldModelCore)
    states = {
        "video": model.initialize({"video": observations["video"]}),
        "audio": model.initialize({"audio": observations["audio"]}),
        "both": model.initialize(observations),
    }
    for W in states.values():
        abi.check_state(W)
    assert not torch.equal(states["video"], states["audio"])
    assert not torch.equal(states["both"], states["video"])
    assert not torch.equal(states["both"], states["audio"])


def test_no_observation_update_is_exactly_the_predictor_prior():
    model = _build().eval()
    observations = _observations()
    W = model.initialize(observations)
    actions = _actions()
    delta_t = torch.full((W.shape[0],), 0.1)

    imagined = model.imagine(W, actions, delta_t)
    filtered_without_evidence = model.observe(W, {}, actions, delta_t)
    assert torch.equal(imagined, filtered_without_evidence)


def test_timestamps_and_unknown_actions_are_not_silently_discarded():
    model = _build().eval()
    observations = _observations()
    shifted = dict(observations)
    video = observations["video"]
    shifted["video"] = contracts.TemporalObservation(
        video.values,
        video.timestamps - 0.5,
        video.valid_mask,
    )

    W = model.initialize(observations)
    W_shifted = model.initialize(shifted)
    assert not torch.equal(W, W_shifted)

    delta_t = torch.full((W.shape[0],), 0.1)
    known = model.imagine(W, _actions(observed=True), delta_t)
    unknown = model.imagine(W, _actions(observed=False), delta_t)
    assert not torch.equal(known, unknown)


def test_joint_belief_has_nonzero_gradient_paths_to_both_encoders():
    model = _build().train()
    W = model.initialize(_observations())
    W.float()[..., 0].mean().backward()

    for modality in ("video", "audio"):
        gradients = [p.grad for p in model.encoders[modality].parameters() if p.requires_grad]
        assert any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in gradients), modality
