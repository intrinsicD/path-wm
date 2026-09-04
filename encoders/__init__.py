"""Encoder family E_m: build_encoder(cfg) selects the architecture by one YAML line (DDR §13.6, CLAUDE.md §3).

What: a plain dict over the encoder architectures that exist today; cfg is the whole parsed spec and the
builder reads cfg["encoder"] and cfg["env"]["resolution"]. Builds on CPU; the runner moves modules.
Why: H1 (§4) swaps the encoder against a frozen predictor (E2); the swap must be `encoder.arch` only.
"""
from __future__ import annotations

import torch.nn as nn

from encoders.audio import SpectrogramAudioEncoder
from encoders.vit import ViT
from encoders.video import TubeletVideoEncoder


def _vit_s8(cfg: dict) -> nn.Module:
    e = cfg["encoder"]
    return ViT(
        dim=int(e["dim"]),
        layers=int(e["layers"]),
        heads=int(e["heads"]),
        resolution=int(cfg["env"]["resolution"]),
        mlp_ratio=int(e.get("mlp_ratio", 4)),  # 4 is the ViT-S constant; a spec may override it
    )


ENCODERS = {"vit_s8": _vit_s8}


def build_encoder(cfg: dict) -> nn.Module:
    arch = cfg["encoder"]["arch"]
    if arch not in ENCODERS:
        raise ValueError(f"unknown encoder.arch {arch!r}; known: {sorted(ENCODERS)}")
    return ENCODERS[arch](cfg)


def _tubelet_transformer(section: dict) -> nn.Module:
    return TubeletVideoEncoder(
        input_channels=int(section["input_channels"]),
        dim=int(section["dim"]),
        layers=int(section["layers"]),
        heads=int(section["heads"]),
        mlp_ratio=int(section.get("mlp_ratio", 4)),
        tubelet=int(section["tubelet"]),
        patch=int(section["patch"]),
    )


def _spectrogram_transformer(section: dict) -> nn.Module:
    return SpectrogramAudioEncoder(
        input_channels=int(section["input_channels"]),
        dim=int(section["dim"]),
        layers=int(section["layers"]),
        heads=int(section["heads"]),
        mlp_ratio=int(section.get("mlp_ratio", 4)),
        sample_rate=int(section["sample_rate"]),
        n_fft=int(section["n_fft"]),
        hop_length=int(section["hop_length"]),
        frequency_patch=int(section["frequency_patch"]),
        time_patch=int(section["time_patch"]),
    )


EVIDENCE_ENCODERS = {
    "tubelet_transformer": _tubelet_transformer,
    "spectrogram_transformer": _spectrogram_transformer,
}


def build_evidence_encoder(cfg: dict, modality: str) -> nn.Module:
    """Build one enabled ABI-v2 modality frontend; adding a sensor is one registry entry."""
    modalities = cfg.get("modalities", {})
    if modality not in modalities or not modalities[modality].get("enabled", False):
        raise ValueError(f"modality {modality!r} is absent or disabled")
    section = modalities[modality]["encoder"]
    arch = section["arch"]
    if arch not in EVIDENCE_ENCODERS:
        raise ValueError(f"unknown evidence encoder {arch!r}; known: {sorted(EVIDENCE_ENCODERS)}")
    encoder = EVIDENCE_ENCODERS[arch](section)
    if encoder.modality != modality:
        raise ValueError(f"encoder {arch!r} emits modality {encoder.modality!r}, configured as {modality!r}")
    return encoder
