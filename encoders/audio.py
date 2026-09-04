"""Spectrogram-patch audio evidence encoder for the common ABI-v2 base.

What: floating waveform spans become native time-frequency tokens with physical timestamps.
How: differentiable STFT magnitude, non-overlapping 2-D patch embedding, continuous (time,frequency)
coordinates, and padding-aware self-attention.
Why: audio has useful temporal geometry but no meaningful location in ABI v1's visual grid; it must
enter the shared model as evidence and be fused into persistent modality-neutral belief slots.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from contracts import EvidenceTokens, TemporalObservation
from encoders.temporal import CoordinateEmbedding, TokenTransformer


class SpectrogramAudioEncoder(nn.Module):
    modality = "audio"

    def __init__(
        self,
        input_channels: int,
        dim: int,
        layers: int,
        heads: int,
        mlp_ratio: int,
        sample_rate: int,
        n_fft: int,
        hop_length: int,
        frequency_patch: int,
        time_patch: int,
    ) -> None:
        super().__init__()
        if min(sample_rate, n_fft, hop_length, frequency_patch, time_patch) < 1:
            raise ValueError("audio sampling, FFT, hop and patch settings must be positive")
        self.input_channels = input_channels
        self.output_dim = dim
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.frequency_patch = frequency_patch
        self.time_patch = time_patch
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)
        self.patch_embed = nn.Conv2d(
            1,
            dim,
            kernel_size=(frequency_patch, time_patch),
            stride=(frequency_patch, time_patch),
        )
        self.position = CoordinateEmbedding(2, dim)
        self.transformer = TokenTransformer(dim, layers, heads, mlp_ratio)

    def encode_observation(self, observation: TemporalObservation) -> EvidenceTokens:
        values, timestamps, valid = observation.values, observation.timestamps, observation.valid_mask
        if values.ndim != 3 or not torch.is_floating_point(values) or values.shape[1] != self.input_channels:
            raise ValueError(
                f"audio must be floating (B,{self.input_channels},S), got {tuple(values.shape)} {values.dtype}"
            )
        batch, _, samples = values.shape
        if tuple(timestamps.shape) != (batch, samples) or not torch.is_floating_point(timestamps):
            raise ValueError("audio timestamps must be floating (B,S)")
        if tuple(valid.shape) != (batch, samples) or valid.dtype != torch.bool:
            raise ValueError("audio valid_mask must be bool (B,S)")
        if samples < self.n_fft:
            raise ValueError(f"audio span {samples} shorter than n_fft {self.n_fft}")

        waveform = values.float().mean(dim=1)
        spectrum = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window,
            center=False,
            return_complex=True,
        ).abs()
        log_spectrum = torch.log1p(spectrum)[:, None]
        if spectrum.shape[1] < self.frequency_patch or spectrum.shape[2] < self.time_patch:
            raise ValueError("audio span produces fewer STFT bins than the configured patch")
        embedded = self.patch_embed(log_spectrum)
        _, _, frequency_tokens, time_tokens = embedded.shape
        tokens = embedded.flatten(2).transpose(1, 2)

        frame_valid = valid.unfold(1, self.n_fft, self.hop_length).all(dim=-1)
        frame_times = timestamps.unfold(1, self.n_fft, self.hop_length).mean(dim=-1)
        used_frames = time_tokens * self.time_patch
        patch_valid = frame_valid[:, :used_frames].reshape(batch, time_tokens, self.time_patch).all(dim=-1)
        patch_times = frame_times[:, :used_frames].reshape(batch, time_tokens, self.time_patch).mean(dim=-1)
        token_valid = patch_valid[:, None, :].expand(-1, frequency_tokens, -1).reshape(batch, -1)
        token_times = patch_times[:, None, :].expand(-1, frequency_tokens, -1).reshape(batch, -1)

        frequency = torch.linspace(-1.0, 1.0, frequency_tokens, device=values.device)
        frequency = frequency[None, :, None].expand(batch, -1, time_tokens).reshape(batch, -1)
        coords = torch.stack((token_times, frequency), dim=-1)
        tokens = tokens + self.position(coords)
        tokens = self.transformer(tokens, token_valid)
        return EvidenceTokens(tokens, token_times, token_valid, self.modality)

    def forward(self, observation: TemporalObservation) -> EvidenceTokens:
        return self.encode_observation(observation)
