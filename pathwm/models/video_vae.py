"""Reuse one spatial image VAE for frames, with optional causal mean refinement.

The posterior factorizes over frame samples conditional on the observed clip:
q(z|x)=prod_t N(mu_t(x_<=t), diag(var_t(x_t))). The prior is iid N(0,I).
Summed KL is therefore valid for this deliberately simple generative model;
it does not constitute a learned temporal prior or future prediction.
"""

from dataclasses import replace

import torch
from torch import nn
from torch.nn import functional as F

from .modalities import observation_values
from .spatial_vae import SpatialVAE


class CausalLatentMixer(nn.Module):
    """Current and two previous latent grids; no state retained across calls."""

    def __init__(self, channels, *, spatial_kernel=3, spatial_iterations=0):
        super().__init__()
        if spatial_kernel < 1 or spatial_kernel % 2 != 1 or spatial_iterations < 0:
            raise ValueError(
                "Positive odd spatial kernel and nonnegative iterations required"
            )
        self.spatial_radius = spatial_kernel // 2
        self.spatial_iterations = spatial_iterations
        self.input = nn.Conv3d(
            channels + 2, 2 * channels, (3, spatial_kernel, spatial_kernel)
        )
        self.output = nn.Conv3d(2 * channels, channels, 1)
        # Shared spatial-only refinement never extends the temporal horizon.
        self.spatial = (
            nn.Conv3d(2 * channels, 2 * channels, (1, 3, 3), padding=(0, 1, 1))
            if spatial_iterations
            else nn.Identity()
        )
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, mu, times, valid):
        residual = self.features(mu, times, valid)
        return (mu + residual).masked_fill(~valid[:, :, None, None, None], 0)

    def features(self, mu, times, valid):
        """Separate causal features, without overwriting the spatial image code."""
        # mu [B,T,Z,H,W]. Time delta is measured since the last valid frame,
        # including across gaps. First valid frame has delta=0. No future scan.
        previous = torch.zeros_like(times[:, 0])
        seen = torch.zeros_like(valid[:, 0])
        deltas = []
        for t in range(mu.shape[1]):
            delta = torch.where(seen & valid[:, t], times[:, t] - previous, 0)
            deltas.append(delta.log1p().to(mu.dtype))
            previous = torch.where(valid[:, t], times[:, t], previous)
            seen = seen | valid[:, t]
        dt = torch.stack(deltas, 1)[:, :, None, None, None].expand_as(mu[:, :, :1])
        mask = valid[:, :, None, None, None]
        features = torch.cat((mu.masked_fill(~mask, 0), dt, mask.expand_as(dt)), 2)
        # Explicit left-only time padding. Spatial padding is symmetric.
        r = self.spatial_radius
        hidden = F.silu(self.input(F.pad(features.transpose(1, 2), (r, r, r, r, 2, 0))))
        for _ in range(self.spatial_iterations):
            hidden = hidden + 0.1 * F.silu(self.spatial(hidden))
        residual = self.output(hidden).transpose(1, 2)
        return residual.masked_fill(~mask, 0)


def local_correlation(previous, current, *, radius=2):
    """Horizontal cosine matches on a common valid interior, without wrap/pad.

    Output[B,2*radius+1,H-2*radius,W-2*radius]; channel i matches current(x)
    to previous(x+i-radius). A negative matching offset indicates rightward motion.
    This is an explicit matching primitive, not learned motion understanding.
    """
    if previous.shape != current.shape or current.ndim != 4:
        raise ValueError("Matching BCHW feature grids required")
    if type(radius) is not int or radius < 1 or min(current.shape[-2:]) <= 2 * radius:
        raise ValueError("Positive radius and nonempty valid interior required")
    previous = F.normalize(previous, dim=1, eps=1e-6)
    current = F.normalize(current, dim=1, eps=1e-6)
    h, w = current.shape[-2:]
    r = radius
    target = current[..., r : h - r, r : w - r]
    return torch.stack(
        [
            (previous[..., r : h - r, r + s : w - r + s] * target).sum(1)
            for s in range(-r, r + 1)
        ],
        1,
    )


class CorrespondenceDirectionHead(nn.Module):
    """Shared signed-offset evidence reader, initially the fixed cosine rule.

    Input is a horizontal correlation volume [B,2*radius+1,H,W]. Geometry defines
    the two sides; learning only refines candidate scores and positive logit scale.
    Channel reversal swaps logits. Arbitrary video reversal need not do so.
    """

    def __init__(self, radius=3):
        super().__init__()
        if type(radius) is not int or radius < 1:
            raise ValueError("Positive integer radius required")
        self.radius = radius
        self.register_buffer(
            "distances",
            torch.arange(-radius, radius + 1).abs().float() / radius,
            persistent=False,
        )
        self.residual = nn.Sequential(
            nn.Linear(3, 16), nn.SiLU(), nn.Linear(16, 1, bias=False)
        )
        nn.init.zeros_(self.residual[-1].weight)
        self.log_scale = nn.Parameter(torch.zeros(()))

    def forward(self, correlation, *, trace=None):
        r = self.radius
        if (
            correlation.ndim != 4
            or correlation.shape[1] != 2 * r + 1
            or min(correlation.shape) < 1
        ):
            raise ValueError("Expected nonempty [B,2*radius+1,H,W] correlation")
        cosine = correlation.mean((-1, -2))
        inputs = torch.stack(
            (
                cosine,
                self.distances.to(cosine).expand_as(cosine),
                cosine[:, r : r + 1].expand_as(cosine),
            ),
            -1,
        )
        residual = self.residual(inputs).squeeze(-1)
        scores = cosine + residual
        scale = self.log_scale.clamp(-4, 6).exp()
        difference = scale * (scores[:, :r].amax(1) - scores[:, r + 1 :].amax(1))
        if trace is not None:
            for name, value in dict(
                cosine=cosine, residual=residual, candidate_scores=scores, scale=scale
            ).items():
                trace[name] = value.detach().cpu().clone()
        return torch.stack((-difference / 2, difference / 2), -1)


class VideoVAE(nn.Module):
    """One owner for image weights; `model.image(rgb)` remains the image path.

    Video input is an Observation with values [B,T,3,H,W], seconds and optional mask.
    The returned Posterior is flattened [B*T,Z,h,w], in batch-major frame order;
    exclude invalid frames when computing losses. Decoder receives only z and H,W.
    This wrapper does not change existing agent token encoders or checkpoints.
    """

    def __init__(self, image, *, temporal=False):
        super().__init__()
        if not isinstance(image, SpatialVAE):
            raise TypeError("Inject an existing SpatialVAE (including HierarchicalVAE)")
        self.image = image
        self.temporal = (
            temporal
            if isinstance(temporal, nn.Module)
            else (
                CausalLatentMixer(image.config["latent_channels"])
                if temporal
                else nn.Identity()
            )
        )

    def encode(self, observation, *, trace=None):
        x, times, valid = observation_values(observation)
        if x.ndim != 5 or x.shape[2] != 3 or min(x.shape) < 1:
            raise ValueError("Expected nonempty video [B,T,3,H,W]")
        for row, mask in zip(times, valid):
            if (row[mask].diff() < 0).any():
                raise ValueError("Valid video timestamps must be nondecreasing")
        p = self.image.encode(x.flatten(0, 1))
        mu = p.mu.unflatten(0, x.shape[:2])
        before = mu.masked_fill(~valid[:, :, None, None, None], 0)
        after = (
            before
            if isinstance(self.temporal, nn.Identity)
            else self.temporal(before, times, valid)
        )
        p = replace(
            p,
            mu=after.flatten(0, 1),
            logvar=p.logvar.masked_fill(~valid.flatten()[:, None, None, None], 0),
        )
        if trace is not None:
            for key, value in dict(
                frame_mu=before,
                video_mu=after,
                logvar=p.logvar,
                times=times,
                valid=valid,
            ).items():
                trace[key] = value.detach().cpu().clone()
        return p

    def decode(self, z, output_size, valid):
        y = self.image.decode(z, output_size).unflatten(0, valid.shape)
        return y.masked_fill(~valid[:, :, None, None, None], 0)

    def forward(self, observation, *, sample=True, generator=None, trace=None):
        p = self.encode(observation, trace=trace)
        valid = observation.valid
        if valid is None:
            valid = torch.ones_like(observation.times, dtype=torch.bool)
        z = p.sample(generator) if sample else p.mu
        return self.decode(z, p.original_size, valid), p
