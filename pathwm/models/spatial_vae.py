"""Spatial VAE with explicit rearrangement, mixing and channel projection.

Only shuffle/unshuffle are guaranteed lossless. Reversible local mixers are an
optional controlled comparison; attention, projections and posterior remain lossy.
The decoder receives only a spatial latent and requested output geometry.
"""

from dataclasses import dataclass
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F


class ResidualMixer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.branch = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
        )
        nn.init.zeros_(self.branch[-1].weight)

    def forward(self, x):
        return x + self.branch(x)


class ReversibleMixer(nn.Module):
    """Additive coupling. Inverse uses the same deterministic subnetworks."""

    def __init__(self, channels):
        super().__init__()
        if channels < 2:
            raise ValueError("Coupling needs at least two channels")
        self.a = channels // 2
        b = channels - self.a

        def subnet(i, o):
            layers = nn.Sequential(
                nn.Conv2d(i, channels, 3, padding=1, bias=False),
                nn.SiLU(),
                nn.Conv2d(channels, o, 3, padding=1, bias=False),
            )
            nn.init.zeros_(layers[-1].weight)
            return layers

        self.f, self.g = subnet(b, self.a), subnet(self.a, b)

    def forward(self, x):
        a, b = x[:, : self.a], x[:, self.a :]
        a = a + self.f(b)
        b = b + self.g(a)
        return torch.cat((a, b), 1)

    def inverse(self, y):
        a, b = y[:, : self.a], y[:, self.a :]
        b = b - self.g(a)
        a = a - self.f(b)
        return torch.cat((a, b), 1)


def processing(channels, depth, reversible=False):
    if not isinstance(depth, int) or depth < 0:
        raise ValueError("Processing depth must be a nonnegative integer")
    block = ReversibleMixer if reversible else ResidualMixer
    return nn.Sequential(*(block(channels) for _ in range(depth)))


class EncoderStage(nn.Module):
    def __init__(self, ci, co, pre=0, post=1, reversible=False):
        super().__init__()
        self.pre_process = processing(ci, pre, reversible)
        self.downsample = nn.PixelUnshuffle(2)
        self.post_process = processing(4 * ci, post, reversible)
        self.projection = nn.Identity() if co is None else nn.Conv2d(4 * ci, co, 1)
        self.projection_kind = (
            "identity"
            if co is None
            else "reduction"
            if co < 4 * ci
            else "expansion"
            if co > 4 * ci
            else "mixing"
        )

    def forward(self, x):
        return self.projection(self.post_process(self.downsample(self.pre_process(x))))


class DecoderStage(nn.Module):
    def __init__(
        self, co, ci, pre=0, post=1, reversible=False, identity_projection=False
    ):
        super().__init__()
        self.projection = (
            nn.Identity() if identity_projection else nn.Conv2d(co, 4 * ci, 1)
        )
        self.processing = processing(4 * ci, post, reversible)
        self.upsample = nn.PixelShuffle(2)
        self.post_process = processing(ci, pre, reversible)

    def forward(self, x):
        return self.post_process(self.upsample(self.processing(self.projection(x))))


class CrossScaleAttention(nn.Module):
    """Coarsest queries read finer processed grids before the posterior."""

    def __init__(self, channels, width=32, heads=4):
        super().__init__()
        if len(channels) < 2 or width % heads:
            raise ValueError("Attention needs at least two scales and compatible width")
        self.query = nn.Conv2d(channels[-1], width, 1)
        self.context = nn.ModuleList(nn.Conv2d(c, width, 1) for c in channels[:-1])
        self.position = nn.Linear(4, width, bias=False)
        self.scale = nn.Parameter(torch.randn(len(channels), width) * 0.02)
        self.qnorm, self.knorm = nn.LayerNorm(width), nn.LayerNorm(width)
        self.attend = nn.MultiheadAttention(width, heads, dropout=0, batch_first=True)
        self.output = nn.Conv2d(width, channels[-1], 1, bias=False)
        nn.init.zeros_(self.output.weight)

    def tokens(self, value, index):
        h, w = value.shape[-2:]
        y = (torch.arange(h, device=value.device, dtype=value.dtype) + 0.5) * 2 / h - 1
        x = (torch.arange(w, device=value.device, dtype=value.dtype) + 0.5) * 2 / w - 1
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        coords = torch.stack(
            (xx, yy, torch.sin(torch.pi * xx), torch.sin(torch.pi * yy)), -1
        ).reshape(-1, 4)
        return (
            value.flatten(2).transpose(1, 2)
            + self.position(coords)[None]
            + self.scale[index]
        )

    def forward(self, grids):
        coarse = grids[-1]
        q = self.qnorm(self.tokens(self.query(coarse), len(grids) - 1))
        kv = self.knorm(
            torch.cat(
                [
                    self.tokens(p(x), i)
                    for i, (p, x) in enumerate(zip(self.context, grids[:-1]))
                ],
                1,
            )
        )
        output = self.attend(q, kv, kv, need_weights=False)[0]
        return coarse + self.output(
            output.transpose(1, 2).reshape(len(coarse), -1, *coarse.shape[-2:])
        )


@dataclass
class Posterior:
    mu: torch.Tensor
    logvar: torch.Tensor
    original_size: tuple
    padded_size: tuple

    def sample(self, generator=None):
        noise = torch.randn(
            self.mu.shape,
            device=self.mu.device,
            dtype=self.mu.dtype,
            generator=generator,
        )
        return self.mu + (self.logvar * 0.5).exp() * noise

    def kl_per_image(self):
        return 0.5 * (
            self.mu.float().square()
            + self.logvar.float().exp()
            - 1
            - self.logvar.float()
        ).sum((1, 2, 3))


class SpatialEncoder(nn.Module):
    def __init__(self, channels, latent_channels, pre, post, variant, logvar_bounds):
        super().__init__()
        self.factor = 2 ** len(channels)
        self.logvar_bounds = logvar_bounds
        self.stages = nn.ModuleList(
            EncoderStage(ci, co, a, b, variant == "reversible")
            for ci, co, a, b in zip((3, *channels[:-1]), channels, pre, post)
        )
        self.attention = CrossScaleAttention(channels) if variant != "base" else None
        self.mu, self.logvar = (
            nn.Conv2d(channels[-1], latent_channels, 1),
            nn.Conv2d(channels[-1], latent_channels, 1),
        )

    def forward(self, x):
        if (
            x.ndim != 4
            or x.shape[1] != 3
            or min(x.shape[0], *x.shape[-2:]) < 1
            or not x.is_floating_point()
        ):
            raise ValueError("Expected nonempty floating RGB B,C,H,W")
        h, w = x.shape[-2:]
        x = F.pad(x, (0, (-w) % self.factor, 0, (-h) % self.factor), mode="replicate")
        padded = x.shape[-2:]
        grids = []
        for stage in self.stages:
            x = stage(x)
            grids.append(x)
        if self.attention is not None:
            x = self.attention(grids)
        return Posterior(
            self.mu(x),
            self.logvar(x).float().clamp(*self.logvar_bounds),
            (h, w),
            tuple(padded),
        )


class SpatialDecoder(nn.Module):
    def __init__(self, channels, latent_channels, pre, post, variant):
        super().__init__()
        self.factor, self.latent_channels = 2 ** len(channels), latent_channels
        self.input = nn.Conv2d(latent_channels, channels[-1], 1)
        stages = [
            DecoderStage(co, ci, a, b, variant == "reversible")
            for ci, co, a, b in zip((3, *channels[:-1]), channels, pre, post)
        ]
        self.stages = nn.ModuleList(reversed(stages))

    def forward(self, z, output_size):
        if len(output_size) != 2 or any(
            type(i) is not int or i < 1 for i in output_size
        ):
            raise ValueError("Positive integer output height and width required")
        if (
            z.ndim != 4
            or z.shape[1] != self.latent_channels
            or tuple((i + self.factor - 1) // self.factor for i in output_size)
            != tuple(z.shape[-2:])
        ):
            raise ValueError("Latent geometry does not match requested output size")
        x = self.input(z)
        for stage in self.stages:
            x = stage(x)
        return x[..., : output_size[0], : output_size[1]]


class SpatialVAE(nn.Module):
    def __init__(
        self,
        channels=(32, 64, 128),
        latent_channels=8,
        pre_depth=0,
        post_depth=1,
        variant="base",
        logvar_bounds=(-12.0, 8.0),
    ):
        super().__init__()
        if (
            not channels
            or any(type(c) is not int or c < 2 for c in channels)
            or type(latent_channels) is not int
            or latent_channels < 1
            or variant not in ("base", "attention", "reversible")
        ):
            raise ValueError("Invalid VAE architecture")
        if (
            len(logvar_bounds) != 2
            or not all(torch.isfinite(torch.tensor(v)) for v in logvar_bounds)
            or logvar_bounds[0] >= logvar_bounds[1]
        ):
            raise ValueError("Invalid log variance bounds")

        def depths(x):
            values = [x] * len(channels) if isinstance(x, int) else list(x)
            if len(values) != len(channels) or any(
                type(n) is not int or n < 0 for n in values
            ):
                raise ValueError("Invalid per-scale depths")
            return values

        pre, post = depths(pre_depth), depths(post_depth)
        self.config = dict(
            channels=list(channels),
            latent_channels=latent_channels,
            pre_depth=pre,
            post_depth=post,
            variant=variant,
            logvar_bounds=list(logvar_bounds),
        )
        self.encoder = SpatialEncoder(
            channels, latent_channels, pre, post, variant, logvar_bounds
        )
        self.decoder = SpatialDecoder(channels, latent_channels, pre, post, variant)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z, output_size):
        return self.decoder(z, output_size)

    def forward(self, x, *, sample=True, generator=None):
        p = self.encode(x)
        return self.decode(p.sample(generator) if sample else p.mu, p.original_size), p

    def save(self, path):
        torch.save(
            dict(
                schema="pathwm-spatial-vae-v1",
                config=self.config,
                model=self.state_dict(),
            ),
            Path(path),
        )

    @classmethod
    def load(cls, path, device="cpu"):
        p = torch.load(path, map_location="cpu", weights_only=True)
        if p["schema"] == "pathwm-spatial-vae-v2":
            from .spatial_vae_v2 import HierarchicalVAE

            m = HierarchicalVAE(**p["config"]).to(device)
            m.load_state_dict(p["model"], strict=True)
            return m
        if p["schema"] != "pathwm-spatial-vae-v1":
            raise ValueError("Wrong VAE schema")
        m = cls(**p["config"]).to(device)
        m.load_state_dict(p["model"], strict=True)
        return m


def vae_loss(reconstruction, target, posterior, beta=1.0, variance=0.5):
    if (
        reconstruction.shape != target.shape
        or tuple(target.shape[-2:]) != posterior.original_size
        or beta < 0
        or variance <= 0
    ):
        raise ValueError("Incompatible reconstruction or loss settings")
    area = target.shape[-2] * target.shape[-1]
    distortion = (reconstruction.float() - target.float()).square().sum((1, 2, 3)) / (
        2 * variance * area
    )
    rate = posterior.kl_per_image() / area
    return (distortion + beta * rate).mean(), dict(
        distortion=distortion.mean(),
        rate=rate.mean(),
        kl_nats_per_sample=posterior.kl_per_image().mean(),
        kl_bits_per_sample=posterior.kl_per_image().mean() / 0.6931471805599453,
        kl_bits_per_latent_position=posterior.kl_per_image().mean()
        / (0.6931471805599453 * posterior.mu.shape[-2] * posterior.mu.shape[-1]),
        kl_bits_per_original_pixel=rate.mean() / 0.6931471805599453,
    )


def build_variant(variant, seed, **kwargs):
    """Match common weights and initial mapping, not different hidden mixer weights."""
    with torch.random.fork_rng(devices=[]):
        # These modules initialize on CPU. Do not reset the CUDA posterior-noise
        # stream: differing numbers of model constructors must not change it.
        torch.random.default_generator.manual_seed(seed)
        anchor = SpatialVAE(variant="base", **kwargs)
        torch.random.default_generator.manual_seed(seed + 1)
        attention = SpatialVAE(variant="attention", **kwargs)
        state = attention.state_dict()
        for name, v in anchor.state_dict().items():
            state[name] = v
        attention.load_state_dict(state, strict=True)
        if variant == "base":
            return anchor
        if variant == "attention":
            return attention
        if variant != "reversible":
            raise ValueError("Unknown variant")
        torch.random.default_generator.manual_seed(seed + 2)
        model = SpatialVAE(variant="reversible", **kwargs)
        state = model.state_dict()
        for name, v in attention.state_dict().items():
            if name in state and state[name].shape == v.shape:
                state[name] = v
        model.load_state_dict(state, strict=True)
        return model
