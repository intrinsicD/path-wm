"""R/P/M/C codec. V2 is opt-in; old spatial VAE exports remain unchanged.

Only R is guaranteed reversible. P and M are shape-preserving, not necessarily
information-preserving. No encoder trace or feature bypass reaches the decoder.
"""

import math
import torch
from torch import nn
from torch.nn import functional as F

from .blocks import TransformerBlock
from .spatial_vae import SpatialVAE, Posterior


class LosslessDownsample2D(nn.Module):
    def forward(self, x):
        if x.ndim != 4 or x.shape[-1] % 2 or x.shape[-2] % 2:
            raise ValueError(
                "PixelUnshuffle requires an even BCHW grid; pad explicitly"
            )
        return F.pixel_unshuffle(x, 2)


class LosslessUpsample2D(nn.Module):
    def forward(self, x):
        if x.ndim != 4 or x.shape[1] % 4:
            raise ValueError("PixelShuffle requires channels divisible by four")
        return F.pixel_shuffle(x, 2)


class LocalProcessingBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.branch = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 1),
        )
        nn.init.zeros_(self.branch[-1].weight)
        nn.init.zeros_(self.branch[-1].bias)

    def forward(self, x):
        y = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return x + self.branch(y)


def local_processing(channels, depth):
    if type(depth) is not int or depth < 0:
        raise ValueError("Nonnegative processing depth required")
    return nn.Sequential(*(LocalProcessingBlock(channels) for _ in range(depth)))


class ChannelCompression(nn.Conv2d):
    def __init__(self, ci, co):
        if not 0 < co < ci:
            raise ValueError(
                "Compression must strictly reduce channels; use Identity otherwise"
            )
        super().__init__(ci, co, 1)


class ChannelExpansion(nn.Conv2d):
    def __init__(self, ci, co):
        if co <= ci:
            raise ValueError("Expansion must strictly increase channels")
        super().__init__(ci, co, 1)


class FourierPosition2D(nn.Module):
    """Two frequency bands at normalized cell centers; no learned spatial table."""

    def __init__(self, channels):
        super().__init__()
        self.projection = nn.Linear(8, channels, bias=False)

    def forward(self, x):
        h, w = x.shape[-2:]
        y = (torch.arange(h, device=x.device, dtype=x.dtype) + 0.5) / h
        xx = (torch.arange(w, device=x.device, dtype=x.dtype) + 0.5) / w
        yy, xx = torch.meshgrid(y, xx, indexing="ij")
        phase = (
            torch.stack([xx, yy], -1)[..., None]
            * x.new_tensor([1.0, 2.0])
            * (2 * math.pi)
        )
        features = torch.cat([phase.sin(), phase.cos()], -1).reshape(h * w, 8)
        return self.projection(features)[None]


class LoopTransformer(nn.Module):
    def __init__(self, channels, heads=4, iterations=1, max_tokens=1024):
        super().__init__()
        if type(iterations) is not int or iterations < 0 or max_tokens < 1:
            raise ValueError("Invalid loop/token budget")
        self.iterations, self.max_tokens = iterations, max_tokens
        self.position = FourierPosition2D(channels)
        self.block = TransformerBlock(channels, heads)

    def forward(self, x, return_states=False):
        if type(self.iterations) is not int or self.iterations < 0:
            raise ValueError("Nonnegative loop count required")
        states = []
        if self.iterations == 0:
            return (x, [x.detach().clone()]) if return_states else x
        if x.shape[-2] * x.shape[-1] > self.max_tokens:
            raise ValueError(
                f"Global attention token budget exceeded ({self.max_tokens})"
            )
        t = x.flatten(2).transpose(1, 2) + self.position(x)
        if return_states:
            states.append(t.transpose(1, 2).reshape_as(x).detach().clone())
        for _ in range(self.iterations):
            t = self.block(t)
            if return_states:
                states.append(t.transpose(1, 2).reshape_as(x).detach().clone())
        y = t.transpose(1, 2).reshape_as(x)
        return (y, states) if return_states else y


def stride2_from_projection(projection):
    """Exact weight bijection: unshuffle(channel,dy,dx) then 1x1 == 2x2/s2."""
    co, cr, _, _ = projection.weight.shape
    if cr % 4:
        raise ValueError("Projection must consume 4*C channels")
    conv = nn.Conv2d(cr // 4, co, 2, stride=2, bias=projection.bias is not None).to(
        projection.weight
    )
    with torch.no_grad():
        conv.weight.copy_(projection.weight.reshape(co, cr // 4, 2, 2))
        if projection.bias is not None:
            conv.bias.copy_(projection.bias)
    return conv


def snapshot(trace, name, x):
    if trace is not None:
        trace[name] = x.detach().clone()


class EncoderStage(nn.Module):
    def __init__(
        self, ci, co, *, depth=1, pre_depth=0, downsample=True, ablation="C", mixer=None
    ):
        super().__init__()
        self.factor = 2 if downsample else 1
        self.out_channels = ci * self.factor**2 if co is None else co
        if self.out_channels > ci * self.factor**2 or self.out_channels < 1:
            raise ValueError(
                "Stage C reduces channels or is Identity; incompatible widths"
            )
        self.pre_process = local_processing(ci, pre_depth)
        self.ablation = ablation
        self.entangled = ablation in ("A_exact", "A_local")
        if self.entangled:
            if not downsample or co is None:
                raise ValueError(
                    "Stride control requires downsampling and output width"
                )
            k = 2 if ablation == "A_exact" else 3
            self.downsample = nn.Conv2d(ci, co, k, stride=2, padding=k // 3)
            self.post_process = self.mixer = self.compress = self.after_process = (
                nn.Identity()
            )
        else:
            cr = ci * self.factor**2
            self.downsample = LosslessDownsample2D() if downsample else nn.Identity()
            self.post_process = local_processing(
                cr, depth if ablation in ("C", "D", "E") else 0
            )
            self.mixer = mixer if mixer is not None else nn.Identity()
            self.compress = (
                nn.Identity()
                if self.out_channels == cr
                else ChannelCompression(cr, self.out_channels)
            )
            self.after_process = local_processing(
                self.out_channels, depth if ablation == "C_after" else 0
            )

    def forward(self, x, trace=None, prefix="stage"):
        snapshot(trace, prefix + ".before_pre", x)
        x = self.pre_process(x)
        snapshot(trace, prefix + ".after_pre", x)
        if self.entangled:
            # Diagnostic equivalent input patch only, not a second executed path.
            if trace is not None:
                k = self.downsample.kernel_size[0]
                unfolded = F.unfold(x, k, padding=k // 3, stride=2)
                snapshot(
                    trace,
                    prefix + ".before_compression",
                    unfolded.reshape(len(x), -1, x.shape[-2] // 2, x.shape[-1] // 2),
                )
            x = self.downsample(x)
        else:
            x = self.downsample(x)
            snapshot(trace, prefix + ".before_post", x)
            x = self.post_process(x)
            snapshot(trace, prefix + ".before_mixer", x)
            if trace is not None and isinstance(self.mixer, LoopTransformer):
                x, states = self.mixer(x, return_states=True)
                for i, state in enumerate(states):
                    trace[f"{prefix}.loop_{i}"] = state
            else:
                x = self.mixer(x)
            snapshot(trace, prefix + ".before_compression", x)
            x = self.compress(x)
        snapshot(trace, prefix + ".after_compression", x)
        x = self.after_process(x)
        snapshot(trace, prefix + ".output", x)
        return x


class DecoderStage(nn.Module):
    def __init__(self, co, ci, factor=2, depth=1):
        super().__init__()
        cr = ci * factor**2
        self.expansion = ChannelExpansion(co, cr) if cr > co else nn.Identity()
        self.processing = local_processing(cr, depth)
        self.upsample = LosslessUpsample2D() if factor == 2 else nn.Identity()
        self.post_process = local_processing(ci, depth)

    def forward(self, x):
        return self.post_process(self.upsample(self.processing(self.expansion(x))))


class HierarchicalEncoder(nn.Module):
    def __init__(
        self,
        stem_channels,
        channels,
        latent_channels,
        ablation,
        depth,
        pre_depth,
        downsample,
        iterations,
        heads,
        max_tokens,
        logvar_bounds,
    ):
        super().__init__()
        self.factor = 2 ** sum(downsample)
        self.logvar_bounds = logvar_bounds
        self.stem = nn.Sequential(
            nn.Conv2d(3, stem_channels, 3, padding=1),
            LocalProcessingBlock(stem_channels),
        )
        self.stages = nn.ModuleList()
        ci = stem_channels
        self.widths = [ci]
        for i, (co, reduce) in enumerate(zip(channels, downsample)):
            mix = (
                LoopTransformer(
                    ci * (4 if reduce else 1), heads, iterations, max_tokens
                )
                if ablation in ("D", "E") and i == len(channels) - 1
                else None
            )
            stage = EncoderStage(
                ci,
                co,
                depth=depth,
                pre_depth=pre_depth,
                downsample=reduce,
                ablation=ablation,
                mixer=mix,
            )
            self.stages.append(stage)
            ci = stage.out_channels
            self.widths.append(ci)
        self.mu = nn.Conv2d(ci, latent_channels, 1)
        self.logvar = nn.Conv2d(ci, latent_channels, 1)

    def forward(self, x, trace=None):
        if (
            x.ndim != 4
            or x.shape[1] != 3
            or min(x.shape[0], *x.shape[-2:]) < 1
            or not x.is_floating_point()
        ):
            raise ValueError("Expected nonempty floating RGB BCHW")
        h, w = x.shape[-2:]
        x = F.pad(x, (0, -w % self.factor, 0, -h % self.factor), mode="replicate")
        padded = tuple(x.shape[-2:])
        snapshot(trace, "input.padded", x)
        x = self.stem(x)
        snapshot(trace, "stem.output", x)
        for i, stage in enumerate(self.stages):
            x = stage(x, trace, f"stage_{i}")
        snapshot(trace, "posterior.input", x)
        mu = self.mu(x)
        raw_logvar = self.logvar(x).float()
        logvar = raw_logvar.clamp(*self.logvar_bounds)
        for name, value in [("mu", mu), ("raw_logvar", raw_logvar), ("logvar", logvar)]:
            snapshot(trace, "posterior." + name, value)
        return Posterior(mu, logvar, (h, w), padded)


class HierarchicalDecoder(nn.Module):
    def __init__(self, widths, downsample, latent_channels, depth):
        super().__init__()
        self.factor, self.latent_channels = 2 ** sum(downsample), latent_channels
        self.input = nn.Conv2d(latent_channels, widths[-1], 1)
        self.stages = nn.ModuleList(
            reversed(
                [
                    DecoderStage(co, ci, 2 if reduce else 1, depth)
                    for ci, co, reduce in zip(widths[:-1], widths[1:], downsample)
                ]
            )
        )
        self.output = nn.Conv2d(widths[0], 3, 3, padding=1)

    def forward(self, z, output_size):
        if len(output_size) != 2 or any(
            type(i) is not int or i < 1 for i in output_size
        ):
            raise ValueError("Positive integer output dimensions required")
        if (
            z.ndim != 4
            or z.shape[1] != self.latent_channels
            or tuple((i + self.factor - 1) // self.factor for i in output_size)
            != tuple(z.shape[-2:])
        ):
            raise ValueError("Latent geometry does not match output")
        x = self.input(z)
        for stage in self.stages:
            x = stage(x)
        return self.output(x)[..., : output_size[0], : output_size[1]]


class HierarchicalVAE(SpatialVAE):
    def __init__(
        self,
        stem_channels=8,
        channels=(16, 24),
        latent_channels=4,
        ablation="C",
        depth=1,
        pre_depth=0,
        decoder_depth=1,
        downsample=None,
        loop_iterations=None,
        heads=4,
        max_tokens=1024,
        logvar_bounds=(-12.0, 8.0),
    ):
        nn.Module.__init__(self)
        if ablation not in ("A_exact", "A_local", "B", "C", "C_after", "D", "E"):
            raise ValueError("Unknown R/P/M/C ablation")
        if not channels or stem_channels < 1 or latent_channels < 1:
            raise ValueError("Positive widths and at least one stage required")
        if (
            len(logvar_bounds) != 2
            or not all(math.isfinite(v) for v in logvar_bounds)
            or logvar_bounds[0] >= logvar_bounds[1]
        ):
            raise ValueError("Invalid log variance bounds")
        downsample = [True] * len(channels) if downsample is None else list(downsample)
        if len(downsample) != len(channels) or any(
            type(v) is not bool for v in downsample
        ):
            raise ValueError("One boolean downsample flag per stage required")
        iterations = (
            (2 if ablation == "E" else 1)
            if loop_iterations is None
            else loop_iterations
        )
        self.config = dict(
            stem_channels=stem_channels,
            channels=list(channels),
            latent_channels=latent_channels,
            ablation=ablation,
            depth=depth,
            pre_depth=pre_depth,
            decoder_depth=decoder_depth,
            downsample=downsample,
            loop_iterations=iterations,
            heads=heads,
            max_tokens=max_tokens,
            logvar_bounds=list(logvar_bounds),
        )
        self.encoder = HierarchicalEncoder(
            stem_channels,
            channels,
            latent_channels,
            ablation,
            depth,
            pre_depth,
            downsample,
            iterations,
            heads,
            max_tokens,
            logvar_bounds,
        )
        self.decoder = HierarchicalDecoder(
            self.encoder.widths, downsample, latent_channels, decoder_depth
        )

    def inspect(self, x):
        """Opt-in detached snapshots; posterior still has the usual gradient path."""
        trace = {}
        p = self.encoder(x, trace)
        return p, trace

    def save(self, path):
        torch.save(
            dict(
                schema="pathwm-spatial-vae-v2",
                config=self.config,
                model=self.state_dict(),
            ),
            path,
        )


def build_hierarchy(ablation, seed, **kwargs):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        anchor = HierarchicalVAE(ablation="C", **kwargs)
        torch.random.default_generator.manual_seed(seed + 1)
        model = HierarchicalVAE(ablation=ablation, **kwargs)
        values = model.state_dict()
        for key, value in anchor.state_dict().items():
            if key in values and values[key].shape == value.shape:
                values[key] = value
        model.load_state_dict(values)
        if ablation == "A_exact":
            for src, dst in zip(anchor.encoder.stages, model.encoder.stages):
                if isinstance(src.compress, nn.Identity):
                    raise ValueError(
                        "A_exact initialization control requires an explicit learned projection at each stage"
                    )
                dst.downsample = stride2_from_projection(src.compress)
        return model
