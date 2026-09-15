"""Small trainable modality adapters. Times denote when a whole item is available.

Video frames and waveform chunks are encoded independently: an observation cutoff
can never admit a feature computed from a later frame/chunk. Raw bytes are a tiny
language interface, not a pretrained language model or speech codec.
"""

from dataclasses import dataclass, replace
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class Observation:
    values: torch.Tensor
    times: torch.Tensor  # [B,T], availability time, in seconds
    valid: torch.Tensor | None = None  # [B,T]; False means entirely unavailable
    provenance: object | None = None  # generated/derived content must retain its tag

    def derive(self, values, *, times=None, valid=None):
        """Retain generated ancestry when a caller transforms an input."""
        return replace(
            self,
            values=values,
            times=self.times if times is None else times,
            valid=self.valid if valid is None else valid,
            provenance=None if self.provenance is None else self.provenance.derived(),
        )


@dataclass(frozen=True)
class TokenBatch:
    values: torch.Tensor  # [B,N,D]
    times: torch.Tensor  # [B,N]
    valid: torch.Tensor  # [B,N]


def position(value, width):
    """Unbounded continuous coordinates; no fixed maximum sequence table."""
    count = (width + 1) // 2
    frequencies = torch.exp(
        torch.arange(count, device=value.device, dtype=value.dtype)
        * (-math.log(10000) / max(count - 1, 1))
    )
    phase = value.unsqueeze(-1) * frequencies
    return torch.cat((phase.sin(), phase.cos()), -1)[..., :width]


def observation_values(observation):
    x, times = observation.values, observation.times
    if x.ndim < 2 or times.shape != x.shape[:2] or times.device != x.device:
        raise ValueError("Observation times must match [B,T] and the values device")
    if not times.is_floating_point():
        raise ValueError("Observation times must be floating point seconds")
    valid = observation.valid
    if valid is None:
        valid = torch.ones_like(times, dtype=torch.bool)
    if (
        valid.shape != times.shape
        or valid.dtype != torch.bool
        or valid.device != x.device
    ):
        raise ValueError(
            "Observation validity must be boolean [B,T] on the values device"
        )
    if not torch.isfinite(times[valid]).all() or not torch.isfinite(x[valid]).all():
        raise ValueError("Valid observations and times must be finite")
    # Remove masked NaNs before projection/attention, including padded text IDs.
    shape = (*valid.shape, *((1,) * (x.ndim - 2)))
    return x.masked_fill(~valid.reshape(shape), 0), times.masked_fill(~valid, 0), valid


class Attend(nn.Module):
    """Attention plus residual MLP. Optional traces are detached CPU measurements."""

    def __init__(self, width, heads=4):
        super().__init__()
        if width < 4 or width % heads:
            raise ValueError("Attention width must be divisible by its head count")
        self.query_norm, self.key_norm = nn.LayerNorm(width), nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(
            width, heads, batch_first=True, dropout=0.0
        )
        self.mlp = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width * 2),
            nn.GELU(),
            nn.Linear(width * 2, width),
        )

    def forward(
        self, query, context, valid=None, causal=False, trace=None, name="attention"
    ):
        if valid is not None:
            if (
                valid.shape != context.shape[:2]
                or valid.dtype != torch.bool
                or valid.device != context.device
            ):
                raise ValueError(
                    "Attention validity must be boolean [B,N] on the context device"
                )
            if not valid.any(1).all():
                raise ValueError("Attention needs at least one valid key per sample")
            # A key-padding mask alone cannot stop NaNs from entering projections
            # or gradients. Remove invalid payload before normalization/mixing.
            context = context.masked_fill(~valid[..., None], 0)
        mask = None
        if causal:
            mask = torch.ones(
                query.shape[1], context.shape[1], device=query.device, dtype=torch.bool
            ).triu(1)
        key = self.key_norm(context)
        normalized_query = self.query_norm(query)
        read, _ = self.attention(
            normalized_query,
            key,
            key,
            key_padding_mask=None if valid is None else ~valid,
            attn_mask=mask,
            need_weights=False,
            average_attn_weights=False,
        )
        out = query + read
        out = out + self.mlp(out)
        if trace is not None:
            # Requesting weights changes PyTorch's attention kernel and floating
            # point results. Keep the native output/gradient path untouched and
            # compute diagnostic pre-dropout probabilities separately, without RNG.
            with torch.no_grad():
                weights = self.attention.in_proj_weight.chunk(3)
                biases = (
                    self.attention.in_proj_bias.chunk(3)
                    if self.attention.in_proj_bias is not None
                    else (None,) * 3
                )
                b, qn, width = normalized_query.shape
                heads = self.attention.num_heads
                depth = width // heads
                q = (
                    F.linear(normalized_query, weights[0], biases[0])
                    .reshape(b, qn, heads, depth)
                    .transpose(1, 2)
                )
                k = (
                    F.linear(key, weights[1], biases[1])
                    .reshape(b, key.shape[1], heads, depth)
                    .transpose(1, 2)
                )
                logits = (q @ k.transpose(-1, -2)) / math.sqrt(depth)
                if valid is not None:
                    logits = logits.masked_fill(~valid[:, None, None, :], -torch.inf)
                if mask is not None:
                    logits = logits.masked_fill(mask[None, None], -torch.inf)
                trace[name] = logits.softmax(-1).nan_to_num().cpu().clone()
        return out


class ImageEncoder(nn.Module):
    def __init__(self, width, patch_size=4):
        super().__init__()
        self.width, self.patch_size = width, patch_size
        self.patch = nn.Conv2d(3, width, patch_size, stride=patch_size)
        self.modality = nn.Parameter(torch.randn(width) * 0.02)

    def forward(self, observation):
        x, times, valid = observation_values(observation)
        if x.ndim != 5 or x.shape[2] != 3 or min(x.shape[-2:]) < self.patch_size:
            raise ValueError("Images/video must be [B,T,3,H,W], at least one patch")
        if x.shape[-2] % self.patch_size or x.shape[-1] % self.patch_size:
            raise ValueError("Image dimensions must be divisible by patch size")
        b, t = x.shape[:2]
        y = self.patch(x.flatten(0, 1))
        h, w = y.shape[-2:]
        row, col = torch.meshgrid(
            torch.linspace(-1, 1, h, device=x.device, dtype=x.dtype),
            torch.linspace(-1, 1, w, device=x.device, dtype=x.dtype),
            indexing="ij",
        )
        coords = torch.cat(
            (
                position(row.flatten(), self.width // 2),
                position(col.flatten(), self.width - self.width // 2),
            ),
            -1,
        )
        y = y.flatten(2).transpose(1, 2).reshape(b, t, h * w, self.width)
        y = y + coords + self.modality
        return TokenBatch(
            y.flatten(1, 2),
            times.repeat_interleave(h * w, 1),
            valid.repeat_interleave(h * w, 1),
        )


class VectorEncoder(nn.Module):
    """Also serves as a minimal raw-waveform chunk encoder."""

    def __init__(self, input_width, width):
        super().__init__()
        self.width, self.input_width = width, input_width
        self.projection = nn.Sequential(
            nn.Linear(input_width, width), nn.GELU(), nn.Linear(width, width)
        )
        self.modality = nn.Parameter(torch.randn(width) * 0.02)

    def forward(self, observation):
        x, times, valid = observation_values(observation)
        if x.ndim != 3 or x.shape[-1] != self.input_width:
            raise ValueError(f"Vector/audio values must be [B,T,{self.input_width}]")
        return TokenBatch(self.projection(x) + self.modality, times, valid)


class TextEncoder(nn.Module):
    def __init__(self, width, vocabulary=259):
        super().__init__()
        self.width, self.vocabulary = width, vocabulary
        self.embedding = nn.Embedding(vocabulary, width, padding_idx=0)
        self.modality = nn.Parameter(torch.randn(width) * 0.02)

    def forward(self, observation):
        x, times, valid = observation_values(observation)
        if (
            x.ndim != 2
            or x.dtype != torch.long
            or ((x < 0) | (x >= self.vocabulary)).any()
        ):
            raise ValueError("Text must contain valid int64 vocabulary IDs [B,T]")
        pos = position(
            torch.arange(
                x.shape[1], device=x.device, dtype=self.embedding.weight.dtype
            ),
            self.width,
        )
        return TokenBatch(
            self.embedding(x) + pos + self.modality, times, valid & (x != 0)
        )


def bytes_batch(strings, device="cpu"):
    sequences = [[1, *(b + 3 for b in s.encode("utf-8")), 2] for s in strings]
    if not sequences:
        raise ValueError("Text batch must be nonempty")
    result = torch.zeros(
        len(sequences), max(map(len, sequences)), dtype=torch.long, device=device
    )
    for i, row in enumerate(sequences):
        result[i, : len(row)] = torch.tensor(row, device=device)
    return result, result != 0


def bytes_text(ids):
    result = []
    for value in ids:
        value = int(value)
        if value == 2:
            break
        if 3 <= value < 259:
            result.append(value - 3)
    return bytes(result).decode("utf-8", errors="replace")


class ImageDecoder(nn.Module):
    def __init__(self, width, image_size=16, patch_size=4):
        super().__init__()
        if image_size % patch_size:
            raise ValueError("Output image size must divide into patches")
        self.image_size, self.patch_size = image_size, patch_size
        side = image_size // patch_size
        self.queries = nn.Parameter(torch.randn(side * side, width) * 0.02)
        self.read = Attend(width)
        self.output = nn.Linear(width, 3 * patch_size * patch_size)

    def forward(self, tokens, trace=None, *, valid=None):
        b, p, side = len(tokens), self.patch_size, self.image_size // self.patch_size
        x = self.read(
            self.queries.expand(b, -1, -1),
            tokens,
            valid=valid,
            trace=trace,
            name="decode.image.attention",
        )
        patches = self.output(x).sigmoid().reshape(b, side, side, 3, p, p)
        return patches.permute(0, 3, 1, 4, 2, 5).reshape(
            b, 3, self.image_size, self.image_size
        )


class AudioDecoder(nn.Module):
    def __init__(self, width, samples=32):
        super().__init__()
        self.samples = samples
        self.queries = nn.Parameter(torch.randn(4, width) * 0.02)
        self.read = Attend(width)
        self.output = nn.Linear(4 * width, samples)

    def forward(self, tokens, trace=None, *, valid=None):
        x = self.read(
            self.queries.expand(len(tokens), -1, -1),
            tokens,
            valid=valid,
            trace=trace,
            name="decode.audio.attention",
        )
        return self.output(x.flatten(1)).tanh()


class TextDecoder(nn.Module):
    def __init__(self, width, vocabulary=259):
        super().__init__()
        self.width, self.vocabulary = width, vocabulary
        self.embedding = nn.Embedding(vocabulary, width, padding_idx=0)
        self.self_attention, self.read = Attend(width), Attend(width)
        self.output = nn.Linear(width, vocabulary)

    def forward(self, tokens, prefix, trace=None, *, valid=None):
        if prefix.ndim != 2 or len(prefix) != len(tokens) or prefix.shape[1] < 1:
            raise ValueError("Text prefix must be nonempty [B,L]")
        if (
            prefix.dtype != torch.long
            or ((prefix < 0) | (prefix >= self.vocabulary)).any()
        ):
            raise ValueError("Text prefix contains invalid vocabulary IDs")
        prefix_valid = prefix != 0
        if (
            not prefix_valid[:, 0].all()
            or ((~prefix_valid[:, :-1]) & prefix_valid[:, 1:]).any()
        ):
            raise ValueError("Text prefixes must be nonempty and right padded")
        x = self.embedding(prefix) + position(
            torch.arange(prefix.shape[1], device=tokens.device, dtype=tokens.dtype),
            self.width,
        )
        x = self.self_attention(
            x,
            x,
            valid=prefix_valid,
            causal=True,
            trace=trace,
            name="decode.text.causal_attention",
        )
        x = self.read(
            x, tokens, valid=valid, trace=trace, name="decode.text.state_attention"
        )
        return self.output(x)

    @torch.no_grad()
    def generate(self, tokens, max_tokens=32, *, valid=None):
        if max_tokens < 1:
            raise ValueError("Text generation budget must be positive")
        prefix = torch.ones(len(tokens), 1, dtype=torch.long, device=tokens.device)
        ended = torch.zeros(len(tokens), dtype=torch.bool, device=tokens.device)
        for _ in range(max_tokens):
            logits = self(tokens, prefix, valid=valid)[:, -1].clone()
            logits[:, :2] = -torch.inf  # PAD/BOS are not generated as content.
            next_id = logits.argmax(-1).masked_fill(ended, 2)
            prefix = torch.cat((prefix, next_id[:, None]), 1)
            ended |= next_id == 2
            if ended.all():
                break
        return prefix
