"""Function-preserving per-grid encoder extensions for a matched continuation.

Both operators see the same existing positional signal before the original
cross-scale exchange. Zero-gated additions preserve the task-trained reference
exactly initially; this compares added packages, not equal parameter counts.
"""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F
from .encoder_variants import ExperimentalEncoder
from world_model.paddle.types import ObservationLatent


class GatedConvolution(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(64)
        self.conv1 = nn.Conv2d(64, 64, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1, bias=False)
        self.gate = nn.Parameter(torch.zeros(()))

    def forward(self, x):
        side = int(x.shape[1] ** .5)
        y = self.norm(x).transpose(1, 2).reshape(-1, 64, side, side)
        y = self.conv2(F.gelu(self.conv1(y))).flatten(2).transpose(1, 2)
        return x + self.gate * y


class GatedTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm_attention = nn.LayerNorm(64)
        self.attention = nn.MultiheadAttention(64, 4, dropout=0., batch_first=True)
        self.norm_mlp = nn.LayerNorm(64)
        self.mlp = nn.Sequential(nn.Linear(64, 256), nn.GELU(), nn.Linear(256, 64))
        self.gate_attention = nn.Parameter(torch.zeros(()))
        self.gate_mlp = nn.Parameter(torch.zeros(()))

    def forward(self, x):
        normalized = self.norm_attention(x)
        attended, _ = self.attention(normalized, normalized, normalized, need_weights=False)
        x = x + self.gate_attention * attended
        return x + self.gate_mlp * self.mlp(self.norm_mlp(x))


class ContinuationEncoder(nn.Module):
    def __init__(self, kind, seed, source_state):
        super().__init__()
        if kind not in ('joint', 'conv', 'transformer'): raise ValueError('extension kind')
        self.kind = kind
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed + 670041)
            self.base = ExperimentalEncoder(depth=2, exchange=True)
            self.base.load_state_dict(source_state)
            block = GatedConvolution if kind == 'conv' else GatedTransformer
            count = 0 if kind == 'joint' else 2
            self.extensions = nn.ModuleDict({name: nn.Sequential(*[block() for _ in range(count)])
                                             for name in ('fine', 'coarse')})

    def forward(self, rgb):
        if rgb.ndim != 4 or rgb.shape[1:] != (3, 64, 64): raise ValueError('RGB64 input required')
        b = self.base
        mid = F.gelu(b.conv2(F.gelu(b.conv1(rgb))))
        coarse = b.coarse_blocks(F.gelu(b.conv3(mid))).flatten(2).transpose(1, 2)
        fine = b.fine_blocks(b.fine_projection(mid)).flatten(2).transpose(1, 2)
        fine = fine + b._positions(b.fine_row, b.fine_column) + b.scale_embeddings[0]
        coarse = coarse + b._positions(b.coarse_row, b.coarse_column) + b.scale_embeddings[1]
        fine, coarse = self.extensions['fine'](fine), self.extensions['coarse'](coarse)
        nf, nc = b.fine_attention_norm(fine), b.coarse_attention_norm(coarse)
        fine, coarse = fine + b.fine_from_coarse(nf, nc, nc), coarse + b.coarse_from_fine(nc, nf, nf)
        return ObservationLatent(fine + b.fine_mlp(b.fine_mlp_norm(fine)),
                                 coarse + b.coarse_mlp(b.coarse_mlp_norm(coarse)))


def encoder_optimizer_groups(encoder):
    no_decay = {id(p) for module in encoder.modules() if isinstance(module, (nn.LayerNorm, nn.GroupNorm))
                for p in module.parameters(recurse=False)}
    groups = {}
    for name, parameter in encoder.named_parameters():
        lr = 3e-4 if name.startswith('extensions.') else 3e-5
        decay = 0. if '.gate' in name or id(parameter) in no_decay else 1e-4
        groups.setdefault((lr, decay), []).append(parameter)
    return [dict(params=parameters, lr=lr, weight_decay=decay) for (lr, decay), parameters in groups.items()]
