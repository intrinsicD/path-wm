"""Exact E/U/P/D/H/R architecture from the version 1 implementation brief.

Persistent activations belong to callers. In particular, freezing U's parameters
does not remove its derivatives with respect to imagined observations and memory.
"""

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .types import ObservationLatent


class Attention(nn.Module):
    """Four-head attention with each Q/K/V/output projection applied once."""

    def __init__(self, query_width: int = 64):
        super().__init__()
        self.query_projection = nn.Linear(query_width, 64)
        self.key_projection = nn.Linear(64, 64)
        self.value_projection = nn.Linear(64, 64)
        self.output_projection = nn.Linear(64, 64)

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        def heads(x):
            return x.reshape(x.shape[0], x.shape[1], 4, 16).transpose(1, 2)

        q = heads(self.query_projection(query))
        k = heads(self.key_projection(key))
        v = heads(self.value_projection(value))
        result = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        result = result.transpose(1, 2).reshape(query.shape[0], query.shape[1], 64)
        return self.output_projection(result)


def _mlp() -> nn.Sequential:
    return nn.Sequential(nn.Linear(64, 128), nn.GELU(), nn.Linear(128, 64))


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.fine_projection = nn.Conv2d(32, 64, 1)
        self.fine_row = nn.Parameter(torch.empty(16, 32))
        self.fine_column = nn.Parameter(torch.empty(16, 32))
        self.coarse_row = nn.Parameter(torch.empty(8, 32))
        self.coarse_column = nn.Parameter(torch.empty(8, 32))
        self.scale_embeddings = nn.Parameter(torch.empty(2, 64))
        for parameter in (self.fine_row, self.fine_column, self.coarse_row, self.coarse_column, self.scale_embeddings):
            nn.init.normal_(parameter, std=.02)
        self.fine_attention_norm = nn.LayerNorm(64, eps=1e-5)
        self.coarse_attention_norm = nn.LayerNorm(64, eps=1e-5)
        self.fine_from_coarse = Attention()
        self.coarse_from_fine = Attention()
        self.fine_mlp_norm = nn.LayerNorm(64, eps=1e-5)
        self.coarse_mlp_norm = nn.LayerNorm(64, eps=1e-5)
        self.fine_mlp = _mlp()
        self.coarse_mlp = _mlp()

    @staticmethod
    def _positions(rows: Tensor, columns: Tensor) -> Tensor:
        width = rows.shape[0]
        return torch.cat((rows[:, None].expand(width, width, 32), columns[None].expand(width, width, 32)), dim=-1).reshape(width * width, 64)

    def forward(self, rgb: Tensor) -> ObservationLatent:
        if rgb.ndim != 4 or rgb.shape[1:] != (3, 64, 64):
            raise ValueError("Encoder expects RGB [B,3,64,64]")
        x = F.gelu(self.conv1(rgb))
        mid = F.gelu(self.conv2(x))
        coarse = F.gelu(self.conv3(mid)).flatten(2).transpose(1, 2)
        fine = self.fine_projection(mid).flatten(2).transpose(1, 2)
        fine = fine + self._positions(self.fine_row, self.fine_column) + self.scale_embeddings[0]
        coarse = coarse + self._positions(self.coarse_row, self.coarse_column) + self.scale_embeddings[1]
        # Both directions read the incoming scales, never the other direction's update.
        nf, nc = self.fine_attention_norm(fine), self.coarse_attention_norm(coarse)
        fine1 = fine + self.fine_from_coarse(nf, nc, nc)
        coarse1 = coarse + self.coarse_from_fine(nc, nf, nf)
        return ObservationLatent(
            fine1 + self.fine_mlp(self.fine_mlp_norm(fine1)),
            coarse1 + self.coarse_mlp(self.coarse_mlp_norm(coarse1)),
        )


class MemoryUpdater(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = Attention(query_width=128)
        self.gru = nn.GRUCell(input_size=67, hidden_size=128)

    def forward(self, memory_before: Tensor, observation: ObservationLatent, previous_action_onehot: Tensor) -> Tensor:
        tokens = observation.tokens()
        read = self.attention(memory_before.unsqueeze(1), tokens, tokens).squeeze(1)
        return self.gru(torch.cat((read, previous_action_onehot), dim=-1), memory_before)


class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention_norm = nn.LayerNorm(64, eps=1e-5)
        self.attention = Attention()
        self.mlp_norm = nn.LayerNorm(64, eps=1e-5)
        self.mlp = _mlp()

    def forward(self, tokens: Tensor) -> Tensor:
        normalized = self.attention_norm(tokens)
        tokens = tokens + self.attention(normalized, normalized, normalized)
        return tokens + self.mlp(self.mlp_norm(tokens))


class Predictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.action_projection = nn.Linear(3, 64)
        self.role_embeddings = nn.Parameter(torch.empty(3, 64))
        nn.init.normal_(self.role_embeddings, std=.02)
        self.blocks = nn.ModuleList([TransformerBlock(), TransformerBlock()])
        self.output_projection = nn.Linear(64, 64)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, observation: ObservationLatent, memory_after: Tensor, candidate_action_onehot: Tensor) -> ObservationLatent:
        original = observation.tokens()
        memory_tokens = memory_after.reshape(memory_after.shape[0], 2, 64) + self.role_embeddings[:2]
        action_token = self.action_projection(candidate_action_onehot).unsqueeze(1) + self.role_embeddings[2]
        tokens = torch.cat((original, memory_tokens, action_token), dim=1)
        for block in self.blocks:
            tokens = block(tokens)
        return ObservationLatent.from_tokens(original + self.output_projection(tokens[:, :320]))


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 16, 3, padding=1)
        self.output_projection = nn.Conv2d(16, 3, 1)

    def forward(self, observation: ObservationLatent) -> Tensor:
        batch = observation.fine.shape[0]
        fine = observation.fine.transpose(1, 2).reshape(batch, 64, 16, 16)
        coarse = observation.coarse.transpose(1, 2).reshape(batch, 64, 8, 8)
        x = torch.cat((fine, F.interpolate(coarse, size=(16, 16), mode="nearest")), dim=1)
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(F.interpolate(x, scale_factor=2, mode="nearest")))
        x = F.gelu(self.conv3(F.interpolate(x, scale_factor=2, mode="nearest")))
        return self.output_projection(x).sigmoid()


class PositionReadout(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(20480, 3)

    def forward(self, observation: ObservationLatent) -> Tensor:
        return self.linear(observation.tokens().flatten(1))


class StateReadout(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(128, 5)

    def forward(self, memory_after: Tensor) -> Tensor:
        return self.linear(memory_after)
