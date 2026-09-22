"""Fixed learned observation queries; lossy access, not a detail-memory store."""

import torch
from torch import nn

from .modalities import Attend


class LatentResampler(nn.Module):
    def __init__(self, width, tokens=64, heads=4):
        super().__init__()
        if not isinstance(tokens, int) or tokens < 1:
            raise ValueError("Observation token budget must be a positive integer")
        self.width, self.tokens = width, tokens
        self.queries = nn.Parameter(torch.randn(tokens, width) * 0.02)
        self.read = Attend(width, heads=heads)

    def forward(self, values, valid=None):
        """Read [B,N,D] once into [B,K,D]; absent members return exact zeros.

        Caller retains ownership of the unchanged detailed source. These outputs
        alone do not preserve all source information or provide later retrieval.
        """
        if values.ndim != 3 or values.shape[-1] != self.width:
            raise ValueError("Detail must be [B,N,width]")
        if valid is None:
            valid = torch.ones(values.shape[:2], dtype=torch.bool, device=values.device)
        if (
            valid.shape != values.shape[:2]
            or valid.dtype != torch.bool
            or valid.device != values.device
        ):
            raise ValueError("Detail validity must be boolean [B,N] on the same device")
        present = valid.any(1)
        # Null source keeps fully empty members finite and has no source gradient.
        source = torch.cat(
            (
                values.masked_fill(~valid[..., None], 0),
                values.new_zeros(len(values), 1, self.width),
            ),
            1,
        )
        mask = torch.cat((valid, ~present[:, None]), 1)
        result = self.read(
            self.queries[None].expand(len(values), -1, -1), source, valid=mask
        )
        return result.masked_fill(~present[:, None, None], 0)
