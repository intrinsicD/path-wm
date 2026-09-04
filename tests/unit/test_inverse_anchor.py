"""Behaviour coverage for the Step 4 tokenwise inverse-dynamics anchor.

What: the selected head must use correspondence between ABI token positions before pooling.
How: independently permuting the successor grid tokens preserves every global mean, so a head that
pre-pools W and W_next gives the same action; the tokenwise head must distinguish the two pairings.
Why: Step 3's pooled inverse head stayed at the zero-action solution and its averaged delta retained
only 4% of the full transition norm, preventing the mandatory E1 inverse loss from anchoring action
information into W (§6.1, §14 action-insensitivity diagnostic).
"""
from __future__ import annotations

import copy
import torch
import torch.nn.functional as F


def _state(abi, seed: int) -> torch.Tensor:
    values = torch.randn(3, abi.n_tokens, abi.dim, generator=torch.Generator().manual_seed(seed))
    return F.layer_norm(values, (abi.dim,)).to(abi.dtype)


def test_selected_inverse_uses_token_correspondence(build, cfg, abi):
    spec = copy.deepcopy(cfg)
    spec["inverse_dynamics"]["kind"] = "mlp_token_pair"
    inverse = build("inverse", cfg=spec)
    W, W_next = _state(abi, 2), _state(abi, 3)
    aligned = inverse.infer_action(W, W_next)

    permuted_next = W_next.clone()
    permuted_next[:, 1:] = W_next[:, 1:].roll(shifts=1, dims=1)
    misaligned = inverse.infer_action(W, permuted_next)

    assert not torch.allclose(aligned, misaligned, rtol=1e-4, atol=1e-5)
