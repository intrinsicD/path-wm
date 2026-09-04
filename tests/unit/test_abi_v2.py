"""ABI-v2 layout and variable token-stream checks for the common multimodal base."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from world_state.abi_v2 import ABIv2Error, load_abi_v2

ROOT = Path(__file__).resolve().parents[2]


def test_v2_is_modality_neutral_and_keeps_variable_evidence_outside_state():
    abi = load_abi_v2(ROOT / "docs/abi/abi_v2.yaml")

    assert abi.version == 2
    assert abi.n_tokens == 65
    assert abi.token_order == ("global", "slots")
    assert abi.state_positions == "learned_slot_id"
    assert abi.evidence_dim == abi.dim == abi.action_dim == 192

    W = torch.zeros(2, abi.n_tokens, abi.dim, dtype=abi.dtype)
    abi.check_state(W)
    for token_count in (1, 17, 73):
        tokens = torch.zeros(2, token_count, abi.evidence_dim, dtype=abi.evidence_dtype)
        timestamps = torch.zeros(2, token_count)
        mask = torch.ones(2, token_count, dtype=torch.bool)
        abi.check_evidence(tokens, timestamps, mask)


def test_v2_rejects_bad_masks_times_and_oversized_action_streams():
    abi = load_abi_v2(ROOT / "docs/abi/abi_v2.yaml")
    tokens = torch.zeros(2, 3, abi.evidence_dim, dtype=abi.evidence_dtype)

    with pytest.raises(ABIv2Error, match="valid_mask"):
        abi.check_evidence(tokens, torch.zeros(2, 3), torch.ones(2, 3))
    with pytest.raises(ABIv2Error, match="timestamps"):
        abi.check_evidence(tokens, torch.zeros(2, 2), torch.ones(2, 3, dtype=torch.bool))

    count = abi.max_action_tokens + 1
    actions = torch.zeros(1, count, abi.action_dim, dtype=abi.dtype)
    times = torch.zeros(1, count)
    valid = torch.ones(1, count, dtype=torch.bool)
    with pytest.raises(ABIv2Error, match="exceeds"):
        abi.check_action_tokens(actions, times, valid, valid)
