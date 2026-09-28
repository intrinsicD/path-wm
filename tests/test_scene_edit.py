"""Memory-conditioned residual editing of native fine codes: software contracts.

Actual native shapes (64-wide 16x16 fine codes, full SlotPerception). The store test
uses the real UnifiedAgent/WorldStore/ExactRetriever with the qualified J6000
perception; the request binding there is supplied by the caller, not learned.
"""
from pathlib import Path

import pytest
import torch
from torch import nn

from pathwm.data import rule_world as rw
from pathwm.models.features import FeatureSpec
from pathwm.models.slots import SlotPerception

J6000 = Path("runs/real_visual_joint_repair_3501_u6000_v1")


def native():
    torch.manual_seed(5)
    return SlotPerception().eval().requires_grad_(False)


def test_reconstruct_bypass_never_calls_generator_or_rng():
    from pathwm.models.conditional_image import edit_code

    class Raising(nn.Module):
        def features(self, *a, **k):
            raise AssertionError("generator called during reconstruct")

    z = torch.randn(2, 256, 64)
    state = torch.get_rng_state().clone()
    out = edit_code("reconstruct", z, generator=Raising())
    assert out is z and torch.equal(state, torch.get_rng_state())
    with pytest.raises(ValueError):
        edit_code("create", z, generator=Raising())


def test_request_context_rejects_bad_inputs_and_null_request_removes_binding():
    from pathwm.models.conditional_image import ComponentRequest
    request = ComponentRequest(64)
    z = torch.randn(3, 256, 64)
    with pytest.raises(ValueError):
        request.context(z, torch.randn(3, 32), torch.zeros(3, dtype=torch.long))
    with pytest.raises(ValueError):
        request.context(z, torch.randn(3, 64), torch.full((3,), 2))
    with pytest.raises(ValueError):
        request.context(z, torch.full((3, 64), float("nan")), torch.zeros(3, dtype=torch.long))
    with pytest.raises(ValueError):
        request.context(z, torch.randn(3, 64), torch.zeros(3))  # float states
    a = request.context(z, torch.randn(3, 64), torch.tensor([0, 1, 0]), null=True)
    b = request.context(z, torch.randn(3, 64), torch.tensor([1, 0, 1]), null=True)
    assert torch.equal(a, b)  # null request carries neither binding nor state


