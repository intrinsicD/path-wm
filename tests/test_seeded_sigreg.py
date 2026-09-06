"""Private sketch numerical/recovery contract, with Claude's RNG test design."""
import copy
import io
from unittest.mock import patch
import pytest
import torch
from third_party.lewm.module import SIGReg
from world_model.seeded_sigreg import SeededSIGReg

@pytest.mark.parametrize('autocast', [False, True])
def test_pinned_loss_and_gradient_parity(autocast):
    g = torch.Generator().manual_seed(7)
    z = torch.randn(3, 8, 12, generator=g, requires_grad=True)
    a = torch.randn(12, 16, generator=g)
    native, seeded = SIGReg(5, 16), SeededSIGReg(5, 16, seed=4)
    with patch('torch.randn', return_value=a.clone()), torch.autocast('cpu', dtype=torch.bfloat16, enabled=autocast):
        expected = native(z)
    with torch.autocast('cpu', dtype=torch.bfloat16, enabled=autocast):
        actual = seeded(z, directions=a / a.norm(p=2, dim=0))
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(torch.autograd.grad(actual, z)[0],
                               torch.autograd.grad(expected, z)[0], atol=0, rtol=0)

def test_projection_count_does_not_change_global_rng_or_dropout():
    sequences = []
    z = torch.ones(2, 4, 8)
    for count in (0, 1024, 4096):
        torch.manual_seed(13)
        module = SeededSIGReg(num_proj=max(count, 1), seed=55)
        outputs = []
        for _ in range(3):
            before = torch.get_rng_state().clone()
            if count: module(z)
            assert torch.equal(before, torch.get_rng_state())
            outputs.append(torch.nn.functional.dropout(z, p=.2))
        sequences.append(torch.stack(outputs))
    assert torch.equal(sequences[0], sequences[1])
    assert torch.equal(sequences[0], sequences[2])

def test_serialization_reproduces_next_sketch_loss_and_gradient():
    z = torch.arange(96, dtype=torch.float32).reshape(2, 6, 8).div(50).requires_grad_()
    source = SeededSIGReg(5, 16, seed=19)
    for _ in range(3): source(z)
    buffer = io.BytesIO()
    torch.save(source.state_dict(), buffer); buffer.seek(0)
    restored = SeededSIGReg(5, 16, seed=19)
    restored.load_state_dict(torch.load(buffer, weights_only=True))
    expected, actual = source(z), restored(z)
    assert torch.equal(expected, actual)
    assert torch.equal(torch.autograd.grad(expected, z)[0], torch.autograd.grad(actual, z)[0])

@pytest.mark.parametrize('change', [dict(seed=20), dict(num_proj=32), dict(knots=9)])
def test_incompatible_sketch_checkpoint_is_rejected(change):
    state = copy.deepcopy(SeededSIGReg(5, 16, seed=19).state_dict())
    target = SeededSIGReg(**{**dict(knots=5, num_proj=16, seed=19), **change})
    with pytest.raises(RuntimeError): target.load_state_dict(state)

def test_incomplete_or_foreign_device_stream_is_rejected():
    module = SeededSIGReg(5, 16, seed=19)
    module(torch.ones(2, 4, 8))
    state = module.get_extra_state()
    incomplete = copy.deepcopy(state); incomplete.pop('generator_device')
    with pytest.raises(RuntimeError): module.set_extra_state(incomplete)
    state['generator_device'] = 'cuda:0'
    module.set_extra_state(state)
    with pytest.raises(RuntimeError, match='device'): module(torch.ones(2, 4, 8))
