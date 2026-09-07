"""Essential invariants for the new perception curriculum."""
import copy
import numpy as np
import pytest
import torch
from world_model.curriculum.data import group_hashes, split_groups
from world_model.curriculum.training import (
    initial_models, objective, backward_batch, selection_key, phase_sampler)

def test_generic_reconstruction_never_calls_task_head():
    models = initial_models(17)
    class Forbidden(torch.nn.Module):
        def forward(self, _):
            raise AssertionError("unlabelled images must not receive pose loss")
    models['H'] = Forbidden()
    loss, metrics = objective(models, torch.rand(2,3,64,64), None)
    loss.backward()
    assert set(metrics) == {'loss','image_mse'}
    assert any(p.grad is not None for p in models['E'].parameters())

def test_physical_selector_cannot_hide_bad_pose_with_good_pixels():
    good = {'position_mae':[4,4,4,4], 'angle_mae_deg':5, 'image_mse':.1}
    bad = {'position_mae':[4,4,4,16], 'angle_mae_deg':5, 'image_mse':.0001}
    assert selection_key(good, 100) < selection_key(bad, 200)
    assert selection_key(good, 100) < selection_key(good, 200)
    with pytest.raises(ValueError):
        selection_key({**good,'angle_mae_deg':float('nan')},0)

@pytest.mark.parametrize('labelled',[False, True])
def test_ragged_microbatches_match_full_batch_gradients(labelled):
    torch.set_num_threads(1)
    full=initial_models(42); micro=copy.deepcopy(full)
    x=torch.rand(3,3,64,64); y=torch.rand(3,6) if labelled else None
    backward_batch(full,x,y,3)
    backward_batch(micro,x,y,2)
    for key in full:
        for a,b in zip(full[key].parameters(),micro[key].parameters()):
            if a.grad is None:
                assert b.grad is None
            else:
                torch.testing.assert_close(a.grad,b.grad,atol=2e-6,rtol=1e-4)

def test_warmup_loads_only_encoder_decoder_and_preserves_fresh_head():
    original=initial_models(2)
    warmup={k:{name:torch.zeros_like(t) for name,t in model.state_dict().items()}
            for k,model in original.items()}
    adapted=initial_models(2,warmup)
    for k in ('E','D'):
        assert all(torch.count_nonzero(t)==0 for t in adapted[k].state_dict().values())
    for name,t in original['H'].state_dict().items():
        assert torch.equal(t,adapted['H'].state_dict()[name])

def test_private_supervised_stream_is_phase_independent_and_resumable():
    a=phase_sampler(4107,'supervised'); b=phase_sampler(4107,'supervised')
    np.testing.assert_array_equal(a.integers(200, size=128),b.integers(200,size=128))
    snapshot=copy.deepcopy(a.bit_generator.state)
    phase_sampler(4107,'warmup').integers(1000,size=5000)
    expected=a.integers(200,size=128)
    restored=phase_sampler(4107,'supervised');restored.bit_generator.state=snapshot
    np.testing.assert_array_equal(expected,restored.integers(200,size=128))

def test_duplicate_components_are_transitive_and_split_as_units():
    # 0->15->255 is a radius-four chain; first/last are distance eight.
    hashes=[0,15,255,2**64-1,0xAAAAAAAAAAAAAAAA,0x5555555555555555]
    groups,edges=group_hashes(hashes,[str(i) for i in range(len(hashes))])
    assert groups[0]==groups[1]==groups[2]
    assert len(set(groups))==4
    split=split_groups(groups,4107)
    membership={i:k for k,ids in split.items() for i in ids}
    assert len(membership)==len(groups)
    assert membership[0]==membership[1]==membership[2]
    assert all(split.values())
