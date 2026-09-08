import pytest
import torch
from world_model.curriculum.perception_decoder import DenseDecoder
from world_model.curriculum.perception_decoder_reliance import routed_features,task_context


def test_donor_routing_keeps_context_levels_together_and_preserves_sources():
    a=tuple(torch.full((2,n,3),float(i)) for i,n in enumerate((256,256,64)))
    b=tuple(torch.full((2,n,3),float(i+7)) for i,n in enumerate((256,256,64)))
    local=routed_features(a,b,'donor_local'); context=routed_features(a,b,'donor_context')
    assert local[0] is b[0] and local[1] is a[1] and local[2] is a[2]
    assert context[0] is a[0] and context[1] is b[1] and context[2] is b[2]
    assert a[0].eq(0).all() and b[0].eq(7).all()


def test_task_override_preserves_typed_outputs_and_restores_hooks_on_failure():
    torch.manual_seed(67); model=DenseDecoder('conditioned',9107)
    features=(torch.randn(1,256,384),torch.randn(1,256,384),torch.randn(1,64,384))
    with torch.no_grad():
        for film in model.films: film.weight.fill_(.05)
        normal=model(features,'rgb')
        with task_context(model,'flipped'):
            flipped=model(features,'rgb')
            assert flipped.shape==normal.shape==(1,3,64,64)
            assert not torch.equal(normal,flipped)
        with task_context(model,'neutral'):
            rgb=model.represent(features,'rgb'); mask=model.represent(features,'mask')
            torch.testing.assert_close(rgb,mask,rtol=0,atol=0)
        with pytest.raises(RuntimeError):
            with task_context(model,'flipped'): raise RuntimeError('intentional')
        torch.testing.assert_close(model(features,'rgb'),normal,rtol=0,atol=0)
    assert all(not film._forward_pre_hooks for film in model.films)
