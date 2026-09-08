import pytest
import torch
from world_model.curriculum.encoder_variants import ExperimentalEncoder
from world_model.curriculum.perception_internal import attention_override


def test_attention_intervention_is_scoped_and_leaves_no_hooks_on_exception():
    torch.manual_seed(31)
    model=ExperimentalEncoder(depth=2,exchange=True).eval()
    rgb=torch.rand(1,3,64,64)
    with torch.no_grad():
        original=model(rgb).tokens()
        with pytest.raises(RuntimeError,match='intentional'):
            with attention_override(model,'zero'):
                assert not torch.equal(model(rgb).tokens(),original)
                raise RuntimeError('intentional')
        assert torch.equal(model(rgb).tokens(),original)
        for module in (model.fine_from_coarse,model.coarse_from_fine):
            assert len(module._forward_hooks)==0


def test_one_direction_does_not_silently_modify_the_other_grid():
    torch.manual_seed(33)
    model=ExperimentalEncoder(depth=2,exchange=True).eval()
    rgb=torch.rand(1,3,64,64)
    with torch.no_grad():
        baseline=model(rgb)
        with attention_override(model,'zero',directions=('fine_from_coarse',)):
            changed=model(rgb)
        assert torch.equal(changed.coarse,baseline.coarse)
        assert not torch.equal(changed.fine,baseline.fine)
