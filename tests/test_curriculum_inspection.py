import numpy as np
import torch
from world_model.curriculum.inspection import pca_fit, pca_transform, attention_weights, grouped_errors
from world_model.paddle.models import Attention


def test_pca_fits_training_only_and_uses_fixed_color_limits():
    train=np.random.default_rng(1).normal(size=(100,4))
    basis=pca_fit(train)
    before={k:np.array(v,copy=True) for k,v in basis.items()}
    z=pca_transform(train,basis)
    assert np.allclose(z.mean(0),0,atol=1e-12)
    pca_transform(np.ones((10,4))*1e6,basis)
    for k in basis:np.testing.assert_array_equal(before[k],basis[k])


def test_attention_weights_reproduce_actual_module():
    torch.manual_seed(1);module=Attention();q=torch.randn(2,3,64);k=torch.randn(2,5,64)
    weights=attention_weights(module,q,k)
    v=module.value_projection(k).reshape(2,5,4,16).transpose(1,2)
    manual=module.output_projection((weights@v).transpose(1,2).reshape(2,3,64))
    torch.testing.assert_close(manual,module(q,k,k))
    torch.testing.assert_close(weights.sum(-1),torch.ones(2,4,3))


def test_group_balancing_does_not_weight_long_episodes_twice():
    raw={'position_abs_error':[[1,1,1,1],[1,1,1,1],[9,9,9,9]],'angle_abs_error_deg':[2,2,10]}
    result=grouped_errors(raw,[{'group':0},{'group':0},{'group':1}])
    assert result['groups']==2
    assert result['position_mae']==[5,5,5,5]
    assert result['angle_mae_deg']==6


def test_mean_image_uses_only_supplied_training_membership():
    from world_model.curriculum.inspection import mean_image
    from world_model.curriculum.data import FrameSet
    frames=np.zeros((3,64,64,3),np.uint8);frames[1]=100;frames[2]=255
    train=FrameSet(frames,[0,1]);mean=mean_image(train)
    np.testing.assert_allclose(mean,50/255)
    frames[2]=0
    np.testing.assert_array_equal(mean_image(train),mean)
