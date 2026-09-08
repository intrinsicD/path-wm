import pytest
import torch
from world_model.curriculum.perception_heads import make_heads,spatial_expectation
from world_model.curriculum.perception_localization import barycentric_targets,location_kl,pose_and_logits


def test_location_targets_recover_endpoints_and_continuous_coordinates():
    torch.manual_seed(53)
    xy=torch.cat((torch.rand(20,2,2,dtype=torch.float64),
        torch.tensor([[[0.,0.],[1.,1.]],[[0.,1.],[1.,0.]],[[7/15,3/15],[7/15+1e-10,3/15-1e-10]]],dtype=torch.float64)))
    target=barycentric_targets(xy)
    assert (target>=0).all()
    torch.testing.assert_close(target.sum((-2,-1)),torch.ones_like(xy[:,:,0]),rtol=0,atol=1e-14)
    expected=spatial_expectation(target.log()).reshape_as(xy)
    torch.testing.assert_close(expected,xy,rtol=0,atol=1e-14)
    with pytest.raises(ValueError): barycentric_targets(torch.tensor([[[1.01,0.]]]))


def test_kl_is_finite_with_finite_gradients_and_zero_at_matching_distribution():
    xy=torch.tensor([[[0.,0.],[.42,.97]]],dtype=torch.float64)
    target=barycentric_targets(xy)
    logits=(target+1e-18).log().requires_grad_(True)
    loss=location_kl(logits,xy); assert abs(float(loss.detach()))<1e-12
    loss.backward(); assert torch.isfinite(logits.grad).all()
    wrong=torch.zeros_like(logits,requires_grad=True)
    penalty=location_kl(wrong,xy); penalty.backward()
    assert penalty>0 and torch.isfinite(wrong.grad).all() and wrong.grad.abs().sum()>0


def test_logit_access_and_zero_regularization_preserve_pose_forward_and_gradients():
    torch.manual_seed(57); head=make_heads(64,9107)['pose']
    fine,coarse=torch.randn(2,256,64),torch.randn(2,64,64)
    target=torch.rand(2,6)
    plain=head(fine,coarse); a=torch.autograd.grad((plain-target).square().mean(),tuple(head.parameters()))
    pose,logits=pose_and_logits(head,fine,coarse)
    loss=(pose-target).square().mean()+0*location_kl(logits,target[:,:4].reshape(-1,2,2))
    b=torch.autograd.grad(loss,tuple(head.parameters()))
    assert torch.equal(pose,plain)
    for x,y in zip(a,b): torch.testing.assert_close(x,y,rtol=0,atol=0)
