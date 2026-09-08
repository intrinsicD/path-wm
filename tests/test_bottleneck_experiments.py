import torch
from world_model.curriculum.bottleneck_models import SpatialPoseHead, ConditionedDecoder, physical_loss
from world_model.paddle.models import Decoder
from world_model.paddle.types import ObservationLatent


def test_spatial_coordinates_and_angle_use_xy_not_row_column():
    coords=torch.tensor([[[.2,.3],[.4,.5],[.4,.5+40/512]]])
    pose=SpatialPoseHead.pose_from_points(coords)
    torch.testing.assert_close(pose,torch.tensor([[.2,.3,.4,.5,1.,0.]]),atol=1e-6,rtol=0)


def test_conditioning_initial_identity_and_domain_isolation():
    torch.manual_seed(3)
    base=Decoder(); conditional=ConditionedDecoder(base)
    z=ObservationLatent(torch.randn(2,256,64),torch.randn(2,64,64))
    expected=base(z)
    torch.testing.assert_close(conditional(z,torch.tensor([0,1])),expected)
    with torch.no_grad():conditional.domain_bias[1].add_(1)
    actual=conditional(z,torch.tensor([0,1]))
    torch.testing.assert_close(actual[0],expected[0]);assert not torch.equal(actual[1],expected[1])


def test_physical_loss_keeps_activation_gradient_through_frozen_head():
    head=torch.nn.Linear(4,3).requires_grad_(False)
    state=torch.randn(2,4,requires_grad=True)
    loss=physical_loss(head(state),torch.zeros(2,3))
    loss.backward()
    assert state.grad is not None and state.grad.abs().sum()>0
    assert all(p.grad is None for p in head.parameters())
