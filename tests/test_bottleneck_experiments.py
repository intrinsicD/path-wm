import torch
from world_model.curriculum.bottleneck_models import SpatialPoseHead
from world_model.paddle.types import ObservationLatent


def test_spatial_coordinates_and_angle_use_xy_not_row_column():
    coords=torch.tensor([[[.2,.3],[.4,.5],[.4,.5+40/512]]])
    pose=SpatialPoseHead.pose_from_points(coords)
    torch.testing.assert_close(pose,torch.tensor([[.2,.3,.4,.5,1.,0.]]),atol=1e-6,rtol=0)


def test_head_training_leaves_feature_source_frozen():
    encoder=torch.nn.Linear(3,320*64).requires_grad_(False)
    head=SpatialPoseHead()
    before={k:v.clone() for k,v in encoder.state_dict().items()}
    source=encoder(torch.randn(2,3)).reshape(2,320,64).detach()
    pose=head(ObservationLatent.from_tokens(source))
    pose.square().mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in head.parameters())
    assert all(p.grad is None for p in encoder.parameters())
    assert all(torch.equal(v,encoder.state_dict()[k]) for k,v in before.items())

def test_landmark_targets_roundtrip_and_spatial_probabilities():
    torch.set_num_threads(1)
    angles=torch.tensor([-.7,1.2])
    truth=torch.cat((torch.full((2,4),.5),angles.sin()[:,None],angles.cos()[:,None]),1)
    torch.testing.assert_close(SpatialPoseHead.pose_from_points(SpatialPoseHead.target_points(truth)),truth)
    head=SpatialPoseHead()
    latent=ObservationLatent(torch.randn(2,256,64),torch.randn(2,64,64))
    _,points,heatmaps=head.details(latent)
    torch.testing.assert_close(heatmaps.sum((-1,-2)),torch.ones(2,3))
    assert ((points>=0)&(points<=1)).all()
