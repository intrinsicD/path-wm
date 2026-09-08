"""Experimental pose readouts; baseline checkpoint interfaces remain unchanged."""
import torch
from torch import nn
from torch.nn import functional as F


class NonlinearPoseHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(20480,64),nn.GELU(),nn.Linear(64,6))

    def forward(self,observation):
        return self.net(observation.tokens().flatten(1))


class SpatialPoseHead(nn.Module):
    """Three expected XY locations on the fine grid, with directed orientation."""
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(128,64,3,padding=1),nn.GELU(),nn.Conv2d(64,3,1))
        # Pixel-center coordinates; image row increases with source world y.
        y,x=torch.meshgrid((torch.arange(16)+.5)/16,(torch.arange(16)+.5)/16,indexing='ij')
        self.register_buffer('grid',torch.stack((x,y),-1).reshape(256,2))

    @staticmethod
    def pose_from_points(points):
        direction=points[:,2]-points[:,1]
        unit=F.normalize(direction,dim=-1,eps=1e-4)
        return torch.cat((points[:,:2].flatten(1),unit[:,[1,0]]),-1)

    @staticmethod
    def target_points(pose):
        center=pose[:,2:4]
        landmark=center+(40/512)*pose[:,[5,4]]
        return torch.stack((pose[:,:2],center,landmark),1)

    def details(self,observation):
        b=len(observation.fine)
        fine=observation.fine.transpose(1,2).reshape(b,64,16,16)
        coarse=observation.coarse.transpose(1,2).reshape(b,64,8,8)
        logits=self.net(torch.cat((fine,F.interpolate(coarse,size=(16,16),mode='nearest')),1))
        probabilities=logits.flatten(2).softmax(-1)
        points=probabilities@self.grid
        return self.pose_from_points(points),points,probabilities.reshape(b,3,16,16)

    def forward(self,observation):
        return self.details(observation)[0]
