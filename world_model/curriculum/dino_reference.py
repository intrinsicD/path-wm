"""Frozen, explicitly sourced DINOv2 patch representation and local adapter."""
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from world_model.paddle.types import ObservationLatent


def preprocess(rgb):
    if rgb.ndim!=4 or rgb.shape[1:]!=(3,64,64):raise ValueError('same RGB64 source views required')
    # Preserve the entire source field of view, unlike the classification crop.
    resized=F.interpolate(rgb,size=(224,224),mode='bicubic',align_corners=False,antialias=True).clamp(0,1)
    mean=rgb.new_tensor([.485,.456,.406])[None,:,None,None]
    std=rgb.new_tensor([.229,.224,.225])[None,:,None,None]
    return (resized-mean)/std


class DinoAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection=nn.Linear(384,64)

    def forward(self,patches):
        if patches.ndim!=3 or patches.shape[1:]!=(256,384):raise ValueError('DINO patches must be [B,256,384]')
        fine=self.projection(patches)
        coarse=F.avg_pool2d(fine.transpose(1,2).reshape(-1,64,16,16),2).flatten(2).transpose(1,2)
        return ObservationLatent(fine,coarse)


class FrozenDino(nn.Module):
    def __init__(self,source,weights):
        super().__init__()
        source,weights=Path(source),Path(weights)
        if not source.is_dir() or not weights.is_file():raise FileNotFoundError('explicit local DINO source and weights required')
        self.backbone=torch.hub.load(str(source.resolve()),'dinov2_vits14',source='local',pretrained=False)
        self.backbone.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True),strict=True)
        self.backbone.eval().requires_grad_(False)

    def train(self,mode=True):
        super().train(False)
        return self

    @torch.no_grad()
    def forward(self,rgb):
        return self.backbone.forward_features(preprocess(rgb))['x_norm_patchtokens']
