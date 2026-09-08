"""A typed dense decoder with explicit local/semantic input slots and task FiLM."""
import torch
from torch import nn
from torch.nn import functional as F
from .perception_heads import Residual


def pack_native(tokens):
    mean = tokens.mean(-1, keepdim=True)
    scale = (tokens.var(-1, unbiased=False, keepdim=True) + 1e-5).sqrt()
    return torch.cat(((tokens - mean) / scale, mean, scale), -1)


def raw_patches(rgb):
    if rgb.shape[1:] != (3, 64, 64): raise ValueError('raw RGB64 patch contract')
    return F.pixel_unshuffle(rgb, 4).flatten(2).transpose(1, 2)


class DenseDecoder(nn.Module):
    def __init__(self, kind, seed):
        super().__init__()
        if kind not in ('late', 'early', 'raw', 'conditioned'): raise ValueError('decoder arm')
        self.kind = kind
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed + 790043)
            self.stages = nn.ModuleList([
                nn.Sequential(nn.Conv2d(384,128,3,padding=1),Residual(128),nn.GroupNorm(8,128),nn.GELU()),
                nn.Sequential(nn.Upsample(scale_factor=2,mode='nearest'),nn.Conv2d(128,64,3,padding=1),nn.GroupNorm(8,64),nn.GELU()),
                nn.Sequential(nn.Upsample(scale_factor=2,mode='nearest'),nn.Conv2d(64,32,3,padding=1),nn.GroupNorm(8,32),nn.GELU())])
            self.outputs = nn.ModuleDict({'rgb':nn.Conv2d(32,3,1),'mask':nn.Conv2d(32,1,1)})
            self.films = nn.ModuleList([nn.Linear(1,2*width) for width in (128,64,32)])
            for film in self.films:
                nn.init.zeros_(film.weight); nn.init.zeros_(film.bias)
            self.projections = nn.ModuleList()
            for index, width in enumerate((48 if kind=='raw' else 384,384,384)):
                torch.random.default_generator.manual_seed(seed + 890051 + index * 1009)
                projection = nn.Linear(width+2,128); nn.init.zeros_(projection.bias)
                self.projections.append(projection)

    def represent(self, features, task):
        if task not in self.outputs: raise ValueError('typed output task')
        maps = []
        for index,(tokens,projection) in enumerate(zip(features,self.projections)):
            side=8 if index==2 else 16
            if tokens.shape[1]!=side*side: raise ValueError('decoder slot spatial contract')
            grid=projection(pack_native(tokens)).transpose(1,2).reshape(-1,128,side,side)
            maps.append(F.interpolate(grid,(16,16),mode='nearest') if side==8 else grid)
        x=torch.cat(maps,1)
        value=(-1. if task=='rgb' else 1.) if self.kind=='conditioned' else 0.
        context=x.new_full((len(x),1),value)
        for stage,film in zip(self.stages,self.films):
            x=stage(x); gamma,beta=film(context).chunk(2,dim=1)
            x=x*(1+gamma[:,:,None,None])+beta[:,:,None,None]
        return x

    def forward(self, features, task):
        output=self.outputs[task](self.represent(features,task))
        return output.sigmoid() if task=='rgb' else output

    def both(self, features):
        if self.kind=='conditioned': return self(features,'rgb'),self(features,'mask')
        x=self.represent(features,'rgb')
        return self.outputs['rgb'](x).sigmoid(),self.outputs['mask'](x)
