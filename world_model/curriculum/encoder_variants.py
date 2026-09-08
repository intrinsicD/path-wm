"""Explicit depth/exchange intervention; legacy Encoder is never modified."""
import torch
from torch import nn
from torch.nn import functional as F
from world_model.paddle.models import Encoder
from world_model.paddle.types import ObservationLatent


class ResidualSpatialBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.GroupNorm(8,64),nn.GELU(),nn.Conv2d(64,64,3,padding=1,bias=False),
                               nn.GroupNorm(8,64),nn.GELU(),nn.Conv2d(64,64,3,padding=1,bias=False))

    def forward(self,x):
        return x+self.net(x)


class ExperimentalEncoder(Encoder):
    def __init__(self,depth=0,exchange=True):
        if depth not in (0,2):raise ValueError('declared residual depths are 0 and 2')
        if not isinstance(exchange,bool):raise TypeError('exchange must be explicit boolean')
        super().__init__()
        self.depth,self.exchange=depth,exchange
        self.fine_blocks=nn.Sequential(*[ResidualSpatialBlock() for _ in range(depth)])
        self.coarse_blocks=nn.Sequential(*[ResidualSpatialBlock() for _ in range(depth)])
        if not exchange:
            for layer in (self.fine_from_coarse,self.coarse_from_fine,self.fine_attention_norm,self.coarse_attention_norm):
                layer.requires_grad_(False)

    def forward(self,rgb):
        if rgb.ndim!=4 or rgb.shape[1:]!=(3,64,64):raise ValueError('Encoder expects RGB [B,3,64,64]')
        mid=F.gelu(self.conv2(F.gelu(self.conv1(rgb))))
        # Branch-local processing: changing the fine residual stack cannot modify
        # the coarse stem input. Both directions of exchange read incoming maps.
        coarse=self.coarse_blocks(F.gelu(self.conv3(mid))).flatten(2).transpose(1,2)
        fine=self.fine_blocks(self.fine_projection(mid)).flatten(2).transpose(1,2)
        fine=fine+self._positions(self.fine_row,self.fine_column)+self.scale_embeddings[0]
        coarse=coarse+self._positions(self.coarse_row,self.coarse_column)+self.scale_embeddings[1]
        if self.exchange:
            nf,nc=self.fine_attention_norm(fine),self.coarse_attention_norm(coarse)
            fine,coarse=fine+self.fine_from_coarse(nf,nc,nc),coarse+self.coarse_from_fine(nc,nf,nf)
        return ObservationLatent(fine+self.fine_mlp(self.fine_mlp_norm(fine)),
                                 coarse+self.coarse_mlp(self.coarse_mlp_norm(coarse)))


def matched_models(seed,depth,exchange):
    from .training import initial_models
    models=initial_models(seed)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed+30011)
        encoder=ExperimentalEncoder(depth,exchange)
    # D/H and every shared E tensor are identical across a paired seed. New
    # blocks have their own RNG stream and cannot perturb readout initialization.
    missing,unexpected=encoder.load_state_dict(models['E'].state_dict(),strict=False)
    if unexpected or any(not k.startswith(('fine_blocks.','coarse_blocks.')) for k in missing):
        raise RuntimeError('unmatched reference encoder tensors')
    models['E']=encoder
    return models
