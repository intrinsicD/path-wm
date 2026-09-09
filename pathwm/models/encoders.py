"""CNN reference and explicit local DINOv2. Inputs are RGB floats in [0,1]."""

from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from .blocks import Attention, SpatialResidual, mlp
from .features import FeatureSpec


class CNNEncoder(nn.Module):
    """Two processed grids; branch-local residual depth and cross-scale exchange.

    Default parameter names preserve the original E checkpoint. Other encoders
    may expose any named grids; this particular architecture has two.
    """

    def __init__(self, image_size=64, width=64, depth=0, exchange=True):
        super().__init__()
        if image_size % 8 or width % 4 or depth < 0:
            raise ValueError("image_size divisible by 8, width by 4, depth nonnegative")
        self.image_size, self.width, self.exchange = image_size, width, exchange
        fine, coarse = image_size // 4, image_size // 8
        self.feature_spec = {
            "fine": FeatureSpec(width, (fine, fine), "CNN fine"),
            "coarse": FeatureSpec(width, (coarse, coarse), "CNN coarse"),
        }
        self.conv1 = nn.Conv2d(3, width // 4, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(width // 4, width // 2, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(width // 2, width, 3, stride=2, padding=1)
        self.fine_projection = nn.Conv2d(width // 2, width, 1)
        self.fine_row = nn.Parameter(torch.empty(fine, width // 2))
        self.fine_column = nn.Parameter(torch.empty(fine, width // 2))
        self.coarse_row = nn.Parameter(torch.empty(coarse, width // 2))
        self.coarse_column = nn.Parameter(torch.empty(coarse, width // 2))
        self.scale_embeddings = nn.Parameter(torch.empty(2, width))
        for p in (
            self.fine_row,
            self.fine_column,
            self.coarse_row,
            self.coarse_column,
            self.scale_embeddings,
        ):
            nn.init.normal_(p, std=0.02)
        self.fine_attention_norm = nn.LayerNorm(width, eps=1e-5)
        self.coarse_attention_norm = nn.LayerNorm(width, eps=1e-5)
        self.fine_from_coarse, self.coarse_from_fine = (
            Attention(width),
            Attention(width),
        )
        self.fine_mlp_norm, self.coarse_mlp_norm = (
            nn.LayerNorm(width, eps=1e-5),
            nn.LayerNorm(width, eps=1e-5),
        )
        self.fine_mlp, self.coarse_mlp = mlp(width), mlp(width)
        self.fine_blocks = nn.Sequential(
            *[SpatialResidual(width, bias=False) for _ in range(depth)]
        )
        self.coarse_blocks = nn.Sequential(
            *[SpatialResidual(width, bias=False) for _ in range(depth)]
        )
        if not exchange:
            for layer in (
                self.fine_from_coarse,
                self.coarse_from_fine,
                self.fine_attention_norm,
                self.coarse_attention_norm,
            ):
                layer.requires_grad_(False)

    def _positions(self, rows, columns):
        n, half = rows.shape
        return torch.cat(
            (rows[:, None].expand(n, n, half), columns[None].expand(n, n, half)), -1
        ).reshape(n * n, self.width)

    def forward(self, rgb):
        if rgb.ndim != 4 or tuple(rgb.shape[1:]) != (
            3,
            self.image_size,
            self.image_size,
        ):
            raise ValueError(
                f"RGB input must be [B,3,{self.image_size},{self.image_size}]"
            )
        mid = F.gelu(self.conv2(F.gelu(self.conv1(rgb))))
        coarse = self.coarse_blocks(F.gelu(self.conv3(mid))).flatten(2).transpose(1, 2)
        fine = self.fine_blocks(self.fine_projection(mid)).flatten(2).transpose(1, 2)
        fine = (
            fine
            + self._positions(self.fine_row, self.fine_column)
            + self.scale_embeddings[0]
        )
        coarse = (
            coarse
            + self._positions(self.coarse_row, self.coarse_column)
            + self.scale_embeddings[1]
        )
        if self.exchange:
            nf, nc = self.fine_attention_norm(fine), self.coarse_attention_norm(coarse)
            # Both directions read incoming features, not the other update.
            fine, coarse = (
                fine + self.fine_from_coarse(nf, nc, nc),
                coarse + self.coarse_from_fine(nc, nf, nf),
            )
        fine, coarse = (
            fine + self.fine_mlp(self.fine_mlp_norm(fine)),
            coarse + self.coarse_mlp(self.coarse_mlp_norm(coarse)),
        )
        return {
            k: x.transpose(1, 2).reshape(
                len(rgb), self.width, *self.feature_spec[k].size
            )
            for k, x in [("fine", fine), ("coarse", coarse)]
        }


class DinoEncoder(nn.Module):
    """Local official ViT-S/14. No download and no historical-run dependency.

    'coarse' is pooled final context, not another transformer stage. 'local' is
    patch embedding before contextual blocks. Freeze/fine-tune in the recipe.
    The original complete 64px view is resized to 224 with the reference transform.
    """

    def __init__(self, source, weights):
        super().__init__()
        source, weights = Path(source), Path(weights)
        if not source.is_dir() or not weights.is_file():
            raise FileNotFoundError(
                "DINO needs explicit local source directory and weights file"
            )
        self.backbone = torch.hub.load(
            str(source.resolve()), "dinov2_vits14", source="local", pretrained=False
        )
        self.backbone.load_state_dict(
            torch.load(weights, map_location="cpu", weights_only=True), strict=True
        )
        self.feature_spec = {
            "local": FeatureSpec(384, (16, 16), "DINO patch embedding"),
            "fine": FeatureSpec(384, (16, 16), "DINO final context"),
            "coarse": FeatureSpec(384, (8, 8), "pooled final context"),
        }

    def forward(self, rgb):
        if rgb.ndim != 4 or tuple(rgb.shape[1:]) != (3, 64, 64):
            raise ValueError("RGB input must be [B,3,64,64] for this DINO reference")
        x = F.interpolate(
            rgb, (224, 224), mode="bicubic", align_corners=False, antialias=True
        ).clamp(0, 1)
        x = (
            x - rgb.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
        ) / rgb.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
        local = self.backbone.patch_embed(x).transpose(1, 2).reshape(-1, 384, 16, 16)
        fine = (
            self.backbone.forward_features(x)["x_norm_patchtokens"]
            .transpose(1, 2)
            .reshape(-1, 384, 16, 16)
        )
        return {"local": local, "fine": fine, "coarse": F.avg_pool2d(fine, 2)}
