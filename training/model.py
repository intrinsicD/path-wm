"""Build, count, save and restore the four trainable modules of E1-a."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

from encoders import build_encoder
from encoders.adapters import build_adapter
from predictors import build_predictor
from world_state.inverse import build_inverse


@dataclass
class ModelBundle:
    encoder: nn.Module
    adapter: nn.Module
    predictor: nn.Module
    inverse: nn.Module
    z_shape: tuple[int, ...]

    def modules(self) -> tuple[nn.Module, ...]:
        return self.encoder, self.adapter, self.predictor, self.inverse

    def parameters(self):
        for module in self.modules():
            yield from module.parameters()

    def train(self) -> None:
        for module in self.modules():
            module.train()

    def eval(self) -> None:
        for module in self.modules():
            module.eval()

    def state_dict(self) -> dict[str, dict[str, torch.Tensor]]:
        return {
            "encoder": self.encoder.state_dict(),
            "adapter": self.adapter.state_dict(),
            "predictor": self.predictor.state_dict(),
            "inverse": self.inverse.state_dict(),
        }

    def load_state_dict(self, state: dict[str, dict[str, torch.Tensor]]) -> None:
        for name in ("encoder", "adapter", "predictor", "inverse"):
            getattr(self, name).load_state_dict(state[name])


def build_models(cfg: dict, device: torch.device, z_shape: tuple[int, ...] | None = None) -> ModelBundle:
    encoder = build_encoder(cfg).to(device)
    if z_shape is None:
        resolution = int(cfg["env"]["resolution"])
        with torch.no_grad():
            z_shape = tuple(encoder.encode(torch.zeros(1, 3, resolution, resolution, dtype=torch.uint8, device=device)).shape[1:])
    adapter = build_adapter(cfg, z_shape).to(device)
    predictor = build_predictor(cfg).to(device)
    inverse = build_inverse(cfg).to(device)
    return ModelBundle(encoder, adapter, predictor, inverse, z_shape)


def parameter_counts(models: ModelBundle) -> dict[str, int]:
    return {name: sum(parameter.numel() for parameter in getattr(models, name).parameters()) for name in ("encoder", "adapter", "predictor", "inverse")}


def check_parameter_targets(cfg: dict, counts: dict[str, int], tolerance: float = 0.25) -> None:
    for section, name in (("encoder", "encoder"), ("predictor", "predictor")):
        target = cfg[section].get("params_target")
        if target is None:
            continue
        relative_error = abs(counts[name] - int(target)) / int(target)
        if relative_error > tolerance:
            raise ValueError(
                f"{name} has {counts[name]:,} parameters, outside {tolerance:.0%} of target {int(target):,}"
            )


def save_checkpoint(
    path: Path,
    cfg: dict,
    models: ModelBundle,
    optimizer: torch.optim.Optimizer,
    step: int,
    seed: int,
    counts: dict[str, int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "cfg": cfg,
            "models": models.state_dict(),
            "optimizer": optimizer.state_dict(),
            "z_shape": models.z_shape,
            "step": step,
            "seed": seed,
            "parameter_counts": counts,
        },
        path,
    )


def load_checkpoint(path: Path, device: torch.device) -> tuple[dict, ModelBundle, dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    cfg = checkpoint["cfg"]
    models = build_models(cfg, device, tuple(checkpoint["z_shape"]))
    models.load_state_dict(checkpoint["models"])
    return cfg, models, checkpoint
