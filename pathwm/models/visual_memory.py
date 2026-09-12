"""Read image histories through the actual belief/memory/working-token path."""

import torch
from torch import nn
from .modalities import Observation, Attend


class VisualMemoryReader(nn.Module):
    def __init__(self, agent):
        super().__init__()
        self.agent = agent
        self.query = nn.Parameter(torch.randn(1, agent.width) * 0.02)
        self.reader = Attend(agent.width)
        self.head = nn.Linear(agent.width, 2)

    def features(self, images):
        if images.ndim != 5 or images.shape[1:] != (4, 3, 32, 32):
            raise ValueError("Visual memory expects four RGB32 frames")
        state = self.agent.initial_state(
            len(images), time=0, session_id="visual-memory"
        )
        for t in range(images.shape[1]):
            observation = Observation(
                images[:, t : t + 1], images.new_full((len(images), 1), float(t))
            )
            state = self.agent.observe(
                state, {"image": observation}, time=float(t), replay=self.training
            )
        working = self.agent.think(
            state, steps=2, goal=self.query.expand(len(images), -1, -1)
        )
        tokens = working.tokens[:, self.agent.layout["working"]]
        return self.reader(self.query.expand(len(images), -1, -1), tokens)[:, 0]

    def forward(self, images):
        return self.head(self.features(images))


@torch.no_grad()
def construct_weights(model, *, timing=False):
    """Handwritten red-salience routing and horizontal signed readout; no data fit."""
    for parameter in model.parameters():
        parameter.zero_()
    width = model.agent.width
    for layer in model.modules():
        if isinstance(layer, nn.LayerNorm):
            layer.weight.fill_(1)
        if isinstance(layer, nn.MultiheadAttention):
            identity = torch.eye(width, device=layer.in_proj_weight.device)
            layer.in_proj_weight[2 * width :].copy_(identity)
            layer.out_proj.weight.copy_(identity)
            for i in range(0, width, width // layer.num_heads):
                layer.in_proj_bias[i] = 6
                layer.in_proj_weight[width + i, 0] = 1
    patch = model.agent.encoders["image"].stem.patch
    area = patch.weight.shape[-1] * patch.weight.shape[-2]
    patch.weight[0, 0].fill_(8 / area)
    patch.weight[0, 1:].fill_(-4 / area)
    model.head.weight[0, 8] = -4
    model.head.weight[1, 8] = 4
    if timing:
        model.agent.dynamics.transition.attention.out_proj.weight.zero_()
        clock = model.agent.memory.clocks[0]
        clock.weight[0, width] = 16
        clock.bias[0] = -8
        for reader in model.agent.memory.readers.values():
            reader.view[0, 0] = 4
            reader.view[1, 0] = -16
            reader.gate.bias.copy_(reader.gate.bias.new_tensor([-8, 8, -8, -8]))


@torch.no_grad()
def fit_centroid_head(model, features, labels):
    """Closed-form label-based adjustment. This is fitting, without backpropagation."""
    means = torch.stack([features[labels == i].double().mean(0) for i in (0, 1)])
    direction = means[1] - means[0]
    norm = direction.square().sum()
    if not torch.isfinite(means).all() or not torch.isfinite(norm) or norm < 1e-16:
        raise ValueError("Fit features have no finite class separation")
    direction *= 2 / norm
    bias = -(direction * means.mean(0)).sum()
    model.head.weight.copy_(torch.stack((-direction, direction)).to(model.head.weight))
    model.head.bias.copy_(torch.stack((-bias, bias)).to(model.head.bias))
