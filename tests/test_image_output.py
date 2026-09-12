from dataclasses import replace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from pathwm.models.features import FeatureSpec


class BaseEncoder(nn.Module):
    feature_spec = {"base": FeatureSpec(3, (2, 2), "test patch means")}

    def forward(self, x):
        return {"base": F.avg_pool2d(x, 4)}


class BaseHead(nn.Module):
    def forward(self, features):
        return F.interpolate(features["base"], scale_factor=4, mode="nearest")


def test_detail_transports_high_frequency_and_keeps_patch_means():
    from pathwm.models.encoders import PatchDetailEncoder
    from pathwm.models.decoders import PatchDetailHead

    encoder = PatchDetailEncoder(BaseEncoder(), image_size=8, patch_size=4)
    decoder = PatchDetailHead(BaseHead(), patch_size=4)
    x = torch.rand(2, 3, 8, 8)
    features = encoder(x)
    reconstructed = decoder(features)
    torch.testing.assert_close(reconstructed, x, atol=1e-6, rtol=0)
    features["detail"] = torch.zeros_like(features["detail"])
    assert F.mse_loss(decoder(features), x) > 0.01
    torch.testing.assert_close(F.avg_pool2d(reconstructed, 4), features["base"])
    with pytest.raises(ValueError, match="detail"):
        decoder({"base": features["base"]})
    with pytest.raises(ValueError, match="divisible"):
        PatchDetailEncoder(BaseEncoder(), image_size=7, patch_size=4)


def test_state_produces_all_features_and_gradients_cross_frozen_decoder():
    from pathwm.models.encoders import PatchDetailEncoder
    from pathwm.models.decoders import PatchDetailHead, StateFeatureDecoder

    encoder = PatchDetailEncoder(BaseEncoder(), image_size=8)
    head = PatchDetailHead(BaseHead()).requires_grad_(False)
    decoder = StateFeatureDecoder(16, encoder.feature_spec, head)
    targets = encoder(torch.rand(3, 3, 8, 8))
    decoder.calibrate(targets)
    tokens = torch.randn(3, 5, 16, requires_grad=True)
    x = decoder(tokens)
    assert x.shape == (3, 3, 8, 8)
    assert set(decoder.features(tokens)) == set(encoder.feature_spec)
    x.square().mean().backward()
    assert tokens.grad.abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in decoder.reader.parameters())
    assert all(p.grad is None for p in head.parameters())
    assert not torch.equal(decoder(tokens), decoder(tokens.flip(0)))
    with pytest.raises(ValueError, match="finite"):
        decoder.calibrate({k: torch.full_like(v, float("nan")) for k, v in targets.items()})


def test_request_path_has_no_image_encoder_and_responds_to_request():
    from experiments.image_output import build_agent
    from pathwm.models.decoders import StateFeatureDecoder

    decoder = StateFeatureDecoder(16, BaseEncoder.feature_spec, BaseHead())
    agent = build_agent(decoder, width=16)
    assert set(agent.encoders) == {"text"}
    from pathwm.models.modalities import Observation, bytes_batch

    ids, valid = bytes_batch(["horizontal", "vertical"])
    obs = Observation(ids, torch.zeros_like(ids, dtype=torch.float), valid)
    state = agent.observe(agent.initial_state(2), {"text": obs}, time=0)
    image = agent.decode(state, modalities=["image"])["image"]
    assert not torch.equal(image[0], image[1])
    assert torch.equal(image, agent.decode(replace(state, tokens=state.tokens.clone()), modalities=["image"])["image"])
