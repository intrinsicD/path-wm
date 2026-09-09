import torch
import pytest
from pathwm.models.encoders import CNNEncoder
from pathwm.models.decoders import ReconstructionDecoder, DenseHead
from pathwm.models.temporal import MemoryUpdater, Predictor, imagine


def test_cnn_and_decoder_have_visible_spatial_contract():
    encoder = CNNEncoder(depth=2)
    features = encoder(torch.zeros(2, 3, 64, 64))
    assert {k: tuple(v.shape) for k, v in features.items()} == {
        'fine': (2, 64, 16, 16), 'coarse': (2, 64, 8, 8)}
    decoder = ReconstructionDecoder(encoder.feature_spec)
    assert decoder(features).shape == (2, 3, 64, 64)
    with pytest.raises(ValueError, match='RGB'):
        encoder(torch.zeros(2, 3, 63, 64))


def test_decoder_uses_selected_scales_only():
    from pathwm.models.features import FeatureSpec
    spec = {'a': FeatureSpec(12, (8, 8), 'test'), 'b': FeatureSpec(20, (4, 4), 'test')}
    head = DenseHead(spec, channels=1, levels=('a',), output_size=(32, 32))
    a = torch.randn(2, 12, 8, 8, requires_grad=True)
    b = torch.randn(2, 20, 4, 4, requires_grad=True)
    head({'a': a, 'b': b}).sum().backward()
    assert a.grad is not None and b.grad is None


def test_imagining_is_functional_and_frozen_memory_transmits_gradients():
    encoder = CNNEncoder()
    updater = MemoryUpdater(encoder.feature_spec, action_width=2).requires_grad_(False)
    predictor = Predictor(encoder.feature_spec, action_width=2)
    features = {k: v.detach() for k, v in encoder(torch.rand(1, 3, 64, 64)).items()}
    original = {k: v.clone() for k, v in features.items()}
    memory = torch.zeros(1, 128)
    states = imagine(features, memory, torch.rand(1, 3, 2), predictor, updater)
    sum(x.square().mean() for x in states[-1][0].values()).backward()
    assert predictor.output_projection.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in updater.parameters())
    assert all(torch.equal(features[k], original[k]) for k in features)
    assert torch.equal(memory, torch.zeros_like(memory))
