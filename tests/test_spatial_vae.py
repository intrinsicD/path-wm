import copy
import torch
import pytest
from torch.nn import functional as F
from pathwm.models.spatial_vae import SpatialVAE, ReversibleMixer, vae_loss, build_variant


def test_shuffle_and_coupling_are_invertible_without_calling_projection_lossless():
    x = torch.randn(2, 12, 5, 7, dtype=torch.float64)
    block = ReversibleMixer(12).double()
    for p in block.parameters():
        torch.nn.init.normal_(p, std=0.02)
    torch.testing.assert_close(block.inverse(block(x)), x, atol=1e-12, rtol=1e-12)
    image = torch.randn(2, 3, 16, 24)
    assert torch.equal(F.pixel_shuffle(F.pixel_unshuffle(image, 2), 2), image)


@pytest.mark.parametrize('size', [(1, 1), (15, 19), (32, 48)])
@pytest.mark.parametrize('variant', ['base', 'attention', 'reversible'])
def test_geometry_and_decoder_needs_only_z(size, variant):
    model = SpatialVAE(channels=(8, 16), latent_channels=3, variant=variant)
    x = torch.randn(2, 3, *size)
    p = model.encode(x)
    assert p.mu.shape == (2, 3, (size[0]+3)//4, (size[1]+3)//4)
    y = model.decode(p.mu, size)
    assert y.shape == x.shape
    saved = copy.deepcopy(model.decoder)
    del model, x
    assert torch.equal(saved(p.mu, size), y)
    with pytest.raises(ValueError):
        saved(p.mu, (size[0]+4, size[1]))


def test_variants_share_initial_function_and_reversible_parameter_count():
    models = [build_variant(v, 17) for v in ['base', 'attention', 'reversible']]
    x = torch.rand(2, 3, 16, 24)
    outputs = [m(x, sample=False)[0] for m in models]
    assert all(torch.equal(outputs[0], y) for y in outputs[1:])
    assert sum(p.numel() for p in models[1].parameters()) == sum(p.numel() for p in models[2].parameters())


def test_sampling_loss_units_and_gradients():
    model = SpatialVAE(channels=(8, 16), latent_channels=3)
    x = torch.rand(2, 3, 15, 19)
    y, p = model(x, generator=torch.Generator().manual_seed(1))
    assert torch.equal(p.sample(torch.Generator().manual_seed(12)), p.sample(torch.Generator().manual_seed(12)))
    assert not torch.equal(p.sample(torch.Generator().manual_seed(12)), p.mu)
    loss, d = vae_loss(y, x, p, beta=0.2)
    kl = .5 * (p.mu.square()+p.logvar.exp()-1-p.logvar).sum((1,2,3)) / (15*19)
    expected = (y-x).square().sum((1,2,3)) / (15*19) + .2*kl
    torch.testing.assert_close(loss, expected.mean())
    torch.testing.assert_close(d['rate'], kl.mean())
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
    assert model.encoder.mu.weight.grad.abs().sum() > 0
    assert model.encoder.logvar.weight.grad.abs().sum() > 0


def test_attention_fine_scale_dependency_and_batch_isolation():
    model = SpatialVAE(channels=(8,16), latent_channels=3, variant='attention')
    torch.nn.init.normal_(model.encoder.attention.output.weight, std=.1)
    x = torch.rand(2,3,16,24)
    p = model.encode(x)
    altered = x.clone(); altered[1] = torch.randn_like(altered[1])
    torch.testing.assert_close(p.mu[0], model.encode(altered).mu[0], atol=0, rtol=0)
    grids = [torch.randn(2,8,8,12), torch.randn(2,16,4,6)]
    out = model.encoder.attention(grids)
    changed = model.encoder.attention([grids[0]+1, grids[1]])
    assert not torch.allclose(out, changed)


def test_checkpoint_has_strict_architecture_and_exact_outputs(tmp_path):
    m = build_variant('reversible', 19)
    p = tmp_path/'weights.pt'; m.save(p)
    loaded = SpatialVAE.load(p)
    x=torch.rand(1,3,19,23)
    assert torch.equal(m(x,sample=False)[0], loaded(x,sample=False)[0])
