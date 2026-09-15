import torch
from torch import nn
import pytest

from pathwm.models.spatial_vae import SpatialVAE, vae_loss
from pathwm.models.spatial_vae_v2 import (
    HierarchicalVAE, LosslessDownsample2D, LosslessUpsample2D,
    LoopTransformer, EncoderStage, stride2_from_projection,
)


def test_exact_rearrangement_and_stride_control():
    x = torch.randn(2, 3, 8, 10, dtype=torch.float64)
    r = LosslessDownsample2D()
    assert torch.equal(LosslessUpsample2D()(r(x)), x)
    c = nn.Conv2d(12, 5, 1).double()
    assert torch.allclose(stride2_from_projection(c)(x), c(r(x)), atol=1e-12)
    with pytest.raises(ValueError):
        r(x[..., :-1])


@pytest.mark.parametrize('ablation', ['A_local', 'A_exact', 'B', 'C', 'C_after', 'D', 'E'])
def test_geometry_backward_trace_export(tmp_path, ablation):
    m = HierarchicalVAE(stem_channels=4, channels=(8, 12), latent_channels=2,
                        ablation=ablation)
    for h, w in [(1, 3), (13, 19), (16, 12)]:
        x = torch.rand(2, 3, h, w)
        y, p = m(x)
        assert y.shape == x.shape
        assert p.mu.shape == (2, 2, (h+3)//4, (w+3)//4)
        assert p.mu.shape == p.logvar.shape == p.sample().shape
        vae_loss(y, x, p)[0].backward()
        assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in m.parameters())
        m.zero_grad(set_to_none=True)
    p, trace = m.inspect(x)
    assert all(not v.requires_grad for v in trace.values())
    assert 'stage_0.before_compression' in trace and 'posterior.mu' in trace
    path = tmp_path/'v2.pt'
    m.save(path)
    restored = SpatialVAE.load(path)
    assert torch.equal(m(x, sample=False)[0], restored(x, sample=False)[0])
    old = SpatialVAE(channels=(4,), latent_channels=2)
    old.save(tmp_path/'old.pt')
    assert torch.equal(old(x, sample=False)[0], SpatialVAE.load(tmp_path/'old.pt')(x, sample=False)[0])


def test_loop_shared_parameters_positions_and_limit():
    loop = LoopTransformer(8, heads=2, iterations=0, max_tokens=20)
    n = sum(p.numel() for p in loop.parameters())
    x = torch.randn(2, 8, 3, 4)
    assert torch.equal(loop(x), x)
    calls = []
    hook = loop.block.register_forward_hook(lambda *args: calls.append(1))
    loop.iterations = 3
    out, states = loop(x, return_states=True)
    assert len(calls) == 3 and len(states) == 4
    assert sum(p.numel() for p in loop.parameters()) == n
    assert not torch.equal(out, x)
    assert all(not s.requires_grad for s in states)
    hook.remove()
    loop(torch.randn(1, 8, 2, 7)).sum().backward()
    assert all(p.grad is not None for p in loop.parameters())
    with pytest.raises(ValueError, match='token'):
        loop(torch.rand(1, 8, 5, 5))


def test_identity_contract():
    stage = EncoderStage(4, None, downsample=False, depth=0)
    x = torch.randn(2, 4, 5, 7)
    assert stage.out_channels == 4 and stage.factor == 1
    assert torch.equal(stage(x), x)
    with pytest.raises(ValueError):
        EncoderStage(4, 7, downsample=False)
