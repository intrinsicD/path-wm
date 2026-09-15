import torch
import pytest

from pathwm.models.spatial_vae import Posterior, vae_loss, reconstruction_penalties


def test_single_kl_and_legacy_values_gradients():
    torch.manual_seed(3)
    y = torch.randn(2, 3, 12, 16, requires_grad=True)
    target = torch.rand_like(y)
    mu = torch.randn(2, 4, 3, 4, requires_grad=True)
    lv = torch.randn_like(mu, requires_grad=True)
    p = Posterior(mu, lv, (12, 16), (12, 16))
    calls = []
    original = p.kl_per_image
    p.kl_per_image = lambda: (calls.append(1), original())[1]
    loss, terms = vae_loss(y, target, p, 0.1)
    assert len(calls) == 1
    nats = 0.5 * (mu.square() + lv.exp() - 1 - lv).sum((1, 2, 3))
    d = (y - target).square().sum((1, 2, 3)) / 192
    old = (d + 0.1 * nats / 192).mean()
    assert torch.equal(loss, old)
    assert torch.equal(terms['kl_nats_per_sample'], nats.mean())
    a = torch.autograd.grad(loss, (y, mu, lv), retain_graph=True)
    b = torch.autograd.grad(old, (y, mu, lv))
    assert all(torch.equal(x, z) for x, z in zip(a, b))


def test_penalties_preserve_true_patterns_and_detect_errors():
    x = torch.zeros(2, 3, 33, 41)
    x[..., ::2, ::2] = 1
    c, p = reconstruction_penalties(x, x)
    assert c == 0 and p == 0
    # Achromatic DC error must not be treated as color or phase imbalance.
    c, p = reconstruction_penalties(x + 0.1, x)
    assert c < 1e-12 and p < 1e-12
    error = torch.zeros_like(x)
    error[..., ::2, ::2] = 0.2
    error[..., 1::2, 1::2] = -0.2
    _, p = reconstruction_penalties(x + error, x)
    assert p == pytest.approx(0.02, abs=1e-6)
    colored = torch.tensor([0.8, 0.2, 0.1])[None, :, None, None].expand_as(x)
    gray = colored.mean(1, keepdim=True).expand_as(x)
    c, p = reconstruction_penalties(gray, colored)
    assert c > 0.01 and p < 1e-12
    for h, w in [(1, 3), (4, 5), (13, 19)]:
        c, p = reconstruction_penalties(x[..., :h, :w], x[..., :h, :w])
        assert torch.isfinite(c + p) and c + p == 0


def test_sampled_objective_gradient_and_invalid_weights():
    from pathwm.models.spatial_vae_v2 import HierarchicalVAE

    torch.manual_seed(3)
    m = HierarchicalVAE(stem_channels=4, channels=[8, 12], latent_channels=2)
    x = torch.rand(2, 3, 13, 19)
    y, p = m(x)
    loss, terms = vae_loss(y, x, p, 0.1, color_weight=6, phase_weight=20)
    loss.backward()
    for name in ['encoder.mu.weight', 'encoder.logvar.weight', 'decoder.output.weight']:
        grad = dict(m.named_parameters())[name].grad
        assert torch.isfinite(grad).all() and grad.abs().sum() > 0
    assert terms['color_weighted'] == 6 * terms['color_error']
    assert terms['phase_weighted'] == 20 * terms['phase_error']
    for value in [-1, float('nan'), float('inf')]:
        with pytest.raises(ValueError):
            vae_loss(y, x, p, color_weight=value)


def test_repair_resume_exact(tmp_path):
    import numpy as np
    from pathwm.data.images import Frames
    from experiments.spatial_vae import train, hierarchy_settings

    data = Frames(np.random.default_rng(7).integers(0, 256, (8, 12, 16, 3), dtype=np.uint8), range(8))
    cfg = dict(hierarchy_settings('C', 0.1), steps=4, batch_size=2, disk_free_gib=0,
               model_config=dict(stem_channels=4, channels=[8, 12], latent_channels=2),
               color_weight=6.0, phase_weight=20.0)
    train(tmp_path/'full', data, data, cfg)
    train(tmp_path/'split', data, data, cfg, stop_after=2)
    train(tmp_path/'split', data, data, cfg, resume=True)
    a, b = [torch.load(tmp_path/n/'last.pt', weights_only=True) for n in ['full', 'split']]
    assert all(torch.equal(v, b['model'][k]) for k, v in a['model'].items())
    import json
    rows = [json.loads(line) for line in (tmp_path/'full/metrics.jsonl').read_text().splitlines()]
    assert all('color_error' in r and 'phase_error' in r for r in rows if r['split']=='train')
