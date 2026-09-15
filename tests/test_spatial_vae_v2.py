import torch
from torch import nn
import pytest

from pathwm.models.spatial_vae import SpatialVAE, vae_loss
from pathwm.models.spatial_vae_v2 import (
    HierarchicalVAE,
    LosslessDownsample2D,
    LosslessUpsample2D,
    LoopTransformer,
    EncoderStage,
    stride2_from_projection,
)


def test_exact_rearrangement_and_stride_control():
    x = torch.randn(2, 3, 8, 10, dtype=torch.float64)
    r = LosslessDownsample2D()
    assert torch.equal(LosslessUpsample2D()(r(x)), x)
    c = nn.Conv2d(12, 5, 1).double()
    assert torch.allclose(stride2_from_projection(c)(x), c(r(x)), atol=1e-12)
    with pytest.raises(ValueError):
        r(x[..., :-1])


@pytest.mark.parametrize(
    "ablation", ["A_local", "A_exact", "B", "C", "C_after", "D", "E"]
)
def test_geometry_backward_trace_export(tmp_path, ablation):
    m = HierarchicalVAE(
        stem_channels=4, channels=(8, 12), latent_channels=2, ablation=ablation
    )
    for h, w in [(1, 3), (13, 19), (16, 12)]:
        x = torch.rand(2, 3, h, w)
        y, p = m(x)
        assert y.shape == x.shape
        assert p.mu.shape == (2, 2, (h + 3) // 4, (w + 3) // 4)
        assert p.mu.shape == p.logvar.shape == p.sample().shape
        vae_loss(y, x, p)[0].backward()
        assert all(
            v.grad is not None and torch.isfinite(v.grad).all() for v in m.parameters()
        )
        m.zero_grad(set_to_none=True)
    p, trace = m.inspect(x)
    assert all(not v.requires_grad for v in trace.values())
    assert "stage_0.before_compression" in trace and "posterior.mu" in trace
    path = tmp_path / "v2.pt"
    m.save(path)
    restored = SpatialVAE.load(path)
    assert torch.equal(m(x, sample=False)[0], restored(x, sample=False)[0])
    old = SpatialVAE(channels=(4,), latent_channels=2)
    old.save(tmp_path / "old.pt")
    assert torch.equal(
        old(x, sample=False)[0],
        SpatialVAE.load(tmp_path / "old.pt")(x, sample=False)[0],
    )


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
    with pytest.raises(ValueError, match="token"):
        loop(torch.rand(1, 8, 5, 5))


def test_identity_contract():
    stage = EncoderStage(4, None, downsample=False, depth=0)
    x = torch.randn(2, 4, 5, 7)
    assert stage.out_channels == 4 and stage.factor == 1
    assert torch.equal(stage(x), x)
    with pytest.raises(ValueError):
        EncoderStage(4, 7, downsample=False)


def test_probe_controls_and_frozen_model():
    from pathwm.evaluation.spatial_vae import probe_readout, collect_stage_samples
    from pathwm.io import seed_everything

    seed_everything(57101)
    x = torch.randn(128, 4)
    y = x @ torch.randn(4, 7)
    held = torch.randn(32, 4)
    target = held @ torch.linalg.lstsq(x, y).solution
    score = probe_readout(x, y, held, target)
    assert score["linear"]["normalized_mse"] < 0.001
    assert score["shuffled_linear"]["mse"] > 100 * score["linear"]["mse"]
    degenerate = probe_readout(x, torch.ones(128, 2), held, torch.ones(32, 2))
    assert degenerate["degenerate"] and degenerate["linear"]["normalized_mse"] is None
    m = HierarchicalVAE()
    saved = {k: v.clone() for k, v in m.state_dict().items()}
    rng = torch.get_rng_state().clone()
    images = torch.linspace(0, 1, 4 * 3 * 12 * 16).reshape(4, 3, 12, 16)
    rows, geometry = collect_stage_samples(m, images, cells_per_image=4)
    assert m.training and torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(saved[k], v) for k, v in m.state_dict().items())
    assert all(not v.requires_grad for stage in rows.values() for v in stage.values())
    assert rows["stage_1"]["rgb"].shape == (16, 3 * 4 * 4)
    assert geometry["stage_1"]["after_compression"][1] == 24


def test_profile_and_new_recipe_resume(tmp_path):
    import numpy as np
    from experiments.spatial_vae import settings, train
    from pathwm.data.images import Frames
    from pathwm.evaluation.spatial_vae import profile_codec
    from pathwm.models.spatial_vae_v2 import build_hierarchy
    from pathwm.io import seed_everything

    seed_everything(57)
    x = torch.rand(2, 3, 12, 16)
    m = build_hierarchy("D", 57, loop_iterations=0)
    first = profile_codec(m, x, repeats=1)
    m.encoder.stages[-1].mixer.iterations = 2
    second = profile_codec(m, x, repeats=1)
    assert first["parameters"] == second["parameters"]
    assert first["active_parameters"] < second["active_parameters"]
    assert first["forward_macs"] < second["forward_macs"]
    data = Frames(
        np.random.default_rng(7).integers(0, 256, (8, 12, 16, 3), dtype=np.uint8),
        range(8),
    )
    config = dict(
        settings("E"),
        model_config=dict(stem_channels=4, channels=[8, 12], latent_channels=2),
        steps=4,
        batch_size=2,
        disk_free_gib=0,
        wall_seconds=120,
    )
    train(tmp_path / "full", data, data, config)
    train(tmp_path / "resumed", data, data, config, stop_after=2)
    train(tmp_path / "resumed", data, data, config, resume=True)
    a, b = [
        torch.load(tmp_path / n / "last.pt", weights_only=False)
        for n in ["full", "resumed"]
    ]

    def equal(x, y):
        if isinstance(x, torch.Tensor):
            assert torch.equal(x, y)
        elif isinstance(x, dict):
            assert x.keys() == y.keys()
            for k in x:
                equal(x[k], y[k])
        elif isinstance(x, (list, tuple)):
            assert len(x) == len(y)
            for left, right in zip(x, y):
                equal(left, right)
        else:
            assert x == y

    for key in ["model", "optimizer", "sampler", "torch", "step"]:
        equal(a[key], b[key])


def test_common_initialization_padding_and_identity_hierarchy():
    from pathwm.models.spatial_vae_v2 import build_hierarchy

    models = [build_hierarchy(v, 57101) for v in ["B", "C", "C_after", "A_exact"]]
    x = torch.rand(2, 3, 13, 19)
    expected = models[0](x, sample=False)[0]
    for m in models[1:]:
        torch.testing.assert_close(
            m(x, sample=False)[0], expected, atol=1e-6, rtol=1e-6
        )
        for key, value in models[0].decoder.state_dict().items():
            assert torch.equal(value, m.decoder.state_dict()[key])
    p, trace = models[1].inspect(x)
    assert p.original_size == (13, 19) and p.padded_size == (16, 20)
    padded = trace["input.padded"]
    assert torch.equal(padded[..., :13, :19], x)
    assert torch.equal(padded[..., -1, -1], x[..., -1, -1])
    restored = LosslessUpsample2D()(LosslessDownsample2D()(padded))
    assert torch.equal(restored[..., :13, :19], x)
    identity = HierarchicalVAE(
        stem_channels=4, channels=(None, 8), downsample=(False, True), depth=0
    )
    assert identity.encoder.factor == 2 and identity.encoder.widths == [4, 4, 8]
    assert identity(x)[0].shape == x.shape


def test_evaluation_rejects_wrong_architecture_label(tmp_path):
    import json
    from experiments.spatial_vae import hierarchy_evaluate, hierarchy_settings

    weights = tmp_path / "model.pt"
    HierarchicalVAE(ablation="C").save(weights)
    output = tmp_path / "wrong_label"
    with pytest.raises(ValueError, match="variant must match"):
        hierarchy_evaluate(output, weights, None, None, hierarchy_settings("E"), "cpu")
    status = json.loads((output / "status.json").read_text())
    assert status["result"] == "failed"
    assert not (output / "report.html").exists()
