import copy
import torch
import pytest
from torch.nn import functional as F
from pathwm.models.spatial_vae import (
    SpatialVAE,
    ReversibleMixer,
    vae_loss,
    build_variant,
)


def test_shuffle_and_coupling_are_invertible_without_calling_projection_lossless():
    x = torch.randn(2, 12, 5, 7, dtype=torch.float64)
    block = ReversibleMixer(12).double()
    for p in block.parameters():
        torch.nn.init.normal_(p, std=0.02)
    torch.testing.assert_close(block.inverse(block(x)), x, atol=1e-12, rtol=1e-12)
    image = torch.randn(2, 3, 16, 24)
    assert torch.equal(F.pixel_shuffle(F.pixel_unshuffle(image, 2), 2), image)


@pytest.mark.parametrize("size", [(1, 1), (15, 19), (32, 48)])
@pytest.mark.parametrize("variant", ["base", "attention", "reversible"])
def test_geometry_and_decoder_needs_only_z(size, variant):
    model = SpatialVAE(channels=(8, 16), latent_channels=3, variant=variant)
    x = torch.randn(2, 3, *size)
    p = model.encode(x)
    assert p.mu.shape == (2, 3, (size[0] + 3) // 4, (size[1] + 3) // 4)
    y = model.decode(p.mu, size)
    assert y.shape == x.shape
    saved = copy.deepcopy(model.decoder)
    del model, x
    assert torch.equal(saved(p.mu, size), y)
    with pytest.raises(ValueError):
        saved(p.mu, (size[0] + 4, size[1]))


def test_variants_share_initial_function_and_reversible_parameter_count():
    models = [build_variant(v, 17) for v in ["base", "attention", "reversible"]]
    x = torch.rand(2, 3, 16, 24)
    outputs = [m(x, sample=False)[0] for m in models]
    assert all(torch.equal(outputs[0], y) for y in outputs[1:])
    assert sum(p.numel() for p in models[1].parameters()) == sum(
        p.numel() for p in models[2].parameters()
    )


def test_sampling_loss_units_and_gradients():
    model = SpatialVAE(channels=(8, 16), latent_channels=3)
    x = torch.rand(2, 3, 15, 19)
    y, p = model(x, generator=torch.Generator().manual_seed(1))
    assert torch.equal(
        p.sample(torch.Generator().manual_seed(12)),
        p.sample(torch.Generator().manual_seed(12)),
    )
    assert not torch.equal(p.sample(torch.Generator().manual_seed(12)), p.mu)
    loss, d = vae_loss(y, x, p, beta=0.2)
    kl = (
        0.5 * (p.mu.square() + p.logvar.exp() - 1 - p.logvar).sum((1, 2, 3)) / (15 * 19)
    )
    expected = (y - x).square().sum((1, 2, 3)) / (15 * 19) + 0.2 * kl
    torch.testing.assert_close(loss, expected.mean())
    torch.testing.assert_close(d["rate"], kl.mean())
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
    assert model.encoder.mu.weight.grad.abs().sum() > 0
    assert model.encoder.logvar.weight.grad.abs().sum() > 0


def test_attention_fine_scale_dependency_and_batch_isolation():
    model = SpatialVAE(channels=(8, 16), latent_channels=3, variant="attention")
    torch.nn.init.normal_(model.encoder.attention.output.weight, std=0.1)
    x = torch.rand(2, 3, 16, 24)
    p = model.encode(x)
    altered = x.clone()
    altered[1] = torch.randn_like(altered[1])
    torch.testing.assert_close(p.mu[0], model.encode(altered).mu[0], atol=0, rtol=0)
    grids = [torch.randn(2, 8, 8, 12), torch.randn(2, 16, 4, 6)]
    out = model.encoder.attention(grids)
    changed = model.encoder.attention([grids[0] + 1, grids[1]])
    assert not torch.allclose(out, changed)


def test_checkpoint_has_strict_architecture_and_exact_outputs(tmp_path):
    m = build_variant("reversible", 19)
    p = tmp_path / "weights.pt"
    m.save(p)
    loaded = SpatialVAE.load(p)
    x = torch.rand(1, 3, 19, 23)
    assert torch.equal(m(x, sample=False)[0], loaded(x, sample=False)[0])


def test_recipe_pause_resume_preserves_noise_optimizer_and_sampler(tmp_path):
    import numpy as np
    from experiments.spatial_vae import settings, train
    from pathwm.data.images import Frames

    data = Frames(
        np.random.default_rng(7).integers(0, 256, (8, 16, 24, 3), dtype=np.uint8),
        range(8),
    )
    config = dict(
        settings("reversible"), steps=4, batch_size=2, disk_free_gib=0, wall_seconds=120
    )
    train(tmp_path / "full", data, data, config)
    train(tmp_path / "resume", data, data, config, stop_after=2)
    train(tmp_path / "resume", data, data, config, resume=True)
    a, b = [
        torch.load(tmp_path / name / "last.pt", weights_only=False)
        for name in ("full", "resume")
    ]

    def equal(left, right):
        if isinstance(left, torch.Tensor):
            assert torch.equal(left, right)
        elif isinstance(left, dict):
            assert left.keys() == right.keys()
            for key in left:
                equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            assert len(left) == len(right)
            for x, y in zip(left, right):
                equal(x, y)
        else:
            assert left == right

    for key in ("model", "optimizer", "sampler", "torch", "step"):
        equal(a[key], b[key])
    assert [r for r in a["rows"] if r["split"] == "train"] == [
        r for r in b["rows"] if r["split"] == "train"
    ]


def test_metrics_match_numpy_and_sampling_preserves_rng_and_batches():
    import numpy as np
    from experiments.spatial_vae import image_metrics, evaluate_images

    rng = np.random.default_rng(9)
    target = rng.random((4, 3, 15, 19)).astype("float32")
    raw = (rng.random(target.shape) * 1.2 - 0.1).astype("float32")
    p = np.clip(raw, 0, 1)
    metrics, _ = image_metrics(torch.from_numpy(raw), torch.from_numpy(target))
    assert metrics["mse"] == pytest.approx(np.mean((p - target) ** 2))
    assert metrics["raw_mse"] == pytest.approx(np.mean((raw - target) ** 2))
    edge = (
        sum(
            np.mean((np.diff(p, axis=axis) - np.diff(target, axis=axis)) ** 2)
            for axis in (-1, -2)
        )
        / 2
    )
    assert metrics["edge_mse"] == pytest.approx(edge)
    model = SpatialVAE(channels=(8, 16), latent_channels=3, variant="attention")
    state = torch.get_rng_state().clone()
    a, _, _ = evaluate_images(model, torch.from_numpy(target), batch_size=1)
    b, _, _ = evaluate_images(model, torch.from_numpy(target), batch_size=4)
    assert torch.equal(state, torch.get_rng_state())
    for name in a:
        torch.testing.assert_close(a[name], b[name], atol=1e-6, rtol=1e-5)


def test_native_crops_use_original_pixels_and_reject_changed_sources(tmp_path):
    import json
    import numpy as np
    from PIL import Image
    from experiments.spatial_vae import native_crops
    from pathwm.data.images import Frames
    from pathwm.io import file_hash

    raw = np.arange(17 * 23 * 3, dtype=np.uint8).reshape(17, 23, 3)
    image = tmp_path / "source.png"
    Image.fromarray(raw).save(image)
    manifest = dict(
        source=str(tmp_path),
        records=[
            dict(file=image.name, source_size=[23, 17], source_sha256=file_hash(image))
        ],
    )
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    frames = Frames(np.zeros((1, 64, 64, 3), dtype=np.uint8), [0])
    crop, _ = native_crops(tmp_path, frames, (9, 11), 1)
    expected = torch.from_numpy(raw[4:13, 6:17].copy()).permute(2, 0, 1).float() / 255
    assert torch.equal(crop[0], expected)
    image.write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash changed"):
        native_crops(tmp_path, frames, (9, 11), 1)
