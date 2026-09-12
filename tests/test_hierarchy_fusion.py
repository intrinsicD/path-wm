import json
from types import SimpleNamespace

import numpy as np
import torch

from experiments.hierarchy_fusion import build_model, objective
from experiments.perception import build_model as perception_model
from pathwm.data.images import Frames
from pathwm.io import state_hash
from pathwm.training.perception import train_perception


def test_constructed_hierarchy_is_deterministic_active_and_reloadable(tmp_path):
    from experiments.hierarchy_fusion import construct_weights

    first, _ = build_model(7498, "deep_fusion")
    second, _ = build_model(7499, "deep_fusion")
    construct_weights(first)
    construct_weights(second)
    assert state_hash(first) == state_hash(second)
    images = torch.zeros(3, 3, 64, 64)
    for c in range(3):
        images[c, c] = 1
    with torch.no_grad():
        output = first(images)
        features = first.encoder(images)
    assert all(torch.isfinite(v).all() for v in output.values())
    colors = output["rgb"].mean((2, 3))
    assert torch.equal(colors.argmax(1), torch.arange(3))
    assert (colors.max(1).values - colors.min(1).values > 0.5).all()
    for block in first.encoder.encoder.pyramid.modules():
        if hasattr(block, "attention") and hasattr(block.attention, "out_proj"):
            assert block.attention.out_proj.weight.abs().sum() > 0
    # Fusion really contributes; this is not an identity-only circuit.
    for block in first.encoder.encoder.pyramid.fusion:
        torch.nn.init.zeros_(block.attention.out_proj.weight)
        torch.nn.init.zeros_(block.mlp[-1].weight)
    with torch.no_grad():
        changed = first.encoder(images)
    assert any(not torch.equal(features[k], changed[k]) for k in features)
    torch.save(second.state_dict(), tmp_path / "weights.pt")
    first.load_state_dict(torch.load(tmp_path / "weights.pt", weights_only=True), strict=True)
    with torch.no_grad():
        assert torch.equal(first(images)["rgb"], output["rgb"])


def test_constructed_readout_fit_only_changes_final_layers_without_optimization(monkeypatch):
    from experiments.hierarchy_fusion import construct_weights, fit_readouts

    def forbidden(*args, **kwargs):
        raise AssertionError("Direct construction must not optimize or backpropagate")

    monkeypatch.setattr(torch.Tensor, "backward", forbidden)
    monkeypatch.setattr(torch.optim.SGD, "step", forbidden)
    monkeypatch.setattr(torch.optim.AdamW, "step", forbidden)
    frames = np.random.default_rng(8881).integers(0, 256, (4, 64, 64, 3), dtype=np.uint8)
    data = Frames(frames, range(4), labels={
        "mask": (frames[:, :, :, 0] > 128)[:, None].astype("float32"),
        "valid": np.ones((4, 1, 64, 64), dtype="float32"),
    })
    model, _ = build_model(7498, "deep_fusion")
    construct_weights(model)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    receipt = fit_readouts(model, data, "cpu")
    changed = {k for k, v in model.state_dict().items() if not torch.equal(v, before[k])}
    assert changed == {f"heads.{name}.trunk.12.{kind}" for name in ("rgb", "mask") for kind in ("weight", "bias")}
    assert receipt["fitted_parameters"] == 132
    assert receipt["training_rows"] == data.rows.tolist()
    assert all(p.grad is None for p in model.parameters())


def test_comparison_copies_all_shared_weights_and_recipe_uses_every_scale():
    shallow, anchor_hash = build_model(7498, "shallow")
    fused, other_hash = build_model(7498, "deep_fusion")
    assert anchor_hash == other_hash == state_hash(shallow)
    other = fused.state_dict()
    for name, value in shallow.state_dict().items():
        assert torch.equal(value, other[name]), name
    args = SimpleNamespace(
        encoder="pyramid",
        width=16,
        levels=3,
        stage_depth=2,
        fusion_depth=2,
        encoder_weights=None,
        freeze_encoder=False,
        dataset="coco",
        masks=True,
        rgb_only=False,
        diagnostic_rgb=False,
    )
    model = perception_model(args)
    assert tuple(model.heads["mask"].input.spec) == tuple(model.encoder.feature_spec)


def test_fused_hierarchy_training_resumes_exactly(tmp_path):
    frames = np.random.default_rng(9).integers(0, 256, (4, 64, 64, 3), dtype=np.uint8)
    data = Frames(
        frames,
        range(4),
        labels={
            "mask": (frames[:, :, :, 0] > 128)[:, None].astype("float32"),
            "valid": np.ones((4, 1, 64, 64), dtype="float32"),
        },
    )

    def run(path, resume=False, stop=None):
        model, _ = build_model(7498, "deep_fusion")
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003)
        settings = dict(
            seed=7498, steps=3, batch_size=1, evaluate_every=3, grad_clip=1.0
        )
        train_perception(
            model,
            data,
            data,
            objective,
            optimizer,
            settings=settings,
            recipe=__file__,
            output=path,
            resume=resume,
            stop_after=stop,
        )
        return torch.load(path / "last.pt", map_location="cpu", weights_only=True)

    full = run(tmp_path / "full")
    run(tmp_path / "resumed", stop=1)
    resumed = run(tmp_path / "resumed", resume=True)

    def same(a, b):
        if isinstance(a, torch.Tensor):
            assert torch.equal(a, b)
        elif isinstance(a, dict):
            assert a.keys() == b.keys()
            for k in a:
                same(a[k], b[k])
        elif isinstance(a, (tuple, list)):
            assert len(a) == len(b)
            for x, y in zip(a, b):
                same(x, y)
        else:
            assert a == b

    for key in ("model", "optimizer", "sampler", "torch", "step"):
        same(full[key], resumed[key])
    assert (
        json.loads((tmp_path / "resumed/status.json").read_text())["report"]
        == "structural_verified"
    )
