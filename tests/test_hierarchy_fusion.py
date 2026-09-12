import json
from types import SimpleNamespace

import numpy as np
import torch
import pytest

from experiments.hierarchy_fusion import build_model, objective
from experiments.perception import build_model as perception_model
from pathwm.data.images import Frames
from pathwm.io import state_hash
from pathwm.training.perception import train_perception


def handwritten_file(tmp_path):
    from experiments.hierarchy_fusion import construct_weights
    model, _ = build_model(7599, "deep_fusion")
    construct_weights(model)
    path = tmp_path / "handwritten.pt"
    torch.save(dict(schema="pathwm-hierarchy-weights-v1", method="constructed", model=model.state_dict()), path)
    return path, state_hash(model)


@pytest.mark.parametrize("part", ["both", "encoder", "decoder"])
def test_handwritten_training_freezes_exact_modules_and_preserves_gradient_path(tmp_path, part):
    from experiments.hierarchy_fusion import configure_training, rgb_objective
    from pathwm.training.perception import step
    path, expected = handwritten_file(tmp_path)
    model, _ = build_model(7598, "deep_fusion")
    info = configure_training(model, initial_weights=path, train_part=part, loss_mode="rgb")
    assert state_hash(model) == expected == info["initial_model_sha256"]
    modules = dict(encoder=model.encoder, rgb=model.heads["rgb"], mask=model.heads["mask"])
    before = {k: state_hash(m) for k,m in modules.items()}
    batch = {"rgb": torch.rand(2,3,64,64)}
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.0003)
    step(model,batch,rgb_objective,optimizer)
    for name,module in modules.items():
        expected_change = name != "mask" and (part == "both" or (part == "encoder" and name == "encoder") or (part == "decoder" and name == "rgb"))
        assert (state_hash(module) != before[name]) == expected_change
        assert any(p.grad is not None and bool(p.grad.abs().sum()) for p in module.parameters()) == expected_change


def test_handwritten_features_collapse_equal_patch_means(tmp_path):
    from experiments.hierarchy_fusion import configure_training, information_probe
    path,_ = handwritten_file(tmp_path)
    model,_ = build_model(7598,"deep_fusion")
    configure_training(model,initial_weights=path,loss_mode="rgb")
    probe = information_probe(model)
    assert probe["input_pair_mse"] > 0.3
    assert probe["patch_projection_rank"] == 3
    assert all(v == 0 for v in probe["scale_pair_max_difference"].values())
    assert probe["rgb_pair_max_difference"] == 0


def test_opening_dormant_decoder_branch_keeps_output_and_enables_weight_gradient(tmp_path):
    from experiments.hierarchy_fusion import configure_training, rgb_objective
    path,_ = handwritten_file(tmp_path)
    plain,_=build_model(7598,"deep_fusion")
    opened,_=build_model(7598,"deep_fusion")
    configure_training(plain,initial_weights=path,loss_mode="rgb")
    configure_training(opened,initial_weights=path,loss_mode="rgb",open_residual_branches=True,seed=7598)
    batch={"rgb":torch.rand(2,3,64,64)}
    assert torch.equal(plain(batch["rgb"])["rgb"],opened(batch["rgb"])["rgb"])
    assert state_hash(plain.heads["mask"]) == state_hash(opened.heads["mask"])
    for model in (plain,opened):
        sum(rgb_objective(model(batch["rgb"]),batch).values()).backward()
    assert plain.heads["rgb"].trunk[1].net[5].weight.grad.count_nonzero() == 0
    assert opened.heads["rgb"].trunk[1].net[5].weight.grad.abs().sum() > 0


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
    first.load_state_dict(
        torch.load(tmp_path / "weights.pt", weights_only=True), strict=True
    )
    with torch.no_grad():
        assert torch.equal(first(images)["rgb"], output["rgb"])


def test_constructed_readout_fit_only_changes_final_layers_without_optimization(
    monkeypatch,
):
    from experiments.hierarchy_fusion import construct_weights, fit_readouts

    def forbidden(*args, **kwargs):
        raise AssertionError("Direct construction must not optimize or backpropagate")

    monkeypatch.setattr(torch.Tensor, "backward", forbidden)
    monkeypatch.setattr(torch.optim.SGD, "step", forbidden)
    monkeypatch.setattr(torch.optim.AdamW, "step", forbidden)
    frames = np.random.default_rng(8881).integers(
        0, 256, (4, 64, 64, 3), dtype=np.uint8
    )
    data = Frames(
        frames,
        range(4),
        labels={
            "mask": (frames[:, :, :, 0] > 128)[:, None].astype("float32"),
            "valid": np.ones((4, 1, 64, 64), dtype="float32"),
        },
    )
    model, _ = build_model(7498, "deep_fusion")
    construct_weights(model)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    receipt = fit_readouts(model, data, "cpu")
    changed = {
        k for k, v in model.state_dict().items() if not torch.equal(v, before[k])
    }
    assert changed == {
        f"heads.{name}.trunk.12.{kind}"
        for name in ("rgb", "mask")
        for kind in ("weight", "bias")
    }
    assert receipt["fitted_parameters"] == 132
    assert receipt["training_rows"] == data.rows.tolist()
    assert all(p.grad is None for p in model.parameters())


def test_direct_hierarchy_run_and_resume_do_not_refit_or_rescore(tmp_path, monkeypatch):
    import experiments.hierarchy_fusion as recipe
    from pathwm.io import file_hash

    frames = np.random.default_rng(8882).integers(
        0, 256, (12, 64, 64, 3), dtype=np.uint8
    )
    labels = dict(
        mask=(frames[:, :, :, 0] > 128)[:, None].astype("float32"),
        valid=np.ones((12, 1, 64, 64), dtype="float32"),
    )
    data = {
        name: Frames(
            frames,
            range(start, start + 4),
            labels={k: v[start : start + 4] for k, v in labels.items()},
        )
        for name, start in [("train", 0), ("validation", 4), ("test", 8)]
    }
    settings = dict(
        weight_method="ridge",
        seed=7498,
        arm="deep_fusion",
        width=32,
        levels=3,
        stage_depth=2,
        fusion_depth=2,
        steps=0,
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("Unexpected fitting, scoring or optimization")

    monkeypatch.setattr(torch.Tensor, "backward", forbidden)
    monkeypatch.setattr(torch.optim.SGD, "step", forbidden)
    model, _ = build_model(7498, "deep_fusion")
    output = tmp_path / "direct"
    recipe.direct_weights(model, data, settings, output, "cpu")
    record = json.loads((output / "direct_weights.json").read_text())
    assert record["fit"]["training_rows"] == [0, 1, 2, 3]
    checkpoint = torch.load(output / "last.pt", weights_only=True)
    assert checkpoint["step"] == 0 and not checkpoint["optimizer"]["state"]
    monkeypatch.setattr(recipe, "fit_readouts", forbidden)
    monkeypatch.setattr(recipe, "score", forbidden)
    other, _ = build_model(7498, "deep_fusion")
    recipe.direct_weights(other, data, settings, output, "cpu", resume=True)
    for name, expected in {**record["files"], **record["report_files"]}.items():
        assert file_hash(output / name) == expected
    assert state_hash(model) == state_hash(other)
    (output / "test_outputs.npz").write_bytes(b"damaged")
    import pytest

    with pytest.raises(ValueError, match="Cached direct-weight file changed"):
        recipe.direct_weights(other, data, settings, output, "cpu", resume=True)


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


def test_handwritten_frozen_training_resumes_exactly(tmp_path):
    from experiments.hierarchy_fusion import configure_training, rgb_objective
    path,_=handwritten_file(tmp_path)
    frames=np.random.default_rng(8891).integers(0,256,(4,64,64,3),dtype=np.uint8)
    data=Frames(frames,range(4),labels={"mask":np.zeros((4,1,64,64),dtype="float32"),"valid":np.ones((4,1,64,64),dtype="float32")})
    def run(output,resume=False,stop=None):
        model,_=build_model(7598,"deep_fusion")
        config=configure_training(model,initial_weights=path,train_part="decoder",loss_mode="rgb",open_residual_branches=True,seed=7598)
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=0.0003)
        settings=dict(seed=7598,steps=3,batch_size=1,evaluate_every=3,grad_clip=1.0,initialization=config)
        train_perception(model,data,data,rgb_objective,optimizer,settings=settings,recipe=__file__,output=output,resume=resume,stop_after=stop)
        return torch.load(output/"last.pt",weights_only=True)
    complete=run(tmp_path/"complete")
    run(tmp_path/"resume",stop=1)
    resumed=run(tmp_path/"resume",resume=True)
    assert complete["step"] == resumed["step"] == 3
    for k,v in complete["model"].items(): assert torch.equal(v,resumed["model"][k])
    for k,v in complete["optimizer"]["state"].items():
        for field,value in v.items(): assert torch.equal(value,resumed["optimizer"]["state"][k][field])
    assert torch.equal(complete["sampler"],resumed["sampler"])
    assert torch.equal(complete["torch"],resumed["torch"])
