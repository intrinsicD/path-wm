import torch
import pytest
from pathwm.models.encoders import CNNEncoder
from pathwm.models.decoders import ReconstructionDecoder, DenseHead
from pathwm.models.temporal import MemoryUpdater, Predictor, imagine


def test_cnn_and_decoder_have_visible_spatial_contract():
    encoder = CNNEncoder(depth=2)
    features = encoder(torch.zeros(2, 3, 64, 64))
    assert {k: tuple(v.shape) for k, v in features.items()} == {
        "fine": (2, 64, 16, 16),
        "coarse": (2, 64, 8, 8),
    }
    decoder = ReconstructionDecoder(encoder.feature_spec)
    assert decoder(features).shape == (2, 3, 64, 64)
    with pytest.raises(ValueError, match="RGB"):
        encoder(torch.zeros(2, 3, 63, 64))


def test_pyramid_spatial_adapter_and_dense_heads_train_through_fusion():
    from pathwm.models.encoders import PyramidEncoder

    encoder = PyramidEncoder(image_size=32, width=16, levels=3, depth=2, fusion_depth=2)
    features = encoder(torch.rand(2, 3, 32, 32))
    assert {k: tuple(v.shape) for k, v in features.items()} == {
        "scale_0": (2, 16, 8, 8), "scale_1": (2, 16, 4, 4), "scale_2": (2, 16, 2, 2)
    }
    head = DenseHead(encoder.feature_spec, channels=3, levels=tuple(features),
                     output_size=(32, 32), activation="sigmoid", retain_statistics=True)
    output = head(features)
    assert output.shape == (2, 3, 32, 32)
    output.square().mean().backward()
    assert encoder.encoder.stem.patch.weight.grad.abs().sum() > 0
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in encoder.parameters())
    assert encoder.encoder.pyramid.fusion[0].attention.in_proj_weight.grad.abs().sum() > 0


def test_decoder_uses_selected_scales_only():
    from pathwm.models.features import FeatureSpec

    spec = {"a": FeatureSpec(12, (8, 8), "test"), "b": FeatureSpec(20, (4, 4), "test")}
    head = DenseHead(spec, channels=1, levels=("a",), output_size=(32, 32))
    a = torch.randn(2, 12, 8, 8, requires_grad=True)
    b = torch.randn(2, 20, 4, 4, requires_grad=True)
    head({"a": a, "b": b}).sum().backward()
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


def test_diagnostic_head_cannot_update_encoder_but_task_head_can():
    from pathwm.models.perception import Perception

    e = CNNEncoder(width=16)
    rgb = ReconstructionDecoder(e.feature_spec)
    task = DenseHead(e.feature_spec, channels=1)
    model = Perception(e, {"rgb": rgb, "mask": task}, detached_heads=("rgb",))
    outputs = model(torch.rand(2, 3, 64, 64))
    outputs["rgb"].mean().backward()
    assert all(p.grad is None for p in e.parameters())
    assert any(p.grad is not None for p in rgb.parameters())
    outputs["mask"].square().mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in e.parameters())


def test_future_targets_cannot_change_rollout_and_action_timing_is_explicit():
    from pathwm.models.temporal import WorldModel, observe

    e = CNNEncoder(width=16).requires_grad_(False)
    u = MemoryUpdater(e.feature_spec, action_width=2, memory_width=32)
    p = Predictor(e.feature_spec, action_width=2, memory_width=32)
    with torch.no_grad():
        p.output_projection.weight.normal_(std=0.02)
    model = WorldModel(e, u, p).eval()
    history = torch.rand(1, 2, 3, 64, 64)
    past, actions, initial = (
        torch.rand(1, 1, 2),
        torch.rand(1, 2, 2),
        torch.full((1, 2), -1.0),
    )
    batch = dict(
        history=history,
        history_actions=past,
        actions=actions,
        initial_previous_action=initial,
        future=torch.rand(1, 2, 3, 64, 64),
    )

    def predict(batch):
        return model(
            batch["history"],
            batch["history_actions"],
            batch["actions"],
            batch["initial_previous_action"],
        )

    a = predict(batch)
    batch["future"] = torch.full_like(batch["future"], 1000.0)
    b = predict(batch)
    assert all(
        torch.equal(a[t][0][k], b[t][0][k]) for t in range(2) for k in e.feature_spec
    )
    with torch.no_grad():
        targets = e(batch["future"][:, 0])
    loss = sum((a[-1][0][k] - targets[k]).square().mean() for k in targets)
    loss.backward()
    assert all(v.grad_fn is None for v in targets.values())
    assert all(v.grad is None for v in e.parameters())
    seen = []
    handle = u.register_forward_pre_hook(lambda mod, args: seen.append(args[2].clone()))
    observe(e, u, history, past, initial)
    handle.remove()
    assert torch.equal(seen[0], initial) and torch.equal(seen[1], past[:, 0])
