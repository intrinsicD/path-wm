"""Conditional output: field algebra, masked context, local RNG and frozen codec."""

from copy import deepcopy
import pytest
import torch
from torch import nn
from pathwm.models.features import FeatureSpec


def generator(kind="flow"):
    from pathwm.models.conditional_image import ConditionalFeatureGenerator

    spec = {
        "fine": FeatureSpec(3, (2, 2), "test"),
        "coarse": FeatureSpec(3, (1, 1), "test"),
    }

    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.ones(()), requires_grad=False)

        def forward(self, values):
            return self.scale * values["fine"]

    return ConditionalFeatureGenerator(
        8, spec, Head(), depth=1, fusion_depth=1, objective=kind, steps=4
    )


def test_flow_algebra_and_integration_reference():
    from pathwm.models.conditional_image import flow_pair, integrate

    target = {"x": torch.tensor([[[[4.0]]], [[[8.0]]]])}
    noise = {"x": torch.tensor([[[[2.0]]], [[[0.0]]]])}
    t = torch.tensor([0.0, 1.0])
    x, v = flow_pair(target, noise, t)
    assert torch.equal(x["x"], torch.tensor([[[[2.0]]], [[[8.0]]]]))
    assert torch.equal(v["x"], target["x"] - noise["x"])
    calls = []

    def field(x, tau):
        calls.append(tau)
        return v

    result = integrate(field, noise, 4)
    assert calls == [0.0, 0.25, 0.5, 0.75]
    assert torch.equal(result["x"], target["x"])
    assert torch.equal(noise["x"], torch.tensor([[[[2.0]]], [[[0.0]]]]))
    with pytest.raises(ValueError):
        integrate(field, noise, 0)


def test_masks_cross_scale_context_gradients_and_frozen_head():
    torch.manual_seed(13)
    g = generator("direct")
    context = torch.randn(2, 3, 8, requires_grad=True)
    valid = torch.tensor([[1, 1, 0], [1, 1, 0]], dtype=torch.bool)
    poisoned = context.detach().clone()
    poisoned[:, 2] = float("nan")
    with torch.no_grad():
        a = g.features(context, valid=valid)
        b = g.features(poisoned, valid=valid)
    assert all(torch.equal(a[k], b[k]) for k in a)
    loss = g.head(g.features(context, valid=valid)).square().mean()
    loss.backward()
    assert (
        context.grad[:, :2].abs().sum() > 0 and context.grad[:, 2].count_nonzero() == 0
    )
    assert g.head.scale.grad is None
    assert all(p.grad is not None for p in g.fusion.parameters())
    with pytest.raises(ValueError, match="context"):
        g.features(context, valid=torch.zeros_like(valid))
    changed = context.detach().clone()
    changed[:, 0, 0] += 2
    with torch.no_grad():
        assert not torch.equal(g.features(changed)["fine"], g.features(context)["fine"])


def test_sampling_is_local_reproducible_and_sample_ids_preserve_partition():
    torch.manual_seed(14)
    g = generator()
    c = torch.randn(3, 2, 8)
    rng = torch.get_rng_state().clone()
    before = {k: v.clone() for k, v in g.state_dict().items()}
    with torch.no_grad():
        a = g.features(c, seed=71, sample_ids=[4, 4, 9])
        b = g.features(c, seed=71, sample_ids=[4, 4, 9])
        parts = [
            g.features(c[i : i + 1], seed=71, sample_ids=[s])
            for i, s in enumerate([4, 4, 9])
        ]
        d = g.features(c, seed=72, sample_ids=[4, 4, 9])
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(a[k], b[k]) for k in a)
    assert all(
        torch.allclose(a[k], torch.cat([p[k] for p in parts]), atol=1e-6, rtol=1e-6)
        for k in a
    )
    assert any(not torch.equal(a[k], d[k]) for k in a)
    assert all(torch.equal(v, g.state_dict()[k]) for k, v in before.items())
    with pytest.raises(ValueError):
        g.features(c, sample_ids=[1])


def test_calibration_and_equal_architecture_objectives():
    torch.manual_seed(15)
    a = generator()
    torch.manual_seed(15)
    b = generator("direct")
    assert all(torch.equal(v, b.state_dict()[k]) for k, v in a.state_dict().items())
    target = {k: torch.randn(4, s.channels, *s.size) for k, s in a.feature_spec.items()}
    a.calibrate(target)
    z = a.standardize(target)
    assert all(
        torch.allclose(a.unstandardize(z)[k], v, atol=1e-6) for k, v in target.items()
    )
    malformed = deepcopy(target)
    malformed["fine"][0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError):
        a.calibrate(malformed)


def test_training_resume_standalone_export_and_target_exclusion(tmp_path):
    from tests.test_tint_readout import centered_export
    from tests.test_runs import equal_tree
    from pathwm.models.memory_output import load_model, frozen_tensors
    from experiments.conditional_image import build, settings, train, evaluate
    from pathwm.io import seed_everything

    seed_everything(21)
    _, data, source_settings = centered_export(tmp_path)
    source = tmp_path / "weights.pt"
    config = dict(source_settings, **settings(seed=21))
    config.update(
        steps=4,
        batch_size=2,
        wall_seconds=120,
        disk_free_gib=0,
        zero_progress_probability=0.5,
        decoded_image_weight=10.0,
        decoded_image_path="sample",
    )
    config["feature_generator"]["steps"] = 2

    def run(name, resume=False, stop_after=None):
        model, cache = build(source, data, config)
        assert all(c % 2 == s for c, s, _ in cache["labels"].tolist())
        frozen = {k: v.clone() for k, v in frozen_tensors(model).items()}
        train(
            model,
            cache,
            data,
            data,
            output=tmp_path / name,
            config=config,
            resume=resume,
            stop_after=stop_after,
        )
        assert all(torch.equal(v, frozen_tensors(model)[k]) for k, v in frozen.items())
        return model, torch.load(tmp_path / name / "last.pt", weights_only=True)

    full, a = run("full")
    run("resume", stop_after=1)
    _, b = run("resume", resume=True)
    for k in ["model", "optimizer", "torch", "sampler", "step"]:
        equal_tree(a[k], b[k])
    loaded = load_model(tmp_path / "full/weights.pt")
    before = evaluate(full, data, "cpu")
    after = evaluate(loaded, data, "cpu")
    for k in before:
        assert torch.equal(before[k], after[k])
    g = loaded.agent.decoders["image"]
    tokens = torch.randn(2, 8, loaded.agent.width)
    # Inference through the new output module has no encoder access.
    loaded.teacher.forward = lambda *_: pytest.fail("target encoder used in generation")
    loaded.agent.encoders["image"].forward = lambda *_: pytest.fail(
        "observation encoder used in generation"
    )
    with torch.no_grad():
        assert torch.isfinite(g(tokens)).all()
    bad = deepcopy(config)
    bad["decoded_image_path"] = "endpoint"
    m, c = build(source, data, bad)
    with pytest.raises(ValueError, match="Incompatible resume"):
        train(m, c, data, data, output=tmp_path / "resume", config=bad, resume=True)


def test_generator_width_and_paired_noise_are_independent_of_context_width():
    from pathwm.models.conditional_image import ConditionalFeatureGenerator

    torch.manual_seed(25)
    base = generator()
    g = ConditionalFeatureGenerator(
        8, base.feature_spec, base.head, hidden_width=16, steps=2
    )
    context = torch.randn(2, 3, 8)
    seen = []
    original = g.field

    def capture(x, t, c, valid=None):
        if t == 0:
            seen.append({k: v.clone() for k, v in x.items()})
        return original(x, t, c, valid)

    g.field = capture
    with torch.no_grad():
        g.features(context, sample_ids=[7, 7])
    assert all(torch.equal(v[0], v[1]) for v in seen[0].values())
    x = {k: torch.zeros(2, s.channels, *s.size) for k, s in g.feature_spec.items()}
    result = original(x, 0.5, context)
    sum(v.square().mean() for v in result.values()).backward()
    assert g.context_projection.weight.grad.abs().sum() > 0


def test_swapped_scores_use_counterfactual_targets_and_composition_groups():
    from pathwm.data.memory_output import MemoryOutputEpisodes
    from experiments.conditional_image import score, MODES

    b = MemoryOutputEpisodes(16, seed=37, curriculum="relocation").batch(range(32))
    a = dict(target=b["target"], labels=b["labels"], teacher_image=b["target"])
    for mode in MODES:
        ids = torch.arange(32) ^ 1 if mode == "reset_swapped" else torch.arange(32)
        a[mode + "_image"] = b["target"][ids]
        a[mode + "_facts"] = b["labels"][ids]
    metrics = score(a)
    assert metrics["reset_swapped/seen"]["weighted_mse"] == 0
    assert metrics["reset_swapped/unseen"]["image_accuracy"] == 1


def test_zero_progress_mixture_has_no_extra_rng_and_keeps_zero_policy_exact():
    from experiments.conditional_image import progress_mixture

    draw = torch.tensor([0.0, 0.1, 0.49, 0.5, 0.75, 0.99])
    rng = torch.get_rng_state().clone()
    assert progress_mixture(draw, 0) is draw
    actual = progress_mixture(draw, 0.5)
    assert torch.equal(actual, torch.tensor([0.0, 0.0, 0.0, 0.0, 0.5, 0.98]))
    assert torch.equal(rng, torch.get_rng_state())
    for p in [-0.1, 1.0, float("nan")]:
        with pytest.raises(ValueError):
            progress_mixture(draw, p)


@pytest.mark.parametrize("kind", ["flow", "direct"])
def test_decoded_supervision_endpoint_and_frozen_gradient_reference(kind):
    from experiments.conditional_image import objective

    torch.manual_seed(71)
    g = generator(kind)
    features = {
        k: torch.randn(4, s.channels, *s.size) for k, s in g.feature_spec.items()
    }
    g.calibrate(features)
    cache = dict(
        ordinary=torch.randn(4, 3, 8),
        reset=torch.randn(4, 3, 8),
        features=features,
        target=torch.rand(4, 3, 2, 2),
    )
    captured = {}
    original = g.field

    def capture(x, t, context):
        out = original(x, t, context)
        captured.update(x=x, t=t, prediction=out)
        return out

    g.field = capture
    torch.manual_seed(73)
    base, base_metrics = objective(g, cache, [0, 1, 2, 3], 1, 0.5)
    base_grad = torch.autograd.grad(base, g.output["fine"].weight)[0]
    torch.manual_seed(73)
    loss, metrics = objective(g, cache, [0, 1, 2, 3], 1, 0.5, decoded_image_weight=10)
    endpoint = (
        captured["prediction"]
        if kind == "direct"
        else {
            k: x + (1 - captured["t"][:, None, None, None]) * captured["prediction"][k]
            for k, x in captured["x"].items()
        }
    )
    pixels = g.head(g.unstandardize(endpoint))
    target = cache["target"]
    w = 1 + 9 * ((target - 40 / 255).abs().amax(1, keepdim=True) > 0.01)
    expected = (
        ((pixels - target).square() * w).sum((1, 2, 3)) / (3 * w.sum((1, 2, 3)))
    ).mean()
    assert torch.allclose(loss, base + 10 * expected)
    assert metrics["decoded_image_loss"] == pytest.approx(float(expected.detach()))
    assert metrics["latent_loss"] == base_metrics["latent_loss"]
    pixel_grad = torch.autograd.grad(
        expected, g.output["fine"].weight, retain_graph=True
    )[0]
    loss.backward()
    assert torch.allclose(
        g.output["fine"].weight.grad, base_grad + 10 * pixel_grad, atol=1e-5
    )
    assert pixel_grad.abs().sum() > 0
    assert g.context_norm.weight.grad.abs().sum() > 0
    assert g.head.scale.grad is None and not g.head.scale.requires_grad


def test_zero_decoded_weight_bypasses_head_and_rejects_invalid_before_rng():
    from experiments.conditional_image import objective

    torch.manual_seed(79)
    g = generator()
    cache = dict(
        ordinary=torch.randn(2, 3, 8),
        reset=torch.randn(2, 3, 8),
        features={
            k: torch.randn(2, s.channels, *s.size) for k, s in g.feature_spec.items()
        },
    )
    g.head.forward = lambda *_: pytest.fail("zero weight called image head")
    rng = torch.get_rng_state().clone()
    a, am = objective(g, cache, [0, 1], 1, 0.5)
    after = torch.get_rng_state().clone()
    torch.set_rng_state(rng)
    b, bm = objective(g, cache, [0, 1], 1, 0.5, decoded_image_weight=0)
    assert torch.equal(a, b) and am == bm and torch.equal(after, torch.get_rng_state())
    for weight in [-1, float("nan"), float("inf")]:
        with pytest.raises(ValueError, match="Decoded-image weight"):
            objective(g, cache, [0, 1], 1, 0.5, decoded_image_weight=weight)
        assert torch.equal(after, torch.get_rng_state())


def test_sampled_image_supervision_matches_full_manual_unroll_gradient():
    from experiments.conditional_image import objective, weighted_error

    torch.manual_seed(83)
    g = generator()
    cache = dict(
        ordinary=torch.randn(2, 3, 8),
        reset=torch.randn(2, 3, 8),
        target=torch.rand(2, 3, 2, 2),
        features={
            k: torch.randn(2, s.channels, *s.size) for k, s in g.feature_spec.items()
        },
    )
    params = [p for p in g.parameters() if p.requires_grad]
    torch.manual_seed(89)
    latent, _ = objective(g, cache, [0, 1], 1, 0.5)
    base = torch.autograd.grad(latent, params)
    rng_after = torch.get_rng_state().clone()
    torch.manual_seed(89)
    torch.rand(2)  # existing progress draw
    x = {k: torch.randn_like(v) for k, v in cache["features"].items()}
    context = torch.stack([cache["reset"][0], cache["ordinary"][1]])
    for i in range(g.steps):
        v = g.field(x, i / g.steps, context)
        x = {k: x[k] + v[k] / g.steps for k in x}
    reference = weighted_error(g.head(g.unstandardize(x)), cache["target"]).mean()
    gradients = torch.autograd.grad(reference, params)
    torch.manual_seed(89)
    actual, metrics = objective(
        g, cache, [0, 1], 1, 0.5, decoded_image_weight=10, decoded_image_path="sample"
    )
    actual.backward()
    assert torch.allclose(actual, latent + 10 * reference)
    assert metrics["decoded_image_loss"] == pytest.approx(float(reference.detach()))
    assert torch.equal(rng_after, torch.get_rng_state())
    for p, a, b in zip(params, base, gradients):
        assert torch.allclose(p.grad, a + 10 * b, atol=2e-5, rtol=1e-5)
    assert all(p.grad is None for p in g.head.parameters())
    with pytest.raises(ValueError, match="Decoded-image path"):
        objective(g, cache, [0, 1], 1, decoded_image_path="unknown")
