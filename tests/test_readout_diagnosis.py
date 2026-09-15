import copy

import numpy as np
import torch


def test_stage_capture_preserves_forward_gradient_and_rng():
    from experiments.modality_readout import Core
    from pathwm.data.modality_readout import dataset, observations
    from pathwm.evaluation.modality_readout import capture_readout_stages

    torch.manual_seed(31)
    core = Core()
    twin = copy.deepcopy(core)
    inputs = observations(dataset("train"), "all", [0, 12])
    rng = torch.get_rng_state().clone()
    expected = core(inputs)
    expected.square().mean().backward()
    after = torch.get_rng_state().clone()
    torch.set_rng_state(rng)
    actual, stages = capture_readout_stages(twin, inputs)
    actual.square().mean().backward()
    assert torch.equal(expected, actual)
    assert torch.equal(after, torch.get_rng_state())
    assert set(stages) == {"encoder", "posterior", "codes", "observed", "thought"}
    assert all(not x.requires_grad for x in stages.values())
    torch.testing.assert_close(
        stages["posterior"].reshape(2, 4, 8).sum(-1), torch.ones(2, 4)
    )
    for p, q in zip(core.parameters(), twin.parameters()):
        assert (p.grad is None) == (q.grad is None)
        if p.grad is not None:
            assert torch.equal(p.grad, q.grad)


def test_probe_recovers_facts_and_uses_training_statistics():
    from pathwm.evaluation.modality_readout import (
        fit_factor_probe,
        predict_factor_probe,
    )

    rng = np.random.default_rng(12)
    labels = np.array([[a, b, c] for a in range(3) for b in range(3) for c in range(2)])
    facts = np.concatenate(
        [np.eye(n)[labels[:, i]] for i, n in enumerate((3, 3, 2))], 1
    )
    train = np.repeat(facts, 8, axis=0)
    y = np.repeat(labels, 8, axis=0)
    train = np.concatenate([train, rng.normal(size=(len(train), 2))], 1)
    validation = np.concatenate([facts, rng.normal(size=(len(facts), 2))], 1)
    reader, choices = fit_factor_probe(train, y, validation, labels)
    assert np.array_equal(predict_factor_probe(reader, validation), labels)
    np.testing.assert_allclose(reader["mean"], train.mean(0))
    assert len(choices) == 5
    assert np.isfinite(reader["weights"]).all()


def test_posterior_auxiliary_is_training_only_and_preserves_initial_rng():
    from experiments.modality_readout import Model
    from pathwm.data.modality_readout import dataset, observations

    torch.manual_seed(61)
    base = Model("native")
    rng = torch.get_rng_state().clone()
    torch.manual_seed(61)
    aux = Model("native", posterior_aux=True)
    assert torch.equal(torch.get_rng_state(), rng)
    for n, p in base.state_dict().items():
        assert torch.equal(p, aux.state_dict()[n])
    x = observations(dataset("train"), "image", [0, 12])
    rng = torch.get_rng_state().clone()
    tokens = base.core(x)
    torch.set_rng_state(rng)
    other, state = aux.core(x, return_state=True)
    assert torch.equal(tokens, other)
    logits = aux.posterior_head(state.logits.softmax(-1).flatten(1))
    logits.square().mean().backward()
    assert aux.core.agent.updater.head.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in aux.outputs.parameters())


def test_variable_text_diagnostic_coordinates_align_without_changing_model():
    from experiments.modality_readout import Core
    from pathwm.data.modality_readout import dataset, observations
    from pathwm.evaluation.modality_readout import capture_readout_stages

    core = Core().eval()
    with torch.no_grad():
        _, known = capture_readout_stages(
            core, observations(dataset("train"), "all", [0])
        )
        _, unseen = capture_readout_stages(
            core, observations(dataset("heldout"), "all", [0])
        )
    assert known["encoder"].shape == unseen["encoder"].shape


def test_raw_capture_and_unit_temperature_preserve_native_path():
    from experiments.modality_readout import Core
    from pathwm.data.modality_readout import dataset, observations

    torch.manual_seed(81)
    core = Core()
    x = observations(dataset("train"), "all", [0, 12])
    rng = torch.get_rng_state().clone()
    native = core(x)
    torch.set_rng_state(rng)
    captured = {}
    output, state = core(
        x, return_state=True, posterior_features=captured, temperature=1.0
    )
    assert torch.equal(native, output)
    assert captured["raw_logits"].requires_grad
    raw = captured["raw_logits"].reshape(2, 4, 8)
    from pathwm.models.belief import distribution

    torch.testing.assert_close(distribution(raw), state.logits, rtol=0, atol=0)
    assert not core.agent.updater.head._forward_hooks
    torch.set_rng_state(rng)
    _, softened = core(x, return_state=True, temperature=10.0)
    assert not torch.equal(state.logits, softened.logits)
    assert torch.all((softened.stochastic == 0) | (softened.stochastic == 1))
    assert not core.agent.updater.head._forward_hooks


def test_encoder_only_initialization_preserves_fresh_core_and_freezes(tmp_path):
    from experiments.modality_readout import Model, load_encoders

    torch.manual_seed(77)
    source = Model("native")
    torch.save({"model": source.state_dict()}, tmp_path / "last.pt")
    torch.manual_seed(78)
    fresh = Model("native")
    before = {k: v.clone() for k, v in fresh.state_dict().items()}
    rng = torch.get_rng_state().clone()
    load_encoders(fresh, tmp_path, torch.device("cpu"))
    assert torch.equal(rng, torch.get_rng_state())
    for k, value in fresh.state_dict().items():
        expected = (
            source.state_dict()[k]
            if k.startswith("core.agent.encoders.")
            else before[k]
        )
        assert torch.equal(value, expected)
    assert all(not p.requires_grad for p in fresh.core.agent.encoders.parameters())


def test_factor_audit_does_not_change_training_and_keeps_direction_coefficient():
    from experiments.modality_readout import (
        Core,
        factor_objective,
        factor_gradient_audit,
    )
    from pathwm.data.modality_readout import dataset, observations

    torch.manual_seed(81)
    core = Core()
    data = dataset("train")
    inputs = observations(data, "all", [0, 12, 24])
    wanted = data["targets"]["factors"][[0, 12, 24]]
    tokens = core(inputs)
    predictions = core.factors(tokens)
    loss, terms = factor_objective(predictions, wanted, "all")
    direction, _ = factor_objective(predictions, wanted, "direction")
    assert torch.equal(direction, terms[2] / 3)
    expected = (
        sum(
            torch.nn.functional.cross_entropy(p, wanted[:, i])
            for i, p in enumerate(predictions)
        )
        / 3
    )
    assert torch.equal(loss, expected)
    params = list(core.agent.updater.parameters())
    expected_grad = torch.autograd.grad(loss, params, retain_graph=True)
    rng = torch.get_rng_state().clone()
    audit = factor_gradient_audit(terms, params)
    assert torch.equal(rng, torch.get_rng_state())
    assert all(p.grad is None for p in core.parameters())
    actual_grad = torch.autograd.grad(loss, params)
    assert all(torch.equal(a, b) for a, b in zip(expected_grad, actual_grad))
    assert len(audit) == 6 and all(np.isfinite(v) for v in audit.values())
    for name, value in audit.items():
        assert -1.00001 <= value <= 1.00001 if "cosine" in name else value >= 0


def test_continuous_working_readout_preserves_stored_state_and_sampling_rng():
    from experiments.modality_readout import Core
    from pathwm.data.modality_readout import dataset, observations

    torch.manual_seed(83)
    hard = Core(belief_readout="sampled")
    soft = copy.deepcopy(hard)
    soft.belief_readout = "probabilities"
    inputs = observations(dataset("train"), "all", [0, 12, 24])
    rng = torch.get_rng_state().clone()
    hard_tokens, hard_state = hard(inputs, return_state=True)
    after = torch.get_rng_state().clone()
    torch.set_rng_state(rng)
    soft_tokens, soft_state = soft(inputs, return_state=True)
    assert torch.equal(after, torch.get_rng_state())
    for name in ("logits", "z", "stochastic", "h", "tokens", "evidence"):
        assert torch.equal(getattr(hard_state, name), getattr(soft_state, name))
    assert torch.equal(soft_state.stochastic, torch.nn.functional.one_hot(soft_state.z, 8).float())
    soft.agent.validate_state(soft_state)
    assert not torch.equal(hard_tokens, soft_tokens)
    soft_tokens.square().mean().backward()
    assert soft.agent.updater.head.weight.grad.abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in soft.parameters() if p.grad is not None)
    assert sum(p.numel() for p in hard.parameters()) == sum(p.numel() for p in soft.parameters())
