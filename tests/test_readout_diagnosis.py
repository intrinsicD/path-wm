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
