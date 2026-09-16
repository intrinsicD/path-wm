import copy
import json

import pytest
import torch


def image_encoder():
    from pathwm.models.multiscale import MultiScaleImageEncoder

    return MultiScaleImageEncoder(24, code_width=8, levels=3)


def enable(encoder):
    from pathwm.models.multiscale import LayerReadout

    encoder.pyramid.layer_readout = LayerReadout(len(encoder.pyramid.stages))


def test_zero_gate_preserves_native_outputs_rng_and_existing_weights():
    from pathwm.models.modalities import Observation

    torch.manual_seed(7341)
    native = image_encoder()
    variant = copy.deepcopy(native)
    rng = torch.get_rng_state().clone()
    enable(variant)
    assert torch.equal(rng, torch.get_rng_state())
    for key, value in native.state_dict().items():
        assert torch.equal(value, variant.state_dict()[key])
    for h, w in ((16, 16), (24, 20)):
        x = Observation(torch.randn(2, 3, h, w), torch.zeros(2, 1))
        a, b = native(x), variant(x)
        for left, right in zip(a.scales, b.scales):
            assert torch.equal(left.values, right.values)
            assert torch.equal(left.times, right.times)
            assert torch.equal(left.valid, right.valid)


def test_nonzero_gate_changes_only_emitted_scale_not_coarser_propagation():
    from pathwm.models.modalities import Observation

    native = image_encoder()
    variant = copy.deepcopy(native)
    enable(variant)
    with torch.no_grad():
        variant.pyramid.layer_readout.gates[0] = 1
    x = Observation(torch.randn(2, 3, 16, 16), torch.zeros(2, 1))
    a, b = native(x), variant(x)
    assert not torch.equal(a.scales[0].values, b.scales[0].values)
    for left, right in zip(a.scales[1:], b.scales[1:]):
        assert torch.equal(left.values, right.values)


def test_gates_have_gradients_while_frozen_encoder_does_not():
    from pathwm.models.modalities import Observation

    encoder = image_encoder().requires_grad_(False)
    enable(encoder)
    x = Observation(torch.randn(2, 3, 16, 16), torch.zeros(2, 1))
    y = encoder(x)
    sum(s.values.square().mean() for s in y.scales).backward()
    assert (encoder.pyramid.layer_readout.gates.grad.abs() > 0).all()
    for name, parameter in encoder.named_parameters():
        if 'layer_readout.' not in name:
            assert parameter.grad is None


def test_temporal_readout_preserves_causality_and_invalid_values():
    from pathwm.models.multiscale import MultiScaleImageEncoder
    from pathwm.models.modalities import Observation

    encoder = MultiScaleImageEncoder(24, code_width=8, levels=3, video=True)
    enable(encoder)
    with torch.no_grad():
        encoder.pyramid.layer_readout.gates.fill_(0.7)
    frames = torch.randn(1, 4, 3, 16, 16)
    times = torch.arange(4.0)[None]
    valid = torch.tensor([[True, True, True, False]])
    a = encoder(Observation(frames, times, valid))
    changed = frames.clone()
    changed[:, 2] += 100
    changed[:, 3] = float('nan')
    b = encoder(Observation(changed, times, valid))
    for left, right in zip(a.scales, b.scales):
        earlier = (left.times < 2) & left.valid
        assert torch.equal(left.values[earlier], right.values[earlier])
        assert torch.isfinite(right.values).all()
        assert torch.equal(right.values[~right.valid], torch.zeros_like(right.values[~right.valid]))


def test_recipe_reload_retains_layer_variant_and_legacy_default(tmp_path):
    from experiments.modality_readout import Model, configure_encoder_readout, load_initial

    model = Model('native')
    configure_encoder_readout(model, 'layers')
    for e in model.core.agent.encoders.values():
        with torch.no_grad():
            e.pyramid.layer_readout.gates.fill_(0.6)
    torch.save({'model': model.state_dict()}, tmp_path / 'last.pt')
    (tmp_path / 'run.json').write_text(json.dumps({'identity': {'settings': {'encoder_readout': 'layers'}}}))
    restored = Model('native')
    load_initial(restored, tmp_path, torch.device('cpu'))
    assert set(model.state_dict()) == set(restored.state_dict())
    for name, value in model.state_dict().items():
        assert torch.equal(value, restored.state_dict()[name])
    old = Model('native')
    assert all(e.pyramid.layer_readout is None for e in old.core.agent.encoders.values())
    with pytest.raises(ValueError):
        configure_encoder_readout(old, 'unknown')
