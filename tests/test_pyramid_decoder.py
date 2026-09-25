"""Native full-config decoder connection contracts; no quality claim from these checks."""
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from experiments.latent_agent import perception_batch
from pathwm.data import rule_world as rw
from pathwm.io import load_component, state_hash
from pathwm.models.slots import SlotPerception

PARENT = Path(__file__).resolve().parents[1] / 'runs/real_visual_joint_repair_3501_u6000_v1/last.pt'


@pytest.fixture
def native():
    model = SlotPerception().eval()
    load_component(model, PARENT, 'perception')
    _, _, rgb, _, _ = perception_batch(torch.Generator().manual_seed(3602), rw.KIND_SPLIT['train'], 2, 'cpu')
    with torch.no_grad():
        pyramid = model.pyramid(rgb)
        percept = model.from_pyramid(pyramid)
    return model, pyramid, percept


def test_enable_preserves_native_output_weights_and_rng(native):
    model, pyramid, before = native
    old = {k:v.clone() for k,v in model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    model.decoder.enable_pyramid_connections()
    assert torch.equal(rng, torch.get_rng_state())
    for key,value in old.items():
        assert torch.equal(value, model.state_dict()[key]), key
    with torch.no_grad():
        after = model.from_pyramid(pyramid)
    assert torch.equal(before.recon, after.recon)
    assert torch.equal(before.alpha, after.alpha)
    assert torch.equal(before.slots, after.slots)


def test_enabled_decoder_requires_pyramid_and_rejects_bad_geometry(native):
    model, pyramid, percept = native
    model.decoder.enable_pyramid_connections()
    with pytest.raises(ValueError):
        model.decoder(percept.slots)
    fine, coarse = pyramid.scales
    wrong = replace(pyramid, scales=(replace(fine, grid=(1,8,32)), coarse))
    with pytest.raises(ValueError):
        model.decoder(percept.slots, wrong)


def test_native_decoder_training_freezes_source_and_reads_features(native):
    model, pyramid, percept = native
    model.requires_grad_(False)
    model.decoder.enable_pyramid_connections()
    model.decoder.requires_grad_(True)
    old_source = state_hash(model.encoder), state_hash(model.slot_attention), state_hash(model.heads)
    original_features = [s.values.clone() for s in pyramid.scales]
    optimizer = torch.optim.AdamW(model.decoder.parameters(), lr=3e-4)
    # Actual pretrained RGB reconstruction is a fixed target; a perturbation ensures
    # a nonzero update. This tests gradient routing, not learned reconstruction quality.
    target = percept.recon.detach() * .9
    for _ in range(2):
        colors, alpha = model.decoder(percept.slots.detach(), pyramid)
        prediction = (alpha.softmax(1)[:,:,None] * colors).sum(1)
        optimizer.zero_grad()
        (prediction-target).square().mean().backward()
        optimizer.step()
    assert old_source == (state_hash(model.encoder), state_hash(model.slot_attention), state_hash(model.heads))
    assert all(p.grad is None for p in model.encoder.parameters())
    for saved,scale in zip(original_features,pyramid.scales):
        assert torch.equal(saved,scale.values)
    zeros = replace(pyramid, scales=tuple(replace(s,values=torch.zeros_like(s.values)) for s in pyramid.scales))
    with torch.no_grad():
        actual = model.decoder(percept.slots, pyramid)[0]
        erased = model.decoder(percept.slots, zeros)[0]
    assert not torch.equal(actual,erased)


@pytest.mark.parametrize("fault", ["batch", "nonfinite", "invalid", "mask_shape", "mask_dtype", "missing_scale"])
def test_native_decoder_rejects_invalid_feature_inputs(native, fault):
    model, pyramid, percept = native
    model.decoder.enable_pyramid_connections()
    fine, coarse = pyramid.scales
    if fault == "batch":
        fine = replace(fine, values=fine.values[:1])
    elif fault == "nonfinite":
        values = fine.values.clone()
        values[0, 0, 0] = float("nan")
        fine = replace(fine, values=values)
    elif fault == "mask_shape":
        fine = replace(fine, valid=torch.tensor(True))
    elif fault == "mask_dtype":
        fine = replace(fine, valid=fine.valid.float())
    elif fault == "invalid":
        valid = fine.valid.clone()
        valid[0, 0] = False
        fine = replace(fine, valid=valid)
    broken = replace(pyramid, scales=(fine,) if fault == "missing_scale" else (fine, coarse))
    with pytest.raises(ValueError):
        model.decoder(percept.slots, broken)


def test_decoder_objective_weight_and_eval_preserve_training_state(native):
    from experiments.latent_agent import decoder_objective, decoder_mask_eval
    from pathwm.io import evaluation_mode
    model,pyramid,percept=native
    _,_,rgb,entity,_=perception_batch(torch.Generator().manual_seed(3602),rw.KIND_SPLIT['train'],2,'cpu')
    a,_=decoder_objective(percept,rgb,entity,0.)
    b,metrics=decoder_objective(percept,rgb,entity,.5)
    assert torch.equal(a,(percept.recon-rgb).square().mean())
    assert abs(float(b-a)-.5*metrics['mask'])<1e-7
    rng=torch.get_rng_state().clone();before=state_hash(model)
    with evaluation_mode(model):result=decoder_mask_eval(model,model,'cpu',2)
    assert torch.equal(rng,torch.get_rng_state()) and state_hash(model)==before
    assert all(v['mean']['pointer_agreement']==1 for v in result.values())


def test_full_resolution_fine_connection_preserves_native_initialization(native):
    model,pyramid,_=native
    model.decoder.enable_pyramid_connections()
    with torch.no_grad():before=model.from_pyramid(pyramid)
    old={k:v.clone() for k,v in model.state_dict().items()}
    rng=torch.get_rng_state().clone()
    model.decoder.enable_fine_subpixels()
    assert torch.equal(rng,torch.get_rng_state())
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in old.items())
    with torch.no_grad():after=model.from_pyramid(pyramid)
    assert torch.equal(before.recon,after.recon) and torch.equal(before.alpha,after.alpha)
    model.requires_grad_(False);model.decoder.fine_subpixel.requires_grad_(True)
    target=before.recon.detach().clone();target[:,:,::4,::4]*=.9
    loss=(model.from_pyramid(pyramid).recon-target).square().mean()
    loss.backward()
    assert model.decoder.fine_subpixel.weight.grad.abs().sum()>0
    assert all(p.grad is None for p in model.encoder.parameters())
