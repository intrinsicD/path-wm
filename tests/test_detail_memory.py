"""Contracts for learned detail codec; training quality lives in the recipe."""
import pytest
import torch
from pathwm.models.detail_memory import DetailCodec, to_parts, from_parts, detail_loss, replace_details


def test_rearrangement_is_exact_and_encoder_cannot_read_missing_pixels():
    x=torch.rand(2,3,16,16)
    assert torch.equal(from_parts(to_parts(x)),x)
    model=DetailCodec(width=16,hidden=32)
    visible=torch.tensor([[True,False,True,False],[False,True,False,True]])
    parts=to_parts(x); contaminated=parts.clone(); contaminated[~visible]=float('nan')
    a=model.encode(parts,visible); b=model.encode(contaminated,visible)
    assert torch.equal(a,b) and torch.equal(a[~visible],torch.zeros_like(a[~visible]))
    with pytest.raises(ValueError): model.encode(parts,visible.float())


def test_replacement_preserves_old_parts_and_training_gradients():
    model=DetailCodec(width=16,hidden=32)
    parts=torch.rand(2,4,192); visible=torch.tensor([[1,0,0,0],[0,0,1,0]],dtype=torch.bool)
    codes=model.encode(parts,visible)
    old=torch.randn_like(codes); seen=torch.ones(2,4,dtype=torch.bool)
    merged,present=replace_details(old,seen,codes,visible)
    assert torch.equal(merged[~visible],old[~visible]) and present.all()
    mu,lv=model.decode(merged,present,torch.tensor([0,3]))
    loss=detail_loss(mu,lv,torch.rand_like(mu)); loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.encoder.parameters())
    assert mu.shape==lv.shape==(2,3,16,16) and torch.isfinite(loss)


def test_missing_codes_do_not_influence_output_and_requests_are_checked():
    model=DetailCodec(width=16,hidden=32).eval()
    z=torch.randn(2,4,16); present=torch.tensor([[1,0,1,0],[0,0,0,0]],dtype=torch.bool)
    pose=torch.tensor([1,2]); changed=z.clone(); changed[~present]=float('nan')
    for a,b in zip(model.decode(z,present,pose),model.decode(changed,present,pose)):
        assert torch.equal(a,b)
    for bad in [torch.tensor([0,4]),torch.tensor([0.,1.]),torch.tensor([0])]:
        with pytest.raises(ValueError): model.decode(z,present,bad)
    with pytest.raises(ValueError): model.decode(z[:,:,:15],present,pose)
