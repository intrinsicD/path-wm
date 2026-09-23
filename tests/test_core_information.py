import torch
from experiments.core_information import make_batch, counterfactual, PropertyReader


def test_counterfactual_changes_only_target_factor():
    b = make_batch(10, 8, heldout=True)
    c = counterfactual(b, 'color')
    m = counterfactual(b, 'motion')
    rows = torch.arange(8)
    expected = b['scenes'].attrs.clone()
    expected[rows, b['target'], 0] = (expected[rows, b['target'], 0]+1)%4
    assert torch.equal(c['scenes'].attrs, expected)
    assert torch.equal(c['scenes'].object_xy, b['scenes'].object_xy)
    assert torch.equal(m['scenes'].attrs, b['scenes'].attrs)
    assert torch.equal(m['xy'][:,1], b['xy'][:,1]+3)
    assert ((b['scenes'].attrs[:,:,0]+b['scenes'].attrs[:,:,1])%4==0).all()
    train = make_batch(10, 8, heldout=False)
    assert ((train['scenes'].attrs[:,:,0]+train['scenes'].attrs[:,:,1])%4!=0).all()


def test_freeze_allows_only_intended_gradients():
    for adapt in (False, True):
        model = PropertyReader(adapt=adapt)
        b = make_batch(12, 2)
        logits, _ = model(b['rgb'], b['xy'])
        logits.square().mean().backward()
        assert all(p.grad is None for p in model.perception.decoder.parameters())
        assert all(p.grad is None for p in model.perception.slot_attention.parameters())
        enc = list(model.perception.encoder.parameters())
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in enc) == adapt
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.core.block.parameters())


def test_constant_answers_cannot_pass_pair_screen():
    from experiments.core_information import paired_metrics, accuracy_screen
    logits=torch.tensor([[10.,0,0,0]]).expand(8,-1)
    target=torch.arange(8)%4
    m=paired_metrics(logits,logits,logits,target,(target+1)%4)
    assert m['color_pair']==0
    assert not accuracy_screen(m)


def test_probe_owns_no_encoder_gradients_or_rng_changes():
    from experiments.core_information import fit_probe
    x=torch.randn(16,8,requires_grad=True);y=torch.arange(16)%4
    before=torch.get_rng_state().clone()
    p,h=fit_probe(x,y,2)
    assert torch.equal(before,torch.get_rng_state())
    assert x.grad is None
    assert torch.isfinite(p(x)).all()
