import copy
import pytest
import torch
from world_model.paddle.models import Encoder
from world_model.paddle.types import ObservationLatent
from world_model.curriculum.encoder_variants import ExperimentalEncoder, matched_models
from world_model.curriculum.pose_diagnostic import IndependentPoseHead
from world_model.curriculum.dino_reference import DinoAdapter, preprocess


def test_variant_preserves_legacy_reference_exactly_and_common_initialization():
    torch.set_num_threads(1)
    m=matched_models(7107,0,True)
    base=Encoder();base.load_state_dict(m['E'].state_dict())
    x=torch.rand(2,3,64,64)
    torch.testing.assert_close(base(x).tokens(),m['E'](x).tokens(),rtol=0,atol=0)
    deep=matched_models(7107,2,False)
    for name in ('E','D','H'):
        for key,value in m[name].state_dict().items():
            assert torch.equal(value,deep[name].state_dict()[key]),(name,key)
    assert sum(p.numel() for p in deep['E'].parameters())-sum(p.numel() for p in m['E'].parameters())==295936


def test_branch_blocks_cannot_change_other_branch_before_exchange():
    torch.set_num_threads(1)
    model=ExperimentalEncoder(2,False)
    other=copy.deepcopy(model)
    with torch.no_grad():
        for p in other.fine_blocks.parameters():p.add_(.2)
    x=torch.rand(2,3,64,64)
    a,b=model(x),other(x)
    torch.testing.assert_close(a.coarse,b.coarse,rtol=0,atol=0)
    assert not torch.equal(a.fine,b.fine)
    a.tokens().square().mean().backward()
    assert all(p.grad is None for p in model.fine_from_coarse.parameters())
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.fine_mlp.parameters())


def test_independent_orientation_does_not_use_position_map_parameters():
    torch.set_num_threads(1)
    h=IndependentPoseHead()
    z=ObservationLatent(torch.randn(2,256,64),torch.randn(2,64,64))
    h(z)[:,4:].square().sum().backward()
    assert all(p.grad is None or not p.grad.any() for p in h.position.parameters())
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in h.orientation.parameters())


def test_dino_adapter_keeps_grid_order_and_pooling_gradients():
    torch.set_num_threads(1)
    a=DinoAdapter();x=torch.zeros(2,256,384,requires_grad=True)
    with torch.no_grad():
        a.projection.weight.zero_();a.projection.bias.zero_();a.projection.weight[0,0]=1
        x[:,:,0]=torch.arange(256)
    z=a(x)
    torch.testing.assert_close(z.fine[:,:,0],x[:,:,0])
    expected=torch.tensor([[8.5,10.5],[40.5,42.5]])
    torch.testing.assert_close(z.coarse[0,:,0].reshape(8,8)[:2,:2],expected)
    z.tokens().sum().backward();assert x.grad is not None
    # A constant input checks normalization independently; no classification crop.
    rgb=torch.ones(1,3,64,64)
    expected=(torch.ones(3)-torch.tensor([.485,.456,.406]))/torch.tensor([.229,.224,.225])
    p=preprocess(rgb);assert p.shape==(1,3,224,224)
    torch.testing.assert_close(p[0,:,0,0],expected)
    torch.testing.assert_close(p[0,:,-1,-1],expected)


def test_exchange_off_equals_zero_attention_outputs_with_same_token_mlps():
    torch.set_num_threads(1)
    on=matched_models(7107,2,True)['E'];off=matched_models(7107,2,False)['E']
    with torch.no_grad():
        for attention in (on.fine_from_coarse,on.coarse_from_fine):
            attention.output_projection.weight.zero_();attention.output_projection.bias.zero_()
    x=torch.rand(2,3,64,64)
    torch.testing.assert_close(on(x).tokens(),off(x).tokens(),rtol=0,atol=0)


def test_frozen_dino_never_enters_training_and_preserves_state(tmp_path,monkeypatch):
    from world_model.curriculum.dino_reference import FrozenDino
    class Stub(torch.nn.Module):
        def __init__(self):
            super().__init__();self.proj=torch.nn.Conv2d(3,384,14,stride=14)
        def forward_features(self,x):
            return {'x_norm_patchtokens':self.proj(x).flatten(2).transpose(1,2)}
    stub=Stub();torch.save(stub.state_dict(),tmp_path/'weights.pt')
    monkeypatch.setattr(torch.hub,'load',lambda *args,**kwargs:Stub())
    model=FrozenDino(tmp_path,tmp_path/'weights.pt');model.train()
    before=copy.deepcopy(model.state_dict());x=torch.rand(2,3,64,64,requires_grad=True)
    adapter=DinoAdapter();adapter(model(x)).tokens().square().mean().backward()
    assert all(not m.training for m in model.modules())
    assert all(p.grad is None and not p.requires_grad for p in model.parameters())
    assert x.grad is None
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items())


def test_variant_resume_preserves_trajectory_and_rejects_changed_architecture(tmp_path):
    import numpy as np
    from world_model.curriculum.data import FrameSet
    from world_model.curriculum.training import train_phase
    from world_model.pusht.checkpoints import read_checkpoint
    torch.set_num_threads(1)
    rng=np.random.default_rng(27)
    ds=FrameSet(rng.integers(0,256,(4,64,64,3),dtype=np.uint8),np.arange(4),rng.random((4,6),dtype=np.float32),fingerprint='tiny-test')
    cfg=dict(seed=7107,phase='supervised',encoder_variant=dict(depth=2,exchange=False),updates=2,batch_size=2,microbatch=2,
             learning_rate=.0003,weight_decay=.0001,grad_clip=1.,validate_every=1,max_seconds=30,device='cpu',cpu_threads=1)
    full=tmp_path/'full';resumed=tmp_path/'resumed'
    train_phase(cfg,ds,ds,[0,1],full)
    train_phase(cfg,ds,ds,[0,1],resumed,stop_after=1)
    changed=copy.deepcopy(cfg);changed['encoder_variant']['exchange']=True
    with pytest.raises(ValueError,match='mismatch'):train_phase(changed,ds,ds,[0,1],resumed,resume=True)
    train_phase(cfg,ds,ds,[0,1],resumed,resume=True)
    a,b=read_checkpoint(full/'last.pt'),read_checkpoint(resumed/'last.pt')
    assert a['model_fingerprint']==b['model_fingerprint']
    assert a['metrics']==b['metrics']
