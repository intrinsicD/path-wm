"""Native full-size code replay: identity, model binding and no RGB dependency."""
from copy import deepcopy
from dataclasses import replace
import pytest
import torch
from pathwm.models.slots import SlotPerception


def model():
    torch.manual_seed(812)
    p = SlotPerception().eval()
    p.decoder.enable_pyramid_connections()
    return p


def test_full_native_code_roundtrip_and_continuation(tmp_path):
    from pathwm.models.image_code import make_image_code, code_pyramid, decode_image_code, save_image_code, load_image_code
    p = model()
    x = torch.rand(2, 3, 64, 64)
    with torch.no_grad():
        direct = p.pyramid(x)
        code = make_image_code(p, direct, provenance=[dict(kind='observed', id='frame-a', available_at=3.), dict(kind='generated', id='sample-b', available_at=4.)])
        save_image_code(tmp_path/'code.pt', code)
        loaded = load_image_code(tmp_path/'code.pt')
        continued = code_pyramid(p, loaded)
        for a,b in zip(direct.scales, continued.scales):
            assert a.grid == b.grid
            for f in ('values','times','content_times','valid','ends'):
                assert torch.equal(getattr(a,f),getattr(b,f))
        expected = p.from_pyramid(direct)
        actual = decode_image_code(p, loaded)
        for f in vars(actual): assert torch.equal(getattr(actual,f),getattr(expected,f))
    assert 'rgb' not in code and 'coarse' not in code and 'slots' not in code
    assert loaded['provenance']==code['provenance']
    assert code['fine']['values'].shape==(2,256,64)


def test_replay_rejects_incompatible_weights_and_metadata():
    from pathwm.models.image_code import make_image_code, code_pyramid
    p = model()
    with torch.no_grad():
        py = p.pyramid(torch.rand(1,3,64,64))
        c = make_image_code(p,py,provenance=[dict(kind='generated',id='test',available_at=0.)])
    for field,value in [('grid',(1,8,32)),('valid',torch.zeros(1,256,dtype=torch.bool)),('times',torch.ones(1,256)),('values',torch.full((1,256,64),float('nan')))]:
        bad=deepcopy(c);bad['fine'][field]=value
        with pytest.raises(ValueError):code_pyramid(p,bad)
    bad=deepcopy(c);bad['condition'].fill_(1)
    with pytest.raises(ValueError):code_pyramid(p,bad)
    with torch.no_grad():p.decoder.network[-1].bias.add_(1)
    code_pyramid(p,c)  # a retrained consumer may use the same encoder codes
    with torch.no_grad():next(p.encoder.parameters()).add_(1)
    with pytest.raises(ValueError,match='weights'):code_pyramid(p,c)


def test_continuation_rejects_changed_export_and_preserves_gradient():
    p=model()
    py=p.pyramid(torch.rand(1,3,64,64))
    fine=replace(py.scales[0],values=py.scales[0].values.detach().requires_grad_())
    out=p.encoder.pyramid.continue_from_fine(fine,torch.zeros(1,16))
    out.scales[-1].values.square().mean().backward()
    assert fine.values.grad is not None and fine.values.grad.abs().sum()>0
    p.encoder.pyramid.layer_readout=torch.nn.Identity()
    with pytest.raises(ValueError,match='readout'):p.encoder.pyramid.continue_from_fine(fine,torch.zeros(1,16))


def test_slot_only_memory_perception_can_export_code():
    from pathwm.models.image_code import make_image_code, decode_image_code
    p=SlotPerception().eval()
    with torch.no_grad():
        pyramid=p.pyramid(torch.rand(1,3,64,64))
        code=make_image_code(p,pyramid,provenance=[dict(kind='observed',id='memory',available_at=1.)])
        assert torch.equal(decode_image_code(p,code).recon,p.from_pyramid(pyramid).recon)
