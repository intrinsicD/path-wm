import numpy as np
import pytest
import torch
from pathwm.data.memory_output import MemoryOutputEpisodes


def test_brightness_variants_keep_complete_histories_targets_and_source():
    from copy import deepcopy
    d = MemoryOutputEpisodes(16, seed=65, curriculum="relocation")
    original = deepcopy(d.identity)
    offsets = [-16, -8, 0, 8, 16]
    a = d.with_input_offsets(offsets)
    control = d.with_input_offsets([0] * 5)
    assert len(a) == 5 * len(d) == len(control)
    assert d.identity == original
    for i, off in enumerate(offsets):
        sl = slice(i*len(d), (i+1)*len(d))
        assert np.array_equal(a.images[sl].astype('int16')-off, d.images)
        assert np.array_equal(control.images[sl], d.images)
        assert np.array_equal(a.labels[sl], d.labels)
        assert np.array_equal(a.targets[sl], d.targets)
    assert a.identity['input_augmentation']['source_data'] == original
    assert a.identity['input_augmentation']['offsets_uint8'] == offsets
    assert a.identity['pairs'] == 5 * original['pairs']
    assert a.shortcut_audit()['hidden_frame_only_joint_accuracy'] == .25
    assert np.array_equal(a.labels, control.labels)


@pytest.mark.parametrize('offsets', [[], [True], [1.5], [21], [-31]])
def test_brightness_variants_reject_invalid_or_clipping(offsets):
    d = MemoryOutputEpisodes(16, seed=65, curriculum='relocation')
    with pytest.raises(ValueError, match='offset'):
        d.with_input_offsets(offsets)


def test_augmented_cache_live_alignment_and_frozen_gradient_contract():
    from experiments.memory_output import cache_readout, readout_objective
    from pathwm.models.memory_output import configure_output_readout, frozen_tensors
    from tests.test_memory_output import model_fixture
    m = configure_output_readout(model_fixture(True), 'native').eval()
    d = MemoryOutputEpisodes(16, seed=66, curriculum='relocation').with_input_offsets([-16,0,16])
    cache = cache_readout(m,d,'cpu',mode='mixed')
    ids = [64,0,33,95]
    b=d.batch(ids)
    with torch.no_grad():
        h=m.observe_history(b['images'])
        for mode, key in [('ordinary','ordinary_tokens'),('reset','tokens')]:
            assert torch.equal(cache[key][ids], m.working(m.query(h['final'],b['images'][:,-1],mode)))
    assert torch.equal(cache['target'][ids],b['target'])
    fixed={k:v.clone() for k,v in frozen_tensors(m).items()}
    loss,_=readout_objective(m,cache,ids,context='mixed',step=1)
    loss.backward()
    assert all(p.grad is None for p in m.parameters() if not p.requires_grad)
    torch.optim.AdamW([p for p in m.parameters() if p.requires_grad]).step()
    assert all(torch.equal(v,fixed[k]) for k,v in frozen_tensors(m).items())


def test_median_centering_is_per_frame_preserves_metadata_and_rejects_clipping():
    from pathwm.models.memory_output import PixelMedianCentering
    from pathwm.models.modalities import Observation
    class Capture(torch.nn.Module):
        code_width=16
        def forward(self, obs, **kw): return obs
    m=PixelMedianCentering(Capture(),[40/255]*3)
    x=MemoryOutputEpisodes(16, seed=67, curriculum='relocation').batch(range(2))['images']
    times=torch.arange(3).float()[None].expand(2,-1)
    valid=torch.ones_like(times,dtype=torch.bool)
    a=m(Observation(x,times,valid)); before=x.clone()
    b=m(Observation(x+12/255,times,valid))
    torch.testing.assert_close(a.values,b.values,atol=1e-7,rtol=0)
    y=x.clone();y[1,2]=.5
    assert torch.equal(a.values[0],m(Observation(y,times,valid)).values[0])
    assert torch.equal(a.values[1,:2],m(Observation(y,times,valid)).values[1,:2])
    assert torch.equal(a.times,times) and torch.equal(a.valid,valid) and torch.equal(x,before)
    invalid=x.clone();invalid[0,1]=float('nan'); mask=valid.clone();mask[0,1]=False
    assert torch.isfinite(m(Observation(invalid,times,mask)).values).all()
    bad=torch.zeros_like(x);bad[...,0,0]=1
    with pytest.raises(ValueError,match='range'):
        m(Observation(bad,times,valid))
    with pytest.raises(ValueError,match='reference'):
        PixelMedianCentering(Capture(),[float('nan')]*3)
