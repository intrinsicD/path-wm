from dataclasses import replace

import pytest
import torch

from tests.test_memory_output import model_fixture
from pathwm.data.memory_output import MemoryOutputEpisodes


def test_latest_stored_working_selection_is_causal_and_order_invariant():
    from pathwm.models.memory_output import configure_output_readout

    m=configure_output_readout(model_fixture(True), 'stored').eval()
    x=MemoryOutputEpisodes(16, seed=41, curriculum='relocation').batch(range(4))['images']
    h=m.observe_history(x);s=h['final'];bank=s.memory;original=bank.values.clone()
    q=m.query(s,x[:,-1],'reset')
    assert torch.equal(m.working(q),m.working(h['stored']))
    reverse=replace(s,memory=replace(bank,keys=bank.keys.flip(1),values=bank.values.flip(1),times=bank.times.flip(1),sources=bank.sources[::-1]))
    assert torch.equal(m.working(m.query(reverse,x[:,-1],'reset')),m.working(q))
    times=bank.times.clone();times[1]=times[1].flip(0)
    mixed=m.query(replace(s,memory=replace(bank,times=times)),x[:,-1],'reset')
    assert torch.equal(m.working(mixed)[1],m.working(h['initial'])[1])
    tied=m.query(s,x[:,-1],'reset_time_erased')
    assert torch.equal(m.working(tied),(m.working(h['initial'])+m.working(h['stored']))/2)
    assert torch.count_nonzero(m.working(m.query(s,x[:,-1],'reset_erased')))==0
    for times in (bank.times+4,torch.full_like(bank.times,float('nan'))):
        with pytest.raises(ValueError):m.query(replace(s,memory=replace(bank,times=times)),x[:,-1],'reset')
    assert torch.equal(original,bank.values)


@pytest.mark.parametrize('stage',['native','stored'])
def test_readout_cache_matches_live_loss_gradients_and_only_heads_learn(stage):
    from pathwm.models.memory_output import configure_output_readout, frozen_tensors
    from experiments.memory_output import cache_readout, readout_objective, objective

    torch.manual_seed(43)
    m=configure_output_readout(model_fixture(True),stage).eval()
    d=MemoryOutputEpisodes(16,seed=44,curriculum='relocation');cache=cache_readout(m,d,'cpu')
    assert not cache['tokens'].requires_grad
    m.output_normalization.calibrate(cache['tokens'])
    z=cache['tokens'].double()
    torch.testing.assert_close(m.output_normalization.mean,z.mean((0,1),keepdim=True).float())
    torch.testing.assert_close(m.output_normalization.std,z.std((0,1),correction=0,keepdim=True).clamp_min(1e-4).float())
    ids=[0,1,2,3];b=d.batch(ids)
    with torch.no_grad():
        h=m.observe_history(b['images']);live=m.working(m.query(h['final'],b['images'][:,-1],'reset'))
    assert torch.equal(cache['tokens'][ids],live)
    fixed={k:v.clone() for k,v in frozen_tensors(m).items()}
    a,_=readout_objective(m,cache,ids);a.backward()
    grads={n:p.grad.clone() for n,p in m.named_parameters() if p.grad is not None}
    m.zero_grad(set_to_none=True);b_loss,_=objective(m,b,True,repair=True);b_loss.backward()
    torch.testing.assert_close(a,b_loss,atol=1e-7,rtol=1e-6)
    for n,p in m.named_parameters():
        if n in grads:torch.testing.assert_close(p.grad,grads[n],atol=1e-6,rtol=1e-5)
        if p.requires_grad:assert n.startswith(('facts.','agent.decoders.image.'))
        else:assert p.grad is None
    torch.optim.AdamW([p for p in m.parameters() if p.requires_grad]).step()
    assert all(torch.equal(v,fixed[k]) for k,v in frozen_tensors(m).items())
