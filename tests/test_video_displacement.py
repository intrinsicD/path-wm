import json

import pytest
import torch

from pathwm.data.video_order import cyclic_pan_pairs


def test_displacement_spec_rejects_ambiguous_or_colliding_groups(tmp_path):
    from experiments.video_order import load_displacements

    path = tmp_path / 'shifts.json'
    spec = dict(schema=1, train=[2,4,6,8], evaluation=dict(known=[2,4],wide=[6,8],intermediate=[3,5,7],extrapolation=[10,11]))
    path.write_text(json.dumps(spec))
    assert load_displacements(path)['evaluation']['extrapolation'] == [10,11]
    for invalid in ([2,2], [True,4], [4,2], [], [0,4], [2,24], [2,3.5]):
        path.write_text(json.dumps(dict(spec, train=invalid)))
        with pytest.raises(ValueError):
            load_displacements(path)
    for name in ('train', 'validation', 'confirm_x', '../x'):
        path.write_text(json.dumps(dict(spec,evaluation=dict(spec['evaluation'], **{name:[1]}))))
        with pytest.raises(ValueError):
            load_displacements(path)


def test_oracle_new_magnitudes_and_ambiguity():
    from experiments.video_order import pixel_oracle

    g = torch.Generator().manual_seed(13)
    shifts=(3,5,7,10,11)
    data=cyclic_pan_pairs(torch.rand(1,3,8,48,generator=g),shifts=shifts)
    predicted,ambiguous=pixel_oracle(data['frames'],shifts=shifts)
    assert torch.equal(predicted,data['labels'])
    assert not ambiguous.any()
    _,ambiguous=pixel_oracle(torch.zeros_like(data['frames']),shifts=shifts)
    assert ambiguous.all()


def test_phase_sampler_matches_content_and_resume_across_support_sizes():
    from experiments.video_order import phase_pair_groups, sample_phase_pairs

    image=torch.zeros(2,3,4,16)
    a=cyclic_pan_pairs(image,shifts=(1,2))
    b=cyclic_pan_pairs(image,shifts=(1,2,3,4))
    ga,gb=(phase_pair_groups(x['metadata']) for x in (a,b))
    rngs=[torch.Generator().manual_seed(41) for _ in range(2)]
    for _ in range(20):
        indices=[sample_phase_pairs(lambda n,k:torch.randint(n,(k,),generator=rng),groups,8) for rng,groups in zip(rngs,(ga,gb))]
        assert torch.equal(a['metadata'][indices[0],:2], b['metadata'][indices[1],:2])
    assert torch.equal(rngs[0].get_state(),rngs[1].get_state())
    state=rngs[0].get_state()
    expected=sample_phase_pairs(lambda n,k:torch.randint(n,(k,),generator=rngs[0]),ga,8)
    rngs[1].set_state(state)
    actual=sample_phase_pairs(lambda n,k:torch.randint(n,(k,),generator=rngs[1]),ga,8)
    assert torch.equal(expected,actual)
    broken=b['metadata'].clone();broken[0,1]=3
    with pytest.raises(ValueError):
        phase_pair_groups(broken)
