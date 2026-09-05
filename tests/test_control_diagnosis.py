"""Protect case populations, action units and matched ranking comparisons."""
import numpy as np
import pytest
from scripts.diagnose_pusht_control import freeze_training_cases, action_blocks, ranking_metrics


def test_training_goals_are_distinct_and_cannot_cross_episode_boundaries():
    lengths=np.array([40,50,25,60,70]); offsets=np.r_[0,np.cumsum(lengths[:-1])]
    cases=freeze_training_cases(lengths,offsets,[0,1,2,3],source_ids=[10,11,12,13,14],count=3)
    assert len({c['episode'] for c in cases})==3
    for c in cases:
        assert c['episode'] in [0,1,3]
        assert c['source_episode']==10+c['episode']
        assert c['row']==offsets[c['episode']]+c['start']
        assert c['start']+25 < lengths[c['episode']]
    with pytest.raises(ValueError):
        freeze_training_cases(lengths,offsets,[0,1,2],source_ids=list(range(5)),count=3)


def test_raw_actions_normalize_before_time_blocking():
    raw=np.arange(50,dtype=np.float32).reshape(25,2)
    stats={'mean':[3.,-2.],'std':[2.,4.]}
    result=action_blocks(raw,stats)
    assert result.shape==(5,10)
    np.testing.assert_allclose(result.reshape(25,2)*stats['std']+stats['mean'],raw)


def test_ranking_uses_lower_cost_and_handles_uninformative_ties():
    result=ranking_metrics([3.,1.,2.],[30.,10.,20.])
    assert result['spearman']==pytest.approx(1.)
    assert result['selected_index']==1
    assert result['selected_distance']==10.
    assert result['regret']==0.
    assert ranking_metrics([0.,0.,0.],[1.,2.,3.])['spearman'] is None
    with pytest.raises(ValueError): ranking_metrics([1.,2.],[3.])
