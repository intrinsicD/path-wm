import numpy as np
import pytest
from world_model.curriculum.control_analysis import paired_delta


def test_bootstrap_preserves_both_members_of_each_pair():
    # Each pair has identical mean; independent member draws would invent variance.
    old=np.array([[0.,1.],[1.,0.]])
    new=1-old
    result,draws=paired_delta(new,old,seed=3)
    assert result['difference']==0
    np.testing.assert_array_equal(draws,np.zeros(2000))
    assert result['ci95']==[0.,0.]
    with pytest.raises(ValueError):paired_delta(new[:,:1],old[:,:1])
