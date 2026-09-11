import pytest
from pathwm.models.source_choice import SourceChoice


def test_feedback_only_choice_and_permutation():
    policy=SourceChoice()
    assert policy.choose() is None
    policy.observe(0,-.05)
    policy.observe(1,.95)
    assert policy.choose()==1
    swapped=SourceChoice()
    swapped.observe(1,-.05)
    swapped.observe(0,.95)
    assert swapped.choose()==0
    before=policy.snapshot()
    for _ in range(5): assert policy.choose()==1
    assert policy.snapshot()==before
    with pytest.raises(ValueError):policy.observe(0,float('nan'))
    assert policy.snapshot()==before
