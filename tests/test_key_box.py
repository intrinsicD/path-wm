import pytest


def test_planner_known_unknown_absent_and_nonmutation():
    from pathwm.models.key_box import plan_key
    belief = (1., 0., 0.)
    opened = (False, False)
    assert plan_key(belief, opened, 4)[0] == ("open", 0)
    assert plan_key(belief, (True, False), 3)[0] == ("retrieve", 0)
    assert plan_key((0., 0., 1.), opened, 4)[0] == ("stop", -1)
    assert plan_key((.5, .5, 0.), opened, 4)[0][0] == "inspect"
    assert plan_key((.5, .5, 0.), opened, 0)[0] == ("stop", -1)
    assert belief == (1., 0., 0.) and opened == (False, False)
    with pytest.raises(ValueError):
        plan_key((1., 1., 0.), opened, 4)
