"""The local controller must consume the same source frames as upstream."""
import numpy as np
from world_model.eval_pusht import initial_observations


def test_source_goal_and_initial_frame_survive_reset_render_difference():
    source=np.full((2,8,8,3),17,dtype=np.uint8)
    goal=np.full((8,8,3),29,dtype=np.uint8)
    initial,target=initial_observations(source[0],goal)
    assert initial.shape==(1,1,3,8,8)
    assert target.shape==(1,1,3,8,8)
    assert initial.unique().tolist()==[17]
    assert target.unique().tolist()==[29]
