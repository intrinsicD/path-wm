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


def test_reference_goal_sampling_respects_episode_boundaries_and_population():
    from world_model.evaluation import sample_source_cases
    lengths=np.array([3,5,8]);offsets=np.array([0,3,8])
    cases=sample_source_cases(lengths,offsets,count=6,seed=42,goal_offset=3)
    # Reference eval.py excludes the final eligible row before sampling.
    assert [x['row'] for x in cases]==[3,4,8,9,10,11]
    assert all(x['start']+3<lengths[x['episode']] for x in cases)
    heldout=sample_source_cases(lengths,offsets,count=1,seed=42,goal_offset=3,population=[1])
    assert heldout==[dict(row=3,episode=1,start=0)]
