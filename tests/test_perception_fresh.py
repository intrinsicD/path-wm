import numpy as np
from world_model.curriculum.perception_fresh import world_pose, nearest_pose_distance


def test_world_pose_preserves_body_origin_units_and_wrapped_orientation():
    theta = np.array([0., -.01, np.pi])
    target = np.column_stack([np.full((3, 4), .5), np.sin(theta), np.cos(theta)])
    world = world_pose(target)
    assert np.allclose(world[:, :4], 256.)
    assert np.allclose(world[:, 4], theta)


def test_near_pose_audit_uses_all_coordinates_and_circular_angle():
    reference = np.array([[100., 100., 200., 200., 2 * np.pi - .01]])
    candidates = np.array([[100., 100., 200., 200., .01], [100., 100., 220., 200., .01]])
    distances = nearest_pose_distance(candidates, reference)
    assert distances[0] < 1
    assert distances[1] == 2.5
