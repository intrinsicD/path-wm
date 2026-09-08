"""Semantic labels and metrics must agree on crop visibility and unknown labels."""
import numpy as np
import pytest
from world_model.curriculum.perception_semantics import category_labels, average_precision


def test_category_crop_visibility_and_crowd_unknown_are_distinct():
    # The centered square crop removes both horizontal margins of this wide view.
    annotations = [
        {'category_id': 1, 'iscrowd': 0, 'segmentation': [[0, 0, 8, 0, 8, 8, 0, 8]]},
        {'category_id': 2, 'iscrowd': 0, 'segmentation': [[55, 20, 70, 20, 70, 40, 55, 40]]},
        {'category_id': 3, 'iscrowd': 1, 'segmentation': [[55, 10, 65, 10, 65, 20, 55, 20]]},
        {'category_id': 4, 'iscrowd': 1, 'segmentation': [[60, 10, 70, 10, 70, 20, 60, 20]]},
        {'category_id': 4, 'iscrowd': 0, 'segmentation': [[60, 30, 70, 30, 70, 40, 60, 40]]},
    ]
    target, known = category_labels(annotations, 64, 128, [1, 2, 3, 4])
    assert target.tolist() == [0, 1, 0, 1]
    assert known.tolist() == [1, 1, 0, 1]


def test_average_precision_groups_ties_and_excludes_unknowns():
    target = np.array([1, 0, 1, 0])
    scores = np.array([.8, .8, .2, .1])
    known = np.array([1, 1, 1, 0])
    # First threshold: recall1/2, precision1/2. Second: recall1, precision2/3.
    assert average_precision(target, scores, known) == pytest.approx(.25 + 1/3)
    assert average_precision(target[::-1], scores[::-1], known[::-1]) == pytest.approx(.25 + 1/3)
    assert average_precision(target, np.ones(4), known) == pytest.approx(2/3)
    assert average_precision(np.zeros(4), scores, np.ones(4)) is None
