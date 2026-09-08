"""Semantic labels and metrics must agree on crop visibility and unknown labels."""
import numpy as np
import pytest
import torch
from world_model.curriculum.perception_semantics import category_labels, average_precision, known_bce, CategoryHead


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


def test_unknown_labels_have_no_gradient_and_classifier_initialization_is_paired():
    logits = torch.tensor([[0., 3.]], requires_grad=True)
    loss = known_bce(logits, torch.tensor([[1., 0.]]), torch.tensor([[1., 0.]]))
    loss.backward()
    assert logits.grad[0, 0] != 0 and logits.grad[0, 1] == 0
    c, v = CategoryHead(128, 78, 9107), CategoryHead(768, 78, 9107)
    assert torch.equal(c.classifier.weight, v.classifier.weight)


def test_cached_normalization_preserves_learned_affine_after_spatial_pool():
    torch.manual_seed(82)
    tokens = torch.randn(3, 8, 12)
    gamma, beta = torch.randn(12), torch.randn(12)
    direct = torch.nn.functional.layer_norm(tokens, (12,), gamma, beta).mean(1)
    pooled = torch.nn.functional.layer_norm(tokens, (12,)).mean(1) * gamma + beta
    assert torch.allclose(direct, pooled, atol=3e-7)
