"""Guard diagnostics against silent probability and projection mistakes."""
import numpy as np
import torch

from world_model.paddle.models import Attention
from world_model.curriculum.encoder_visual_audit import attention_details, entropy, variance_parts


def test_manual_attention_matches_real_operation_with_distinct_values():
    torch.manual_seed(317)
    module = Attention().double()
    query = torch.randn(2, 7, 64, dtype=torch.float64)
    key = torch.randn(2, 11, 64, dtype=torch.float64)
    value = torch.randn(2, 11, 64, dtype=torch.float64) * 3 + 2
    details = attention_details(module, query, key, value)
    torch.testing.assert_close(details['output'], module(query, key, value), rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(details['probabilities'].sum(-1), torch.ones(2, 4, 7, dtype=torch.float64))


def test_entropy_limits_and_head_average_are_not_interchangeable():
    uniform = torch.full((2, 8), 1 / 8, dtype=torch.float64)
    torch.testing.assert_close(entropy(uniform), torch.ones(2, dtype=torch.float64))
    # Two heads selecting opposite keys have zero individual entropy, while
    # their averaged distribution is uniform. This catches averaging too early.
    heads = torch.eye(2, dtype=torch.float64)
    torch.testing.assert_close(entropy(heads), torch.zeros(2, dtype=torch.float64))
    torch.testing.assert_close(entropy(heads.mean(0)), torch.tensor(1., dtype=torch.float64))


def test_image_and_spatial_variance_have_known_additive_decomposition():
    # Image means -3,+3 have variance 9; within-image positions -2,+2 have 4.
    tokens = np.array([[[-5.], [-1.]], [[1.], [5.]]])
    result = variance_parts(tokens)
    assert np.isclose(result['total_variance'], 13)
    assert np.isclose(result['image_mean_variance'], 9)
    assert np.isclose(result['within_image_variance'], 4)
    assert np.isclose(result['image_mean_fraction'], 9 / 13)
