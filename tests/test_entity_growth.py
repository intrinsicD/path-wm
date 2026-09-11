import torch
from pathwm.evaluation.entity_growth import growth_inputs, evaluate_growth
from tests.test_entity_memory import Scorer


def test_nested_populations_and_lifecycle():
    families = growth_inputs(seed=61, count=3)
    assert families == growth_inputs(seed=61, count=3)
    for family in families:
        points = torch.tensor(family['descriptors'])
        distances = torch.cdist(points, points)
        assert (distances[~torch.eye(9, dtype=torch.bool)] >= .9).all()
    result = evaluate_growth(Scorer(), families)
    assert result['passed']
    assert len(result['episodes']) == 12
    assert all(row['retry_equal'] and row['restore_equal'] for row in result['episodes'])
