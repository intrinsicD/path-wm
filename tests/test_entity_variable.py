import torch
from pathwm.data.entities import VariableEntityMatches
from pathwm.models.entities import EntityMatchReader
from pathwm.evaluation.entities import matching_metrics


def test_variable_masks_targets_and_permutation():
    data = VariableEntityMatches('train', 32)
    assert data.inputs.shape == (32, 3, 8, 15)
    counts = data.inputs[:, 0, :, 14].sum(-1)
    assert set(counts.tolist()) == set(range(1, 9))
    model = EntityMatchReader()
    logits = model(data.inputs)[0]
    assert torch.all(logits[:, :8][data.inputs[:, 0, :, 14] == 0] < -1e8)
    reverse = data.inputs.clone()
    reverse[:, 0] = reverse[:, 0].flip(1)
    swapped = model(reverse)[0]
    assert torch.allclose(swapped[:, :8].flip(1), logits[:, :8])
    assert torch.equal(swapped[:, -1], logits[:, -1])
    perfect = 30*data.targets[0]
    assert matching_metrics((perfect,), data.targets, data.cohorts)['gates']['passed']
    other = VariableEntityMatches('validation', 32)
    assert set(data.descriptor_ids).isdisjoint(other.descriptor_ids)
