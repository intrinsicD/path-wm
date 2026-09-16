import copy

import numpy as np
import pytest
import torch

from experiments import modality_readout as recipe
from pathwm.data.understanding import UnderstandingData, synthetic_records


def fixtures():
    data = UnderstandingData.__new__(UnderstandingData)
    data.records, data.arrays, data.cases = synthetic_records("full")
    return data


def test_grounded_training_never_uses_validation_or_answers_as_inputs():
    data = fixtures()
    rows = recipe.grounded_records(data, "VID.order")
    assert len(rows) == 16 and {r["split"] for r in rows} == {"calibration"}
    assert set(r["id"] for r in rows).isdisjoint(
        r["id"] for r in data.records if r["split"] != "calibration"
    )
    inputs, targets = recipe.grounded_batch(data, rows, [0, 1], "cpu")
    changed = copy.deepcopy(rows)
    for row in changed:
        row["answer"] = 1 - row["answer"]
    other, other_targets = recipe.grounded_batch(data, changed, [0, 1], "cpu")
    assert not torch.equal(targets, other_targets)
    for kind in inputs:
        assert torch.equal(inputs[kind].values, other[kind].values)
        assert torch.equal(inputs[kind].times, other[kind].times)
        assert inputs[kind].provenance is None
    assert torch.equal(inputs["video"].values[0, -1], inputs["video"].values[1, -1])
    leaked = [r for r in data.records if r["case"] == "VID.order" and r["split"] == "test"]
    with pytest.raises(ValueError, match="calibration"):
        recipe.grounded_batch(data, leaked, [0], "cpu")
    with pytest.raises(ValueError, match="controlled"):
        recipe.grounded_records(data, "MIX.agreement.text+video")


@pytest.mark.parametrize("scope", ["decoder", "core"])
def test_grounded_loss_respects_frozen_boundaries(scope):
    torch.manual_seed(75)
    torch.set_num_threads(2)
    data = fixtures()
    rows = recipe.grounded_records(data, "VID.order")
    inputs, targets = recipe.grounded_batch(data, rows, [0, 1], "cpu")
    model = recipe.Model("native")
    recipe.configure_grounded_training(model, scope)
    frozen = {n: p.clone() for n, p in model.named_parameters() if not p.requires_grad}
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=.001)
    tokens = model.core(inputs)
    loss = recipe.grounded_objective(model, tokens, targets)
    logits = model.outputs("text", tokens, targets[:, :-1])
    reference = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), targets[:, 1:].flatten(), ignore_index=0
    ) / np.log(259)
    torch.testing.assert_close(loss, reference)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.outputs.decoders["text"].parameters())
    core_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.core.parameters())
    assert core_grad == (scope == "core")
    assert all(p.grad is None for p in model.core.agent.encoders.parameters())
    optimizer.step()
    for n, p in model.named_parameters():
        if n in frozen:
            assert p.grad is None
            torch.testing.assert_close(p, frozen[n], rtol=0, atol=0)
