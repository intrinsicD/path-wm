from copy import deepcopy

import pytest
import torch

from tests.test_memory_output import model_fixture
from pathwm.data.memory_output import MemoryOutputEpisodes


def test_auxiliary_reader_changes_only_training_gradients_and_stays_frozen():
    from pathwm.models.memory_output import (
        TokenProbe, configure_recall_repair, frozen_tensors,
    )
    from experiments.memory_output import objective

    torch.manual_seed(34)
    model = configure_recall_repair(model_fixture(True)).eval()
    batch = MemoryOutputEpisodes(16, seed=35, curriculum="relocation").batch(range(4))
    original = model.observe_history(batch["images"])["final"].memory.values.clone()
    model.workspace_reference = TokenProbe(model.working(
        model.observe_history(batch["images"])["stored"]
    )).requires_grad_(False)
    frozen = {k: v.clone() for k, v in frozen_tensors(model).items()}
    base, _ = objective(model, batch, True, repair=True, reference_weight=0)
    assisted, metrics = objective(model, batch, True, repair=True, reference_weight=1)
    assert metrics["reference_loss"] > 0
    torch.testing.assert_close(assisted - base, base.new_tensor(metrics["reference_loss"]))
    (assisted - base).backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in model.agent.thinker.parameters())
    assert all(p.grad is None for p in model.workspace_reference.parameters())
    assert all(p.grad is None for p in model.facts.parameters())
    assert all(p.grad is None or torch.count_nonzero(p.grad) == 0
               for p in model.agent.decoders["image"].parameters())
    torch.optim.AdamW([p for p in model.parameters() if p.requires_grad]).step()
    assert all(torch.equal(v, frozen[k]) for k, v in frozen_tensors(model).items())
    assert torch.equal(model.observe_history(batch["images"])["final"].memory.values, original)
    with torch.no_grad():
        expected = model(batch["images"], "reset")
        for p in model.workspace_reference.parameters():
            p.add_(10)
        actual = model(batch["images"], "reset")
    for key in ("facts", "image"):
        assert torch.equal(expected[key], actual[key])


def test_reference_requires_valid_weight_and_matching_finite_export(tmp_path):
    from pathwm.models.memory_output import TokenProbe, load_workspace_reference
    from experiments.memory_output import objective

    model = model_fixture()
    batch = MemoryOutputEpisodes(2, seed=35).batch(range(4))
    for weight in (-1, float("nan"), float("inf"), 1):
        with pytest.raises(ValueError):
            objective(model, batch, True, repair=True, reference_weight=weight)
    probe = TokenProbe(torch.randn(4, 8, 16))
    export = dict(width=16, stages=["stored_working"], model={
        "stored_working." + k: v for k, v in probe.state_dict().items()
    })
    path = tmp_path / "probe.pt"
    torch.save(export, path)
    restored = load_workspace_reference(path, 16)
    x = torch.randn(4, 8, 16)
    assert torch.equal(probe(x), restored(x))
    assert all(not p.requires_grad for p in restored.parameters())
    with pytest.raises(ValueError):
        load_workspace_reference(path, 32)
    corrupt = deepcopy(export)
    corrupt["model"]["stored_working.std"].zero_()
    torch.save(corrupt, path)
    with pytest.raises(ValueError):
        load_workspace_reference(path, 16)
