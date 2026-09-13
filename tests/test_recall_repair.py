from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from tests.test_memory_output import model_fixture
from pathwm.data.memory_output import MemoryOutputEpisodes


def test_memory_calibration_changes_read_only_and_is_atomic():
    from pathwm.models.memory_output import CalibratedMemory
    from pathwm.models.agent_state import EpisodicMemory

    model = model_fixture().eval()
    images = MemoryOutputEpisodes(2, seed=28).batch(range(4))["images"]
    state = model.observe_history(images)["final"]
    raw = state.memory.values.clone()
    memory = CalibratedMemory(16, capacity=4, retrieve_count=2)
    expected = EpisodicMemory(4, 2).read(state)
    assert torch.equal(memory.read(state), expected)
    memory.calibrate([raw[:2], raw[2:]])
    flat = raw.flatten(0, 2).double()
    mean = flat.mean(0).float()
    std = flat.std(0, correction=0).clamp_min(1e-4).float()
    torch.testing.assert_close(memory.mean.flatten(), mean)
    torch.testing.assert_close(memory.std.flatten(), std)
    torch.testing.assert_close(memory.read(state), (expected - mean) / std)
    assert torch.equal(state.memory.values, raw)
    assert memory.read(replace(state, memory=None)).shape == (4, 0, 16)
    before = deepcopy(memory.state_dict())
    for values in ([], [raw, torch.full_like(raw, float("nan"))]):
        with pytest.raises(ValueError):
            memory.calibrate(values)
        assert all(torch.equal(before[k], v) for k, v in memory.state_dict().items())


def test_repair_freezes_writer_and_both_codec_paths_but_trains_output():
    from pathwm.models.memory_output import configure_recall_repair, frozen_tensors
    from experiments.memory_output import objective

    torch.manual_seed(28)
    model = model_fixture(True).eval()
    batch = MemoryOutputEpisodes(16, seed=29, curriculum="relocation").batch(range(4))
    original = model.observe_history(batch["images"])["final"].memory.values.clone()
    configure_recall_repair(model)
    model.agent.memory.calibrate([original])
    frozen = {k: v.clone() for k, v in frozen_tensors(model).items()}
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad])
    loss, _ = objective(model, batch, reset=True, repair=True)
    loss.backward()
    for prefix in ("agent.thinker.", "facts.", "agent.decoders.image.projections."):
        assert any(p.grad is not None and p.grad.abs().sum() > 0
                   for n, p in model.named_parameters() if n.startswith(prefix))
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    optimizer.step()
    assert all(torch.equal(v, frozen[k]) for k, v in frozen_tensors(model).items())
    assert torch.equal(model.observe_history(batch["images"])["final"].memory.values, original)
    with torch.no_grad():
        recall = model(batch["images"], "reset")
        swapped = model(batch["images"], "reset_swapped")
        erased = model(batch["images"], "reset_erased")
    for key in ("image", "facts"):
        torch.testing.assert_close(swapped[key], recall[key][[1, 0, 3, 2]], atol=1e-6, rtol=0)
        assert torch.equal(erased[key][0], erased[key][1])
