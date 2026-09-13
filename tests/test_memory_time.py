from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F

from pathwm.models.agent_state import EpisodicMemory, LatentState, MemoryBank
from pathwm.models.modalities import position


def state_fixture():
    torch.manual_seed(63)
    tokens = torch.randn(2, 4, 16)
    keys = F.normalize(torch.randn(2, 3, 16), dim=-1)
    values = torch.randn(2, 3, 4, 16)
    return LatentState(
        tokens,
        torch.zeros_like(tokens),
        torch.full((2,), 3.0),
        torch.full((2,), 3.0),
        observation_count=3,
        memory=MemoryBank(
            keys,
            values,
            torch.tensor([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]]),
            ("first", "second", "third"),
        ),
    )


def test_time_tags_follow_selected_values_and_preserve_storage():
    state = state_fixture()
    raw = state.memory.values.clone()
    plain = EpisodicMemory(3, 2)
    timed = EpisodicMemory(3, 2, relative_time=True)
    trace = {}
    baseline = plain.read(state, trace=trace)
    indices = trace["memory_indices"]
    assert not torch.equal(indices[0], indices[1])
    dt = state.memory.times.gather(1, indices) - state.time[:, None]
    code = position(dt, 16) - position(torch.zeros_like(dt), 16)
    expected = baseline.reshape(2, 2, 4, 16) + code[:, :, None]
    torch.testing.assert_close(
        timed.read(state), expected.flatten(1, 2), atol=0, rtol=0
    )
    assert torch.equal(state.memory.values, raw)
    assert not list(timed.parameters())
    zero = replace(state, memory=replace(state.memory, times=torch.full((2, 3), 3.0)))
    assert torch.equal(timed.read(zero), plain.read(zero))


def test_time_read_is_clock_invariant_and_storage_permutation_equivariant():
    state = state_fixture()
    memory = EpisodicMemory(3, 2, relative_time=True)
    expected = memory.read(state)
    shifted = replace(
        state,
        time=state.time + 16,
        observed_time=state.observed_time + 16,
        memory=replace(state.memory, times=state.memory.times + 16),
    )
    assert torch.equal(memory.read(shifted), expected)
    p = [2, 0, 1]
    bank = state.memory
    reordered = replace(
        state,
        memory=replace(
            bank,
            keys=bank.keys[:, p],
            values=bank.values[:, p],
            times=bank.times[:, p],
            sources=tuple(bank.sources[i] for i in p),
        ),
    )
    assert torch.equal(memory.read(reordered), expected)
    assert not torch.equal(
        memory.read(replace(state, memory=replace(bank, times=bank.times.flip(1)))),
        expected,
    )
    assert memory.read(replace(state, memory=None)).shape == (2, 0, 16)
    for times in (bank.times + 10, torch.full_like(bank.times, float("nan"))):
        with pytest.raises(ValueError):
            memory.read(replace(state, memory=replace(bank, times=times)))


def test_timestamp_interventions_leave_untimed_query_unchanged():
    from tests.test_memory_output import model_fixture
    from pathwm.data.memory_output import MemoryOutputEpisodes
    from pathwm.models.memory_output import configure_recall_repair

    model = configure_recall_repair(model_fixture(True)).eval()
    images = MemoryOutputEpisodes(16, seed=64, curriculum="relocation").batch(range(4))[
        "images"
    ]
    with torch.no_grad():
        reference = model(images, "reset")
        for mode in ("reset_time_erased", "reset_time_swapped"):
            actual = model(images, mode)
            assert torch.equal(actual["facts"], reference["facts"])
            assert torch.equal(actual["image"], reference["image"])
        configure_recall_repair(model, relative_time=True)
        history = model.observe_history(images)["final"]
        bank = history.memory.to_dict()
        regular = model.query(history, images[:, -1], "reset")
        reordered = replace(
            history,
            memory=replace(
                history.memory,
                keys=history.memory.keys.flip(1),
                values=history.memory.values.flip(1),
                times=history.memory.times.flip(1),
                sources=history.memory.sources[::-1],
            ),
        )
        permuted = model.query(reordered, images[:, -1], "reset")
        torch.testing.assert_close(permuted.tokens, regular.tokens, atol=1e-5, rtol=0)
        collapsed = model.query(history, images[:, -1], "reset_time_erased")
        assert not torch.equal(regular.tokens, collapsed.tokens)
        for k, v in bank.items():
            assert (
                torch.equal(getattr(history.memory, k), v)
                if isinstance(v, torch.Tensor)
                else getattr(history.memory, k) == v
            )
