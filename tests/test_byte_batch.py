import pytest
import torch

from pathwm.models.modalities import bytes_batch, bytes_text


def original_batch(strings, device="cpu"):
    sequences = [[1, *(b + 3 for b in s.encode("utf-8")), 2] for s in strings]
    if not sequences:
        raise ValueError("Text batch must be nonempty")
    result = torch.zeros(
        len(sequences), max(map(len, sequences)), dtype=torch.long, device=device
    )
    for i, row in enumerate(sequences):
        result[i, : len(row)] = torch.tensor(row, device=device)
    return result, result != 0


@pytest.mark.parametrize(
    "strings",
    [
        [""],
        ["", "rot", "grün", "Tür\nöffnen 🎨", "a\x00b"],
        ["same"] * 8,
        ["x" * 257, "y"],
    ],
)
def test_byte_batches_preserve_independent_reference_and_roundtrip(strings):
    before = torch.get_rng_state().clone()
    actual, mask = bytes_batch(iter(strings))
    expected, expected_mask = original_batch(strings)
    assert torch.equal(actual, expected) and torch.equal(mask, expected_mask)
    assert actual.dtype == torch.long and mask.dtype == torch.bool
    assert actual.is_contiguous() and actual.device.type == "cpu"
    assert [bytes_text(row) for row in actual] == strings
    assert torch.equal(before, torch.get_rng_state())


def test_byte_batch_rejects_empty_iterable():
    with pytest.raises(ValueError, match="nonempty"):
        bytes_batch(iter(()))


def test_task_metadata_roundtrip_and_gradients_match_original_byte_builder(monkeypatch):
    import copy
    from pathwm.models import tasks

    torch.manual_seed(84)
    new = tasks.MetadataEncoder(16)
    old = copy.deepcopy(new)
    records = [{"caller": "Älice", "task": ""}, {"caller": "B", "task": "read"}] * 4
    output = new(records)
    with monkeypatch.context() as patch:
        patch.setattr(tasks, "bytes_batch", original_batch)
        expected = old(records)
    assert torch.equal(output, expected)
    output.square().sum().backward()
    expected.square().sum().backward()
    assert all(
        torch.equal(p.grad, q.grad) for p, q in zip(new.parameters(), old.parameters())
    )
