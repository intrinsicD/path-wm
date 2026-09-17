import copy

import pytest
import torch

from pathwm.models.tasks import MetadataEncoder


def test_frozen_metadata_reuses_only_identical_batches_and_returns_owned_values():
    torch.manual_seed(87)
    model = MetadataEncoder(16).requires_grad_(False).eval()
    calls = []
    handle = model.sequence.register_forward_hook(lambda *args: calls.append(1))
    records = [{"caller": "Älice", "value": "same"}] * 8
    first = model(records)
    reference = first.clone()
    first.zero_()
    assert torch.equal(model(copy.deepcopy(records)), reference)
    assert len(calls) == 1
    model(records[:1])
    model(records)
    assert len(calls) == 3  # Exactly one cached batch, not an unbounded dictionary.
    model.cache_frozen = False
    model(records)
    assert len(calls) == 4
    handle.remove()


@pytest.mark.parametrize("change", ["parameter", "load", "trainable", "train_mode", "child_train_mode", "dtype"])
def test_frozen_cache_invalidates_and_keeps_uncached_reference(change):
    torch.manual_seed(89)
    model = MetadataEncoder(16).requires_grad_(False).eval()
    records = [{"id": "one"}, {"id": "two"}]
    model(records)
    if change == "parameter":
        with torch.no_grad():
            model.sequence.weight_ih_l0.add_(.1)
    elif change == "load":
        model.load_state_dict(MetadataEncoder(16).state_dict())
    elif change == "trainable":
        model.requires_grad_(True)
    elif change == "train_mode":
        model.train()
    elif change == "child_train_mode":
        model.sequence.train()
    else:
        model.double()
    reference = copy.deepcopy(model)
    reference.cache_frozen = False
    actual, expected = model(records), reference(records)
    assert torch.equal(actual, expected)
    if change == "trainable":
        actual.square().sum().backward()
        expected.square().sum().backward()
        assert all(torch.equal(p.grad, q.grad) for p, q in zip(model.parameters(), reference.parameters()))


def test_cache_is_not_checkpoint_state_and_frozen_calls_preserve_rng():
    model = MetadataEncoder(16).requires_grad_(False).eval()
    before = copy.deepcopy(model.state_dict())
    rng = torch.get_rng_state().clone()
    model([{"text": "x"}])
    model([{"text": "x"}])
    assert before.keys() == model.state_dict().keys()
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in before.items())
    assert torch.equal(rng, torch.get_rng_state())
