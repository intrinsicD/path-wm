import copy

import pytest
import torch

from pathwm.models.tasks import MetadataEncoder


def test_frozen_metadata_reuses_only_identical_batches_and_returns_owned_values():
    torch.manual_seed(87)
    model = MetadataEncoder(16).requires_grad_(False).eval()
    records = [{"caller": "Älice", "value": "same"}] * 8
    with torch.autograd.profiler.profile() as profile:
        first = model(records)
        reference = first.clone()
        first.zero_()
        assert torch.equal(model(copy.deepcopy(records)), reference)
        model(records[:1])
        model(records)  # Exactly one cached batch, not an unbounded dictionary.
        model.cache_frozen = False
        model(records)
    assert sum(e.count for e in profile.key_averages() if e.key == "aten::gru") == 4


@pytest.mark.parametrize(
    "change",
    ["parameter", "load", "trainable", "train_mode", "child_train_mode", "dtype"],
)
def test_frozen_cache_invalidates_and_keeps_uncached_reference(change):
    torch.manual_seed(89)
    model = MetadataEncoder(16).requires_grad_(False).eval()
    records = [{"id": "one"}, {"id": "two"}]
    model(records)
    if change == "parameter":
        with torch.no_grad():
            model.sequence.weight_ih_l0.add_(0.1)
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
        assert all(
            torch.equal(p.grad, q.grad)
            for p, q in zip(model.parameters(), reference.parameters())
        )


def test_cache_is_not_checkpoint_state_and_frozen_calls_preserve_rng():
    model = MetadataEncoder(16).requires_grad_(False).eval()
    before = copy.deepcopy(model.state_dict())
    rng = torch.get_rng_state().clone()
    model([{"text": "x"}])
    model([{"text": "x"}])
    assert before.keys() == model.state_dict().keys()
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in before.items())
    assert torch.equal(rng, torch.get_rng_state())


def test_hooks_are_not_skipped_after_warming_cache():
    model = MetadataEncoder(16).requires_grad_(False).eval()
    records = [{"text": "same"}]
    model(records)
    calls = []
    handle = model.sequence.register_forward_hook(lambda *args: calls.append(1))
    model(records)
    model(records)
    assert len(calls) == 2
    handle.remove()


def test_inference_and_autocast_modes_do_not_leak_cached_tensor_semantics():
    model = MetadataEncoder(16).requires_grad_(False).eval()
    records = [{"text": "same"}]
    with torch.inference_mode():
        model(records)
        model(records)
    actual = model(records)
    downstream = torch.nn.Linear(16, 1)
    downstream(actual).sum().backward()
    assert downstream.weight.grad is not None and not actual.is_inference()
    reference = copy.deepcopy(model)
    reference.cache_frozen = False
    with torch.autocast("cpu", dtype=torch.bfloat16):
        assert torch.equal(model(records), reference(records))
    with torch.inference_mode():
        created = MetadataEncoder(16).requires_grad_(False).eval()
        assert torch.equal(created(records), created(records))
