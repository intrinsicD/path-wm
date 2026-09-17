import copy

import pytest
import torch

from pathwm.models.tasks import MetadataEncoder
from pathwm.models.modalities import bytes_batch


def reference_output(model, records):
    # Independent unchanged expression retained after rejecting memoization.
    import json

    ids, valid = bytes_batch(
        [json.dumps(r, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
         for r in records], model.embedding.weight.device,
    )
    values, _ = model.sequence(model.embedding(ids))
    return ((values * valid[..., None]).sum(1) / valid.sum(1)[:, None])[:, None]


def test_frozen_metadata_repeated_batches_return_owned_values():
    torch.manual_seed(87)
    model = MetadataEncoder(16).requires_grad_(False).eval()
    records = [{"caller": "Älice", "value": "same"}] * 8
    first = model(records)
    reference = first.clone()
    first.zero_()
    assert torch.equal(model(copy.deepcopy(records)), reference)
    model(records[:1])
    assert torch.equal(model(records), reference)



@pytest.mark.parametrize(
    "change",
    ["parameter", "load", "trainable", "train_mode", "child_train_mode", "dtype"],
)
def test_frozen_metadata_changes_keep_independent_reference(change):
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
    actual, expected = model(records), reference_output(reference, records)
    assert torch.equal(actual, expected)
    if change == "trainable":
        actual.square().sum().backward()
        expected.square().sum().backward()
        assert all(
            torch.equal(p.grad, q.grad)
            for p, q in zip(model.parameters(), reference.parameters())
        )


def test_frozen_metadata_preserves_checkpoint_state_and_rng():
    model = MetadataEncoder(16).requires_grad_(False).eval()
    before = copy.deepcopy(model.state_dict())
    rng = torch.get_rng_state().clone()
    model([{"text": "x"}])
    model([{"text": "x"}])
    assert before.keys() == model.state_dict().keys()
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in before.items())
    assert torch.equal(rng, torch.get_rng_state())


def test_metadata_hooks_are_observed_after_repeated_inputs():
    model = MetadataEncoder(16).requires_grad_(False).eval()
    records = [{"text": "same"}]
    model(records)
    calls = []
    handle = model.sequence.register_forward_hook(lambda *args: calls.append(1))
    model(records)
    model(records)
    assert len(calls) == 2
    handle.remove()


def test_metadata_inference_and_autocast_tensor_semantics():
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
    with torch.autocast("cpu", dtype=torch.bfloat16):
        assert torch.equal(model(records), reference_output(reference, records))
    with torch.inference_mode():
        created = MetadataEncoder(16).requires_grad_(False).eval()
        assert torch.equal(created(records), created(records))


def test_frozen_metadata_after_cudnn_rnn_import():
    # Importing this PyTorch submodule changes its parent attribute. Exercise the
    # ordinary frozen path in a fresh interpreter, without precision setup.
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "-c", "import torch.backends.cudnn.rnn; "
         "from pathwm.models.tasks import MetadataEncoder; "
         "m = MetadataEncoder(8).requires_grad_(False).eval(); "
         "assert m([{'text': 'same'}]).shape == (1, 1, 8)"],
        check=True,
        capture_output=True,
        text=True,
    )
