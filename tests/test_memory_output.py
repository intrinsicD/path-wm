from dataclasses import replace

import numpy as np
import pytest
import torch


def model_fixture():
    from experiments.memory_output import build_model
    from pathwm.models.encoders import PyramidEncoder, PatchDetailEncoder
    from pathwm.models.decoders import DenseHead, PatchDetailHead
    from pathwm.models.perception import Perception

    encoder = PyramidEncoder(width=16, levels=2)
    head = DenseHead(encoder.feature_spec, channels=3, activation="sigmoid")
    codec = Perception(PatchDetailEncoder(encoder), {"rgb": PatchDetailHead(head)})
    return build_model(codec.requires_grad_(False), width=16)


def test_pairs_require_history_and_target_combinations_are_disjoint():
    from pathwm.data.memory_output import MemoryOutputEpisodes, templates, image_labels

    train = MemoryOutputEpisodes(16, seed=17, split="train")
    test = MemoryOutputEpisodes(16, seed=18, split="test")
    tuples = []
    for data, parity in ((train, 0), (test, 1)):
        b = data.batch(range(len(data)))
        assert torch.equal(b["images"][0::2, 1:], b["images"][1::2, 1:])
        assert (b["labels"][0::2] != b["labels"][1::2]).all()
        assert not torch.equal(b["images"][0::2, 0], b["images"][1::2, 0])
        assert torch.equal(image_labels(b["target"]), b["labels"])
        assert torch.equal(image_labels(templates()[0]), templates()[1])
        c, shape, side = b["labels"].T
        assert ((c % 2) ^ shape ^ side == parity).all()
        tuples.append(set(map(tuple, b["labels"].tolist())))
    assert len(tuples[0]) == len(tuples[1]) == 8
    assert not tuples[0] & tuples[1]
    assert train.identity != test.identity
    with pytest.raises(ValueError):
        MemoryOutputEpisodes(0)


def test_recalled_state_depends_only_on_selected_bank_and_has_no_target_path():
    from pathwm.data.memory_output import MemoryOutputEpisodes
    from pathwm.models.agent_state import LatentState

    torch.manual_seed(12)
    model = model_fixture()
    b = MemoryOutputEpisodes(2, seed=14).batch(range(4))
    history = model.observe_history(b["images"])
    bank = history["final"].memory
    assert bank.values.grad_fn is None and bank.values.shape[1] == 2
    before = bank.values.clone()
    recalled = model.query(history["final"], b["images"][:, -1], "reset")
    changed = replace(history["final"], tokens=torch.randn_like(history["final"].tokens))
    other = model.query(changed, b["images"][:, -1], "reset")
    torch.testing.assert_close(recalled.tokens, other.tokens, atol=0, rtol=0)
    erased = model.query(changed, b["images"][:, -1], "reset_erased")
    torch.testing.assert_close(erased.tokens[0::2], erased.tokens[1::2], atol=0, rtol=0)
    swapped = model.query(history["final"], b["images"][:, -1], "reset_swapped")
    torch.testing.assert_close(swapped.tokens, recalled.tokens[torch.arange(4) ^ 1])
    assert torch.equal(bank.values, before)
    restored = LatentState.from_dict(history["final"].to_dict())
    torch.testing.assert_close(model.query(restored, b["images"][:, -1], "reset").tokens, recalled.tokens)
    with pytest.raises(ValueError, match="future"):
        future = replace(restored, memory=replace(bank, times=bank.times + 100))
        model.query(future, b["images"][:, -1], "reset")


def test_supervised_writes_and_frozen_decoder_have_intended_gradients():
    from pathwm.data.memory_output import MemoryOutputEpisodes
    from experiments.memory_output import objective

    torch.manual_seed(12)
    model = model_fixture()
    b = MemoryOutputEpisodes(2, seed=14).batch(range(4))
    with torch.no_grad():
        target_features = model.teacher(b["target"])
    model.agent.decoders["image"].calibrate(target_features)
    loss, _ = objective(model, b, reset=True)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.agent.updater.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.agent.thinker.parameters())
    assert all(p.grad is None for p in model.teacher.parameters())
    assert all(p.grad is None for p in model.agent.decoders["image"].head.parameters())
    assert all(p.grad is None for p in model.direct.parameters())


def test_training_resume_and_standalone_reload(tmp_path):
    from pathwm.data.memory_output import MemoryOutputEpisodes
    from experiments.memory_output import train, default_settings, load_model
    from tests.test_runs import equal_tree

    train_data = MemoryOutputEpisodes(4, seed=17)
    val = MemoryOutputEpisodes(4, seed=18)
    test = MemoryOutputEpisodes(4, seed=19, split="test")

    def run(path, resume=False, stop_after=None):
        torch.manual_seed(21)
        model = model_fixture()
        settings = default_settings(21)
        settings.update(steps=4, batch_size=2, width=16, levels=2, depth=1, fusion_depth=0)
        with torch.no_grad():
            model.agent.decoders["image"].calibrate(model.teacher(train_data.batch(range(8))["target"]))
        result = train(model, train_data, val, test, output=path, settings=settings,
                       resume=resume, stop_after=stop_after)
        return result, torch.load(path / "last.pt", weights_only=True)

    full_model, full = run(tmp_path / "full")
    run(tmp_path / "resumed", stop_after=2)
    _, resumed = run(tmp_path / "resumed", resume=True)
    for key in ("model", "optimizer", "sampler", "torch", "step"):
        equal_tree(full[key], resumed[key])
    loaded = load_model(tmp_path / "full" / "weights.pt")
    b = test.batch(range(4))
    with torch.no_grad():
        expected = full_model(b["images"], mode="reset")
        actual = loaded(b["images"], mode="reset")
    torch.testing.assert_close(actual["image"], expected["image"], atol=0, rtol=0)
    saved = np.load(tmp_path / "full" / "predictions.npz")
    assert np.isfinite(saved["reset_image"]).all()
