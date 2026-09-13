from dataclasses import replace

import numpy as np
import pytest
import torch


def test_relocation_curriculum_breaks_appearance_and_initial_frame_shortcuts():
    from collections import Counter
    import hashlib
    from pathwm.data.memory_output import MemoryOutputEpisodes, image_labels

    datasets = [
        MemoryOutputEpisodes(16, seed=s, curriculum="relocation") for s in (17, 18)
    ]
    frame_sets = []
    for data in datasets:
        b = data.batch(range(len(data)))
        assert torch.equal(image_labels(b["target"]), b["labels"])
        counts = Counter(map(tuple, data.labels.tolist()))
        assert len(counts) == 16 and set(counts.values()) == {2}
        for start in range(0, len(data), 4):
            frames = b["images"][start : start + 4]
            labels = b["labels"][start : start + 4]
            assert torch.equal(frames[:2, 0], frames[2:, 0])
            assert torch.equal(frames[0, 1:], frames[1, 1:])
            assert torch.equal(frames[2, 1:], frames[3, 1:])
            assert not torch.equal(frames[0, 1], frames[2, 1])
            assert torch.equal(labels[:2, :2], labels[2:, :2])
            assert (labels[:2, 2] != labels[2:, 2]).all()
        audit = data.shortcut_audit()
        assert audit["appearance_only_side_accuracy"] == 0.5
        assert audit["initial_frame_only_side_accuracy"] == 0.5
        assert audit["final_frame_only_joint_accuracy"] == 0.5
        assert audit["hidden_frame_only_side_accuracy"] == 0.5
        assert audit["hidden_frame_only_joint_accuracy"] == 0.25
        frame_sets.append(
            {hashlib.sha256(x.tobytes()).hexdigest() for x in data.images[:, 0]}
        )
    assert not frame_sets[0] & frame_sets[1]
    with pytest.raises(ValueError, match="complete"):
        MemoryOutputEpisodes(17, curriculum="relocation")


def test_relocation_metrics_detect_correct_appearance_at_the_wrong_location():
    from pathwm.data.memory_output import MemoryOutputEpisodes
    from experiments.memory_output import score, passes

    b = MemoryOutputEpisodes(16, seed=17, curriculum="relocation").batch(range(32))
    labels = b["labels"]
    logits = torch.cat(
        [torch.nn.functional.one_hot(labels[:, i], n) for i, n in enumerate((4, 2, 2))],
        1,
    ).float()
    arrays = dict(
        target=b["target"],
        labels=labels,
        direct_logits=logits,
        stored_logits=logits,
        teacher_image=b["target"],
    )
    modes = [
        "ordinary",
        "reset",
        "reset_erased",
        "erased_history",
        "reset_swapped",
        "cue_erased",
        "last_seen_erased",
    ]
    for mode in modes:
        ids = torch.arange(32)
        if mode == "reset_swapped":
            ids ^= 1
        elif mode in (
            "reset_erased",
            "erased_history",
            "cue_erased",
            "last_seen_erased",
        ):
            ids *= 0
        arrays[mode + "_logits"] = logits[ids].clone()
        arrays[mode + "_image"] = b["target"][ids].clone()
    metrics = score(arrays, modes, relocation=True)
    assert passes(metrics)
    assert metrics["reset"]["factual_relocation_pair_accuracy"] == 1
    # Copying the static counterpart preserves appearance but misses every move.
    moving = torch.arange(32) % 4 >= 2
    arrays["reset_logits"][moving] = logits[torch.arange(32)[moving] - 2]
    arrays["reset_image"][moving] = b["target"][torch.arange(32)[moving] - 2]
    metrics = score(arrays, modes, relocation=True)
    assert metrics["reset"]["factual_color_accuracy"] == 1
    assert metrics["reset"]["factual_shape_accuracy"] == 1
    assert metrics["reset"]["factual_side_accuracy"] == 0.5
    assert metrics["reset"]["factual_relocation_pair_accuracy"] == 0
    assert metrics["reset"]["image_relocation_pair_accuracy"] == 0
    assert metrics["reset"]["moved_factual_accuracy"] == 0
    assert not passes(metrics)


def model_fixture(normalize=False):
    from experiments.memory_output import build_model
    from pathwm.models.encoders import PyramidEncoder, PatchDetailEncoder
    from pathwm.models.decoders import DenseHead, PatchDetailHead
    from pathwm.models.perception import Perception

    encoder = PyramidEncoder(width=16, levels=2)
    head = DenseHead(
        encoder.feature_spec,
        levels=tuple(encoder.feature_spec),
        channels=3,
        activation="sigmoid",
        retain_statistics=True,
    )
    codec = Perception(PatchDetailEncoder(encoder), {"rgb": PatchDetailHead(head)})
    return build_model(codec.requires_grad_(False), width=16, normalize_input=normalize)


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
    changed = replace(
        history["final"], tokens=torch.randn_like(history["final"].tokens)
    )
    other = model.query(changed, b["images"][:, -1], "reset")
    torch.testing.assert_close(recalled.tokens, other.tokens, atol=0, rtol=0)
    erased = model.query(changed, b["images"][:, -1], "reset_erased")
    torch.testing.assert_close(erased.tokens[0::2], erased.tokens[1::2], atol=0, rtol=0)
    swapped = model.query(history["final"], b["images"][:, -1], "reset_swapped")
    torch.testing.assert_close(swapped.tokens, recalled.tokens[torch.arange(4) ^ 1])
    assert torch.equal(bank.values, before)
    restored = LatentState.from_dict(history["final"].to_dict())
    torch.testing.assert_close(
        model.query(restored, b["images"][:, -1], "reset").tokens, recalled.tokens
    )
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
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.agent.updater.parameters()
    )
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.agent.thinker.parameters()
    )
    assert all(p.grad is None for p in model.teacher.parameters())
    assert all(p.grad is None for p in model.agent.decoders["image"].head.parameters())
    assert all(p.grad is None for p in model.direct.parameters())


@pytest.mark.parametrize(
    "normalize,curriculum,repair",
    [
        (False, "parity", False),
        (True, "parity", False),
        (True, "relocation", False),
        (True, "relocation", True),
    ],
)
def test_training_resume_and_standalone_reload(tmp_path, normalize, curriculum, repair):
    from pathwm.data.memory_output import MemoryOutputEpisodes
    from experiments.memory_output import train, default_settings, load_model
    from tests.test_runs import equal_tree

    pairs = 16 if curriculum == "relocation" else 4
    train_data = MemoryOutputEpisodes(pairs, seed=17, curriculum=curriculum)
    val = MemoryOutputEpisodes(pairs, seed=18, curriculum=curriculum)
    test = MemoryOutputEpisodes(pairs, seed=19, split="test", curriculum=curriculum)

    def run(path, resume=False, stop_after=None):
        torch.manual_seed(21)
        model = model_fixture(normalize)
        settings = default_settings(21)
        settings.update(
            steps=4,
            batch_size=2,
            width=16,
            levels=2,
            depth=1,
            fusion_depth=0,
            normalize_input=normalize,
            curriculum=curriculum,
        )
        with torch.no_grad():
            model.agent.decoders["image"].calibrate(
                model.teacher(train_data.batch(range(8))["target"])
            )
        if normalize:
            from pathwm.models.modalities import Observation

            images = train_data.batch(range(8))["images"]
            model.agent.encoders["image"].calibrate(
                [
                    Observation(images[:, t : t + 1], torch.full((8, 1), float(t)))
                    for t in range(3)
                ]
            )
        if repair:
            from pathwm.models.memory_output import configure_recall_repair

            configure_recall_repair(model)
            with torch.no_grad():
                history = model.observe_history(train_data.batch(range(8))["images"])
            model.agent.memory.calibrate([history["final"].memory.values])
            settings["recall_repair"] = "calibrated"
        result = train(
            model,
            train_data,
            val,
            test,
            output=path,
            settings=settings,
            resume=resume,
            stop_after=stop_after,
        )
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
    html = (tmp_path / "full" / "report.html").read_text()
    assert "Recorded result" in html and "factual accuracy" in html
    assert "Labeled observation, target and output comparison" in html


def test_removing_each_visible_frame_erases_its_counterfactual_signal():
    from pathwm.data.memory_output import MemoryOutputEpisodes

    torch.manual_seed(12)
    model = model_fixture(True).eval()
    images = MemoryOutputEpisodes(16, seed=17, curriculum="relocation").batch(range(4))[
        "images"
    ]
    for mode, permutation in (
        ("cue_erased", [1, 0, 3, 2]),
        ("last_seen_erased", [2, 3, 0, 1]),
    ):
        before = images.clone()
        with torch.no_grad():
            out = model(images, mode)
        assert torch.equal(images, before)
        for key in ("facts", "image"):
            torch.testing.assert_close(out[key], out[key][permutation], atol=0, rtol=0)


def test_input_calibration_preserves_noop_metadata_and_frozen_model():
    from pathwm.data.memory_output import MemoryOutputEpisodes
    from pathwm.models.modalities import Observation
    from pathwm.io import state_hash

    torch.manual_seed(91)
    plain = model_fixture()
    torch.manual_seed(91)
    normalized = model_fixture(True)
    p = {k: v for k, v in plain.named_parameters() if v.requires_grad}
    q = {k: v for k, v in normalized.named_parameters() if v.requires_grad}
    assert p.keys() == q.keys()
    assert all(torch.equal(p[k], q[k]) for k in p)
    images = MemoryOutputEpisodes(4, seed=17).batch(range(8))["images"]
    obs = Observation(images[:, :1], torch.zeros(8, 1))
    encoder = normalized.agent.encoders["image"]
    raw = plain.agent.encoders["image"](obs)
    noop = encoder(obs)
    for a, b in zip(raw.scales, noop.scales):
        for key in ("values", "valid", "times", "content_times", "ends"):
            torch.testing.assert_close(getattr(a, key), getattr(b, key), atol=0, rtol=0)
        assert a.grid == b.grid
    a, b = plain(images)["image"], normalized(images)["image"]
    torch.testing.assert_close(a, b, atol=0, rtol=0)
    a.square().mean().backward()
    b.square().mean().backward()
    for k in p:
        if p[k].grad is None:
            assert q[k].grad is None
        else:
            torch.testing.assert_close(p[k].grad, q[k].grad, atol=0, rtol=0)
    before = state_hash(normalized.teacher)
    encoder.calibrate([obs])
    fixed = state_hash(encoder)
    trace = {}
    out = encoder(obs, trace=trace)
    torch.testing.assert_close(
        trace["scale.0.values"], out.scales[0].values, atol=0, rtol=0
    )
    for a, b in zip(raw.scales, out.scales):
        torch.testing.assert_close(
            b.values.mean((0, 1)), torch.zeros(16), atol=2e-5, rtol=0
        )
        assert torch.equal(a.times, b.times) and torch.equal(a.valid, b.valid)
    assert state_hash(encoder) == fixed and state_hash(normalized.teacher) == before
    invalid = Observation(
        torch.full_like(obs.values, float("nan")),
        obs.times,
        torch.zeros_like(obs.times, dtype=torch.bool),
    )
    encoder.calibrate([obs, invalid])
    assert state_hash(encoder) == fixed
    with pytest.raises(ValueError):
        encoder.calibrate([])
    with pytest.raises(ValueError):
        encoder.calibrate(
            [Observation(torch.full_like(obs.values, float("nan")), obs.times)]
        )
