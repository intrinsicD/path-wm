"""Real photographs: split/provenance, causal history and reproducible adaptation."""

import json
import numpy as np
import pytest
import torch


def test_photo_split_selection_keeps_groups_disjoint(tmp_path):
    from pathwm.io import file_hash
    from experiments.real_photo import photo_data

    np.save(tmp_path / "frames.npy", np.zeros((12, 64, 64, 3), dtype=np.uint8))
    manifest = dict(
        schema="curriculum-coco-v1",
        frames_sha256=file_hash(tmp_path / "frames.npy"),
        transform="fixture",
        records=[
            dict(group=i, file=str(i), canonical_sha256=str(i)) for i in range(12)
        ],
        splits=dict(train=list(range(6)), validation=[6, 7, 8], test=[9, 10, 11]),
    )
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    a = photo_data(tmp_path, (4, 2, 2), 13)
    b = photo_data(tmp_path, (4, 2, 2), 13)
    assert all(np.array_equal(a[k].rows, b[k].rows) for k in a)
    assert not set(a["train"].rows) & set(a["test"].rows)
    manifest["records"][9]["group"] = 0
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="group overlap"):
        photo_data(tmp_path, (4, 2, 2), 13)


def test_photo_history_hides_current_target_and_swaps_metrics():
    from experiments.real_photo import photo_history, score

    x = torch.stack([torch.full((3, 64, 64), 0.2), torch.full((3, 64, 64), 0.8)])
    history = photo_history(x)
    assert torch.equal(history[:, 0], x) and torch.equal(history[:, 1], x)
    assert torch.equal(history[0, 2], history[1, 2])
    arrays = dict(
        target=x,
        teacher=x,
        mean=torch.full_like(x, 0.5),
        ordinary=x,
        reset=x,
        erased=torch.full_like(x, 0.5),
        swapped=x.flip(0),
    )
    metrics = score(arrays)
    assert metrics["reset"]["rgb_mse"] == 0
    assert metrics["swapped_target"]["rgb_mse"] == 0
    assert metrics["swapped_original"]["rgb_mse"] > 0.3


def test_real_photo_continuation_freeze_resume_and_export(tmp_path):
    from tests.test_tint_readout import centered_export
    from tests.test_runs import equal_tree
    from pathwm.models.conditional_image import configure_generator
    from pathwm.models.memory_output import (
        load_model,
        frozen_tensors,
        PixelMedianCentering,
    )
    from pathwm.data.images import Frames
    from experiments.real_photo import prepare, settings, train, evaluate

    model, _, config = centered_export(tmp_path)
    config["feature_generator"] = dict(
        objective="flow", depth=1, fusion_depth=1, steps=2, hidden_width=16
    )
    configure_generator(model, config["feature_generator"])
    source = tmp_path / "source.pt"
    torch.save(dict(model=model.state_dict(), settings=config), source)
    frames = np.random.default_rng(21).integers(0, 256, (8, 64, 64, 3), dtype=np.uint8)
    data = Frames(frames, range(4), identity=dict(split="train"))
    validation = Frames(frames, range(4, 8), identity=dict(split="validation"))
    cfg = dict(config, **settings())
    cfg.update(steps=4, batch_size=2, disk_free_gib=0, wall_seconds=120)

    def fit(name, resume=False, stop_after=None):
        m, cache, c = prepare(source, data, cfg)
        assert not isinstance(m.agent.encoders["image"], PixelMedianCentering)
        frozen = {k: v.clone() for k, v in frozen_tensors(m).items()}
        initial = {k: p.clone() for k, p in m.named_parameters() if p.requires_grad}
        train(
            m,
            cache,
            data,
            validation,
            output=tmp_path / name,
            config=c,
            resume=resume,
            stop_after=stop_after,
        )
        assert all(torch.equal(v, frozen_tensors(m)[k]) for k, v in frozen.items())
        assert any(
            not torch.equal(v, dict(m.named_parameters())[k])
            for k, v in initial.items()
        )
        return m, torch.load(tmp_path / name / "last.pt", weights_only=True)

    full, a = fit("full")
    fit("resume", stop_after=1)
    _, b = fit("resume", resume=True)
    for key in ["model", "optimizer", "torch", "sampler", "step"]:
        equal_tree(a[key], b[key])
    loaded = load_model(tmp_path / "full/weights.pt")
    mean = frames[:4].mean(0).transpose(2, 0, 1) / 255
    before = evaluate(full, validation, torch.tensor(mean, dtype=torch.float32), "cpu")
    after = evaluate(loaded, validation, torch.tensor(mean, dtype=torch.float32), "cpu")
    assert all(torch.equal(before[k], after[k]) for k in before)
    loaded.teacher.forward = lambda *_: pytest.fail("target encoder called")
    loaded.agent.encoders["image"].forward = lambda *_: pytest.fail(
        "image encoder called"
    )
    assert torch.isfinite(loaded.agent.decoders["image"](torch.randn(2, 4, 16))).all()
