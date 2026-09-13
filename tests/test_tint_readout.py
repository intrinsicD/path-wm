"""Scene learning preserves task binding and persisted input preprocessing."""

from copy import deepcopy
import json
import numpy as np
import pytest
import torch
from pathwm.data.memory_output import MemoryOutputEpisodes


def test_scene_blocks_preserve_binding_provenance_and_neutral_control():
    d = MemoryOutputEpisodes(16, seed=81, curriculum="relocation")
    identity = deepcopy(d.identity)
    scenes = [
        {},
        {"background_offset": [12, 0, -8]},
        {"background_offset": [-8, 0, 12]},
    ]
    a, c = d.with_scenes(scenes), d.with_scenes([{}] * 3)
    assert len(a) == len(c) == 96
    assert d.identity == identity
    for i, scene in enumerate(scenes):
        sl = slice(i * len(d), (i + 1) * len(d))
        assert np.array_equal(a.images[sl], d.with_scene(**scene).images)
        assert np.array_equal(c.images[sl], d.images)
        assert np.array_equal(a.targets[sl], d.targets)
        assert np.array_equal(a.labels[sl], d.labels)
    actual, expected = a.shortcut_audit(), d.shortcut_audit()
    assert actual.pop("tuple_counts") == {
        k: 3 * v for k, v in expected.pop("tuple_counts").items()
    }
    assert actual == expected
    assert a.identity["input_augmentation"]["source_data"] == identity
    snapshot = deepcopy(a.identity)
    scenes[1]["background_offset"][0] = 10
    d.identity["seed"] = -1
    assert a.identity == snapshot
    with pytest.raises(ValueError, match="untransformed"):
        a.with_scenes([{}])


@pytest.mark.parametrize(
    "scenes", [[], "neutral", [None], [{"background_offset": [255, 0, 0]}]]
)
def test_scene_blocks_reject_invalid_or_clipping(scenes):
    d = MemoryOutputEpisodes(16, seed=82, curriculum="relocation")
    with pytest.raises(ValueError):
        d.with_scenes(scenes)


def centered_export(tmp_path):
    from experiments.memory_output import center_from_training, default_settings
    from tests.test_memory_output import model_fixture

    m = model_fixture(True)
    d = MemoryOutputEpisodes(16, seed=83, curriculum="relocation")
    config = center_from_training(m, d)
    settings = dict(
        default_settings(83),
        width=16,
        levels=2,
        depth=1,
        fusion_depth=0,
        normalize_input=True,
        curriculum="relocation",
        input_centering=config,
    )
    torch.save(dict(model=m.state_dict(), settings=settings), tmp_path / "weights.pt")
    (tmp_path / "run.json").write_text(
        json.dumps(dict(identity=dict(settings=settings, data=dict(train=d.identity))))
    )
    return m, d, settings


def test_centering_standalone_reload_and_mismatch_guards(tmp_path):
    from pathwm.models.memory_output import load_model, configure_input_centering

    m, d, settings = centered_export(tmp_path)
    loaded = load_model(tmp_path / "weights.pt")
    assert all(
        torch.equal(v, loaded.state_dict()[k]) for k, v in m.state_dict().items()
    )
    base = loaded.agent.encoders["image"].base
    configure_input_centering(loaded, settings["input_centering"])
    assert loaded.agent.encoders["image"].base is base
    bad = deepcopy(settings["input_centering"])
    bad["reference_rgb"][0] += 0.01
    with pytest.raises(ValueError, match="reference"):
        configure_input_centering(loaded, bad)
    bad["kind"] = "unknown"
    with pytest.raises(ValueError, match="kind"):
        configure_input_centering(loaded, bad)
    record = torch.load(tmp_path / "weights.pt", weights_only=True)
    record["model"]["agent.encoders.image.reference"].add_(0.01)
    torch.save(record, tmp_path / "bad.pt")
    with pytest.raises(ValueError, match="reference"):
        load_model(tmp_path / "bad.pt")


@pytest.mark.parametrize("explicit", [False, True])
def test_embedded_centering_coverage_applies_without_refitting(
    tmp_path, monkeypatch, explicit
):
    import experiments.memory_output as r

    m, d, settings = centered_export(tmp_path)
    monkeypatch.setattr(
        r, "center_from_training", lambda *a: pytest.fail("refit reference")
    )
    monkeypatch.setattr(
        r, "evaluate", lambda *a, **k: pytest.fail("scored rejected data")
    )
    bad = d.with_scene(background_offset=(64, 64, 64))
    out = tmp_path / "evaluation"
    assert (
        r.evaluate_export(
            tmp_path / "weights.pt", bad, output=out, center_input=explicit
        )
        is None
    )
    result = json.loads((out / "result.json").read_text())
    assert result["completed"] and not result["task_scored"] and not result["gate"]
    assert not (out / "predictions.npz").exists()
    assert json.loads((out / "run.json").read_text())["identity"]["settings"][
        "input_centering"
    ] == json.loads(json.dumps(settings["input_centering"]))


def test_centered_scene_cache_live_gradients_and_preflight(tmp_path):
    from experiments.memory_output import cache_readout, readout_objective, train
    from pathwm.models.memory_output import configure_output_readout, frozen_tensors

    m, d, settings = centered_export(tmp_path)
    configure_output_readout(m, "native")
    a = d.with_scenes(
        [{}, {"background_offset": (12, 0, -8)}, {"background_offset": (-8, 0, 12)}]
    )
    cache = cache_readout(m, a, "cpu", mode="mixed")
    ids = [95, 0, 33, 64]
    b = a.batch(ids)
    with torch.no_grad():
        h = m.observe_history(b["images"])
        for mode, key in [("ordinary", "ordinary_tokens"), ("reset", "tokens")]:
            assert torch.equal(
                cache[key][ids],
                m.working(m.query(h["final"], b["images"][:, -1], mode)),
            )
    fixed = {k: v.clone() for k, v in frozen_tensors(m).items()}
    loss, _ = readout_objective(m, cache, ids, context="mixed", step=1)
    loss.backward()
    assert all(p.grad is None for p in m.parameters() if not p.requires_grad)
    torch.optim.AdamW([p for p in m.parameters() if p.requires_grad]).step()
    assert all(torch.equal(v, fixed[k]) for k, v in frozen_tensors(m).items())
    settings.update(readout_stage="native", readout_context="mixed", batch_size=2)
    with pytest.raises(ValueError, match="coverage"):
        train(
            m,
            a,
            d,
            d.with_scene(background_offset=(64,) * 3),
            output=tmp_path / "bad",
            settings=settings,
        )
    assert not (tmp_path / "bad").exists()


@pytest.mark.parametrize(
    "flags",
    [
        ["--center-input"],
        ["--train-scenes", "neutral"],
        ["--evaluate-only", "--train-scenes", "neutral"],
        ["--train-input-offsets", "0", "--train-scenes", "neutral"],
        ["--writer-learning", "trainable", "--train-scenes", "neutral"],
        ["--standardize-output", "--train-scenes", "neutral"],
    ],
)
def test_scene_cli_rejects_incompatible_policies(monkeypatch, flags):
    import sys
    import experiments.memory_output as r

    args = ["recipe", "--weights", "unused.pt", "--output", "unused"]
    if flags[0] not in ("--center-input", "--train-scenes", "--evaluate-only"):
        args += [
            "--repair",
            "identity",
            "--readout-stage",
            "native",
            "--readout-context",
            "mixed",
        ]
    monkeypatch.setattr(sys, "argv", args + flags)
    with pytest.raises(SystemExit) as e:
        r.main()
    assert e.value.code == 2


def test_scene_cli_uses_original_centering_and_correct_augmented_data(
    tmp_path, monkeypatch
):
    import sys
    import experiments.memory_output as r
    from pathwm.models.memory_output import PixelMedianCentering

    _, d, s = centered_export(tmp_path)
    out = tmp_path / "fit"

    def capture(model, training, validation, test, **kw):
        out.mkdir()
        assert isinstance(model.agent.encoders["image"], PixelMedianCentering)
        assert not isinstance(model.agent.encoders["image"].base, PixelMedianCentering)
        assert kw["settings"]["input_centering"] == s["input_centering"]
        assert (
            len(training) == 768
            and training.identity["input_augmentation"]["source_data"]["seed"] == 17701
        )
        assert (
            training.identity["input_augmentation"]["kind"] == "whole-history-scenes-v1"
        )
        assert kw["settings"]["train_scenes"] == [
            "neutral",
            "background-tint",
            "background-cool",
        ]
        assert (
            kw["settings"]["steps"] == 16 and not kw["settings"]["standardize_output"]
        )

    monkeypatch.setattr(r, "train", capture)
    monkeypatch.setattr(
        r,
        "center_from_training",
        lambda *a: pytest.fail("recalibrated embedded centering"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "recipe",
            "--weights",
            str(tmp_path / "weights.pt"),
            "--output",
            str(out),
            "--repair",
            "identity",
            "--readout-stage",
            "native",
            "--readout-context",
            "mixed",
            "--center-input",
            "--train-scenes",
            "neutral",
            "background-tint",
            "background-cool",
            "--development",
        ],
    )
    r.main()


def test_resume_rejects_restored_reference_mismatch_before_updates(
    tmp_path, monkeypatch
):
    from experiments.memory_output import train
    from pathwm.models.memory_output import load_model, configure_output_readout
    from pathwm.io import file_hash

    _, data, settings = centered_export(tmp_path)
    settings.update(
        steps=4,
        batch_size=2,
        readout_stage="native",
        readout_context="mixed",
        recall_repair="identity",
        wall_seconds=120,
    )

    def run(resume=False):
        model = configure_output_readout(load_model(tmp_path / "weights.pt"), "native")
        return train(
            model,
            data,
            data,
            data,
            output=tmp_path / "fit",
            settings=settings,
            resume=resume,
            stop_after=1,
        )

    run()
    path = tmp_path / "fit" / "last.pt"
    record = torch.load(path, weights_only=True)
    record["model"]["agent.encoders.image.reference"].add_(0.001)
    torch.save(record, path)
    before = file_hash(path)
    rows = (tmp_path / "fit" / "metrics.jsonl").read_bytes()
    monkeypatch.setattr(
        torch.optim.AdamW,
        "step",
        lambda *a, **k: pytest.fail("Updated with a mismatched centering reference"),
    )
    with pytest.raises(ValueError, match="reference"):
        run(True)
    assert file_hash(path) == before
    assert (tmp_path / "fit" / "metrics.jsonl").read_bytes() == rows
    status = json.loads((tmp_path / "fit" / "status.json").read_text())
    assert (
        status["result"] == "failed"
        and status["step"] == 1
        and "reference" in status["error"]
    )
