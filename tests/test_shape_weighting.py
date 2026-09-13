"""Training shape weights preserve inference and isolate the image producer."""

import json
import numpy as np
import pytest
import torch
from pathwm.data.memory_output import templates, MemoryOutputEpisodes


def test_rectangle_loss_is_symmetric_for_shape_errors_and_keeps_legacy_default():
    from experiments.memory_output import training_image_error, weighted_error

    t, y = templates()
    for color in range(4):
        for side in range(2):
            a = t[((y == torch.tensor([color, 0, side])).all(1))]
            b = t[((y == torch.tensor([color, 1, side])).all(1))]
            assert torch.equal(training_image_error(a, b), weighted_error(a, b))
            assert weighted_error(a, b) < weighted_error(b, a) / 5
            torch.testing.assert_close(
                training_image_error(a, b, "box"),
                training_image_error(b, a, "box"),
                atol=0,
                rtol=0,
            )
            # Independently hand-construct the union rectangle for these templates.
            fg = (a - 40 / 255).abs().amax(1, keepdim=True) > 0.01
            weight = 1 + 9 * fg
            expected = ((a - b).square() * weight).sum((1, 2, 3)) / (
                3 * weight.sum((1, 2, 3))
            )
            torch.testing.assert_close(
                training_image_error(a, b, "box"), expected, atol=0, rtol=0
            )
    blank = torch.full((2, 3, 8, 8), 40 / 255)
    x = blank + 0.05
    torch.testing.assert_close(
        training_image_error(x, blank, "box"), (x - blank).square().mean((1, 2, 3))
    )
    with pytest.raises(ValueError, match="weighting"):
        training_image_error(x, blank, "unknown")


def test_cached_image_only_box_gradients_and_default_objective_match():
    from tests.test_memory_output import model_fixture
    from pathwm.models.memory_output import configure_output_readout, frozen_tensors
    from experiments.memory_output import (
        cache_readout,
        readout_objective,
        weighted_error,
        factual_loss,
    )

    torch.manual_seed(53)
    m = configure_output_readout(model_fixture(True), "native", image_only=True).eval()
    d = MemoryOutputEpisodes(16, seed=54, curriculum="relocation")
    c = cache_readout(m, d, "cpu", mode="mixed")
    ids = [0, 1, 2, 3]
    before = {k: v.clone() for k, v in frozen_tensors(m).items()}
    plain, metrics = readout_objective(m, c, ids, context="mixed", step=1)
    tokens = torch.where(
        torch.tensor([False, True, False, True])[:, None, None],
        c["ordinary_tokens"][ids],
        c["tokens"][ids],
    )
    out = m.output_tokens(tokens)
    expected = (
        factual_loss(out["facts"], c["labels"][ids])
        + weighted_error(out["image"], c["target"][ids]).mean()
        + 0.1
        * m.agent.decoders["image"].latent_loss(
            out["features"], {k: v[ids] for k, v in c["features"].items()}
        )
    )
    assert torch.equal(plain, expected)
    loss, values = readout_objective(
        m, c, ids, context="mixed", step=1, image_weighting="box"
    )
    assert values["foreground_rgb_loss"] == metrics["rgb_loss"]
    loss.backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in m.parameters()
        if p.requires_grad
    )
    assert all(p.grad is None for p in m.parameters() if not p.requires_grad)
    torch.optim.AdamW([p for p in m.parameters() if p.requires_grad]).step()
    assert all(torch.equal(v, frozen_tensors(m)[k]) for k, v in before.items())
    assert torch.equal(m.output_tokens(tokens)["facts"], out["facts"])


def test_cached_box_resume_reload_and_training_report(tmp_path):
    from tests.test_tint_readout import centered_export
    from tests.test_runs import equal_tree
    from pathwm.models.memory_output import (
        configure_output_readout,
        load_model,
        frozen_tensors,
    )
    from experiments.memory_output import train

    _, base, settings = centered_export(tmp_path)
    data = base.with_scenes([{}, {"background_offset": (-8, 0, 12)}])
    settings.update(
        steps=4,
        batch_size=2,
        wall_seconds=120,
        readout_stage="native",
        readout_context="mixed",
        image_only=True,
        image_weighting="box",
        writer_learning=None,
        standardize_output=False,
        recall_repair="identity",
    )

    def run(name, resume=False, stop_after=None):
        torch.manual_seed(55)
        m = configure_output_readout(
            load_model(tmp_path / "weights.pt"), "native", image_only=True
        )
        fixed = {k: v.clone() for k, v in frozen_tensors(m).items()}
        train(
            m,
            data,
            base,
            base,
            settings=settings,
            output=tmp_path / name,
            resume=resume,
            stop_after=stop_after,
        )
        assert all(torch.equal(v, frozen_tensors(m)[k]) for k, v in fixed.items())
        return m, torch.load(tmp_path / name / "last.pt", weights_only=True)

    m, a = run("full")
    run("resumed", stop_after=1)
    _, b = run("resumed", resume=True)
    for key in ["model", "optimizer", "sampler", "torch", "step"]:
        equal_tree(a[key], b[key])
    loaded = load_model(tmp_path / "full/weights.pt")
    assert not any(p.requires_grad for p in loaded.facts.parameters())
    with torch.no_grad():
        for mode in ["ordinary", "reset"]:
            x = base.batch(range(4))["images"]
            for key in ["facts", "image"]:
                assert torch.equal(m(x, mode)[key], loaded(x, mode)[key])
    assert (
        json.loads((tmp_path / "full/run.json").read_text())["identity"]["settings"][
            "image_weighting"
        ]
        == "box"
    )
    assert "Training fit by scene" in (tmp_path / "full/report.html").read_text()
    with np.load(tmp_path / "full/predictions.npz") as z:
        assert all(np.isfinite(z[k]).all() for k in z.files)


def test_cli_accepts_cached_image_only_box_and_preserves_centering(
    tmp_path, monkeypatch
):
    import sys
    import experiments.memory_output as r
    from tests.test_tint_readout import centered_export

    _, _, settings = centered_export(tmp_path)
    out = tmp_path / "fit"

    def capture(model, training, validation, test, **kw):
        out.mkdir()
        assert (
            kw["settings"]["image_weighting"] == "box" and kw["settings"]["image_only"]
        )
        assert kw["settings"]["writer_learning"] is None
        assert kw["settings"]["input_centering"] == settings["input_centering"]
        assert len(training) == 512
        assert not any(p.requires_grad for p in model.facts.parameters())

    monkeypatch.setattr(r, "train", capture)
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
            "--image-only",
            "--image-weighting",
            "box",
            "--train-scenes",
            "neutral",
            "background-cool",
            "--development",
        ],
    )
    r.main()


@pytest.mark.parametrize(
    "flags",
    [
        ["--evaluate-only", "--image-weighting", "box", "--test-seed", "1"],
        ["--image-weighting", "box"],
        [
            "--repair",
            "identity",
            "--readout-stage",
            "native",
            "--readout-context",
            "mixed",
            "--image-weighting",
            "box",
        ],
        [
            "--repair",
            "identity",
            "--readout-stage",
            "native",
            "--readout-context",
            "mixed",
            "--writer-learning",
            "frozen",
            "--image-only",
            "--image-weighting",
            "box",
        ],
    ],
)
def test_cli_rejects_unsupported_weighting_paths(monkeypatch, flags):
    import sys
    from experiments.memory_output import main

    monkeypatch.setattr(
        sys,
        "argv",
        ["recipe", "--weights", "not-needed.pt", "--output", "unused", *flags],
    )
    with pytest.raises(SystemExit) as e:
        main()
    assert e.value.code == 2
