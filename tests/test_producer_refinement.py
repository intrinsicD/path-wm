"""Optional image refinement must preserve old models and freeze boundaries."""

import json
import pytest
import torch
from tests.test_memory_output import model_fixture


def test_refinement_starts_exact_preserves_rng_and_learns_in_two_steps():
    torch.manual_seed(97)
    m = model_fixture(True).eval()
    d = m.agent.decoders["image"]
    x = torch.randn(4, 6, m.agent.width)
    before = {k: v.clone() for k, v in d.state_dict().items()}
    with torch.no_grad():
        expected = d.features(x)
    rng = torch.get_rng_state().clone()
    d.enable_refinement()
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(v, d.state_dict()[k]) for k, v in before.items())
    with torch.no_grad():
        actual = d.features(x)
    assert all(torch.equal(expected[k], v) for k, v in actual.items())
    ids = [id(p) for p in d.refinements.parameters()]
    d.enable_refinement()
    assert ids == [id(p) for p in d.refinements.parameters()]
    optimizer = torch.optim.AdamW(d.refinements.parameters(), lr=0.001)
    loss = sum(v.square().mean() for v in d.features(x).values())
    loss.backward()
    for layer in d.refinements.values():
        assert layer[3].weight.grad.abs().sum() > 0
        assert torch.count_nonzero(layer[1].weight.grad) == 0
    optimizer.step()
    optimizer.zero_grad()
    loss = sum(v.square().mean() for v in d.features(x).values())
    loss.backward()
    for layer in d.refinements.values():
        assert layer[1].weight.grad.abs().sum() > 0
    with torch.no_grad():
        assert any(not torch.equal(expected[k], v) for k, v in d.features(x).items())


def test_refined_cached_training_freeze_resume_and_strict_reload(tmp_path):
    from tests.test_tint_readout import centered_export
    from tests.test_runs import equal_tree
    from pathwm.models.memory_output import (
        configure_output_readout,
        load_model,
        frozen_tensors,
    )
    from experiments.memory_output import train

    _, data, settings = centered_export(tmp_path)
    source = tmp_path / "weights.pt"
    legacy = load_model(source)
    assert not legacy.agent.decoders["image"].refinements
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
        producer_refinement=True,
    )

    def run(name, resume=False, stop_after=None):
        torch.manual_seed(98)
        m = load_model(source)
        m.agent.decoders["image"].enable_refinement()
        configure_output_readout(m, "native", image_only=True)
        frozen = {k: v.clone() for k, v in frozen_tensors(m).items()}
        train(
            m,
            data,
            data,
            data,
            output=tmp_path / name,
            settings=settings,
            resume=resume,
            stop_after=stop_after,
        )
        assert all(torch.equal(v, frozen_tensors(m)[k]) for k, v in frozen.items())
        return m, torch.load(tmp_path / name / "last.pt", weights_only=True)

    full, a = run("full")
    run("resume", stop_after=1)
    _, b = run("resume", resume=True)
    for key in ["model", "optimizer", "sampler", "torch", "step"]:
        equal_tree(a[key], b[key])
    loaded = load_model(tmp_path / "full/weights.pt")
    with torch.no_grad():
        x = data.batch(range(4))["images"]
        for mode in ["ordinary", "reset"]:
            for key in ["facts", "image"]:
                assert torch.equal(full(x, mode)[key], loaded(x, mode)[key])
    record = torch.load(tmp_path / "full/weights.pt", weights_only=True)
    record["settings"].pop("producer_refinement")
    torch.save(record, tmp_path / "mismatch.pt")
    with pytest.raises(RuntimeError, match="Unexpected key"):
        load_model(tmp_path / "mismatch.pt")
    assert json.loads((tmp_path / "full/run.json").read_text())["identity"]["settings"][
        "producer_refinement"
    ]


def test_refinement_cli_and_settings_mismatch(tmp_path, monkeypatch):
    import sys
    import experiments.memory_output as r
    from tests.test_tint_readout import centered_export

    _, data, settings = centered_export(tmp_path)
    captured = []

    def capture(model, training, validation, test, **kw):
        captured.append(kw["settings"])
        assert kw["settings"]["producer_refinement"]
        assert model.agent.decoders["image"].refinements
        assert not any(p.requires_grad for p in model.facts.parameters())

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "recipe",
            "--weights",
            str(tmp_path / "weights.pt"),
            "--output",
            str(tmp_path / "fit"),
            "--repair",
            "identity",
            "--readout-stage",
            "native",
            "--readout-context",
            "mixed",
            "--image-only",
            "--image-weighting",
            "box",
            "--refine-image",
        ],
    )
    original = r.train
    monkeypatch.setattr(r, "train", capture)
    r.main()
    assert len(captured) == 1
    m = model_fixture(True)
    with pytest.raises(ValueError, match="refinement"):
        original(
            m,
            data,
            data,
            data,
            settings=dict(settings, producer_refinement=True),
            output=tmp_path / "bad",
        )


@pytest.mark.parametrize(
    "flags",
    [
        ["--evaluate-only", "--test-seed", "1"],
        [
            "--repair",
            "identity",
            "--readout-stage",
            "native",
            "--readout-context",
            "mixed",
        ],
        [
            "--repair",
            "identity",
            "--readout-stage",
            "native",
            "--readout-context",
            "mixed",
            "--image-only",
            "--writer-learning",
            "frozen",
        ],
    ],
)
def test_refinement_cli_rejects_unsupported_paths(monkeypatch, flags):
    import sys
    from experiments.memory_output import main

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "recipe",
            "--weights",
            "not-needed.pt",
            "--output",
            "unused",
            "--refine-image",
            *flags,
        ],
    )
    with pytest.raises(SystemExit) as e:
        main()
    assert e.value.code == 2
