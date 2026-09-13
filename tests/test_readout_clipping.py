"""Factual learning must not silently change the image branch's optimizer step."""

from copy import deepcopy
import pytest
import torch
from tests.test_memory_output import model_fixture
from pathwm.models.memory_output import configure_output_readout


def test_separate_clipping_matches_reference_and_isolates_factual_gradients():
    from experiments.memory_output import clip_readout_gradients

    base = configure_output_readout(model_fixture(True), "native", image_only=True)
    solo, joint, global_joint = [deepcopy(base) for _ in range(3)]
    joint.facts.requires_grad_(True)
    global_joint.facts.requires_grad_(True)
    for m in [solo, joint, global_joint]:
        for n, p in m.named_parameters():
            if p.requires_grad:
                p.grad = torch.full_like(p, 10.0 if n.startswith("facts.") else 0.25)
    torch.nn.utils.clip_grad_norm_(global_joint.parameters(), 1.0)
    reference = deepcopy(solo)
    for n, p in reference.named_parameters():
        if p.requires_grad:
            p.grad = dict(solo.named_parameters())[n].grad.clone()
    torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.0)
    a, b = clip_readout_gradients(solo), clip_readout_gradients(joint)
    assert a["image_grad_norm"] == b["image_grad_norm"] > 1
    assert a["facts_grad_norm"] == 0 and b["facts_grad_norm"] > 1
    assert a["image_grad_clipped"] == b["image_grad_clipped"] == 1
    for n, p in solo.named_parameters():
        if p.requires_grad:
            assert torch.equal(p.grad, dict(joint.named_parameters())[n].grad)
            assert torch.equal(p.grad, dict(reference.named_parameters())[n].grad)
            assert not torch.equal(
                p.grad, dict(global_joint.named_parameters())[n].grad
            )
    unexpected = deepcopy(joint)
    unexpected.agent.initial.requires_grad_(True)
    with pytest.raises(ValueError, match="readout"):
        clip_readout_gradients(unexpected)
    next(joint.facts.parameters()).grad.fill_(float("nan"))
    with pytest.raises(RuntimeError, match="non-finite"):
        clip_readout_gradients(joint)


def test_factual_backward_has_no_image_producer_path():
    from experiments.memory_output import factual_loss

    m = configure_output_readout(model_fixture(True), "native")
    tokens = torch.randn(4, 6, m.agent.width)
    out = m.output_tokens(tokens)
    factual_loss(
        out["facts"], torch.tensor([[0, 0, 0], [1, 0, 1], [2, 1, 0], [3, 1, 1]])
    ).backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in m.facts.parameters()
    )
    assert all(p.grad is None for p in m.agent.decoders["image"].parameters())
    assert all(p.grad is None for p in m.parameters() if not p.requires_grad)


def test_joint_box_training_preserves_image_updates_and_resumes(tmp_path):
    from tests.test_tint_readout import centered_export
    from tests.test_runs import equal_tree
    from pathwm.models.memory_output import load_model, frozen_tensors
    from experiments.memory_output import train

    _, data, settings = centered_export(tmp_path)
    settings.update(
        steps=4,
        batch_size=2,
        wall_seconds=120,
        readout_stage="native",
        readout_context="mixed",
        writer_learning=None,
        standardize_output=False,
        recall_repair="identity",
        image_weighting="box",
        separate_readout_clipping=True,
    )

    def run(name, image_only=False, resume=False, stop_after=None):
        torch.manual_seed(101)
        m = configure_output_readout(
            load_model(tmp_path / "weights.pt"), "native", image_only=image_only
        )
        before = {k: v.clone() for k, v in frozen_tensors(m).items()}
        train(
            m,
            data,
            data,
            data,
            output=tmp_path / name,
            settings=dict(settings, image_only=image_only),
            resume=resume,
            stop_after=stop_after,
        )
        assert all(torch.equal(v, frozen_tensors(m)[k]) for k, v in before.items())
        return m, torch.load(tmp_path / name / "last.pt", weights_only=True)

    full, a = run("joint")
    run("resume", stop_after=1)
    _, b = run("resume", resume=True)
    for key in ["model", "optimizer", "sampler", "torch", "step"]:
        equal_tree(a[key], b[key])
    solo, c = run("solo", image_only=True)
    for n, p in full.agent.decoders["image"].state_dict().items():
        assert torch.equal(p, solo.agent.decoders["image"].state_dict()[n])
    assert any(
        not torch.equal(p, dict(solo.facts.named_parameters())[n])
        for n, p in full.facts.named_parameters()
    )

    # Optimizer entries are in model parameter order, not assumed numeric IDs.
    def image_state(model, checkpoint):
        names = [n for n, p in model.named_parameters() if p.requires_grad]
        ids = checkpoint["optimizer"]["param_groups"][0]["params"]
        return {
            n: checkpoint["optimizer"]["state"][i]
            for n, i in zip(names, ids)
            if n.startswith("agent.decoders.image.")
        }

    equal_tree(image_state(full, a), image_state(solo, c))
    loaded = load_model(tmp_path / "joint/weights.pt")
    assert all(p.requires_grad for p in loaded.facts.parameters())
    with torch.no_grad():
        x = data.batch(range(4))["images"]
        for mode in ["ordinary", "reset"]:
            for key in ["facts", "image"]:
                assert torch.equal(full(x, mode)[key], loaded(x, mode)[key])


def test_joint_box_cli_and_guarded_scope(tmp_path, monkeypatch):
    import sys
    import experiments.memory_output as r
    from tests.test_tint_readout import centered_export

    centered_export(tmp_path)
    got = []

    def capture(model, training, validation, test, **kw):
        assert (
            kw["settings"]["separate_readout_clipping"]
            and not kw["settings"]["image_only"]
        )
        assert kw["settings"]["image_weighting"] == "box"
        assert all(p.requires_grad for p in model.facts.parameters())
        got.append(True)

    monkeypatch.setattr(r, "train", capture)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "recipe",
            "--weights",
            str(tmp_path / "weights.pt"),
            "--output",
            str(tmp_path / "joint"),
            "--repair",
            "identity",
            "--readout-stage",
            "native",
            "--readout-context",
            "mixed",
            "--image-weighting",
            "box",
            "--separate-readout-clipping",
        ],
    )
    r.main()
    assert got == [True]


@pytest.mark.parametrize(
    "flags",
    [
        ["--evaluate-only", "--test-seed", "1"],
        ["--repair", "identity", "--readout-stage", "stored"],
        [
            "--repair",
            "identity",
            "--readout-stage",
            "native",
            "--readout-context",
            "mixed",
            "--writer-learning",
            "frozen",
        ],
        [
            "--repair",
            "identity",
            "--readout-stage",
            "native",
            "--readout-context",
            "mixed",
            "--standardize-output",
        ],
    ],
)
def test_separate_clipping_cli_rejects_unsupported_paths(monkeypatch, flags):
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
            "--separate-readout-clipping",
            *flags,
        ],
    )
    with pytest.raises(SystemExit) as e:
        main()
    assert e.value.code == 2
