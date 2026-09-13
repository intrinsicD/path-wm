import torch
import numpy as np


def test_cached_stages_are_frozen_causal_and_match_native_readouts():
    from experiments.memory_probes import probe_tokens
    from tests.test_memory_output import model_fixture
    from pathwm.data.memory_output import MemoryOutputEpisodes
    from pathwm.io import state_hash

    model = model_fixture(True).eval()
    before = state_hash(model)
    images = MemoryOutputEpisodes(16, seed=117, curriculum="relocation").batch(
        range(4)
    )["images"]
    values, native = probe_tokens(model, images)
    assert set(values) == {
        "encoder",
        "initial_working",
        "stored_working",
        "stored_all",
        "recall_working",
        "recall_all",
    }
    assert all(x.grad_fn is None and not x.requires_grad for x in values.values())
    assert torch.equal(values["initial_working"][:2], values["initial_working"][2:])
    history = model.observe_history(images)
    assert torch.equal(values["stored_all"], history["final"].memory.values[:, 1])
    with torch.no_grad():
        original = model(images, mode="reset")["facts"]
    torch.testing.assert_close(native["recall"], original, atol=0, rtol=0)
    assert state_hash(model) == before


def test_probe_calibration_and_gradients_are_isolated():
    from experiments.memory_probes import make_probes

    torch.manual_seed(117)
    train = {"a": torch.randn(16, 4, 16), "b": torch.randn(16, 9, 16) * 3 + 5}
    probes = make_probes(train, seed=119)
    pa, pb = (
        dict(probes["a"].head.named_parameters()),
        dict(probes["b"].head.named_parameters()),
    )
    assert all(torch.equal(pa[k], pb[k]) for k in pa)
    assert all(pa[k].data_ptr() != pb[k].data_ptr() for k in pa)
    for name in train:
        p = probes[name]
        assert torch.allclose(p.mean, train[name].mean((0, 1), keepdim=True))
        normalized = (train[name] - p.mean) / p.std
        assert torch.allclose(normalized.mean((0, 1)), torch.zeros(16), atol=1e-5)
    frozen = {k: v.clone() for k, v in probes["a"].named_buffers()}
    probes["a"](train["a"] + 100).sum().backward()
    assert any(x.grad is not None for x in pa.values())
    assert all(x.grad is None for x in pb.values())
    assert all(torch.equal(v, frozen[k]) for k, v in probes["a"].named_buffers())


def test_probe_scores_require_both_relocation_outcomes():
    from experiments.memory_probes import probe_score

    labels = torch.tensor([[0, 0, 0], [2, 1, 1], [0, 0, 1], [2, 1, 0]])
    logits = torch.cat(
        [torch.nn.functional.one_hot(labels[:, i], n) for i, n in enumerate((4, 2, 2))],
        1,
    ).float()
    assert probe_score(logits, labels)["side_relocation_pair_accuracy"] == 1
    logits[2:] = logits[:2].clone()
    result = probe_score(logits, labels)
    assert result["side_accuracy"] == 0.5
    assert result["color_accuracy"] == result["shape_accuracy"] == 1
    assert result["side_relocation_pair_accuracy"] == 0


def test_probe_training_resume_and_standalone_reload(tmp_path):
    from experiments.memory_probes import (
        train_probes,
        default_settings,
        load_probes,
        evaluate,
    )
    from tests.test_runs import equal_tree

    torch.manual_seed(121)
    labels = torch.tensor([[0, 0, 0], [2, 1, 1], [0, 0, 1], [2, 1, 0]]).repeat(4, 1)
    names = (
        "encoder",
        "initial_working",
        "stored_working",
        "stored_all",
        "recall_working",
        "recall_all",
    )
    caches = {
        split: dict(
            features={k: torch.randn(16, 3, 16) for k in names},
            labels=labels,
            native={k: torch.randn(16, 8) for k in ("encoder", "stored", "recall")},
        )
        for split in ("train", "validation", "test")
    }
    settings = default_settings(122)
    settings.update(steps=4, batch_size=4, wall_seconds=30)

    def run(name, **kwargs):
        train_probes(
            caches,
            output=tmp_path / name,
            settings=settings,
            identity={"fixture": 1},
            **kwargs,
        )
        return torch.load(tmp_path / name / "last.pt", weights_only=True)

    full = run("full")
    run("resumed", stop_after=2)
    resumed = run("resumed", resume=True)
    for key in ("model", "optimizer", "sampler", "torch", "step"):
        equal_tree(full[key], resumed[key])
    model = load_probes(tmp_path / "full/weights.pt")
    _, actual = evaluate(model, caches["test"], "cpu")
    with np.load(tmp_path / "full/predictions.npz") as expected:
        assert all(np.array_equal(v, expected[k]) for k, v in actual.items())
    html = (tmp_path / "full/report.html").read_text()
    assert (
        "side relocation pair accuracy" in html and "probe accessibility only" in html
    )
