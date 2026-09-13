import torch


def test_cached_stages_are_frozen_causal_and_match_native_readouts():
    from experiments.memory_probes import probe_tokens
    from tests.test_memory_output import model_fixture
    from pathwm.data.memory_output import MemoryOutputEpisodes
    from pathwm.io import state_hash

    model = model_fixture(True).eval()
    before = state_hash(model)
    images = MemoryOutputEpisodes(16, seed=117, curriculum="relocation").batch(range(4))["images"]
    values, native = probe_tokens(model, images)
    assert set(values) == {"encoder", "initial_working", "stored_working", "stored_all", "recall_working", "recall_all"}
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
    pa, pb = dict(probes["a"].head.named_parameters()), dict(probes["b"].head.named_parameters())
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
