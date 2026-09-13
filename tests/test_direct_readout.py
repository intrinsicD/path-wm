from dataclasses import replace

import pytest
import torch

from tests.test_memory_output import model_fixture
from pathwm.data.memory_output import MemoryOutputEpisodes


def test_latest_stored_working_selection_is_causal_and_order_invariant():
    from pathwm.models.memory_output import configure_output_readout

    m = configure_output_readout(model_fixture(True), "stored").eval()
    x = MemoryOutputEpisodes(16, seed=41, curriculum="relocation").batch(range(4))[
        "images"
    ]
    h = m.observe_history(x)
    s = h["final"]
    bank = s.memory
    original = bank.values.clone()
    q = m.query(s, x[:, -1], "reset")
    assert torch.equal(m.working(q), m.working(h["stored"]))
    reverse = replace(
        s,
        memory=replace(
            bank,
            keys=bank.keys.flip(1),
            values=bank.values.flip(1),
            times=bank.times.flip(1),
            sources=bank.sources[::-1],
        ),
    )
    assert torch.equal(m.working(m.query(reverse, x[:, -1], "reset")), m.working(q))
    times = bank.times.clone()
    times[1] = times[1].flip(0)
    mixed = m.query(replace(s, memory=replace(bank, times=times)), x[:, -1], "reset")
    assert torch.equal(m.working(mixed)[1], m.working(h["initial"])[1])
    tied = m.query(s, x[:, -1], "reset_time_erased")
    assert torch.equal(
        m.working(tied), (m.working(h["initial"]) + m.working(h["stored"])) / 2
    )
    assert torch.count_nonzero(m.working(m.query(s, x[:, -1], "reset_erased"))) == 0
    for times in (bank.times + 4, torch.full_like(bank.times, float("nan"))):
        with pytest.raises(ValueError):
            m.query(replace(s, memory=replace(bank, times=times)), x[:, -1], "reset")
    assert torch.equal(original, bank.values)


@pytest.mark.parametrize("mode", ["reset", "reset_erased", "reset_time_erased"])
@pytest.mark.parametrize("stage", ["native", "stored"])
def test_readout_cache_matches_live_loss_gradients_and_only_heads_learn(stage, mode):
    from pathwm.models.memory_output import configure_output_readout, frozen_tensors
    from experiments.memory_output import (
        cache_readout,
        readout_objective,
        factual_loss,
        weighted_error,
    )

    torch.manual_seed(43)
    m = configure_output_readout(model_fixture(True), stage).eval()
    d = MemoryOutputEpisodes(16, seed=44, curriculum="relocation")
    cache = cache_readout(m, d, "cpu", batch_size=4, mode=mode)
    assert not cache["tokens"].requires_grad
    m.output_normalization.calibrate(cache["tokens"])
    z = cache["tokens"].double()
    torch.testing.assert_close(
        m.output_normalization.mean, z.mean((0, 1), keepdim=True).float()
    )
    torch.testing.assert_close(
        m.output_normalization.std,
        z.std((0, 1), correction=0, keepdim=True).clamp_min(1e-4).float(),
    )
    ids = [0, 1, 2, 3]
    b = d.batch(ids)
    with torch.no_grad():
        h = m.observe_history(b["images"])
        live = m.working(m.query(h["final"], b["images"][:, -1], mode))
    assert torch.equal(cache["tokens"][ids], live)
    fixed = {k: v.clone() for k, v in frozen_tensors(m).items()}
    a, _ = readout_objective(m, cache, ids)
    a.backward()
    grads = {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}
    m.zero_grad(set_to_none=True)
    live_out = m.output_tokens(live)
    with torch.no_grad():
        teacher = m.teacher(b["target"])
    b_loss = (
        factual_loss(live_out["facts"], b["labels"])
        + weighted_error(live_out["image"], b["target"]).mean()
        + 0.1 * m.agent.decoders["image"].latent_loss(live_out["features"], teacher)
    )
    b_loss.backward()
    torch.testing.assert_close(a, b_loss, atol=1e-7, rtol=1e-6)
    for n, p in m.named_parameters():
        if n in grads:
            torch.testing.assert_close(p.grad, grads[n], atol=1e-6, rtol=1e-5)
        if p.requires_grad:
            assert n.startswith(("facts.", "agent.decoders.image."))
        else:
            assert p.grad is None
    torch.optim.AdamW([p for p in m.parameters() if p.requires_grad]).step()
    assert all(torch.equal(v, fixed[k]) for k, v in frozen_tensors(m).items())


def test_switching_readout_stage_resets_previous_scaling():
    from pathwm.models.memory_output import configure_output_readout

    m = configure_output_readout(model_fixture(True), "stored")
    m.output_normalization.mean.fill_(5)
    m.output_normalization.std.fill_(2)
    configure_output_readout(m, "native")
    assert torch.count_nonzero(m.output_normalization.mean) == 0
    assert torch.equal(
        m.output_normalization.std, torch.ones_like(m.output_normalization.std)
    )


@pytest.mark.parametrize("step", [1, 2])
def test_mixed_cache_alignment_live_gradients_and_context_phase(step):
    from experiments.memory_output import cache_readout, readout_objective
    from pathwm.models.memory_output import configure_output_readout, frozen_tensors

    torch.manual_seed(48)
    m = configure_output_readout(model_fixture(True), "native").eval()
    d = MemoryOutputEpisodes(16, seed=49, curriculum="relocation")
    cache = cache_readout(m, d, "cpu", batch_size=4, mode="mixed")
    ids = [11, 2, 11, 0]  # reordered and repeated histories must retain their targets
    ordinary = (torch.arange(4) + step) % 2 == 0
    b = d.batch(ids)
    with torch.no_grad():
        h = m.observe_history(b["images"])
        a = m.working(m.query(h["final"], b["images"][:, -1], "ordinary"))
        r = m.working(m.query(h["final"], b["images"][:, -1], "reset"))
        assert torch.equal(cache["tokens"][ids], r)
        assert torch.equal(cache["ordinary_tokens"][ids], a)
        assert not torch.equal(a, r)
        assert torch.equal(cache["labels"][ids], b["labels"])
        assert torch.equal(cache["target"][ids], b["target"])
        teacher = m.teacher(b["target"])
        assert all(
            torch.equal(v, cache["features"][k][ids]) for k, v in teacher.items()
        )
    assert not cache["ordinary_tokens"].requires_grad
    fixed = {k: v.clone() for k, v in frozen_tensors(m).items()}
    loss, metrics = readout_objective(m, cache, ids, context="mixed", step=step)
    assert metrics["ordinary_examples"] == metrics["reset_examples"] == 2
    assert metrics["first_ordinary"] == int(ordinary[0])
    loss.backward()
    grads = {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}
    m.zero_grad(set_to_none=True)
    live = dict(
        cache,
        tokens=torch.where(ordinary[:, None, None], a, r),
        labels=b["labels"],
        target=b["target"],
        features=teacher,
    )
    expected, _ = readout_objective(m, live, range(4))
    expected.backward()
    torch.testing.assert_close(loss, expected, atol=1e-7, rtol=1e-6)
    for n, p in m.named_parameters():
        if n in grads:
            torch.testing.assert_close(p.grad, grads[n], atol=1e-6, rtol=1e-5)
        else:
            assert p.grad is None
    torch.optim.AdamW([p for p in m.parameters() if p.requires_grad]).step()
    assert all(torch.equal(v, fixed[k]) for k, v in frozen_tensors(m).items())
    with pytest.raises(ValueError, match="even"):
        readout_objective(m, cache, [0], context="mixed", step=step)


def test_training_replay_preserves_runtime_bank_and_exposes_earlier_write_gradients():
    from pathwm.models.memory_output import configure_output_readout, frozen_tensors
    from experiments.memory_output import live_readout_objective
    from pathwm.models.modalities import Observation

    torch.manual_seed(54)
    m = configure_output_readout(
        model_fixture(True), "native", train_writer=True
    ).eval()
    b = MemoryOutputEpisodes(16, seed=55, curriculum="relocation").batch(range(4))
    fixed = {k: v.clone() for k, v in frozen_tensors(m).items()}
    previous = None
    optimizer = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad])
    for step in (1, 2):
        h = m.observe_history(b["images"], memory_grad=True)
        runtime = m.observe_history(b["images"])
        bank = h["final"].memory
        assert (
            bank.values.requires_grad
            and not runtime["final"].memory.values.requires_grad
        )
        for k in ("values", "keys", "times"):
            assert torch.equal(getattr(bank, k), getattr(runtime["final"].memory, k))
        assert bank.sources == runtime["final"].memory.sources
        assert not bank.keys.requires_grad and not bank.times.requires_grad
        assert torch.equal(bank.values[:, 0], h["initial"].tokens)
        assert torch.equal(bank.values[:, 1], h["stored"].tokens)
        for mode in ("ordinary", "reset"):
            a = m.query(h["final"], b["images"][:, -1], mode)
            r = m.query(runtime["final"], b["images"][:, -1], mode)
            assert torch.equal(a.tokens, r.tokens)
        # Cut the live query/state path: only the actual past values can teach writes.
        with torch.no_grad():
            fresh = m.agent.observe(
                m.agent.initial_state(4, time=2),
                {"image": Observation(b["images"][:, -1:], torch.full((4, 1), 2.0))},
                time=2,
            )
        q = replace(fresh, memory=bank)
        out = m.output(m.agent.think(q, steps=2))
        loss = out["facts"].square().mean() + out["image"].square().mean()
        grads = torch.autograd.grad(
            loss, (h["initial"].tokens, h["stored"].tokens), retain_graph=True
        )
        assert all(g.abs().sum() > 0 for g in grads)
        erased = m.output(m.agent.think(replace(q, memory=None), steps=2))
        grads = torch.autograd.grad(
            erased["facts"].square().mean(),
            (h["initial"].tokens, h["stored"].tokens),
            allow_unused=True,
        )
        assert grads == (None, None)
        if previous is not None:
            assert not torch.equal(previous, bank.values)
        previous = bank.values.detach().clone()
        optimizer.zero_grad(set_to_none=True)
        actual, metrics = live_readout_objective(m, b, step=step)
        assert metrics["ordinary_examples"] == metrics["reset_examples"] == 2
        actual.backward()
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in m.agent.updater.parameters()
        )
        assert all(p.grad is None for p in m.parameters() if not p.requires_grad)
        optimizer.step()
        assert all(torch.equal(v, fixed[k]) for k, v in frozen_tensors(m).items())
        assert m.agent.initial_state(4).memory is None
        assert all(v.grad_fn is None for v in m.buffers())


def test_frozen_live_writer_policy_equals_cached_mixed_head_training():
    from pathwm.models.memory_output import configure_output_readout
    from experiments.memory_output import (
        cache_readout,
        readout_objective,
        live_readout_objective,
    )

    torch.manual_seed(56)
    m = configure_output_readout(model_fixture(True), "native").eval()
    d = MemoryOutputEpisodes(16, seed=57, curriculum="relocation")
    cache = cache_readout(m, d, "cpu", batch_size=4, mode="mixed")
    ids = [10, 7, 2, 7]
    a, _ = readout_objective(m, cache, ids, context="mixed", step=3)
    a.backward()
    grads = {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}
    m.zero_grad(set_to_none=True)
    b, _ = live_readout_objective(m, d.batch(ids), step=3)
    b.backward()
    torch.testing.assert_close(a, b, atol=1e-7, rtol=1e-6)
    for n, p in m.named_parameters():
        if n in grads:
            torch.testing.assert_close(p.grad, grads[n], atol=1e-6, rtol=1e-5)


def test_joint_observer_reader_gradients_and_freeze_contract():
    from pathwm.models.memory_output import configure_output_readout, frozen_tensors
    from experiments.memory_output import live_readout_objective

    torch.manual_seed(58)
    m = configure_output_readout(
        model_fixture(True), "native", train_writer=True, train_thinker=True
    )
    fixed = {k: v.clone() for k, v in frozen_tensors(m).items()}
    b = MemoryOutputEpisodes(16, seed=59, curriculum="relocation").batch(range(4))
    loss, _ = live_readout_objective(m, b, step=1)
    loss.backward()
    for module in (m.agent.updater, m.agent.thinker, m.facts):
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters()
        )
    assert all(p.grad is None for p in m.parameters() if not p.requires_grad)
    torch.optim.AdamW([p for p in m.parameters() if p.requires_grad]).step()
    assert all(torch.equal(v, fixed[k]) for k, v in frozen_tensors(m).items())
    configure_output_readout(m, "native", train_writer=True)
    assert all(not p.requires_grad for p in m.agent.thinker.parameters())
    with pytest.raises(ValueError):
        configure_output_readout(m, "native", train_thinker=True)


def test_image_only_learning_preserves_workspace_facts_and_frozen_buffers():
    from pathwm.models.memory_output import configure_output_readout, frozen_tensors
    from experiments.memory_output import live_readout_objective

    torch.manual_seed(60)
    m = configure_output_readout(model_fixture(True), "native", image_only=True).eval()
    b = MemoryOutputEpisodes(16, seed=61, curriculum="relocation").batch(range(4))
    fixed = {k: v.clone() for k, v in frozen_tensors(m).items()}
    before = {mode: m(b["images"], mode) for mode in ("ordinary", "reset")}
    h = m.observe_history(b["images"])
    states = {
        mode: m.query(h["final"], b["images"][:, -1], mode).tokens.clone()
        for mode in before
    }
    trainable = {n: p.clone() for n, p in m.named_parameters() if p.requires_grad}
    assert trainable and all(
        n.startswith("agent.decoders.image.")
        and not n.startswith("agent.decoders.image.head.")
        for n in trainable
    )
    loss, _ = live_readout_objective(m, b, step=1)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.parameters())
    assert all(p.grad is None for p in m.parameters() if not p.requires_grad)
    torch.optim.AdamW([p for p in m.parameters() if p.requires_grad]).step()
    assert all(torch.equal(v, fixed[k]) for k, v in frozen_tensors(m).items())
    assert any(
        not torch.equal(p, trainable[n])
        for n, p in m.named_parameters()
        if n in trainable
    )
    for mode in before:
        assert torch.equal(m(b["images"], mode)["facts"], before[mode]["facts"])
        h = m.observe_history(b["images"])
        assert torch.equal(
            m.query(h["final"], b["images"][:, -1], mode).tokens, states[mode]
        )
    for kw in ({"stage": "stored"}, {"stage": "native", "train_writer": True}):
        with pytest.raises(ValueError, match="Image-only"):
            configure_output_readout(m, image_only=True, **kw)
    configure_output_readout(m, "native")
    assert all(p.requires_grad for p in m.facts.parameters())
