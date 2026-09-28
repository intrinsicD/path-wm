"""Contracts of the shared core (docs/shared-core-plan.md, S1; core-design check 1).

All tests use the full configuration (width 128, 2 blocks, 2 rounds, 8 slots)."""

import pytest
import torch

from pathwm.models.core import CoreConfig, CoreState, SharedCore

B, N, E = 3, 10, 24  # batch, evidence tokens, evidence input width


def make(**kw):
    torch.manual_seed(0)
    return SharedCore(CoreConfig(evidence_width=E, action_width=E, **kw)).eval()


def observed(core, g=None):
    g = g or torch.Generator().manual_seed(1)
    state = core.initial_state(B, generator=g)
    evidence = torch.randn(B, N, E, generator=g)
    return core.observe(state, evidence, torch.ones(B, N, dtype=torch.bool)), evidence


def permute_state(state, order):
    kept = {k: getattr(state, k) for k in ("scene", "work", "revision", "work_revision", "hypothetical")}
    logits = None if state.logits is None else state.logits[:, order]
    return CoreState(h=state.h[:, order], logits=logits, presence=state.presence[:, order], **kept)


def test_full_configuration_block_parameters():
    core = make()
    assert sum(p.numel() for p in core.blocks.parameters()) == 529_664
    c = core.config
    assert (c.width, c.heads, c.blocks, c.rounds, c.slots, c.scene, c.work) == (128, 4, 2, 2, 8, 4, 16)
    assert (c.groups, c.classes, c.concept_tokens) == (4, 16, 4)


def test_tied_shares_one_stack_and_untied_has_one_per_operation():
    tied, untied = make(), make(tied=False)
    stacks = {op: id(tied.stack(op)) for op in SharedCore.OPERATIONS}
    assert len(set(stacks.values())) == 1
    stacks = {op: id(untied.stack(op)) for op in SharedCore.OPERATIONS}
    assert len(set(stacks.values())) == 5
    block = sum(p.numel() for p in tied.blocks.parameters())
    assert sum(p.numel() for p in untied.parameters()) - sum(p.numel() for p in tied.parameters()) == 4 * block


def test_zero_duration_without_action_is_exact_identity():
    core = make()
    with torch.no_grad():
        state, _ = observed(core)
        same = core.predict(state, action=None, dt=torch.zeros(B))
    for name in ("h", "logits", "presence", "scene", "work"):
        assert torch.equal(getattr(same, name), getattr(state, name))


def test_action_at_zero_duration_is_applied_once_and_rows_are_independent():
    core = make()
    with torch.no_grad():
        state, _ = observed(core)
        action = torch.randn(B, E)
        present = torch.tensor([True, False, True])
        out = core.predict(state, action=action, action_present=present, dt=torch.zeros(B))
    assert not torch.equal(out.h[0], state.h[0])
    assert torch.equal(out.h[1], state.h[1])  # no action, no time: that row stays exact
    assert not torch.equal(out.h[2], state.h[2])


def test_missing_action_differs_from_zero_action():
    core = make()
    with torch.no_grad():
        state, _ = observed(core)
        dt = torch.ones(B)
        missing = core.predict(state, action=None, dt=dt)
        zero = core.predict(state, action=torch.zeros(B, E), dt=dt)
    assert not torch.allclose(missing.h, zero.h)


def test_slot_permutation_equivariance():
    core = make()
    order = torch.tensor([3, 0, 7, 1, 6, 2, 5, 4])
    with torch.no_grad():
        state, evidence = observed(core)
        valid = torch.ones(B, N, dtype=torch.bool)
        action, dt = torch.randn(B, E), torch.ones(B)
        p = core.predict(state, action=action, dt=dt)
        pp = core.predict(permute_state(state, order), action=action, dt=dt)
        torch.testing.assert_close(pp.h, p.h[:, order], atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(pp.logits, p.logits[:, order], atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(pp.scene, p.scene, atol=1e-5, rtol=1e-4)
        o = core.observe(p, evidence, valid)
        op = core.observe(permute_state(p, order), evidence, valid)
        torch.testing.assert_close(op.logits, o.logits[:, order], atol=1e-5, rtol=1e-4)
        t = core.think(o)
        tp = core.think(permute_state(o, order))
        torch.testing.assert_close(tp.state.work, t.state.work, atol=1e-5, rtol=1e-4)


def test_branch_isolation():
    core = make()
    with torch.no_grad():
        state, evidence = observed(core)
        before = {k: getattr(state, k).clone() for k in ("h", "logits", "presence", "scene", "work")}
        branch = core.imagine(state, action=torch.randn(B, E), dt=torch.ones(B))
    assert branch.hypothetical and not state.hypothetical
    for k, v in before.items():
        assert torch.equal(getattr(state, k), v)
    with pytest.raises(ValueError, match="hypothetical"):
        core.observe(branch, evidence, torch.ones(B, N, dtype=torch.bool))
    # A branch fed to thinking changes only the workspace of the live state.
    with torch.no_grad():
        result = core.think(state, branches=[branch])
    assert not result.state.hypothetical
    assert result.state.h is state.h and result.state.logits is state.logits


def test_invalid_evidence_is_ignored():
    core = make()
    g = torch.Generator().manual_seed(2)
    with torch.no_grad():
        state = core.initial_state(B, generator=g)
        evidence = torch.randn(B, N, E, generator=g)
        valid = torch.ones(B, N, dtype=torch.bool)
        valid[:, 6:] = False  # e.g. support that becomes available later
        changed = evidence.clone()
        changed[:, 6:] = torch.randn(B, N - 6, E, generator=g) * 100
        a = core.observe(state, evidence, valid)
        b = core.observe(state, changed, valid)
    torch.testing.assert_close(a.logits, b.logits)
    torch.testing.assert_close(a.assignment[..., :6], b.assignment[..., :6])
    assert torch.isneginf(a.assignment[..., 6:]).all()


def test_write_rights_of_think_and_apply():
    core = make()
    with torch.no_grad():
        state, _ = observed(core)
        result = core.think(state, goal=torch.randn(B, 2, core.config.width))
        z = core.induce(torch.randn(B, 5, core.config.width), torch.ones(B, 5, dtype=torch.bool))
        out = core.apply(torch.randn(B, 3, core.config.width), z)
    s = result.state
    assert s.h is state.h and s.logits is state.logits and s.presence is state.presence and s.scene is state.scene
    assert not torch.equal(s.work, state.work)
    assert s.work_revision == state.revision
    assert result.finish.shape == (B,) and result.retrieve.shape == (B, core.config.width)
    assert z.shape == (B, core.config.concept_tokens, core.config.width)
    assert out.outcome.shape == (B, 3) and not hasattr(out, "state")


def test_correction_invalidates_workspace():
    core = make()
    with torch.no_grad():
        state, evidence = observed(core)
        thought = core.think(state).state
        assert core.work_valid(thought)
        corrected = core.observe(core.predict(thought, action=torch.randn(B, E), dt=torch.ones(B)),
                                 evidence, torch.ones(B, N, dtype=torch.bool))
    assert corrected.revision == thought.revision + 1
    assert not core.work_valid(corrected)


def test_induce_ignores_invalid_rows_and_empty_support_is_finite():
    core = make()
    w = core.config.width
    tokens = torch.randn(B, 6, w)
    valid = torch.tensor([[1, 1, 1, 0, 0, 0], [1] * 6, [0] * 6], dtype=torch.bool)
    changed = tokens.clone()
    changed[~valid] = 50.0
    with torch.no_grad():
        a, b = core.induce(tokens, valid), core.induce(changed, valid)
    torch.testing.assert_close(a, b)
    assert torch.isfinite(a).all()


def test_state_round_trip_is_exact():
    core = make()
    with torch.no_grad():
        state, _ = observed(core)
        state = core.think(state).state
        back = CoreState.from_dict(state.to_dict())
        a = core.predict(state, action=torch.ones(B, E), dt=torch.ones(B))
        b = core.predict(back, action=torch.ones(B, E), dt=torch.ones(B))
    assert back.revision == state.revision and back.work_revision == state.work_revision
    assert torch.equal(a.h, b.h) and torch.equal(a.logits, b.logits)


def test_gradients_reach_blocks_from_prior_and_posterior():
    core = make().train()
    state, evidence = observed(core)
    prior = core.predict(state, action=torch.randn(B, E), dt=torch.ones(B))
    post = core.observe(prior, evidence, torch.ones(B, N, dtype=torch.bool))
    loss = core.kl(post, prior).mean()
    loss.backward()
    grads = [p.grad for p in core.blocks.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)


def test_continuous_only_has_no_code():
    core = make(continuous_only=True)
    with torch.no_grad():
        state, evidence = observed(core)
        prior = core.predict(state, action=torch.randn(B, E), dt=torch.ones(B))
    assert state.logits is None and prior.logits is None
    assert core.tokens(prior).shape == (B, 8, 128)


def test_no_step_counter_input():
    core = make()
    with torch.no_grad():
        state, _ = observed(core)
        one = core.think(state).state
        again = core.think(state).state
    assert torch.equal(one.work, again.work)  # same input, same output: no hidden counter
