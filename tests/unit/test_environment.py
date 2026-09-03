"""E0 engine behaviour: reset is a function of the seed, save/restore is bit-exact, worlds are independent.

What: the unit tests CLAUDE.md §4 lists for the environment ("save/restore bit-exact including the RNG
stream, >= 2 homotopy classes on the fixed layout"), minus the two parts that cannot land yet (below).
How: the engine is obtained through `build("env")` (tests/conftest.py) and driven only through the
contracts.py Environment Protocol: reset, step, save, restore, render, ground_truth. Every tensor kept
across a call is a clone, so a live buffer handed out by the engine cannot make a comparison pass
(DDR §18: what the engine hands out is the caller's). Actions come from a private torch.Generator.
Why: paired interventions (E0 spec `interventions.paired_by_save_restore`, §9 E0) and planner forks (§7)
are exact only if restore(save()) is; a drifting restore would put the intervention's effect and the
drift into the same counterfactual pair. Deterministic reset is what makes the probe set regenerable
from a committed seed and count (§2 first slice; abi_v1.yaml conformance.probe_set) and a dataset
reproducible from its spec (DDR §19). Independent worlds are what lets N parallel worlds stand in for
N episodes (DDR §18 batched layout); a coupled engine would read off the §14 panel as a model failure.

Deferred, and where that is recorded (DDR §13 step-2 additions; CLAUDE.md Now block):
  - The RNG-stream half of the E0 gate (docs/preregistration.md E0 row: "save/restore exact" including the
    RNG stream, E0 spec `engine.save_restore`). In the deterministic variant the RNG is consumed only in
    reset, so no sequence of contract calls after save() can tell a restored stream from an unrestored
    one; the test lands with the stochastic variant, whose step draws from the stream.
  - The >= 2 homotopy classes on the fixed layout (E0 spec `gate.homotopy_classes_min`) lands with the E0
    freeze task, which fixes the layout it is a property of.
"""
from __future__ import annotations

import torch

import contracts


def _gen(seed: int) -> torch.Generator:
    """A private stream, so the actions do not shift when a builder consumes the global RNG."""
    return torch.Generator().manual_seed(seed)


def _actions(gen: torch.Generator, n: int, abi) -> torch.Tensor:
    """(n, action_dims) float32 uniform in the ABI action range: the 2-D form Environment.step takes."""
    return torch.rand(n, abi.action_dims, generator=gen) * 2 - 1


def _clone_gt(g) -> dict[str, torch.Tensor]:
    """A snapshot of a ground_truth() mapping the engine cannot touch afterwards."""
    return {k: v.clone() for k, v in g.items()}


def _assert_gt_equal(got, want) -> None:
    """Same keys, same dtypes, every ground-truth tensor bit-equal (torch.equal, not allclose: the contract is
    exact; the dtype check because torch.equal promotes, so an int tensor rebuilt as float would compare equal)."""
    assert set(got.keys()) == set(want.keys()), f"ground-truth keys {sorted(got)} != {sorted(want)}"
    for k in want:
        assert got[k].dtype == want[k].dtype, f"ground_truth[{k!r}] dtype {got[k].dtype} != {want[k].dtype}"
        assert torch.equal(got[k], want[k]), f"ground_truth[{k!r}] differs"


def test_reset_is_deterministic_in_the_seed(build):
    """reset(seed) is a function of the seed alone: same seed, same obs and same ground truth; another seed,
    another obs. The probe set (§2 first slice) and the dataset (DDR §19) are regenerated from a seed."""
    env = build("env")
    assert isinstance(env, contracts.Environment)
    obs_a = env.reset(seed=0).clone()      # clone: the second reset must not be able to overwrite it
    gt_a = _clone_gt(env.ground_truth())
    obs_b = env.reset(seed=0)
    assert torch.equal(obs_a, obs_b), "reset(0) twice gives different obs"
    _assert_gt_equal(env.ground_truth(), gt_a)
    assert not torch.equal(obs_a, env.reset(seed=1)), "reset(1) renders the same obs as reset(0)"


def test_save_restore_is_bit_exact_through_steps(build, cfg, abi):
    """restore(save()) reproduces the obs, the ground truth and the whole following trajectory bit for bit.

    Warm-up steps move the state away from the spawn configuration so that the snapshot is not a reset in
    disguise; the recorded trajectory must itself move the state (non-vacuity), or an engine that ignores
    actions would pass. The restore-and-replay is done twice: a restore that installs the snapshot's own
    tensors as the live state (instead of copying them) passes the first pass, and the replay then steps
    the snapshot itself in place, so the second pass restores a moved state and fails.
    """
    env = build("env")
    assert isinstance(env, contracts.Environment)
    N = cfg["env"]["n_worlds"]
    gen = _gen(1000)
    env.reset(seed=0)
    for _ in range(3):
        env.step(_actions(gen, N, abi))
    snap = env.save()
    o_save = env.render().clone()
    g_save = _clone_gt(env.ground_truth())
    actions = [_actions(gen, N, abi) for _ in range(5)]
    rec_obs, rec_gt = [], []
    for a in actions:
        rec_obs.append(env.step(a).clone())
        rec_gt.append(_clone_gt(env.ground_truth()))
    # non-vacuity: the five steps actually moved the physical state
    assert not torch.equal(g_save["full_state"], rec_gt[-1]["full_state"]), "the state did not move in 5 steps"
    for pass_ in range(2):
        env.restore(snap)
        assert torch.equal(env.render(), o_save), f"pass {pass_}: obs after restore"
        _assert_gt_equal(env.ground_truth(), g_save)
        for i, a in enumerate(actions):
            assert torch.equal(env.step(a), rec_obs[i]), f"pass {pass_}: replay step {i} obs differs"
            _assert_gt_equal(env.ground_truth(), rec_gt[i])


def test_worlds_do_not_interact(build, cfg, abi):
    """Changing one world's actions changes that world only.

    From one snapshot, two 4-step runs whose actions differ only in world j = N // 2 (+1 force in the
    first, -1 in the second) give bit-equal obs and ground truth in every other world; full_state is the
    sensitive one, since cross-world coupling is sub-pixel at 64 px and would not show in obs. World j's
    full_state must differ: opposite forces over 4 steps guarantee a difference unless world j's agent
    is fully pinned at spawn, which the fixed layout's spawn region excludes. An engine that ignores
    actions would otherwise show on the §14 panel as model action insensitivity (s(w) = 0, §4 H1).
    """
    env = build("env")
    assert isinstance(env, contracts.Environment)
    N = cfg["env"]["n_worlds"]
    assert N >= 2, "the independence check needs another world to compare"  # torch.equal on empty slices is True
    j = N // 2
    env.reset(seed=0)
    snap = env.save()
    A = _actions(_gen(1000), N, abi)
    A[j] = 1.0
    for _ in range(4):
        obs1 = env.step(A)
    obs1 = obs1.clone()
    gt1 = _clone_gt(env.ground_truth())
    env.restore(snap)
    A2 = A.clone()
    A2[j] = -1.0
    for _ in range(4):
        obs2 = env.step(A2)
    obs2 = obs2.clone()
    gt2 = _clone_gt(env.ground_truth())
    mask = torch.ones(N, dtype=torch.bool)
    mask[j] = False                          # every world except j
    assert torch.equal(obs1[mask], obs2[mask]), "another world's obs changed with world j's actions"
    assert set(gt1) == set(gt2)
    for k in gt1:
        assert torch.equal(gt1[k][mask], gt2[k][mask]), f"ground_truth[{k!r}] of another world changed with world j"
    assert not torch.equal(gt1["full_state"][j], gt2["full_state"][j]), "opposite forces left world j unchanged"
