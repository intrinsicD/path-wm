"""Structural conformance of the E0 engine against contracts.py Environment (§9 E0, §16.1, DDR §18).

What: with the dev spec's N worlds at its resolution, reset/step/render hand out batched uint8 RGB
(N, 3, res, res); step takes the ABI action space (abi_v1.yaml actions) over its whole range; what the
engine hands out belongs to the caller and never changes under a later step (DDR §18); render is the
current obs and does not step; ground_truth carries the E0 spec's keys in the layouts the §14 probes and
the viewer consume (Invariant 11: ground truth is for probes, overlays and goal masks only).
How: env = build("env") from configs/dev/first_slice.yaml; synthetic actions from the private
Generator; aliasing is caught by keeping clones and comparing after further steps and a further
render / ground_truth call. Structural layer (CLAUDE.md §4): random init, CPU, seconds, no marker.
Why: the slice's numbers, s(w) (§4 H1) and transition error (§6.2), are computed on transitions this
engine emits; a layout or aliasing bug there would be read off the §14 panel as a model symptom.
Save/restore bit-exactness is NOT tested here: CLAUDE.md §4 lists it under the unit tests
(tests/unit/test_environment.py); ≥ 2 homotopy classes lands with the E0 freeze task (§20 gate).
"""
from __future__ import annotations

from pathlib import Path

import torch
import yaml

import contracts

ROOT = Path(__file__).resolve().parents[2]
E0_SPEC = ROOT / "experiments" / "E0_causal_world.yaml"


def _actions(gen: torch.Generator, n: int, abi) -> torch.Tensor:
    """(n, action_dims) float32 uniform in the ABI range: the 2-D form Environment.step takes."""
    return torch.rand(n, abi.action_dims, generator=gen) * 2 - 1


def _bound(n: int, abi, value: float) -> torch.Tensor:
    """(n, action_dims) float32 with every entry at one end of the action range."""
    return torch.full((n, abi.action_dims), value)


def _assert_obs(x: torch.Tensor, n: int, res: int) -> None:
    assert isinstance(x, torch.Tensor)
    assert tuple(x.shape) == (n, 3, res, res), f"obs shape {tuple(x.shape)} != ({n}, 3, {res}, {res})"
    assert x.dtype is torch.uint8, f"obs dtype {x.dtype} != torch.uint8"


def _clone_gt(g) -> dict[str, torch.Tensor]:
    return {k: v.clone() for k, v in g.items()}


def test_reset_returns_batched_uint8_rgb(cfg, build):
    """reset seeds every world and returns obs (N, 3, res, res) uint8 with N = the spec's n_worlds."""
    env = build("env")
    assert isinstance(env, contracts.Environment)
    N, res = cfg["env"]["n_worlds"], cfg["env"]["resolution"]
    obs = env.reset(seed=0)
    _assert_obs(obs, N, res)
    assert N == env.n_worlds


def test_step_takes_abi_actions_and_returns_obs(cfg, abi, build, gen):
    """step consumes (N, action_dims) float32 in the ABI range, the bounds included, and returns the next obs."""
    env = build("env")
    assert isinstance(env, contracts.Environment)
    N, res = cfg["env"]["n_worlds"], cfg["env"]["resolution"]
    env.reset(seed=0)
    a = _actions(gen, N, abi)
    abi.check_actions(a)  # the synthetic action is a valid ABI action before the engine sees it
    _assert_obs(env.step(a), N, res)
    for value in abi.action_range:  # all -1 and all +1: the range is closed
        _assert_obs(env.step(_bound(N, abi, value)), N, res)


def test_returned_tensors_do_not_alias_live_state(cfg, abi, build, gen):
    """Every tensor the engine hands out is the caller's: a later step never changes it (DDR §18).

    Why: a collector storing live buffers would produce a dataset with o_t == o_{t+1}, read off the
    panel as "action insensitivity" (§14) instead of an engine bug.
    """
    env = build("env")
    assert isinstance(env, contracts.Environment)
    N = cfg["env"]["n_worlds"]
    obs0 = env.reset(seed=0)
    obs1 = env.step(_actions(gen, N, abi))
    r = env.render()
    g = env.ground_truth()
    obs0_c, obs1_c, r_c, g_c = obs0.clone(), obs1.clone(), r.clone(), _clone_gt(g)
    # consecutive observations must not share a buffer, or o_t == o_{t+1} by construction
    assert obs1.untyped_storage().data_ptr() != obs0.untyped_storage().data_ptr(), (
        "reset obs and step obs share one storage: the engine hands out its own frame buffer (DDR §18)"
    )
    # one max-force step, then a random one: (+1, -1) is an exact round trip for a position-controlled
    # engine, which would leave a live full_state buffer back at its checked value
    env.step(_bound(N, abi, abi.action_range[1]))
    env.step(_actions(gen, N, abi))
    # a buffer rewritten only on the next render / ground_truth call is live too
    env.render()
    env.ground_truth()
    assert torch.equal(obs0, obs0_c), "reset obs changed under a later step"
    assert torch.equal(obs1, obs1_c), "step obs changed under a later step"
    assert torch.equal(r, r_c), "render obs changed under a later step"
    for k in g_c:
        assert torch.equal(g[k], g_c[k]), f"ground_truth[{k!r}] changed under a later step"


def test_render_is_the_current_obs_and_does_not_step(cfg, abi, build, gen):
    """render returns exactly the obs of the last step and leaves the world where it is."""
    env = build("env")
    assert isinstance(env, contracts.Environment)
    N = cfg["env"]["n_worlds"]
    env.reset(seed=0)
    obs1 = env.step(_actions(gen, N, abi))
    assert torch.equal(env.render(), obs1)
    g = _clone_gt(env.ground_truth())
    assert torch.equal(env.render(), obs1), "render advanced the world"
    g_again = env.ground_truth()
    for k in g:
        assert torch.equal(g_again[k], g[k]), f"ground_truth[{k!r}] changed across render calls"


def test_ground_truth_layout(cfg, build):
    """ground_truth carries the E0 spec's keys, each batched over N, in the layout the probes consume.

    The `ground_truth:` list is read from experiments/E0_causal_world.yaml: this reads, not runs, a spec
    (CLAUDE.md §4: experiments are not tests), and E0 is frozen by hash at the phase gate, after which
    the dependency is on an immutable file. segmentation is (N, res, res) with an integer dtype (pixel
    ids, the §14 per-token occupancy probe target); full_state is floating; homotopy_signature has
    leading dim N only, its dtype being the engine's, since winding numbers accumulate as floats
    along a path (contracts.py Environment.ground_truth, DDR §13 step-2 additions).
    """
    env = build("env")
    assert isinstance(env, contracts.Environment)
    N, res = cfg["env"]["n_worlds"], cfg["env"]["resolution"]
    required = yaml.safe_load(E0_SPEC.read_text())["ground_truth"]
    env.reset(seed=0)
    g = env.ground_truth()
    assert set(required) <= set(g.keys()), f"ground_truth keys {sorted(g)} lack {sorted(set(required) - set(g))}"
    for k, v in g.items():
        assert isinstance(v, torch.Tensor), f"ground_truth[{k!r}] is not a tensor"
        assert v.ndim >= 1 and v.shape[0] == N, f"ground_truth[{k!r}] shape {tuple(v.shape)} has no leading dim N = {N}"
    seg = g["segmentation"]
    assert tuple(seg.shape) == (N, res, res), f"segmentation shape {tuple(seg.shape)} != ({N}, {res}, {res})"
    assert not torch.is_floating_point(seg), f"segmentation dtype {seg.dtype} is not an integer dtype"
    assert torch.is_floating_point(g["full_state"]), f"full_state dtype {g['full_state'].dtype} is not floating"
    assert g["homotopy_signature"].shape[0] == N  # dtype deliberately unconstrained: it is the engine's
