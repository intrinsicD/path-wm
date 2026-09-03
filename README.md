# PATH-WM

**Modular JEPA World Models with Transition-Compatible Latents and Path-Space Planning.**

A research program on whether perception, predictive dynamics and planning can be made independently replaceable by communicating through a transition-compatible learned world-state interface (the *World Latent ABI*), and whether planning over learned world models should treat trajectories as persistent, diverse objects in path space.

Status: pre-registration draft v0.3 (2026-09-03). No code yet.

## Documents

| File | What it is |
|---|---|
| `CLAUDE.md` | The agentic workflow: how the code gets built (working rule 6), code and testing rules, and the **Now** block with the current target, phase and slice. Loaded by the coding agent at the start of every session. |
| `docs/PATH-WM_v0.3.md` | The research document: hypotheses, reference architecture, objectives, planner, experiments E0–E10, evaluation, falsification criteria, novelty boundaries, roadmap. Read this first. |
| `docs/design-decisions.md` | Decision register: for every open architecture question, the options, the v0.3 default, the reason, and the test that would overturn it. |
| `docs/literature-2026.md` | Annotated bibliography by problem area, with `[read]` / `[sweep]` verification flags. |
| `docs/preregistration.md` | Frozen experiment specs (by hash) and the numeric thresholds committed before each experiment runs. |
| `docs/abi/abi_v1.yaml` | The ABI specification: token layout, dimension, normalization, positional convention, action and Δt tokens. |
| `experiments/*.yaml` | One spec per experiment. A spec is frozen by committing its hash into `docs/preregistration.md`. |

## The first gate

> Can a predictor trained with one encoder plan using another?

Freeze the predictor and the inverse-dynamics head trained with encoder A; train only an adapter for encoder B (a CNN, a hybrid, a frozen pretrained ViT) with transition consistency, inverse dynamics and a counterfactual contrast; run the identical planner at equal rollout budget; report success as a curve over adapter size. Details in `docs/PATH-WM_v0.3.md` §18.

## Working rules

1. Interfaces are predesigned; implementations evolve one module per experiment against a frozen reference.
2. Every implementation passes the conformance tests in `tests/conformance/` before it enters an experiment.
3. Planner comparisons report predictor calls and critic calls separately and use budget curves.
4. Labels never enter the world state's objective; decoders are diagnostics only.
5. Thresholds are fixed from pilot variance before an experiment is frozen; negative results are results.
6. We build in vertical slices: tests for the essential parts against the interfaces of rule 1, the simplest thing that works end-to-end, then widen. Every task names the experiment it serves. Details in `CLAUDE.md`.
