# PATH-WM

**Modular JEPA World Models with Transition-Compatible Latents and Path-Space Planning.**

A research program on whether perception, predictive dynamics and planning can be made independently replaceable by communicating through a transition-compatible learned world-state interface (the *World Latent ABI*), and whether planning over learned world models should treat trajectories as persistent, diverse objects in path space.

Status: pre-registration draft v0.3 (2026-09-03). First slice in progress (`CLAUDE.md` Now block).

Run an experiment or dev spec with `python run.py <spec.yaml>`; plain `pytest` is the fast test run (CLAUDE.md §4). Every completed seed refreshes the offline visual instrument panel at `runs/experiment_dashboard.html`.

**Reused code.** E1 starts from LeWorldModel (arXiv 2603.19312): https://github.com/lucas-maes/le-wm at commit `8edfeb336732b5f3ce7b8b210d0ba370a09e2cac` (MIT). Adapted parts carry the header line `# adapted from lucas-maes/le-wm@8edfeb3:<path>`; the ViT is written here because LeWM's comes from the `stable_pretraining` dependency.

## Documents

| File | What it is |
|---|---|
| `CLAUDE.md` | The agentic workflow: how the code gets built (working rule 6), code and testing rules, and the **Now** block with the current target, phase and slice. Loaded by the coding agent at the start of every session. |
| `docs/PATH-WM_v0.3.md` | The research document: hypotheses, reference architecture, objectives, planner, experiments E0–E10, evaluation, falsification criteria, novelty boundaries, roadmap. Read this first. |
| `docs/design-decisions.md` | Decision register: for every open architecture question, the options, the v0.3 default, the reason, and the test that would overturn it. |
| `docs/literature-2026.md` | Annotated bibliography by problem area, with `[read]` / `[sweep]` verification flags. |
| `docs/preregistration.md` | Frozen experiment specs (by hash) and the numeric thresholds committed before each experiment runs. |
| `contracts.py` | The §16.1 module signatures as `typing.Protocol`s, plus the environment. Every module and every test imports from it; `world_state/abi.py` is the typed view of the ABI spec. |
| `docs/abi/abi_v1.yaml` | The ABI specification: token layout, dimension, normalization, positional convention, action and Δt tokens. |
| `configs/dev/*.yaml` | Dev copies of a spec for one slice, with the plan in the header comment. Never experiment results. |
| `experiments/*.yaml` | One spec per experiment. A spec is frozen by committing its hash into `docs/preregistration.md`. |
| `viewer/` | Read-only views: the experiment dashboard over run ledgers and a raw-data microscope over the RGB/action inputs selected by a spec. |

## Experiment dashboard

Open `runs/experiment_dashboard.html` directly in a browser after any successful run. It compares every completed seed, keeps development evidence visibly labeled, and provides:

- selected-run outcome cards and action-correctness controls;
- cross-run action sensitivity, counterfactual accuracy, transition error, and parameter comparisons;
- training-objective, counterfactual-accuracy, gradient, and curriculum trajectories;
- visible direction indicators on every chart (`↑` higher is better, `↓` lower is better, `↔` context only);
- exact final-result, run-configuration, and logged-training tables beneath the charts.

The JSON/YAML/JSONL files under `runs/<spec>/<seed>/` remain the authoritative DDR §19 ledger; the HTML is a generated, read-only view and is gitignored with the other run artifacts. To backfill or manually refresh it without rerunning training:

```bash
python -m viewer.dashboard
```

The command uses the installed Data Analytics portable-artifact builder to produce a self-contained file with no server, CDN, or sidecars. It locates the builder from the Codex plugin cache; outside that environment, point `PATH_WM_ARTIFACT_BUILDER` at `deliver_portable_artifact.mjs`. Dashboard validation is part of experiment completion: if packaging fails, raw run artifacts remain intact and the run command exits with the dashboard error instead of silently leaving a stale view.

## Raw data viewer

To inspect what a particular spec actually feeds to training and evaluation, build the offline data viewer:

```bash
python -m viewer.data configs/dev/first_slice_counterfactual.yaml
```

Open `runs/data_viewer.html`. Its tabs separate stored training episodes, same-state counterfactual
training branches, regenerated fixed transition probes, and regenerated held-out counterfactual probes.
Every view includes exact actions, tensor shapes, full population sizes, seeds, and source paths. The
embedded examples are deterministic and evenly spaced; they are only a visual subset, while the model
continues to consume the full validated populations. If the selected spec has counterfactual training
disabled, that tab says so even when another run left a `counterfactual.pt` file in the shared dataset.

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
