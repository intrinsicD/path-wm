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

## Common-base real A/V development

The first ABI-v2 representation slice uses the official TAU Urban Audio-Visual Scenes 2021 corpus. The
small `examples` archive is only an end-to-end development fixture; promotion requires the complete
development archive. The resumable full-corpus downloader verifies every official byte size and MD5,
extracts one archive at a time, and removes each verified staging ZIP by default:

```bash
scripts/download_tau_urban_av_2021.sh
```

Partial downloads and completion markers live under the ignored
`data/tau_urban_av_2021/.archives/` directory. Rerun the command after an interruption to resume. Set
`PATH_WM_TAU_KEEP_ARCHIVES=1` if the 100.2 GiB of verified ZIPs should be retained after extraction.
The resulting source layout is:

```text
data/tau_urban_av_2021/raw/audio/*.wav
data/tau_urban_av_2021/raw/video/*.mp4
data/tau_urban_av_2021/raw/examples/*.mp4
data/tau_urban_av_2021/metadata/meta.csv
data/tau_urban_av_2021/metadata/evaluation_setup/*.csv
```

Build the checksummed manifest and normalized per-clip shards once, then run the gated R0 training:

```bash
python -m training.av_data configs/dev/common_base.yaml
python run.py configs/dev/common_base.yaml --device auto
```

Raw media, normalized shards, manifests, checkpoints, and run ledgers are local ignored artifacts. Scene
labels are retained only as manifest provenance; representation batches never expose them.

## Experiment dashboard

Open `runs/experiment_dashboard.html` directly in a browser after any successful run. It compares every completed seed, keeps development evidence visibly labeled, and provides:

- selected-run outcome cards and action-correctness controls;
- held-out representation collapse, temporal prediction/retrieval, and A/V synchrony controls;
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


### Full-corpus R0 and recovery

The common evidence frontend serves H1 through E1_common_base. Its next comparison keeps the existing
R0 panel fixed and varies only the covariance weight at matched seeds and training budget (DDR §24).

```bash
scripts/download_tau_urban_av_2021.sh --jobs 3
python -m training.av_data configs/dev/common_base_full_balanced.yaml --workers 4
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python run.py configs/dev/common_base_full_balanced.yaml --device cuda --resume
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python run.py configs/dev/common_base_full_control.yaml --device cuda --resume
```

`--resume` restores completed optimizer/EMA/random-stream snapshots and rejects changed specs or
manifest fingerprints. `--seed 0` selects one seed already declared in the spec. Fresh runs refuse to
overwrite a training ledger. Full-corpus shards use a separate directory from the example data.

For a bounded overnight queue, run `python scripts/run_common_base_overnight.py --deadline
2026-09-05T09:25:00+02:00` with the desired absolute deadline. It runs the predeclared example-duration
controls during acquisition, waits for all verified archives, ingests the complete official fold,
then executes full-corpus paired seeds. Supply `--download-session <tmux-name>` only for a downloader
owned by that run; the queue interrupts it if the deadline expires. Logs, status and comparison JSON
are written to `runs/overnight/common_base_20260905/`. Each completed seed refreshes the experiment
dashboard. A failed R0 leaves R1 gated; no automatic architecture promotion occurs.

To overlap decoding with acquisition, run `python scripts/prefill_tau_cache.py
configs/dev/common_base_full_balanced.yaml --deadline 2026-09-05T09:25:00+02:00`
in another local process with the same desired deadline. It caches up to 128 available pairs per batch
and never publishes a partial manifest; final ingestion verifies the entire source corpus again.

The full-corpus specs use `train.prefetch_batches: 1` to overlap one CPU batch with GPU training.
Checkpoint random states include only consumed batches. Exact CPU/CUDA recovery and a 250-step replay
against the earlier saved smoke validate this runtime change (DDR §29); scientific settings are fixed.
