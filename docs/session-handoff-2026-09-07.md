# World-model continuation handoff — 2026-09-07

This document and the package manifest remove any dependence on the chat for
continuation. The repository is already on `main`; its configured remote is
named `origin`. The coordinator owns the final commit and authorized push.
The package is prospective until its manifest exists. Do not
claim that a running experiment or a package has finished from this draft.

Read [CLAUDE.md](../CLAUDE.md), [project state](project-state.md), and the current
experiment plans before starting another run. The original LeWM work remains
preserved in Git; this package targets the paddle and new E/U/P PushT task.

## Restore, install, and inspect at home

Run from the repository root after pulling the final `main` commit. Restoration
needs only the standard library and preserves existing files that already match.
The package path below is the proposed final location; its `manifest.json` must
exist before these commands can run.

```bash
git pull --ff-only origin main
python3 scripts/session_handoff.py restore --package artifacts/session-2026-09-07 --destination .
python3 scripts/session_handoff.py requirements --package artifacts/session-2026-09-07 > /tmp/path-wm-observed-constraints.txt
python3.14 -m venv .venv
.venv/bin/python -m pip install -c /tmp/path-wm-observed-constraints.txt -e '.[paddle,pusht,dev]'
.venv/bin/python -m world_model doctor
.venv/bin/python scripts/session_handoff.py verify-restored --root . --output runs/restored-session-verification.json
```

Use Python 3.14.7 for the recorded environment. The constraints pin observed
package distribution versions; inspect the recorded Torch/CUDA build as well,
since wheel source and platform affect the runtime. The original runtime is
Torch `2.14.0+cu130`, CUDA `13.0`. Do not describe a different environment as an
exact numerical continuation. Open `runs/experiment_dashboard.html` to inspect
the carried report. Its package contains the original browser verification
receipt. `verify-restored` performs full offline paddle replay, PushT source
verification with the relocated HDF5, supplement hash/label verification, and
CPU E/U/P/D/H/R forward checks for the four carried inference bundles. PushT's
bundles are software smoke evidence; no passing trained PushT bundle is implied.
Run this check before substantive training.

All training stages scheduled for this session have finished. Inspect
`runs/paddle/history_start_v1/memory/paddle_result.json` for the selected and final
history results, `runs/pusht_world_model/baseline/predictor_1/pusht_result.json`
for the retained failed P1 gate, and
`runs/pusht_world_model/perception_coverage_v1/perception/pusht_result.json` for
the negative coverage experiment. New training choices belong in a new declared
run; a completed receipt is not an invitation to extend its budget with resume.

## Portable package scope and implementation plan

Restore **exact recorded bytes**, including original relative paths and
checkpoint aliases. Never regenerate a prepared dataset merely because it is
ignored by Git: that can change fingerprints, split membership or preprocessing.
Preserve selected checkpoints, committed resume checkpoints, their referenced
immutable snapshots, initial snapshots, checkpoint dependencies cited by the
current diagnostics, raw ledgers, qualitative figures, and the canonical HTML.

An explicit reviewed file selection is packaged as SHA256-addressed objects in a
compressed archive, split into files no larger than 45,000,000 bytes. Identical
`best.pt`/`last.pt`/snapshot bytes share one object while their restored names
remain distinct. `manifest.json` records each relative path, size and SHA256,
each archive part's size/hash, environment inventory, Git revision, selected
scope, and final run status. Packing waits until the coordinator stops writers
or they finish. Check source metadata before/after packing; fail if it changes.

The restoration script uses only Python's standard library. Validate every part,
object and destination before installing files; reject missing/corrupt bytes,
unrecognized paths, archive traversal, symlinks and conflicting existing files.
Repeated restoration may skip an already identical file. Preserve all original
data/run directories on the source computer. Unit tests exercise exact roundtrip,
deduplication, bounded parts, corruption, conflicting files and unsafe paths.

The initial read-only inventory measured:

| Material | Logical bytes | Gzip level 6 estimate | Decision |
|---|---:|---:|---|
| Exact paddle baseline NPZ episodes + manifest | 27,394,459 | 15,364,223 | Include |
| Exact prepared PushT episodes, manifest, RGB64 store | 317,172,698 | 8,957,044 | Include |
| Exact converted CCHI source HDF5 | 99,922,306 | 14,897,536 | Include for offline source verification |
| Completed paddle evaluation JSON/CSV ledgers | 437,401,526 | 50,021,322 | Include authoritative evidence |
| Selected/resume/referenced snapshots at inventory time | About 43,600,000 | About 20,000,000 after alias deduplication | Recompute after final stages |
| Paddle raw RGB cache | About 2.6 GB | Not measured | Rebuild from exact NPZ files |
| Observer caches | About 240 MB | Not measured | Rebuild under exact E/U fingerprints |

The total estimate is 130–180 MB compressed before final experiment growth, split
into ordinary Git-sized parts. This is an estimate, not a completed artifact.
Do not include unrelated historical data/runs, credentials, `.idea`, `.codex`,
virtual environments, compiled caches, or installed plugin caches. The small
CCHI acquisition/conversion receipts and current task collaboration evidence
belong with the source provenance. Keep source ZIP/Zarr optional: exact prepared
files plus the exact converted HDF5 suffice for training and full verification.

## Exact data identities and relocation

- Paddle: `data/paddle/baseline`, fingerprint
  `c6d255dc9919b1bb9ce38182f9d180047f964754a32c616db072bc965a3be54f`.
  This is the already verified 5,000/500/500 episode population.
- PushT: `data/pusht_world_model/cchi_v1`, fingerprint
  `e6ff0cad101c15ba3e2b839df28bc0bd75dca4063a7d6293141976ab44b4fa9c`.
  This is 206 episodes/25,650 frames and the recorded 164/20/22 configuration split.
- PushT RGB64 flat store SHA256:
  `7b52ba5b8499ee62e6c15201dc6f22c88f080e8ed38e321d5436986d5773c86a`.
- Exact converted CCHI HDF5 SHA256:
  `30e442fedacaa4a0d36662b951c326b411a15c3a365b514587b9a738229da8f6`.
- The independent PushT perception supplement is
  `data/pusht_world_model/pose_supplement_v1`: 20,493 canonical images and actual
  post-reset pose labels, with fingerprint
  `34e6e5a352a1355fd6ac7a1a6471bb77f5e5a466e18bf9882a8d9215afb9b7f3`.
  Include its exact arrays and manifest; its roughly 253 MB array storage is
  additional to the initial inventory table above.

PushT's manifest intentionally retains absolute source provenance paths from the
original machine. **Do not edit them.** Training reads the restored relative
prepared-data directory. For complete verification, override the HDF5 location
without rewriting the manifest:

```bash
.venv/bin/python -m world_model verify-data --data data/paddle/baseline
.venv/bin/python -c "from world_model.pusht.data import verify_dataset; print(verify_dataset('data/pusht_world_model/cchi_v1', source='data/pusht_cchi/pusht_cchi.h5'))"
```

The ordinary `world_model.pusht.cli verify-data` command currently uses the old
absolute source path, so use the explicit Python call above after relocation.
Preprocessing and acquisition code are tracked in `world_model/pusht/data.py`,
`scripts/prepare_data.py`, and `configs/datasets/pusht_cchi.yaml`; the latter
records the original public source URL. They are an audit/recovery recipe, not
permission to silently replace the restored prepared population.

## Environment and reporting prerequisites

The final package must include an exact installed-distribution/version inventory,
Python/OS/architecture, Torch/CUDA/runtime, and Node/browser versions where
available. Record package names/versions only; omit environment variables,
authentication files and package-index credentials. The current training machine
uses Python 3.14.7, Torch 2.14.0+cu130, NumPy 2.5.2, OpenCV 5.0.0 and an NVIDIA RTX 4090.
`pyproject.toml` defines the project extras. The package environment inventory is
the authority if later installs change this provisional list.

Create a Python environment matching that inventory and install the local project
with its paddle/PushT/development dependencies. Install the recorded Torch build
from its appropriate package source; installing a generic current Torch version
does not establish an exact-resume environment. CPU tests work without CUDA.
Changing GPU/runtime may preserve functional checkpoint loading while changing
floating-point trajectories; record that explicitly. The history trainer rejects
resuming on a different resolved device instead of silently changing execution.

`runs/experiment_dashboard.html` is self-contained and opens offline immediately.
Generating a fresh canonical dashboard additionally requires Node, Chromium, and
the Data Analytics plugin's portable-artifact builder. The installed plugin is
deliberately not exported. `viewer.dashboard.find_portable_artifact_builder()`
locates it; `PATH_WM_ARTIFACT_BUILDER` can point at a compatible installation.
`PATH_WM_CHROMIUM` selects the browser. Check these prerequisites before a long
run so completed raw results are not mistaken for completed reporting:

```bash
.venv/bin/python -c "from viewer.dashboard import find_portable_artifact_builder; print(find_portable_artifact_builder())"
.venv/bin/python -m pytest tests/paddle tests/pusht
```

## Continue from recorded status

The final package's stage receipts and `docs/project-state.md` determine what is
complete, running, failed, or still planned. Inspect each stage's `status.json`,
`paddle_result.json`/`pusht_result.json`, `last.pt`, and immutable selected snapshot.
Do not infer a completed stage from the presence of `best.pt` alone.

Paddle's fixed reference completed 10,000 perception updates, 10,000 memory
updates, 20,000 total P1 updates across the original run and its continuation,
and 10,000 P5 updates. Its learned control targets were
not demonstrated by a falling latent loss; retain the recorded negative outcomes.
The separate [history-start experiment](paddle-history-start-plan.md) subsequently
finished 10,000 U updates and selected update 9,750. Its exact config is
`configs/paddle/history_starts.yaml`, and its initial E is the immutable
`runs/paddle/baseline/perception/checkpoints/best_00009750.pt`.

If its last checkpoint is incomplete, resume the same run/config:

```bash
.venv/bin/python run.py -m world_model train-history-memory \
  --config configs/paddle/history_starts.yaml --data data/paddle/baseline \
  --perception runs/paddle/baseline/perception/checkpoints/best_00009750.pt \
  --run runs/paddle/history_start_v1/memory --resume
```

Use `train-history-memory` explicitly: generic `train-memory` and `run-all` reject
this config. After its fixed U budget, use the selected **new** U for fresh P1
and P5 as specified in the plan. P1 has a 20,000-update ceiling; P5 has 10,000. The
unchanged K1 quality gate still controls whether P5 can start. Existing baseline
P weights and observer caches have incompatible U dependencies. Preserve the
ordinary versus suffix joint-validation weighting and paired-diagnostic isolation.

PushT's initial CCHI baseline reached its declared perception/U/P1 budgets and
failed its one-step quality gate; P5/control must not be represented as completed
baseline stages. The perception-coverage follow-up also completed and was
negative, so no further PushT U/P stages were launched from it.
Use `python -m world_model.pusht.cli ...` for this separate task. Resume only a
checkpoint's identical resolved config; use a new named run for a changed
training distribution or budget.

Use `python run.py ...` for each completed stage/evaluation, or explicitly refresh
and verify the canonical dashboard afterward. Restore the exact raw evidence
before rebuilding it. Historical rows embedded in the carried HTML may refer to
older tasks outside this deliberately bounded data package; they are preserved
viewing evidence, not newly restored training populations.
