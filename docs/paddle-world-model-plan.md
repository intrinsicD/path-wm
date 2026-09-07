# Paddle world-model baseline — 2026-09-07

Implement `world_model_codex_implementation_brief.md` as supplied. Preserve the
existing LeWM package and evidence. New implementation lives in
`world_model/paddle/`; `python -m world_model` dispatches its CLI. Configurations
live in `configs/paddle/`, data in `data/paddle/`, and runs in `runs/paddle/`.

## Slice, hypothesis, and budget

First establish that event-based simulation, exact replay, E/U/P timing, staged
optimization, exhaustive planning, serialization, and reporting execute without
scientific leakage. The 20/4/4-episode, 20-update smoke is execution evidence only.
Then run the fixed 5000/500/500 baseline and four bounded 10000-update stages if
the measured local resource budget permits. Preserve the one-step latent and
moving-position improvement gate. No architecture/loss change is authorized by
an implementation failure. Report target failures honestly and diagnose on
validation, keeping final test results separate.

## Shared implementation interfaces

- `types.py`: `ObservationLatent(fine, coarse)` with `tokens()`,
  `from_tokens`, `detach`, `clone`, `__getitem__`, `cat`; and
  `PlanningState(observation, memory)`. Batch dimension is always retained by
  callers. `action_one_hot(ids)` returns float32 `[B,3]`.
- `env.py`: `PaddleEnv(seed=0, state=None)`, `reset(seed=..., state=...)`,
  `step(action)` -> `(frame, terminated, truncated, info)`, `render()`, `clone()`;
  `state` is a float64 array ordered x,y,vx,vy,paddle_x. Expose `step_index`,
  `hit_count`, `terminated`, `truncated`. Reset returns the raw frame.
- `data.py`: `generate_dataset(config, output)`, `verify_dataset(path)`;
  `EpisodeDataset(path, split)` with `__len__`, `__getitem__` returning NPZ
  fields as arrays and `.manifest`, `.fingerprint`. Config has `dataset` mapping
  with `train`, `validation`, `test` counts, fixed split seed bases 0/5000/5500.
  `history_pairs(count, seed)` yields two frame histories and initial states per
  pair; document concrete returned mapping. `test_history_cases(output)`.
- `models.py`: `Encoder`, `MemoryUpdater`, `Predictor`, `Decoder`,
  `PositionReadout`, `StateReadout` use exact brief shapes and default constructors.
- `rollout.py`: `rollout_step(state, action_onehot, predictor, updater)`.
- `planner.py`: `ExhaustivePlanner(predictor, updater, readout,
  candidate_batch_size=243).plan(state, goal_config=None)` for a single real
  state. `PlanResult` has `action_id`, `sequence`, `score`, optional `states`.
- `training.py`: stage functions use `config, data, run`, checkpoint dependencies;
  shared checkpoint utilities in `checkpoints.py`. Public inference
  `load_observer(perception, memory, device)` returns dict E,D,H,U,R;
  `load_system(perception, memory, predictor, device)` adds P and statistics.
- `evaluation.py`: `evaluate(config, data, perception, memory, predictor, output)`
  and `demo(perception, memory, predictor, output, device='auto')`.
- All source paths and config choices are recorded. Smoke gate bypass is explicit.
  Mandatory canonical dashboard must include new ledgers and pass verification.

## Work allocation

Codex independent tasks implement simulator/data, exact models/planner, and
evaluation. The coordinator owns staged training, checkpoints, CLI, integration,
and machine validation. Independent Claude tasks review physics and learning
contracts, then review integrated code; their findings and changes are recorded.
Each owner writes essential behavioral tests before implementation and reports
the informative initial failure. Coordinator commits completed test/plan and
implementation slices; collaborators do not commit other owners' files.

## Measured environment

Sandbox cannot see NVIDIA devices. Outside-sandbox read-only `nvidia-smi` reports
RTX 4090, 24564 MiB, driver 580.159.04. Existing `.venv` is Python 3.14.7,
PyTorch 2.14.0+cu130, NumPy 2.5.2, PyYAML 6.0.3, pytest 9.1.1.
Added Pillow 12.3.0 and matplotlib 3.11.1; preserved PyTorch. Actual synchronized
CUDA forward/backward passes: E/D/H batch128 used428.74MiB and0.17151s; P5 batch16
used216.35MiB and0.01825s. Single-pass engineering profiles, not sustained rates.
Local Chrome for Testing152.0.7977.82 supplies mandatory browser verification.
The old optional pygame2.6.1 has no Python3.14 wheel and its source build lacks
SDL development libraries; pygame-ce2.5.8 supplies the compatible import for
legacy regression checks. No previous LeWM training recipe was changed.

## Numerical contact correction from independent review

Claude's adversarial review identified an inclusive-contact roundoff case:
the prescribed `(toward, toward, stay)` history sequence reaches physical offset
8, but accumulated float64 arithmetic produced 8.000000000000004 for 19 of the
300 validation/test pair members. Regression tests first failed on seeds 7007
and 8004. The comparator now uses `offset <= 8 + 1e-10`, retaining the physical
threshold and recording both contact and event-time comparison epsilons in the
environment fingerprint. An offset 8 + 1e-8 still misses. All 150 continuously
sampled centers remain unchanged; the test enumerates every three-action prefix
and requires exactly two catching sequences for each member. Completed datasets
with the old fingerprint remain immutable and require distinct output paths.
This is a numerical correctness repair, with no training-distribution change.

## Completion evidence

The complete20-update smoke pipeline passed at `runs/paddle/smoke_v2`, using
`data/paddle/smoke_v2` (28episodes,1212frames). All five controller definitions,
prediction/copy/reset comparisons, paired probes, PNG/GIF panels, checkpoint
loading and a self-contained inference bundle executed. CPU bundle weights,
planning score and action match the dependent checkpoints exactly. Canonical
desktop/mobile/source HTML checks pass and preserve25 earlier development runs.
Learned smoke control1/2 ordinary and0/4 paired is not learning evidence; all
engineering targets fail. Inspected reconstructions mostly omit both objects.
Earlier preliminary `smoke` data/run and failed reporting logs remain preserved.

Full data are collected and exactly replay-verified: train5000/184808frames,
validation500/18115frames, test500/17831frames. Overall214754transitions,
25949103compressed NPZ bytes;2039catches,20.254%ascending frames,
5979terminated/21truncated episodes. Generation71.05s and replay19.66s.
`runs/paddle/data_baseline/coverage.json` contains complete counts and identities.
Dataset fingerprint: c6d255dc9919b1bb9ce38182f9d180047f964754a32c616db072bc965a3be54f.

Independent reviews repaired contact roundoff, validation denominators, cache
split identity, duplicate resumed logs, checkpoint best/last transaction recovery,
completed-stage resume, exact K1 statistics reuse and initialization provenance.
Eight transaction tests used explicit negative controls because corrections
landed concurrently before their first execution. Three independent Claude calls
produced physics, learning and actual-code reviews; receipts are under
`runs/paddle/collaboration/` (reported API-equivalent total$9.16633925).
Cold-start distribution shift in paired histories remains a labelled diagnostic
limitation; no training-start distribution, architecture or loss was changed.

## Frozen baseline execution

The baseline uses configs/paddle/baseline.yaml unchanged scientific settings:
seed1701, FP32/TF32 off, AdamW3e-4, full episode batches and four10000-update caps,
validation every250. Optional early stopping uses eight successive validations
without0.1% improvement over the prior best after at least1000updates.
The K1 gate is strictly lower fixed-validation latent loss AND lower MAE for
each ball-x, ball-y and paddle-x coordinate than matched copy-S. No10% margin or
bootstrap requirement was added. K5 uses fresh AdamW with selectedK1 weights,
records its fingerprint and retains identical K1 scale statistics. GPU resume
restores RNG/optimizer/sampler; bitwise GPU determinism is not promised.

Full baseline training/evaluation is next; empirical target achievement remains
unassessed until actual held-out measurements. Results and limitations will be
recorded separately from software completion.

### Measured input-loading amendment before continuation

The first250 full perception updates produced validation H MAE
(0.997806,0.965587,1.057109) pixels. Median original loader cost was0.36037s per
128-frame batch versus0.00753s for warm GPU E/D/H forward/backward. Separate frame
and label loops evicted the32-episode LRU, decoding most episodes twice.
The trainer was deliberately interrupted and retains the250-update checkpoint,
optimizer and RNG. Uncheckpointed logs are archived on resume.

An implementation-only raw RGB/state/action cache now streams source episodes
into read-only disk mappings with dataset/split/order/schema and file checksums.
All three splits require2725122695bytes and built in10.67s. The matched loader
median is0.0009967s. Twelve128-frame batches and100 additional selected frames
match source inputs/labels and normalized tensors bitwise. Peak profiling RSS
was810MiB including PyTorch; no full-population RAM tensor or18GB feature cache
is created. Records: runs/paddle/frame_cache_profile.json. All eight cache tests
pass, including partial-build rejection, ordering, immutability and corruption.
NPZ source, scientific config, model parameters, sample indices, objectives,
validation and budgets are unchanged. Continue from the same250-update state;
subsequent checkpoints record the revised source-code fingerprint.

Full dashboard capacity passed in isolated synthetic fixtures, never added to
the scientific ledger: four10000-update stages, all3500 controller outcomes,
matched prediction and all-frame readouts. Maximum artifact1.27MB against3MB;
desktop/mobile/source checks pass. Failure-case identifiers are conserved across
bounded detail tables instead of overflowing one renderer cell.
