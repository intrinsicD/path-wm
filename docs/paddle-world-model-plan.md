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
PyTorch 2.14.0+cu130, NumPy 2.5.2, PyYAML 6.0.3, pytest 9.1.1; Pillow is missing.
Preserve working PyTorch and verify CUDA with an actual forward/backward.

## Completion evidence

Pending. Software completion and empirical target achievement will be reported
separately, including dataset counts, fixed validation, held-out controls,
collision diagnostics, exact checkpoint identities, and measured latency.
