# Broader PushT training pilot

The user approved this bounded pilot on 2026-09-05 after the released checkpoint
passed full-source control validation. It tests whether the unchanged,
random-initialized LeWM baseline learns useful control from more diverse data.
This is subset training and local development evaluation, not paper reproduction.

## Frozen protocol

- Source: `quentinll/lewm-pusht`, revision
  `655cd446b9929369d7d406001da85c15d1457850`, verified full archive; 18,685 episodes.
- Audit: the full archive has 185 exact initial configurations. Group exact and
  near initial states using connected components: agent and block positions each
  within 5 pixels, circular block angle within 0.05 radians. This produces the
  same 185 groups; the closest distinct pair has maximum normalized distance 7.61.
- NumPy seed 3072 selects 160 groups without replacement and one random episode
  from each. The first 128 groups train; the remaining 32 are held out. Source
  episode IDs span 66–18,684. No outcome filtering. Grouping initial states does
  not prove absence of later shared trajectory segments.
- Derived HDF5: `data/pusht/pilot_128_32/trajectories.h5`, 20,015 frames;
  SHA256 `af224b18f0bf54039983618bddf62ec4f8ca446ee549570ab824caa7a31d0ae5`.
  Every extracted pixel/action/state/proprio/step-index block was compared to
  the source. Local episode indices map to original IDs in `split.json`.
- Configuration: `configs/diagnostics/pusht_broader_pilot.yaml`. Batch 128,
  random seed 3072, unchanged published model/loss, bf16 training, at most
  1,000 optimizer updates or 1,800 seconds. Training time includes in-run
  validation, excludes data preparation and subsequent evaluation. The deadline
  is checked before each update; one in-flight operation can cross it. Resume
  retains previously spent time. Save immutable snapshots at 0/250/500/1000,
  plus the latest checkpoint (including a time-limited final checkpoint).
- Validation uses the same 512 held-out windows in float32 at initialization,
  every 250 updates, and completion. Report one-step, copy, shuffled-action,
  zero-action and rollout MSE plus embedding variance. No BatchNorm calibration
  or buffer replacement. MSE comparisons across checkpoints are descriptive:
  the learned target representation also changes.
- Freeze 20 distinct held-out episodes and random starts with seed 42 before
  training. Targets are 25 source frames later. All checkpoint/control methods
  share cases and reset/CEM seeds 1234–1253. Evaluate retained 250/500/final
  checkpoints with 300 CEM samples, 30 iterations, 30 elites, horizon 5,
  five-frame action blocks, receding horizon 5, and a 50-step environment budget.
- Compare stationary actions (50-step budget), recorded replay (the 25 actions
  leading to each source target), and released weights on these same goals.
  Released weights retain full-source sklearn normalization; trained weights use
  saved training-only unbiased action statistics. These cases are held out
  for the pilot, but belong to the released model's source training population.
  Record initial successes and raw per-case outcomes; verify checkpoint hashes
  are unchanged after evaluation. Absolute latent MSE across separately trained
  models is not a calibrated comparison.

The local control evaluator has already matched the pinned SWM evaluator on ten
paired cases with released weights. See [reference validation](reference-validation.md).
This pilot uses explicit reset seeds, unlike the earlier 50-case upstream run.

## Reproduction commands

```bash
.runtime/lewm/bin/python -m scripts.prepare_pusht_pilot
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .runtime/lewm/bin/python -m pytest
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .runtime/lewm/bin/python -m world_model.train configs/diagnostics/pusht_broader_pilot.yaml
.runtime/lewm/bin/python -m scripts.evaluate_pusht_pilot runs/diagnostics/pusht_broader_pilot runs/diagnostics/pusht_broader_pilot/checkpoint_000250.pt runs/diagnostics/pusht_broader_pilot_control_250
```

Use separate output directories for each checkpoint. Data preparation refuses to
overwrite an existing subset. Checkpoint evaluation refuses existing output
directories. Preparation and all 13 CPU tests passed before the training launch.

## Results

The pilot completed **1,000 updates in 1,506.89 seconds (25.11 minutes)**,
including its in-run validations, under the 30-minute cap. Training used clean
commit `0a81029f1ae6dd118d473eaef3901063e4d78b07`, 13,615 training windows,
3,360 held-out windows, and 128,000 sampled training windows over the run.
Normalization used only the 16,047 source-action rows in the 128 training
trajectories. All 13 CPU tests passed before launch.

The final checkpoint modestly beats predictive controls but reaches **0/20
held-out control goals**. The 250- and 500-update checkpoints also reach 0/20.
The released model reaches 17/20 on the same cases. The learned-control gate
remains unmet; this pilot does not justify claiming a reliable trained baseline.

### Fixed float32 prediction windows

All rows use the same 512 held-out windows, saved weights and BatchNorm buffers.

| Updates | Prediction MSE | Copy MSE | Shuffled-action MSE | Rollout MSE | Embedding std |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.077519 | 0.00000118 | 0.077519 | 0.076615 | 0.001193 |
| 250 | 9.570510 | 0.381401 | 9.571372 | 10.651162 | 0.930674 |
| 500 | 0.970264 | 0.312354 | 0.971932 | 1.223923 | 0.853572 |
| 750 | 0.235894 | 0.203680 | 0.247254 | 0.369799 | 0.762922 |
| 1,000 | 0.187671 | 0.213567 | 0.203553 | 0.291436 | 0.795040 |

At 1,000 updates, prediction MSE is 12.1% below copying and 7.8% below shuffled
actions; zero-action MSE is 0.195898. The initially tiny embedding variance
makes the untrained copy error tiny: absolute MSE across changing representations
is not a standalone learning curve. The 250- and 500-update checkpoints fail the
copy-control comparison. Repeat evaluation of all three retained trained
checkpoints reproduces the in-run float32 metrics within 1e-6.

### Paired control outcomes

| Method | Successes | Mean environment steps |
| --- | ---: | ---: |
| Trained, 250 updates | 0/20 | 50.00 |
| Trained, 500 updates | 0/20 | 50.00 |
| Trained, 1,000 updates | 0/20 | 50.00 |
| Released LeWM | 17/20 | 29.45 |
| Stationary | 0/20 | 50.00 |
| Recorded replay | 19/20 | See per-case records; at most 25 |

No case is successful at reset. All 80 model-control evaluations use identical
cases and reset/CEM seeds; manifests, outcomes and checkpoint hashes were
reconciled. This is one seed and 20 fixed development goals, not a population
success-rate estimate or a matched-data training comparison with released weights.
The earlier 45/50 upstream result uses a different case set and reset policy.

Replay fails on source episode 8051, start 76. Repeating the same 25 actions
ends 30.21 pixels away in the combined agent/block position metric, exceeding
the 20-pixel success threshold; the angular error is only 0.02247 radians.
The case remains included. Its replay mismatch does not establish that the goal
is unreachable. No source frames, goals or outcomes were filtered after freezing.

### Final-checkpoint mode diagnostic

The existing prediction-only probe was run after all scored control tests. It
uses 512 training windows and the same 512 held-out windows, in batches of 128
instead of the scored batch size 32. Shuffled pairings and batchwise embedding
statistics therefore differ slightly from the primary measurements.

- Untouched float32 prediction MSE: train 0.155230, held out 0.187671.
- On the same first 128 held-out windows: float32 0.180468, bf16 0.175220
  (2.9% lower); using current held-out batch statistics on a discarded probe
  gives 0.253954, which is worse.
- A separate clone with BatchNorm statistics estimated from 512 training windows
  gives held-out MSE 0.180559, only 3.8% below the untouched result. No optimizer
  updates, checkpoint modifications, calibrated control scoring or adoption.

The large early float32 prediction mismatch is substantially reduced by the end.
These small final precision/calibration differences do not establish the cause
of the control failure. Prediction gains alone are insufficient to establish
useful multi-step dynamics or a useful planning cost.

### Decision and evidence

Keep research extensions deferred. Recommend a focused diagnosis of rollout
predictions and action-plan ranking on fixed cases, with training-versus-held-out
control to separate memorization/generalization from a planning failure, before
committing to more training. This is a recommendation; no additional training or
research run was launched.

All checkpoints and raw evidence are local under `runs/diagnostics/`:

- `pusht_broader_pilot/`: training manifest, metrics, status, retained checkpoints,
  `pilot_summary.json`, `diagnostics.json`, and `diagnostic_integrity.json`.
- `pusht_broader_pilot_control_{250,500,1000,released}/`: frozen case manifests,
  float32 predictions, per-case outcomes, executed action arrays and summaries.
- `pusht_broader_pilot_action_baselines/`: stationary/replay records and the
  repeated replay-failure diagnostic.
- `data/pusht/pilot_128_32/`: derived source subset, full split/source mapping and
  extraction verification. These local data/run files are intentionally ignored
  by Git; protocol, code and results narrative are committed.

Checkpoint SHA256 values (all unchanged through evaluation):

| Checkpoint | SHA256 |
| --- | --- |
| 250 | `5d0a3e931d91befc4adadc49aec6a3b1db80fa7135f33c871050d0378012f637` |
| 500 | `91dd5e47628cbae38e0ff91071ad57cced739ecb8a3656741ec53ace1c4221dd` |
| 1,000 | `1988e4646c81376d065c513a8c3ac33bae6dd77cce8a342b4b35f38f78348d96` |
| Released | `48938400ae3464c9680731287f583a9cb516f55a8ec64ea13a91be47fb15b607` |
