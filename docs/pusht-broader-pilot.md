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

Pending the approved training and paired control evaluation.
