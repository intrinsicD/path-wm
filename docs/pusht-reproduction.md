# Source-scale LeWM reproduction preparation

Preparation is authorized; no long training run is launched. This candidate
addresses the pilot's data/update shortfall while retaining the published
architecture, objective, batch size and optimizer. It is not a completed
reproduction or proof of the released checkpoint's training history.

## Explicit data and schedule contract

- Use the verified full `quentinll/lewm-pusht` HDF5 archive at revision
  `655cd446b9929369d7d406001da85c15d1457850`: 18,685 episodes and 2,336,736 frames.
- Training windows contain four observations five source frames apart, with
  each paired action block beginning at its observation. There are 1,981,721
  valid windows. Fit float32 unbiased action mean/std on all finite source rows
  before splitting, following the pinned training normalizer.
- Randomly permute all window indices with PyTorch generator seed 3072. Explicitly
  allocate `floor(0.9 * N)` windows to training and the remainder to validation.
  Save split hashes. This fixes local rounding; the exact stable-pretraining
  version used for the released checkpoint is unknown. Random-window validation
  shares episodes/configurations with training and measures interpolation.
- Batch 128, ten epochs, drop the final incomplete training batch: 13,933 updates
  per epoch, 139,330 updates total. AdamW 5e-5, weight decay 1e-3, gradient clip 1,
  bf16, prediction MSE plus 0.09 SIGReg (1,024 projections, 17 knots).
- Explicit per-update linear warmup for 1,393 updates, then cosine decay through
  update 139,330. This is a documented scheduler choice: the release declares an
  epoch-interval scheduler, while inspected dependency code uses step-derived
  lengths. The paper reports ten epochs and release YAML declares 100. Neither
  ambiguity is silently interpreted as the released checkpoint's exact history.
- Keep the existing random initialization and model components. Save initialization,
  each epoch and final checkpoint; retain the resume state and checkpoint hashes.
  Use fixed 512-window float32 validation every epoch and every 5,000 updates.

## Implementation slice and tests

Add `prepare_training_data` with explicit `random_windows` and `full_source`
normalization options; preserve default episode-split training behavior. The
configuration's data-only preparation must use the same function as training,
verify source receipts, calculate exact update/warmup budgets, and freeze source
control goals without constructing a model or taking an optimizer step. Tests
must catch overlapping/missing windows, wrong normalization populations and
nonreproducible split hashes. Existing tiny CPU training verifies the default path.

## Matched evaluation

Freeze 50 source goals using the pinned authors' valid-window sampling rule and
seed 42. Preserve the existing 20 configuration-held-out pilot goals as a
separately named diagnostic; after full-source training they are source-training
cases, not held-out generalization. Each evaluated epoch uses the same goal and
reset/CEM seeds, horizon 5, five-action blocks, 300 samples, 30 iterations, 30
elites and a 50-step budget. Run released weights, replay and stationary actions
on that same case set before interpreting learned performance. Fresh weights use
saved training normalization; released weights retain their full-source sklearn
normalizer. These normalization conventions differ slightly and stay explicit.

Primary behavior is successes/50 with per-case outcomes and initial successes.
Report prediction/copy/shuffled-action/rollout metrics separately. No success-rate
pass threshold is set by this preparation. Exact upstream batch-of-50 evaluation
with unseeded resets remains a separate protocol from deterministic paired local
evaluation. Unseen-configuration generalization requires a separate group-held-out
training run; the full-source reproduction does not measure it.

## Remaining numerical and budget considerations

The 1,000-update pilot took 1,506.89 seconds with a cached subset. Scaling that
observed rate to 139,330 updates gives about 58.3 hours, before full-source disk
I/O and the additional control evaluations. This is a rough planning estimate,
not a runtime guarantee. Source preparation allocates no model and does not
launch this work.

The real-batch numerical diagnostic compares native JEPA with local backward on
discarded clones in float32 and bf16. Its result and any recomputation discrepancy
must accompany the candidate configuration; see the control-diagnosis report.
An eventual long run still requires an explicit compute schedule and review of
these remaining recipe differences.

## Pinned sources

- [Authors' training function and split](https://github.com/lucas-maes/le-wm/blob/8edfeb336732b5f3ce7b8b210d0ba370a09e2cac/train.py)
- [Training normalizer](https://github.com/lucas-maes/le-wm/blob/8edfeb336732b5f3ce7b8b210d0ba370a09e2cac/utils.py)
- [Release optimizer/training YAML](https://github.com/lucas-maes/le-wm/blob/8edfeb336732b5f3ce7b8b210d0ba370a09e2cac/config/train/lewm.yaml)
- [Release data YAML](https://github.com/lucas-maes/le-wm/blob/8edfeb336732b5f3ce7b8b210d0ba370a09e2cac/config/train/data/pusht.yaml)

## Prepared commands

The configuration is `configs/reproduction/pusht_source_scale.yaml`. Preparation
uses `python -m scripts.prepare_pusht_reproduction` and writes the frozen window
indices, normalization, exact budget and 50-goal manifest under
`runs/reproduction/pusht_source_scale_preparation/`. It constructs no model.

After a separate long-run schedule is approved, the training command is:

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .runtime/lewm/bin/python run.py -m world_model.train configs/reproduction/pusht_source_scale.yaml
```

For each retained epoch, evaluate its checkpoint using a fresh output directory:

```bash
.runtime/lewm/bin/python run.py -m scripts.evaluate_prepared_control runs/reproduction/pusht_source_scale_preparation/control_cases.json runs/reproduction/pusht_source_scale/checkpoint_139330.pt runs/reproduction/pusht_source_scale_control_final
```

Use the same case manifest with `data/reference/lewm-pusht/weights.pt` and
`--released` for the paired released control. Run
`python run.py -m scripts.check_control_baselines <evaluation-directory>` for
stationary/replay controls. These commands are prepared, not launched here.

The candidate enables nonreentrant encoder activation checkpointing and disables
encoder batch slicing. This preserves full-batch kernels, BatchNorm and SIGReg
semantics. A real-data batch-4 native comparison matches bf16 gradients to a
relative L2 difference of 1.36e-7; slicing that batch in half gave 0.06124. This
isolates a numerical difference, not the causal origin of the pilot's failure.
The separate batch-128 memory preflight and its limits are recorded in
`runs/diagnostics/pusht_control_diagnosis/reproduction_memory.json`.
