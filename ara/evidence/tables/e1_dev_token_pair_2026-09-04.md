# E1 Token-Pair Inverse Development Iteration — 2026-09-04

- **Baseline implementation:** `4556ea8`, `runs/dev/first_slice/0/`
- **Iteration implementation:** `1148cc6`, `runs/dev/first_slice_token_pair/0/`
- **Sole model change:** `inverse_dynamics.kind: mlp_pooled_pair -> mlp_token_pair`
- **Device:** NVIDIA RTX 3050 8 GB, CUDA
- **Training:** 2,000 optimizer steps, seed 0, the same 20,480 deterministic E0 transitions
- **Probe set:** the same 256 fixed samples, rollout horizon 4

## Pre-intervention diagnosis

| Diagnostic | Value |
|---|---:|
| Zero-action target MSE | 0.336418 |
| Two-frame raster-position linear-probe MSE | 0.235068 |
| Three-frame raster-position linear-probe MSE | 0.112431 |
| Step 3 inverse-head held-out MSE | 0.338520 |
| Step 3 inverse-output correlation, x / y | -0.00285 / -0.02412 |
| Mean-pooled / full latent-delta norm | 0.040109 |
| Frozen-W pooled-MLP action-probe MSE | 0.344028 |
| Frozen-W attentive-token action-probe MSE | 0.345371 |

The raw frame pair contains action information, but the Step 3 representation and inverse
head remain at the zero-action solution. A third frame substantially improves recoverability,
which also leaves a partial-observability/history question open for the force-driven world.

## Matched before/after panel

| Metric | Step 3 pooled | Step 4 token pair | Change |
|---|---:|---:|---:|
| Action-sensitivity ratio | 0.002276306 | 0.002877012 | +26.39% |
| One-step transition error | 0.000345123 | 0.000636088 | +84.31% (worse) |
| Four-step transition error | 0.002701576 | 0.004243674 | +57.08% (worse) |
| Final-batch inverse loss | 0.348572 | 0.234204 | -32.81% |
| Held-out inverse MSE | 0.338520 | 0.210544 | -37.80% |
| Held-out inverse correlation, x / y | -0.00285 / -0.02412 | 0.63874 / 0.63642 | improved |

## Held-out action-correctness controls

These use all 1,024 one-step pairs in the fixed probe trajectories.

| Transition MSE | Step 3 pooled | Step 4 token pair |
|---|---:|---:|
| Identity (no predictor) | 0.000129298 | 0.000546195 |
| Predictor, correct action | 0.000385505 | 0.000791805 |
| Predictor, zero action | 0.000384851 | 0.000791497 |
| Predictor, shuffled action | 0.000385366 | 0.000792345 |

The token-pair head makes action genuinely decodable from W, but the predictor's correct
action is no better than zero or shuffled actions. The increased opposite-action separation
therefore does not establish correct action conditioning, and this checkpoint is not promoted.

## Validation and raw artifacts

- Fast suite: 57 passed, 2 threshold tests deselected.
- Threshold suite on the iteration checkpoint: 2 skipped with `threshold_unset`, 57 deselected.
- Ruff: all checks passed. Staged diff whitespace check: clean.
- Raw run files: `runs/dev/first_slice_token_pair/0/{checkpoint.pt,metrics.json,run_summary.json,training.jsonl,threshold_record.json,spec.yaml}`.
- Baseline raw files remain under `runs/dev/first_slice/0/`.
