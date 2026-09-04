# E1 Counterfactual-Anchor Development Iteration — 2026-09-04

- **Matched control:** `configs/dev/first_slice_token_pair.yaml`, `runs/dev/first_slice_token_pair/0/`
- **Intervention:** `configs/dev/first_slice_counterfactual.yaml`, `runs/dev/first_slice_counterfactual/0/`
- **Sole training change after step 1,000:** add weight-1, K=4 counterfactual InfoNCE at κ=0.1
- **Device:** NVIDIA RTX 4090, CUDA bf16 for both fresh runs
- **Training:** 2,000 optimizer steps, seed 0, the same 20,480 episode transitions
- **Paired data:** 2,048 training groups; 256 disjoint fixed-probe groups

## Paired-data validity

Every action branch is generated after restoring the same E0 snapshot. On the 256-group fixed probe,
all groups have at least two distinct rendered successors, 79.69% have all four successors distinct,
96.22% of off-diagonal branch pairs differ, and 97.75% of branches differ from their initial frame.
Raster quantization therefore does not make the intervention vacuous. The exact-replay unit test checks
every stored branch against a fresh restore/step replay.

## Hardware-matched panel

The two training logs are bit-identical through logged step 950, before the counterfactual term starts
at step 1,000. This verifies that the separate counterfactual sampling stream did not perturb ordinary
episode-window sampling, chunk draws, or initialization.

| Metric | Token-pair control | + counterfactual κ=0.1 | Change |
|---|---:|---:|---:|
| Action-sensitivity ratio | 0.003059450 | 0.003642225 | +19.05% |
| One-step correct-action error | 0.000740863 | 0.000749412 | +1.15% (worse) |
| Four-step transition error | 0.003914789 | 0.004121069 | +5.27% (worse) |
| Identity error | 0.000530905 | 0.000490669 | -7.58% |
| Zero-action error | 0.000739576 | 0.000748925 | +1.26% (worse) |
| Shuffled-action error | 0.000741616 | 0.000751882 | +1.38% (worse) |
| Counterfactual accuracy | 0.248047 | 0.250977 | chance in both (1/K = 0.25) |
| Final-batch inverse loss | 0.230776 | 0.236260 | +2.38% (worse) |

Correct-action error remains 0.000000488 above zero-action error and 1.53 times identity error in the
counterfactual run. The candidate therefore fails the action-correctness requirement and is not promoted.

## Temperature-scale diagnosis

| Fixed-probe diagnostic | Token-pair control | + counterfactual κ=0.1 |
|---|---:|---:|
| Mean target-branch off-diagonal MSE | 0.000546284 | 0.000500560 |
| Mean prediction-branch off-diagonal MSE | 0.000006945 | 0.000009176 |
| Mean prediction/candidate distance-row span | 0.000671727 | 0.000597437 |
| Mean logit-row span after division by κ=0.1 | 0.006717 | 0.005974 |

The predicted branches occupy only 1.83% of the target branch separation after training. At κ=0.1,
the candidate logits are almost uniform: training loss ends at 1.386298 (approximately log 4) and every
logged stage-2 minibatch has accuracy 0.25. On the same checkpoint and batch, counterfactual-only raw
gradient norm is 0.000273 at κ=0.1, 0.0114 at κ=0.01, and 1.048 at κ=0.001. Thus this run establishes
that the inherited inactive placeholder temperature supplies effectively no anchor; it does not reject
the §6.4 objective at a temperature commensurate with the observed 10^-3 latent-distance scale.

## Validation and raw artifacts

- Fast suite: 65 passed, 2 threshold tests deselected.
- Threshold suite on each checkpoint: 2 skipped with `threshold_unset`, 65 deselected.
- Ruff and git diff whitespace checks: clean.
- Raw runs: `runs/dev/{first_slice_token_pair,first_slice_counterfactual}/0/`.
- Paired data: `data/e0_dev/counterfactual.pt`; episode data: `data/e0_dev/episodes.pt`.

All numbers are development diagnostics, not experiment results; ABI thresholds remain unset.
