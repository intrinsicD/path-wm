# E1 Counterfactual-Anchor Development Iteration — 2026-09-04

- **Matched control:** `configs/dev/first_slice_token_pair.yaml`, `runs/dev/first_slice_token_pair/0/`
- **Intervention:** `configs/dev/first_slice_counterfactual.yaml`, `runs/dev/first_slice_counterfactual/0/`
- **Calibrated intervention:** `configs/dev/first_slice_counterfactual_scaled.yaml`, `runs/dev/first_slice_counterfactual_scaled/0/`
- **Sole training change after step 1,000:** add weight-1, K=4 counterfactual InfoNCE at κ=0.1
- **Calibration change:** κ=0.1 → 0.001 only; a spec regression test enforces equality of every other parsed field
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

## Calibrated-temperature result

The first 20 log records (through step 950) have the same SHA-256 in the control, κ=0.1, and
κ=0.001 runs (`757cffc7...8178`). The temperature change is therefore isolated to the intended
stage-2 intervention.

| Metric | Token-pair control | κ=0.1, weight 1 | κ=0.001, weight 1 |
|---|---:|---:|---:|
| Action-sensitivity ratio | 0.003059450 | 0.003642225 | 0.192504078 |
| One-step correct-action error | 0.000740863 | 0.000749412 | 0.019133145 |
| Four-step transition error | 0.003914789 | 0.004121069 | 0.034507081 |
| Identity error | 0.000530905 | 0.000490669 | 0.001878535 |
| Zero-action error | 0.000739576 | 0.000748925 | 0.008107607 |
| Shuffled-action error | 0.000741616 | 0.000751882 | 0.021818783 |
| Counterfactual accuracy | 0.248047 | 0.250977 | 0.649414 |
| Final-batch inverse loss | 0.230776 | 0.236260 | 0.242766 |

Calibration resolved the inactivity: the fixed probe correctly retrieves 665 of 1,024 branches,
and correct actions are 12.31% better than shuffled actions. It did not pass the transition-compatible
reference gate. Correct-action error is 2.36 times zero-action and 10.19 times identity error; one-step
and four-step errors are 25.83 and 8.81 times the matched control.

The first active batch at step 1,050 had counterfactual loss 5.60584 and raw gradient norm 178.74
before the global 1.0 clip. Across the 20 logged stage-2 batches, raw gradient norm averaged 30.94
(median 6.59), compared with 2.54 for the matched control. Thus κ=0.001 supplies meaningful ranking
gradients, but weight 1 lets them dominate the shared update and trade absolute accuracy for branch
separation. The checkpoint is not promoted. This motivated the informed grid below rather than a
single follow-up weight.

## Informed parameter grid

At κ=0.001, weights `{0.001, 0.003, 0.01, 0.015, 0.02, 0.03, 0.1}` bracket estimated onset
counterfactual contributions below, near, and above the ordinary post-stage raw gradient norm. Existing
weight-0 and weight-1 runs close the endpoints. Every full no-inheritance grid config has a regression
test proving that only its declared parameters differ.

| κ | Contrastive weight | CF accuracy | One-step error | Four-step error | Correct beats zero? |
|---:|---:|---:|---:|---:|:---:|
| 0.001 | 0.001 | 0.250977 | 0.000725371 | 0.004012406 | no |
| 0.001 | 0.003 | 0.250977 | 0.000721691 | 0.003981284 | no |
| 0.001 | 0.010 | 0.253906 | 0.000688771 | 0.003747892 | no |
| 0.001 | 0.015 | 0.259766 | 0.000679549 | 0.003703346 | yes |
| 0.001 | 0.020 | 0.296875 | 0.000717301 | 0.003817476 | yes |
| 0.001 | 0.030 | 0.448242 | 0.001140863 | 0.004122191 | no |
| 0.001 | 0.100 | 0.662109 | 0.005820910 | 0.008974137 | no |
| 0.001 | 1.000 | 0.649414 | 0.019133145 | 0.034507081 | no |

The learning transition occurs between weights 0.015 and 0.03, with no scalar point jointly producing
strong discrimination and all absolute controls. An equal-near-uniform-gradient slice held
`weight/κ = 20` while varying the expected distance-row logit span:

| κ | Contrastive weight | CF accuracy | One-step error | Four-step error | Correct-vs-zero gain |
|---:|---:|---:|---:|---:|---:|
| 0.0005 | 0.01 | 0.308594 | 0.000701391 | 0.003767039 | 0.89% |
| 0.0010 | 0.02 | 0.296875 | 0.000717301 | 0.003817476 | 1.11% |
| 0.0020 | 0.04 | 0.297852 | 0.000678858 | 0.003569567 | 1.24% |
| 0.0040 | 0.08 | 0.291992 | 0.000693545 | 0.003794172 | 1.34% |

κ=0.002, weight 0.04 is the best pure-InfoNCE balance: counterfactual accuracy is 3.54 binomial
standard errors above 1/4 chance and one-/four-step errors improve 8.37%/8.82% over control. Its correct
prediction nevertheless remains 51.06% worse than identity.

## Objective audit and tested remedies

On the matched control checkpoint at κ=0.002, the raw InfoNCE gradient norm is 53.06 when the context
encoder is trainable: encoder, adapter, and predictor component norms are 52.71, 5.55, and 2.45.
More than 99% of squared norm is therefore outside the predictor. A predictor-only route retains norm
2.45. Paired diagonal MSE has component norms 0.328, 0.036, and 0.017 (total 0.330). These measurements
set predictor-only weight 1 and paired-positive weights `{2, 6, 12, 24, 48}`; they were not guessed.

| Remedy | CF accuracy | One-step error | Four-step error | Correct-vs-identity gap | Correct-vs-zero gain | Correct-vs-shuffle gain |
|---|---:|---:|---:|---:|---:|---:|
| Pure InfoNCE (κ=.002, λ=.04) | 0.297852 | 0.000678858 | 0.003569567 | +51.06% | 1.24% | 8.60% |
| Predictor-only InfoNCE (λ=1) | 0.468750 | 0.018022582 | 0.029327536 | +8741.15% | -54.79% | 1.19% |
| + paired MSE weight 2 | 0.305664 | 0.000629607 | 0.003478306 | +45.86% | 2.44% | 10.23% |
| + paired MSE weight 6 | **0.315430** | 0.000555387 | 0.003081677 | +25.16% | **6.22%** | **14.04%** |
| + paired MSE weight 12 | 0.305664 | 0.000545781 | 0.003319908 | +26.94% | 6.67% | 11.94% |
| + paired MSE weight 24 | 0.292969 | 0.000505679 | 0.003076825 | +23.69% | 5.68% | 9.36% |
| + paired MSE weight 48 | 0.260742 | **0.000450445** | **0.002891624** | +28.83% | 1.64% | 2.90% |

Predictor-only routing proves that encoder absorption is not the whole failure: relative-only InfoNCE
can learn ranking while making absolute dynamics unusable. Paired MSE fixes much of that geometry.
Weight 6 is the semantic/accuracy Pareto choice (the strongest discrimination and control margins with
four-step error within 0.16% of weight 24); weight 48 closes the high-absolute bracket by returning
discrimination to within one standard error of chance. No remedy beats identity, so none is promoted.

## Observability audit

There is an exact architectural impossibility witness. Two E0 snapshots with equal positions and
opposite hidden agent velocities render bit-identical current RGB, yet under the same zero action their
next RGB differs in 100% of 64 worlds (mean next-position gap 0.0688). A deterministic function of one
RGB frame and action therefore cannot represent the environment transition.

On 3,840 fresh random transitions, a finite-difference velocity estimate from two frames nearly closes
the information gap:

| Physical next-position predictor | MSE |
|---|---:|
| Identity | 0.000215142 |
| Current action, zero assumed velocity | 0.000148172 |
| Two-frame velocity estimate + current action | 0.000002944 |
| Full hidden-velocity oracle + current action | 0.000001315 |

Two frames reduce error 98.63% versus identity. The preferred architectural solution is the already
predesigned E1-b predict-then-correct updater, which puts history into W while keeping P Markov and the
ABI unchanged. The registered comparison alternatives remain a K-frame context encoder and a
block-causal encoder; exposing velocity in RGB is rejected because it changes E0's partial-observation
task rather than solving belief-state formation.

## Validation and raw artifacts

- Scoped fast suite after grid, objective, and observability regressions: 104 passed, 2 threshold tests deselected.
- Scoped threshold suite on the paired-MSE-weight-6 checkpoint: 2 skipped with `threshold_unset`, 104 deselected.
- Ruff and git diff whitespace checks: clean.
- Raw runs: `runs/dev/first_slice_{token_pair,counterfactual,counterfactual_scaled}/0/` and every
  `runs/dev/first_slice_counterfactual_{grid,solution}_*/0/` directory named by the tables above.
- Paired data: `data/e0_dev/counterfactual.pt`; episode data: `data/e0_dev/episodes.pt`.

All numbers are development diagnostics, not experiment results; ABI thresholds remain unset.
