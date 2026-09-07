# Paired projection-count training screen — final results

Completed 2026-09-07. **More accurate SIGReg sketches produced modest final gains, but did not demonstrate much faster learning or learning from fewer independent examples.** The earlier roughly fourfold conditional gradient-noise reduction translated into an average of one additional calibrated success out of 50 at 1500 updates on each dataset. Early PushT control did not improve; early TwoRoom calibrated differences varied across seeds, and saved-buffer control consistently deteriorated. Keep 4096 projections as an optional research setting, without adopting it as the sample-efficiency solution.

All **12 runs completed 1500 updates**, and all **48 planned control outcomes** are recorded. There were no training cap stops or missing outcomes. The larger, repeated effect was TwoRoom normalization: at 750 updates, updating only BatchNorm buffers added **13–33 successes out of 50** across the six trained checkpoints. That is evidence of an inference-state problem contributing to apparent undertraining on these source cases; the responsible layer remains unidentified.

The [verified dashboard](../runs/experiment_dashboard.html), [fixed-scale control figure](../runs/projection_training_2026-09-06/figures/control_final/projection_learning_curves.png), and [immutable descriptive snapshot](../runs/projection_training_2026-09-06/report_assets/report_summary.json) contain the results. Exact paired tables, raw metric rows and source hashes are preserved with the snapshot. The [frozen plan](projection-training-plan-2026-09-06.md) records the design and reporting amendments; the [preceding investigation](sample-efficiency-2026-09-06.md) contains the original gradient and literature evidence.

## What was compared

Two full-source datasets, three paired seeds (3072–3074), 1024 versus 4096 **resampled** SIGReg directions. Each pair has identical initial weights, train/validation populations, window order and global CPU/CUDA random streams. A separate checkpointed sketch generator prevents projection count from changing model/dropout randomness. Configuration differences are restricted to projection count and output directory.

Both arms use latent dimension 192, batch 128, regularizer weight 0.09, AdamW learning rate 5e-5, weight decay 0.001, clipping threshold 1, bf16 training, FP32 evaluation and full-batch encoder activation checkpointing. The 1500-update cosine schedule has 15 warmup updates. **It is a short schedule, not a prefix of the full reproduction schedule.** Original checkpoints and reproduction configurations remain unchanged.

Control uses the same frozen 50 source cases per dataset at 750 and 1500 updates. Training-only, layerwise calibration on the same 512 windows within each pair was the predeclared primary policy. Saved-buffer control is reported separately. Each calibrated clone is diagnostic-only and changes six BN buffer tensors, with no parameter or optimizer update. These cases share the source population; they do not measure unseen-configuration generalization or establish long-goal planning.

## Primary: calibrated control

Scores are successes out of 50. Differences are **4096 minus 1024** in successful cases.

| Dataset | Updates | Seed | 1024 | 4096 | Difference |
|---|---:|---:|---:|---:|---:|
| PushT | 750 | 3072 | 3 | 2 | -1 |
| PushT | 750 | 3073 | 1 | 1 | +0 |
| PushT | 750 | 3074 | 0 | 0 | +0 |
| PushT | 1500 | 3072 | 2 | 4 | +2 |
| PushT | 1500 | 3073 | 1 | 1 | +0 |
| PushT | 1500 | 3074 | 1 | 2 | +1 |
| TwoRoom | 750 | 3072 | 41 | 37 | -4 |
| TwoRoom | 750 | 3073 | 35 | 41 | +6 |
| TwoRoom | 750 | 3074 | 41 | 43 | +2 |
| TwoRoom | 1500 | 3072 | 46 | 46 | +0 |
| TwoRoom | 1500 | 3073 | 45 | 45 | +0 |
| TwoRoom | 1500 | 3074 | 43 | 46 | +3 |

Final mean success rates are 2.67% versus 4.67% on PushT and 89.33% versus 91.33% on TwoRoom. PushT remains weak at this budget. TwoRoom already supports strong short-goal source control, with small final projection-count differences. Three seeds and two control checkpoints cannot establish a time-to-target acceleration factor.

## Saved-buffer control

| Dataset | Updates | Seed | 1024 | 4096 | Difference |
|---|---:|---:|---:|---:|---:|
| PushT | 750 | 3072 | 3 | 1 | -2 |
| PushT | 750 | 3073 | 1 | 2 | +1 |
| PushT | 750 | 3074 | 1 | 1 | +0 |
| PushT | 1500 | 3072 | 4 | 2 | -2 |
| PushT | 1500 | 3073 | 2 | 1 | -1 |
| PushT | 1500 | 3074 | 1 | 4 | +3 |
| TwoRoom | 750 | 3072 | 18 | 11 | -7 |
| TwoRoom | 750 | 3073 | 22 | 11 | -11 |
| TwoRoom | 750 | 3074 | 21 | 10 | -11 |
| TwoRoom | 1500 | 3072 | 43 | 47 | +4 |
| TwoRoom | 1500 | 3073 | 44 | 44 | +0 |
| TwoRoom | 1500 | 3074 | 40 | 41 | +1 |

The early TwoRoom deficit with 4096 occurs in every seed (−7, −11, −11 cases), while the final difference becomes nonnegative (+4, 0, +1). The normalization policy therefore materially changes the apparent learning curve. Keep both observations; do not choose the policy after seeing a favorable score.

## Descriptive uncertainty

Each row has **n = 3 paired training seeds**. SD and SE describe variation in the paired differences, not independent control-case uncertainty. The same 50 cases recur across seeds; 2400 case executions are not 2400 independent samples. No confidence interval, significance test or passing threshold was frozen.

| Dataset | Updates | BN policy | Mean Δ (pp) | Min–max (pp) | Sample SD | SE |
|---|---:|---|---:|---:|---:|---:|
| PushT | 750 | calibrated | -0.67 | -2 to +0 | 1.15 | 0.67 |
| PushT | 750 | saved | -0.67 | -4 to +2 | 3.06 | 1.76 |
| PushT | 1500 | calibrated | +2.00 | +0 to +4 | 2.00 | 1.15 |
| PushT | 1500 | saved | +0.00 | -4 to +6 | 5.29 | 3.06 |
| TwoRoom | 750 | calibrated | +2.67 | -8 to +12 | 10.07 | 5.81 |
| TwoRoom | 750 | saved | -19.33 | -22 to -14 | 4.62 | 2.67 |
| TwoRoom | 1500 | calibrated | +2.00 | +0 to +6 | 3.46 | 2.00 |
| TwoRoom | 1500 | saved | +3.33 | +0 to +8 | 4.16 | 2.40 |

The [full exact tables](../runs/projection_training_2026-09-06/report_assets/tables.md) and source snapshot retain unrounded values and initially-unsolved comparisons. PushT has no initially satisfied cases. TwoRoom has four, but the 4096 saved-buffer controllers at 750 retain only three in seeds 3072 and 3073: their 11 total successes include eight newly solved cases. Newly solved counts are derived from cases, never obtained by blindly subtracting four.

## What the gradients and batches support

Every run logs 61 full-objective pre-clip gradient norms: **all 732 recorded samples exceed 1**. The [gradient figure](../runs/projection_training_2026-09-06/figures/diagnostics_final/gradient_norms.png) shows individual observations on a common log scale. These are neither a gradient-variance trajectory nor a record of clipping on every update. Adam's moment normalization also prevents interpreting a clipping multiplier as the same multiplier on effective learning rate. Raising the threshold is not justified by this observation alone.

The earlier frozen-weight experiment reduced conditional total-gradient covariance by 3.96× on PushT and 4.13× on TwoRoom using the same 128 windows, four projection-seed replicates and fixed dropout. That mechanism remains valid in its measured setting. This training screen weakens the stronger hypothesis that sketch noise is the dominant explanation for the large iteration requirement. It does not measure early-training conditional noise or rule out benefits under other schedules.

Saved-buffer FP32 prediction/copy ratios at 1500 updates are:

| Dataset | Seed | Final prediction/copy, 1024 | Final prediction/copy, 4096 |
|---|---:|---:|---:|
| PushT | 3072 | 0.7469 | 0.7273 |
| PushT | 3073 | 0.7934 | 0.7949 |
| PushT | 3074 | 0.7575 | 0.7316 |
| TwoRoom | 3072 | 0.3641 | 0.2130 |
| TwoRoom | 3073 | 0.2958 | 0.2741 |
| TwoRoom | 3074 | 0.4192 | 0.3233 |

4096 improves this ratio in all three TwoRoom seeds and two PushT seeds, but the control gains remain modest. The [prediction figure](../runs/projection_training_2026-09-06/figures/diagnostics_final/prediction_copy.png) preserves all six scheduled trained-checkpoint observations without smoothing. Intermediate ratios can be nonmonotonic and very large. Raw latent MSEs live in separately learned coordinate systems; neither a lower MSE nor rank alone validates useful dynamics. Saved-buffer prediction must not be treated as the forward model used by calibrated control.

An inherited label was corrected: training validation `rollout_mse / identity_mse` divides autoregressive error by adjacent one-step copying error. The dashboard now calls it **rollout / one-step copy**, with different horizons disclosed. It is not matched multi-step prediction skill. Scientific metrics and evaluation numerics were preserved.

The batch audit found about 127 distinct episodes and almost 512 distinct encoded frame rows per batch of 128 windows. Four observations are encoded per window, with three supervised transitions. The source global shuffle already gives diverse batches; I/O waits are negligible relative to compute. The earlier gradient consistency probe favored batch 128 over 32/64 in its measured regime. BatchNorm and SIGReg couple examples, so classical additive-loss critical-batch formulas do not directly apply.

Exact [window-exposure accounting](../runs/projection_training_2026-09-06/window_exposure.json) was verified against all completed pairs. At 750/1500 updates, each arm processes 96,000/192,000 distinct window starts. At 1500, the 768,000 encoded frame presentations cover:

| Dataset | Episodes encountered | Unique encoded source-frame rows | Presentations per unique row |
|---|---:|---:|---:|
| PushT | 18,660–18,664 of 18,685 | 671,146–671,678 | about 1.14× |
| TwoRoom | 9,999–10,000 of 10,000 | 540,753–541,370 | about 1.42× |

The calibration set adds 452–459 PushT or 344–370 TwoRoom window starts outside the optimized 1500-update prefix. Full-source action normalization also accesses the source population. Random-window splits have disjoint starts but can share episodes and frames. Source identities are not statistical effective sample sizes. This screen compares optimization at matched exposure; it does **not** demonstrate training from a smaller independent dataset. Repeated frame processing alone is an inadequate explanation for weak early PushT control, since this short prefix has little exact-row reuse and already encounters almost every source episode.

## Working theory and plausible next solution

The evidence supports several distinct contributors: learning representations and action-conditioned dynamics from scratch takes optimization; stochastic sketches add conditional noise; the learning-rate schedule determines what a short run can accomplish; and inference normalization can obscure useful learned weights. Bad within-batch shuffling, loader throughput and severe encoder-branch gradient cancellation are not supported as dominant explanations by the completed probes. No single cause of sample complexity has been identified.

**First isolate the normalization effect.** A bounded follow-up should compare preserved clones with projector-only, prediction-projector-only, both-layer and unchanged buffers, using a fixed training-only calibration population and matched prediction/control cases. Then test the identified correction prospectively across training checkpoints, with saved-buffer results retained. The current calibration averages per-batch unbiased variances; compare it explicitly with aggregate population moments before treating it as PreciseBN. The empirical intervention is strong on early TwoRoom and inconsistent on PushT, so it is not adopted universally.

**Then test independent-data efficiency directly.** Freeze configuration-disjoint training/development/test groups before normalization or subset selection. Compare a coverage-preserving subset with a size-matched random subset and full training data. Preserve action/contact diversity for PushT and doorway-crossing coverage for TwoRoom. Count unique experience, repeated processing and elapsed time separately; keep the existing source cases as a separate regression panel. The earlier proposed half-processed-window / quarter-unique-data criterion requires a predeclared held-out control target and compute ceiling before execution. This follow-up is proposed, not launched.

A faster predictor timescale remains a secondary hypothesis. Smaller batches, a 32/64-dimensional latent, removing target gradients, fixed sketch directions and changing clipping are not established fixes. The current results do not justify expanding the 4096 sweep as the primary route to much fewer examples.

## Literature grounding

Ten primary papers informed the preceding investigation and this follow-up; conclusions are scoped to their assumptions.

- [LeWorldModel (2026), Appendix G](https://arxiv.org/html/2603.19312v1): final-score projection ablations show weak sensitivity, and reducing latent size is not a free improvement. Those experiments do not settle early learning speed.
- [LeJEPA (2025), §4](https://arxiv.org/html/2511.08544v3): resampled sketches support distribution matching; more directions reduce estimator noise without removing finite-data error.
- [McCandlish et al. (2018)](https://arxiv.org/abs/1812.06162): noise-scale batch arguments require care for this coupled objective.
- [Shallue et al. (2019)](https://www.jmlr.org/papers/v20/18-789.html): batch and efficiency comparisons depend on optimizer tuning and the chosen budget.
- [Tang et al. (2022)](https://arxiv.org/html/2212.03319v1): predictor timescales motivate an ablation under different assumptions, not an automatic target-gradient change.
- [SPR (2020/2021)](https://arxiv.org/abs/2007.05929): multi-step latent supervision is a sample-efficiency precedent in reward-driven Atari, not a directly comparable PushT result.
- [GRAD-MATCH (2021)](https://proceedings.mlr.press/v139/killamsetty21a.html): gradient-aware subset selection motivates a comparison; supervised guarantees do not transfer directly to a changing, batch-coupled representation.
- [Klindt, LeCun & Balestriero (2026)](https://arxiv.org/html/2605.26379v1): identifiability requires explicit transition/distribution assumptions; Gaussian-looking embeddings alone do not establish useful control.
- [Wu & Johnson (2021), §3 and Appendix A](https://arxiv.org/html/2105.07576v1): historical EMA statistics can mismatch an evolving model; their layerwise PreciseBN discussion motivates inference-state checks. Our current variance estimator is not their exact aggregate-moment procedure.
- [Batch Renormalization (2017), §3–4](https://proceedings.neurips.cc/paper/2017/file/c54e7837e0cd0ced286cb5995327d1ab-Paper.pdf): scheduled corrections address training/inference differences while retaining derivatives through batch moments. Simply training through moving statistics is not the method; no renormalization intervention was adopted here.

## Qualitative checks, implementation and QA

The first frozen case, seed 3072 and calibrated 1500-update checkpoints were selected before scores. Both local PushT arms fail the fixed case after 50 steps, while released weights succeed in 16 and recorded-action replay in 14. Both local TwoRoom arms and released weights succeed in 21 steps; replay succeeds in 20. Saved-action replays exactly reproduce the recorded outcomes. All four panels were visually inspected:

[PushT 1024](../runs/projection_training_2026-09-06/qualitative/pusht_s3072_m1024/case_0_rollouts.png), [PushT 4096](../runs/projection_training_2026-09-06/qualitative/pusht_s3072_m4096/case_0_rollouts.png), [TwoRoom 1024](../runs/projection_training_2026-09-06/qualitative/tworoom_s3072_m1024/case_0_rollouts.png), [TwoRoom 4096](../runs/projection_training_2026-09-06/qualitative/tworoom_s3072_m4096/case_0_rollouts.png). These fixed examples illustrate behavior, not representative success rates.

Claude contributed private-RNG implementation/test design and the production paired-statistics implementation through two restricted neutral MCP tasks. Reported API-equivalent usage was **$1.132410**, below the $4 ceiling; account balances were unavailable. No private repository data or local results were transferred, and Claude did not review these measured outcomes. Integration preserved the exact pinned LeWM quadrature and precision, correcting generic draft differences. Absence of `sigreg_seed` preserves legacy behavior; opt-in checkpoints require and restore the independent stream.

The scientific screen used 18,000 optimizer updates, plus four separate 20-update GPU preflights. Summed recorded training-loop elapsed time was 22,446.52 seconds; individual runs ranged from 1774.53 to 2106.36 seconds. The latter completed because its last update started before the 2100-second cap; final validation/checkpointing can extend beyond that boundary. Serial shared-GPU timings are observational. Preflight update times for 1024/4096 were 1.3265/1.3368 seconds on PushT and 1.3198/1.3354 on TwoRoom; peak allocated memory was 3.2822/3.2830 GB. The coordinator ran from September 6 at 19:39:40Z to September 7 at 05:35:44Z, including a reporting repair interruption.

**103 tests passed, including three installed-browser checks.** Tests cover native loss/input-gradient parity, private/global RNG isolation, exact resume, calibrated-clone integrity, paired population/count validation, exact dashboard records and reporting recovery. A tiny CUDA check established bitwise bf16 loss/gradient parity and private-stream recovery without optimizer updates. The [final integrity audit](../runs/projection_training_2026-09-06/integrity_audit_final.json) confirms all six pairs match initialization/populations and global CPU/CUDA RNG at 750/1500, all 24 clones change only BN buffers, and all 13 protected code/reference/case inputs are unchanged. Full source HDF5 bytes were not rehashed in this screen; declared revisions/hashes remain recorded.

One completed evaluation triggered a static-HTML output-limit failure: fallback cell tooltips repeated the entire global file list. Reporting now uses a compact inventory join for repeated record identities and retains the full raw-file union once in source metadata. A stale prior receipt cannot mark a failed current stage verified. Completed scientific output was reused; no evaluation was repeated to obtain a different score. The first coordinator restart failed during imports after interpreter symlink resolution selected system Python; retrying the workspace runtime succeeded. All failure and recovery logs are preserved.

The final portable payload is **2,516,589 bytes**, below the unchanged 3 MB cap. Canonical packaging, source interaction and browser verification pass at 1440/390 pixels. Native charts display adaptive success-rate fractions; standalone PNG/SVG figures use fixed 0–100% control scales. All four native paired charts render all 12 points at both widths. Final scientific figures and actual desktop/mobile charts were visually inspected. No external publication occurred, and no further training is queued.
