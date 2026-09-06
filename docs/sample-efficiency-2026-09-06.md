# Learning from fewer examples: evidence and proposed experiment

2026-09-06. Research with Claude through MCP; local training-gradient measurements on PushT and TwoRoom. **No optimizer updates or new training data. Original checkpoints are unchanged.**

## Answer

There is a plausible route to learning more from the same experience. We have not yet demonstrated a better learning curve with fewer independent trajectories. The strongest measured lead is **noise in SIGReg's random projection estimate**, alongside repeated processing of existing trajectory frames. The batches themselves are already diverse. Smaller batches, a much smaller latent space, removing target gradients, or raising the clipping threshold are not justified as the first intervention.

Keep the working 192-dimensional model, batch 128, action alignment and normalization policy. First test a lower-variance estimate of the same regularizer; then test a smaller, coverage-preserving training population against a size-matched random subset. Separate those two claims: fewer processed examples and fewer independently collected experiences.

## What an example means here

| Quantity at inspected checkpoint | PushT, step 13933 | TwoRoom, step 4074 |
|---|---:|---:|
| Source episodes | 18,685 | 10,000 |
| Source frames | 2,336,736 | 920,809 |
| Valid training windows | 1,783,548 | 657,728 |
| Processed windows | 1,783,424 | 521,472 |
| Encoded frame slots | 7,133,696 | 2,085,888 |
| Distinct encoded source frame rows | 2,241,226 | 837,706 |
| Encodings per encountered frame row | 3.183 | 2.490 |
| Supervised transition slots | 5,350,272 | 1,564,416 |
| Distinct transition start rows | 2,146,373 | 778,990 |
| Supervisions per transition start row | 2.493 | 2.008 |
| Mean distinct episodes in a batch of 128 | 127.50 | 127.23 |
| Mean distinct frame rows in its 512 frame slots | 511.93 | 511.89 |

Batch-diversity means use the first 100 exact training batches. Whole-checkpoint counts reconstruct the original seed-3072 split and epoch-zero permutation. PushT is one complete dropped-last epoch; TwoRoom is partway through its first epoch. There is no repeated *window identity* within either processed prefix. Repeated transitions can appear with different history positions, so these repetitions are not necessarily redundant supervision.

PushT has **185 distinct exact initial configurations** under first-frame `state[:5]` (agent xy, block xy and angle), with 90.3 configurations per sampled batch on average. Different episodes from one initial configuration can contain different actions and contacts; this is not permission to discard them as duplicate trajectories. The raw-row counts above are coverage accounting, **not statistical effective sample sizes** or visual-content deduplication.

The loader constructs four observations at raw offsets 0, 5, 10 and 15, and four blocks of five actions. Prediction supervises three transitions; the fourth action block is unused. Window starts advance by one raw frame, so neighboring starts are highly related but do not necessarily share identical encoded rows unless their offsets align modulo five. SIGReg operates on 128 embeddings at each time position; projector BatchNorm sees 512 embeddings and prediction-projector BatchNorm sees 384. The source-compatible window eligibility reserves the unused last action block, excluding four otherwise potentially usable end windows per sufficiently long episode. This is a small coverage inefficiency, not an explanation for orders of magnitude of training.

The uniform global shuffle already prevents substantial within-batch overlap. A new sampler must improve coverage or reduce processing across the run; merely enforcing one episode per batch addresses little measured waste.

## What the gradients say

The probes use 512 fixed **training** windows per dataset, four disjoint batches of 128, nested prefixes of 32 and 64, actual training BatchNorm/dropout behavior, bf16, the source objective and full-batch encoder activation checkpointing. Buffers and RNG are restored between probes. Combined-objective gradients are computed directly, separately from component gradients. This differs from the older dashboard gradient measurements, which used an eval-mode validation batch of eight.

| Mean pairwise cosine of total gradients | Batch 32 | Batch 64 | Batch 128 |
|---|---:|---:|---:|
| PushT: different data, fixed RNG seed | 0.174 | 0.354 | 0.488 |
| TwoRoom: different data, fixed RNG seed | 0.530 | 0.670 | 0.795 |

Larger batches produced more consistent directions in this snapshot. That does not establish an optimal batch size or steps-to-target. A fixed seed across different batch shapes does not fix identical dropout masks or sketch matrices. BatchNorm changes the forward map and SIGReg is batch coupled, so the classical per-example gradient-noise scaling law cannot be applied directly.

At batch 128, mean weighted-SIGReg gradient norm divided by mean prediction-gradient norm is **2.84 on the PushT projector and 1.94 on the TwoRoom projector**. On the encoder the corresponding ratios are 0.65 and 0.67. The two terms are nearly orthogonal on average; this is not evidence of broad antagonistic gradient conflict. The predictor and action encoder get no direct SIGReg gradient.

Prediction's shared encoder input/target branch cosine was -0.148 on PushT and +0.127 on TwoRoom in one bf16 batch. A full-float32 control gave -0.114 and -0.029. These are not near -1: the theory that learning is mostly stalled by two almost-canceling encoder branches is not supported at these checkpoints. The float32 decomposition residuals were 1.7e-6 and 1.0e-5, versus up to roughly 4% in bf16. Float32-only SIGReg barely changes the full gradient direction (cosines 0.99996 and 0.99937); changing all forward arithmetic to float32 has a larger effect, but is not by itself evidence of improved learning. The precision comparison also includes any backend-dependent dropout realization differences.

Every one of 41 logged TwoRoom training updates was clipped. Only 2 of 55 logged updates in the final PushT continuation were clipped. These are sampled logs, not an exhaustive clipping history. An analytical next AdamW update using saved moments gives module-relative parameter changes around 0.015–0.027% in both models, despite TwoRoom's example clip multiplier of 0.54. Adam's moment normalization means that multiplier is not an effective-learning-rate multiplier. Raw module norms alone do not show an optimizer bottleneck.

Steady-state loading waits were roughly 0.0003–0.0004 seconds versus about 1.17 seconds of training computation. More loader workers cannot explain or fix the iteration count. The full PushT schedule also has 1,393 warmup updates (178,304 window presentations), around 27 minutes at the observed update rate; a short run on that schedule does not sample mature optimization. This schedule must remain frozen for reproduction and be retuned explicitly for a new short-budget experiment.

## Is the noise coming from the examples?

Freeze the same 128 windows and independently vary either dropout seed or SIGReg projection seed. Four replicates per condition:

| Conditional total-gradient covariance trace | PushT | TwoRoom |
|---|---:|---:|
| Dropout changes; projection directions fixed | 0.01064 | 0.04179 |
| Projection directions change; dropout fixed | 0.11388 | 0.47020 |
| Projection/dropout variance ratio | 10.70 | 11.25 |

This isolates a substantial source of stochastic variation that needs no new environmental experience. These are conditional variances at a single data batch and checkpoint; they cannot be added to or subtracted from variances measured on different batches as a full variance decomposition. Neither variance nor cosine establishes useful downstream control.

The predeclared candidate increased SIGReg directions from 1024 to 4096, with the same batch, original weights, dropout seed, λ and four projection-seed replicates:

| Dataset | Covariance trace, 1024 | Covariance trace, 4096 | Variance reduction | Pairwise cosine |
|---|---:|---:|---:|---:|
| pusht | 0.11388 | 0.02878 | 3.96× | 0.746 → 0.923 |
| tworoom | 0.47020 | 0.11378 | 4.13× | 0.860 → 0.965 |

This is a successful **conditional noise-reduction measurement**, consistent with the inverse-number-of-projections Monte Carlo variance expectation. It uses zero additional examples. It does not demonstrate fourfold faster learning, a smaller independent-data requirement, or a better final control score. Full training-step throughput and memory at 4096 still need benchmarking; these diagnostic runtimes include repeated backward passes and are not optimizer-update timings. [Comparison figure](../runs/diagnostics/sample_efficiency_2026-09-06/projection_candidate_4096/gradient_geometry.png).


## Joint theory, with its limits

A useful working model is:

**new information in a batch → prediction/representation gradients + stochastic regularization gradients → Adam update → gradually improved latent geometry and dynamics → control.**

Many iterations can be needed even when more independent experience is not the limiting factor. The encoder starts from scratch, its prediction targets evolve with it, the prior constrains its representation, and random sketches add variance to the update. The large window count partly counts repeated source frames and transitions. Meanwhile, the old TwoRoom result already showed that changing only BatchNorm buffers improved primary control from 14/50 to 48/50. Apparent undertraining can therefore include an inference-normalization problem. That intervention did not collect more data or learn better weights.

This is a **multifactor hypothesis**, not an identified single cause of sample complexity. We did not measure the early-training gradient trajectory, learning curves over independent dataset sizes, action/contact coverage of candidate subsets, or curvature/preconditioned noise. The current probes weaken the bad-shuffling, I/O and severe-branch-cancellation explanations. They motivate testing sketch noise and data reuse, without proving that either sets the final control ceiling.

For fixed, independently sampled embeddings, write the sketch discrepancy as Δ and its finite-sample term as F. The empirical-characteristic-function identity gives:

`E[SIGReg_B] = B × Δ + F`, hence `E[∇SIGReg_B] = B × ∇Δ + ∇F` under the usual interchange assumptions.

This follows from `E|φ_hat − φ_target|² = |φ − φ_target|² + (1 − |φ|²)/B`, then multiplying by B and integrating/averaging. It explains why a batch-size sweep at fixed λ also changes leading regularization pressure. A leading-order comparison would hold `λ × B` fixed, while still accounting for changed BatchNorm statistics and finite-sample effects. The formula is explanatory for iid, batch-independent encodings; training BatchNorm prevents treating it as an exact prediction for this network. Empirical rank at B<192 is also bounded by B−1 even for genuine Gaussian samples, so a rank-deficient individual batch is not proof of collapse.

## Literature that affected the decision

Primary sources were checked directly; the HF markdown endpoint for LeWM failed, so its arXiv HTML was used.

- **LeWorldModel (Maes et al., 2026)**: Appendix G reports weak sensitivity to projection count, a drop in control below the usual embedding size (text says around 184), and a useful range of regularization weights. Its limitations discuss difficulty matching a high-dimensional Gaussian in low-diversity environments. This supports investigating the objective while **arguing against treating a 32/64-dimensional bottleneck or more projections as an established win**. Our question concerns early sample efficiency, which these final-score ablations do not settle. [Paper, §§4, 6, Appendix G](https://arxiv.org/html/2603.19312v1)
- **LeJEPA (Balestriero & LeCun, 2025)**: resampling sketch directions across updates is part of its distribution-matching argument; a permanently fixed small set can leave directions unconstrained. Keep resampling. More directions per update can reduce Monte Carlo variance while preserving the objective in expectation; it does not remove finite-data error. [Paper, §4.3–4.4 and implementation](https://arxiv.org/html/2511.08544v3)
- **McCandlish et al. (2018)**: gradient noise scale predicts where increased batch parallelism stops paying off in the studied additive-loss settings. A larger noise-to-signal ratio corresponds to a larger useful batch scale, not a smaller one. Our coupled objective and four replicates do not yield that critical batch estimate. [Paper](https://arxiv.org/abs/1812.06162)
- **Shallue et al. (2019)**: batch-size conclusions depend on workload, optimizer/hyperparameter tuning and budget. Compare steps, examples and time separately, retuning schedules explicitly rather than reading optimality off one frozen checkpoint. [JMLR paper](https://www.jmlr.org/papers/v20/18-789.html)
- **Tang et al. (2022)**: in their self-predictive learning analysis, predictor timescale and semi-gradient representation updates matter for collapse avoidance. This motivates a later faster-predictor/target-dynamics ablation, but its assumptions and objective differ from SIGReg-trained LeWM. Our branch probe does not establish a need for stop-gradient. [Paper](https://arxiv.org/html/2212.03319v1)
- **Schwarzer et al. (2020/2021), SPR**: multi-step latent prediction and augmentation improved sample efficiency in reward-driven Atari learning. This is a precedent for extracting more supervision from experience, not a directly comparable reward-free PushT result; coordinate-changing augmentation must preserve action semantics. [Paper](https://arxiv.org/abs/2007.05929)
- **Killamsetty et al. (2021), GRAD-MATCH**: adaptive subsets can approximate useful training gradients. It motivates comparing a coverage/gradient-aware subset with random selection, but a changing JEPA representation and batch-coupled regularizer require fresh validation; supervised guarantees do not transfer automatically. [ICML paper](https://proceedings.mlr.press/v139/killamsetty21a.html)
- **Klindt, LeCun & Balestriero (2026)**: identifiability results require explicit transition/distribution assumptions. Their experiments separate good covariance matching from useful alignment. A Gaussian-looking embedding alone cannot validate our dynamics or planning. [Paper, theory and limitations](https://arxiv.org/html/2605.26379v1)

## Plausible solution and a clean test

**First intervention: estimate the same SIGReg objective more accurately per encoder pass.** Keep batch 128, latent 192, λ=0.09, resampled directions and all learned components. Compare 1024 with 4096 directions; an implementation may average four independent 1024-direction estimates on the same live embeddings to bound intermediate memory. It must accumulate the mean regularizer gradient before the encoder update, preserve the full-batch statistics and never cache detached embeddings as a replacement for encoder gradients. Verify loss/gradient equivalence for a fixed projection matrix. This adds no environment examples, but its compute/memory cost must be measured.

Run the baseline and this single change with paired initialization, data order and independent model/sketch RNG streams. Use three training seeds and the same explicitly budgeted LR schedule/warmup in both arms. Save checkpoints at equal processed-window counts and equal wall-clock budgets; total encoder passes, optimizer updates and all tuning costs remain visible. Start with a short screening budget, then extend both arms if their curves have not reached useful control. A short-run loss improvement is not a passing gate.

**Second intervention, only after isolating the first: a smaller coverage-preserving dataset.** Freeze a new configuration-disjoint train/development/test split before selection or fitting normalization. For PushT, retain diversity of actions and block motion within each initial-configuration family; for TwoRoom, verify layouts and doorway-crossing coverage. Compare a 25% unique-frame/trajectory budget selected for coverage with a size-matched random subset and the full training population. The exact sampling unit and budget must be fixed: 25% of windows is not 25% of independent experience. Do not identify or tune a subset using the locked test cases. Repeated passes are permitted but count toward processed examples.

Preserve the existing 50 control cases as a separate regression panel; they share source configurations and cannot validate unseen-configuration generalization. Generate a new locked evaluation case set from the held-out groups, with source replay and stationary controls. Use one fixed training-only BatchNorm calibration policy across all arms, and report saved-buffer results separately. Retain the long-goal control check as a guard against optimizing only short-horizon prediction.

**Proposed confirmation criterion (not yet an executed gate):** reach a predeclared held-out control-success target with at most half the processed windows, while staying within five percentage points of the matched baseline final success, over paired seeds and without a wall-time regression. For the independent-data claim, additionally meet that target using at most 25% of the baseline unique-data budget. Freeze the target, sample counts, intervals and compute ceiling before starting; a pilot with three seeds only screens candidates. If improved gradient estimates do not improve these learning curves, reject the sketch-noise bottleneck hypothesis rather than collecting favorable gradient plots.

A much smaller predictive latent remains a secondary hypothesis because the source ablation contradicts an easy dimension reduction. A separate decoder, arbitrary image augmentation, permanent sketch directions, and uncalibrated small-batch accumulation are not part of the proposed first change.

## Collaboration, reproducibility and QA

Claude supplied gradient-estimator mathematics and an initial implementation, competing hypotheses, a corrected derivation of SIGReg's batch scaling, and intervention/ablation ideas. Calls used the registered MCP tool in a neutral directory with tools disabled; no repository source, dataset images or internal measurements were transferred. I applied the code locally, corrected mathematical errors, checked the primary literature, and performed the dataset experiments. The saved Claude text includes unverified or incorrect suggestions; it is a contribution record, not an authoritative report. In particular, general invertible linear transformations do not preserve effective rank, and prediction gradients remain batch coupled when BatchNorm is active.

Two calls reported **$0.649057 total API-equivalent cost**; actual account balances were unavailable. Raw receipts and contributions: `runs/diagnostics/sample_efficiency_2026-09-06/claude/`.

Reproducible commands and all manifests/results are under `runs/diagnostics/sample_efficiency_2026-09-06/`. Main modules: `scripts.audit_sample_efficiency` (checkpoint and output arguments), `scripts.audit_gradient_noise` (independent stochastic/precision controls), and its `--projection-candidate` mode. All standalone runs use `run.py`.

- PushT checkpoint SHA256: `151b356addea1a9bc7c939fcd102986ed1e7dfca693212463308b8456b3ca4f0`.
- TwoRoom checkpoint SHA256: `46ce474a1f978bc5f515f5ed770c514bd675afed6f80b36646bbb4616b0839c0`.
- Raw results include complete indices, source revisions, before/after integrity checks, module gradients, stochastic geometry, analytical Adam updates and source log quantiles. Snapshot tables show explicit per-regime/batch means; individual probes remain in the linked raw JSON/JSONL.
- One stochastic-control attempt was rejected because it ran only one projection replicate and used an inaccurate long-vector cosine reduction. Its raw output is preserved in `noise_controls_rejected/`, excluded from valid measurements and disclosed in dashboard notices. The corrected run enforces four replicates and uses float64 comparison reductions.
- Initial HTML packaging exceeded the 3 MB limit. Reporting was repaired by removing unused SQL intermediate datasets while retaining view/card dependencies, then by using explicit gradient summary means with full raw provenance. A missing metric-card dependency was caught and fixed. Original experiment outputs were preserved.

Final verification: **84 tests passed, including all three installed-browser checks**. The refreshed [canonical dashboard](../runs/experiment_dashboard.html) passed packaging, validation, source interaction and browser verification at widths 1440 and 390. It contains 31 charts, eight tables and five image blocks; the new gradient block includes both dataset audits and the projection-count comparison. All three new figures were visually inspected. The receipt and source/image hashes are in `runs/diagnostics/sample_efficiency_2026-09-06/qa.json`. No experiment or training process remains queued.

The main two probes took 41.73 and 39.15 seconds, and the 4096 candidate took 34.04 seconds; the complete diagnostic work, including the rejected/corrected stochastic checks, stayed well below the predeclared 30 GPU-minute ceiling. No claim of training acceleration is inferred from those diagnostic times.
