# Frozen PushT pose-head comparison — 8 September 2026

The frozen encoder contains substantially more usable orientation information than the original head exposed. The spatial head reduces held-out orientation MAE from **27.31° to 12.98° (52.49%)**, while improving pusher localization. However, object-position errors worsen, so this head is not ready to replace the original system. **None of the heads passes the existing perception-readiness target.**

The user narrowed execution to this first experiment and reassessment. All three head fits and evaluation are complete. Paddle, decoder conditioning, encoder retraining, registers and PushT dynamics remain deferred.

| Head, selected on validation | Test angle MAE ↓ | Test pusher x / y MAE ↓ | Test object x / y MAE ↓ | Validation q ↓ |
|---|---:|---:|---:|---:|
| Original task-only A head | 27.31° | 20.39 / 16.09 | 13.33 / 13.63 | 3.026 |
| Fresh linear head | 24.73° | 14.51 / 13.65 | 13.36 / 16.23 | 2.384 |
| Fresh nonlinear head | 23.88° | 18.02 / 17.57 | 14.78 / 19.07 | 2.220 |
| Spatial head | 12.98° | 7.36 / 6.52 | 25.67 / 28.08 | 4.073 |

Positions use the source world's 512-unit coordinate system, not RGB64 pixels. Readiness is q=max(each position MAE/8, orientation MAE/10), with q≤1 required. Linear and nonlinear heads pass the predeclared ≥10% relative validation improvement criterion; spatial does not. None passes absolute readiness. The spatial head's test q is 3.510 versus original 2.731, despite its better orientation and pusher position.

## Interpretation and proposed reassessment

Successful orientation decoding from unchanged encoder outputs establishes accessibility under this head and budget. It makes a blanket claim that the encoder cannot represent geometry untenable. It does not establish complete geometry, reliable dynamics, or that encoder changes could never help.

The spatial head has 73,987 trainable parameters, versus 122,886 for linear and 1,311,174 for nonlinear. Its orientation gain therefore does not require a larger head than the original linear architecture. Spatial structure, normalization and added landmark supervision change together; this experiment does not isolate which causes the gain. Fresh linear refitting also helps, suggesting the original jointly trained readout was not the best achievable readout of the final fixed encoder.

The spatial object's position errors are already large on the training sample: 23.40 / 28.11 world units, similar to its test errors. This differs from a purely held-out generalization failure. The probability maps are diffuse for object center and orientation landmark, while the pusher map is concentrated. The shared-scale and row-scaled figures show both facts. The landmark is a virtual point 40 world units along the labelled angle, not an independently annotated physical corner.

My proposed next experiment, **not started**, is to separate object-position estimation from orientation estimation and express their losses in the existing physical tolerance units. That would test whether coupling direction to two moving point estimates and their loss balance causes the localization tradeoff. The spatial validation curve is still improving at the fixed endpoint, so insufficient optimization time is also plausible; it is not proven that a larger budget alone fixes the problem. A bounded budget extension would be a separate control, not an automatic continuation. No test-set results were used to select a different checkpoint or extend training.

The practical choice now is to preserve all parents and use this result to design the next geometry comparison, rather than start PushT dynamics with a failing observer. Paddle and retention work can remain queued conceptually until the user reassesses priorities.

## Protocol, selection and uncertainty

The [predeclared protocol and scope amendment](world-model-next-experiments-2026-09-08.md) freeze the selected task-only A encoder from the prior curriculum. The cache stores exact FP32 features for all 20,493 training, 2,651 validation and 2,506 test frames. Configuration groups are disjoint: 164 / 20 / 22. Test populations were already inspected in prior work; this is an exploratory one-seed study, not independent confirmation.

Each fresh head receives the same 2,000 AdamW updates, 128 frames per update, seed 6107, learning rate 0.0003, weight decay 0.0001 and gradient clipping 1. All 6,000 updates and 768,000 training presentations completed. Identical frame-draw hashes match across arms. Head-only training from cached features took 20.3 / 17.7 / 24.4 seconds for linear/nonlinear/spatial; feature extraction took 34.0 seconds. These describe this hardware/runtime and exclude software development, tests and dashboard work.

Selection uses the same fixed 2,048 validation frames every 100 updates: lowest q, then pose MSE, then earliest update. Linear and nonlinear select update 1,700; spatial selects final update 2,000. The final linear/nonlinear checkpoints happen to score better on test orientation (23.62° / 23.12°); they remain secondary and do not replace the validation-selected results above. Training diagnostics use a fixed 2,048-frame sample, so its original-head orientation MAE 5.37° need not equal the earlier study's differently sampled 5.51°.

The 2,000-draw paired bootstrap resamples whole test configuration groups, preserving paired frames and frame-weighted means. Selected-head minus original orientation MAE differences, with descriptive 95% intervals, are:

- Linear: −2.59° [−4.79°, −0.51°].
- Nonlinear: −3.43° [−5.24°, −1.78°].
- Spatial: −14.34° [−22.34°, −7.29°].

These intervals are conditional on the trained models and 22 sampled groups; they do not quantify training-seed uncertainty or correct for exploring multiple heads. Raw per-frame and per-configuration errors remain available.

## Figures, verification and reproduction

[Verified dashboard](../runs/experiment_dashboard.html) · [Raw metrics](../runs/bottlenecks_2026-09-08/pose/experiment/evaluation/metrics.json) · [Integrity audit](../runs/bottlenecks_2026-09-08/pose/experiment/evaluation/integrity_audit.json) · [Run navigation](../runs/bottlenecks_2026-09-08/pose/experiment/README.md)

![Pose errors by head and population](../runs/bottlenecks_2026-09-08/pose/experiment/evaluation/pose_errors.png)

![Spatial landmarks and probability maps](../runs/bottlenecks_2026-09-08/pose/experiment/evaluation/spatial_heatmaps.png)

![Spatial maps with explicitly different row scales](../runs/bottlenecks_2026-09-08/pose/experiment/evaluation/spatial_heatmaps_detail.png)

![Validation learning curves](../runs/bottlenecks_2026-09-08/pose/experiment/evaluation/learning_curves.png)

All **362 software tests pass**, including installed-browser checks. Each completed development/full stage refreshed and verified the canonical dashboard. Final raw-metric reconciliation, validation-selection audit, cache/parent hashes, disjoint groups and paired sample identities pass. The first development browser attempt failed inside the sandbox and was repaired by rerunning only reporting with Chromium access; no completed science was discarded. Final figures are visually inspected. The original decoder-recovery dashboard is preserved under `runs/bottlenecks_2026-09-08/pose/prior_reports`.
