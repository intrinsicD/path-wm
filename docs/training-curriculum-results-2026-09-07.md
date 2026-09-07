# Training curriculum results — 7 September 2026

Perception training and its frozen evaluation are complete. All scheduled training is complete. The full Paddle controller evaluation is
still running and will be added before this execution is closed.

The task-only PushT arm is the best of the three at equal total updates, but none
meets the declared pose targets. COCO E/D warmup does learn useful image
reconstruction; subsequent task adaptation severely damages that generic
reconstruction capability. No pretraining policy was adopted and no PushT U/P
training was launched.

[Verified offline dashboard](../runs/experiment_dashboard.html) ·
[Accepted protocol](training-curriculum-2026-09-07.md) ·
[Execution and amendments](training-curriculum-execution-2026-09-07.md) ·
[Frozen selection and matching audit](../runs/curriculum_2026-09-07/seed_4107/screen_decision.json)

## What ran

The restored task data and four inference bundles passed full source verification.
The home environment uses Python 3.12.9, Torch 2.9.0+cu128 and an RTX 3050 with
8 GiB. It is a different runtime from the historical work-machine experiments.

COCO preparation decoded all 82,783 source images. Exact/canonical matches and
all dHash64 pairs at distance at most four were grouped transitively before the
internal train/validation/test split: **74,501 / 4,136 / 4,146 images**.
The grouping is conservative, including some visually unrelated sky images.
It does not guarantee detection of every possible transformed duplicate.
These are internal splits, not the official COCO validation benchmark.

CCHI retained the verified **20,493 / 2,651 / 2,506 frames** across
**164 / 20 / 22 source episodes**. Original grouped split membership and
normalization were preserved. The physical selector used the original fixed
2,048 validation frame indices, independently of the subsequent full-split
diagnostics.

| Arm | Reconstruction-only updates | Supervised CCHI updates | Total updates | Training + validation time |
|---|---:|---:|---:|---:|
| A: task-only | 0 | 4,000 | 4,000 | 232.9 s |
| B: COCO warmup | 2,000 | 2,000 | 4,000 | 219.5 s |
| C: CCHI-image warmup | 2,000 | 2,000 | 4,000 | 215.4 s |

All arms used effective batch 128, FP32, AdamW learning rate 0.0003,
weight decay 0.0001 and gradient clipping at 1. The matching audit verifies
identical initial E/D tensors, fresh identical task heads, and identical first
2,000 supervised batch draws. No pose loss or fake task targets were supplied
during either warmup. Each arm processed 512,000 image presentations, with
different numbers of labelled presentations.

The 100-update profile took 6.01 s with peak allocated GPU memory 1.39 GB.
The 64-frame training diagnostic completed 500 updates. Its selected update 300
reached position MAEs of 1.99–2.62 world units and angle MAE 2.43°. At update
500, position error regressed sharply even as reconstruction improved.
This is training-set capacity/optimization evidence, not held-out readiness.

## Physical accuracy and the frozen decision

The prospective selector is
`q = max(pusher-x/8, pusher-y/8, block-x/8, block-y/8, angle-degrees/10)`,
using mean absolute errors. Numeric readiness requires **q ≤ 1**,
plus inspected object reconstructions. Eight world units is one RGB64 pixel.

| Arm | Selected supervised update | Selection-validation q | Test pusher x/y MAE | Test block x/y MAE | Test angle MAE | Test q |
|---|---:|---:|---:|---:|---:|---:|
| A | 3,800 | 3.009 | 20.39 / 16.09 | 13.33 / 13.63 | 27.31° | 2.731 |
| B | 1,900 | 3.546 | 20.85 / 20.43 | 25.61 / 18.23 | 33.81° | 3.381 |
| C | 1,900 | 5.159 | 35.03 / 29.65 | 20.55 / 23.00 | 37.28° | 4.379 |

Every selected physical validation error is worse in B and C than in A.
Neither warmup passes the predeclared improvement rule, and all three fail
readiness. The decision was frozen before new test inference. The protocol
therefore stops PushT expansion and does not trigger seeds 4108/4109.

The equal-supervised-exposure comparison is more nuanced. At exactly 2,000
supervised updates, test angle MAE is 41.05° for A, 36.08° for B and 35.13° for C.
B has lower test q than A at that point, but its block-y error is worse and it
has already spent another 2,000 updates on warmup. This does not establish a
benefit at equal total compute or satisfy the nonworsening adoption rule.

The historical selected reference has test q 6.682 and angle MAE 45.37°.
The new A model improves on that historical snapshot, but seed, runtime,
training budget and checkpoint selector differ. This is context, not a
controlled estimate of one intervention's effect.

## What the internals and reconstruction checks show

[Comparison figure](../runs/curriculum_2026-09-07/perception_analysis/perception_comparison.png) ·
[A perception states](../runs/curriculum_2026-09-07/inspection/A_selected/perception_states.png) ·
[B perception states](../runs/curriculum_2026-09-07/inspection/B_selected/perception_states.png) ·
[C perception states](../runs/curriculum_2026-09-07/inspection/C_selected/perception_states.png) ·
[A pose readout](../runs/curriculum_2026-09-07/inspection/A_selected/pose_readout.png)

The selected task-only model has training angle MAE **5.51°** versus test
**27.31°**. Its training pusher position MAEs remain about 11.1 world units,
so the result combines remaining position underfitting with a substantial
orientation generalization gap.

All selected models reconstruct held-out task images much better than a
train-only mean image. Their full-test RGB MSE ratios to that baseline are
**0.0794 / 0.0581 / 0.0966** for A/B/C. Separate 256-frame pusher and block
region checks also beat the mean-image control. Good reconstruction has not
translated into the required pose accuracy.

Fine token maps visibly respond to moving objects. Coarse maps contain strong
spatial structure and, particularly after COCO warmup, much less variation
between frames. Mean across-frame token/channel standard deviation is
fine/coarse **0.0597/0.0513** for A, **0.0751/0.00605** for B and
**0.0625/0.0185** for C. These are descriptive values in different learned
coordinate systems; they do not by themselves prove collapse or causation.

The maps use PCA bases and color limits fitted only to training tokens.
PCA colors are independently defined for each checkpoint. Cross-scale attention
shows measured weights for the marked pusher query, averaged over four heads;
attention is not causal attribution. Raw per-head weights are retained.

The error tails matter. Selected A's test angle-error 95th percentile is
**143.68°**. Its group-balanced angle MAE is 26.21°, while the three worst
test groups have means around 65.8°, 65.1° and 49.3°. There are 78 coordinate
estimates outside the nominal 0–512 world bounds. The corrected readout plots
include those estimates; no prediction is clipped in the metrics.

The train-only linear ridge probes use only 256 training frames and fixed
regularization. They are limited diagnostic controls, not a determination that
E is or is not the sole bottleneck. The channel spectra include both spatial
and across-frame variation; they are not the rank of the entire 20,480-feature
representation.

## Generic reconstruction retention

[COCO before/after image grid](../runs/curriculum_2026-09-07/generic_reconstruction/coco_retention.png) ·
[Exact COCO metrics](../runs/curriculum_2026-09-07/generic_reconstruction/metrics.json)

On all 4,146 internal COCO test images, reconstruction MSE is:

| E/D checkpoint | RGB MSE |
|---|---:|
| Train-only COCO mean image | 0.069517 |
| After 2,000 COCO warmup updates | 0.006072 |
| After selected task adaptation | 0.272196 |
| After final task adaptation | 0.265026 |

The generic warmup succeeds at reconstruction. Task adaptation increases its
COCO reconstruction error roughly **45-fold**, with the decoder output becoming
dominated by the task's white background and limited colors. This is forgetting
in the combined E/D system; the experiment does not isolate E from D.
The immutable generic warmup checkpoint remains available for future work.

## Paddle follow-up

P1 completed its 20,000-update budget in 893.9 seconds. Its selected update 19,750
has normalized latent loss 0.19725 versus copy 1.07739 and position MAEs
[2.38197, 2.38395, 1.59502] pixels versus copy [3.83335, 2.38499, 2.52450].
All four original comparisons strictly improve, so P5 was launched. Ball-y's
margin is only 0.00105 pixels; the gate's pass should not be mistaken for a robust
quality margin. The final P1 update 20,000 has ball-y 2.60224 and would fail the
gate; selection remains the original fixed-validation objective.

At exact update 10,000, new P1 MAE is [3.81218, 3.74440, 1.82866]. The historical
10,000-update reference is [3.70127, 3.59783, 1.79505]. At their selected
20,000-budget checkpoints, the historical/new values are
[2.42536, 2.22583, 1.65811] versus [2.38197, 2.38395, 1.59502].
The new memory does not uniformly improve prediction. Validation window indices,
E/D/H dependency, dataset and E-only variance bytes match; the U dependency
differs as intended. Runtime differences remain a limitation of the
cross-session comparison.

P5 completed 10,000 updates in 609.4 seconds and selected 9,250 by its unchanged
validation objective. Its five-step validation MAE is [5.82656, 4.30762, 4.93063]
pixels. On 1,024 matched test windows, five-step MAE is [5.84882, 4.04752, 4.80097],
versus copy [16.16791, 11.07811, 5.54455]. All three coordinates fail the two-pixel
engineering target. One-step test MAE is [3.37767, 2.29149, 1.53336].

Across 16,831 actual test observations after warmup, H position MAE remains
[0.07125, 0.07575, 0.09343] pixels. U/R velocity MAE is [0.86914, 0.62626] pixels
per interval, above the 0.5 target. Memory reset degrades five-step ball-x MAE
from 5.85 to 14.79 pixels, evidence that the predictor uses its history, without
establishing that the learned history is sufficiently accurate.

Collision stratification shows that the error is not confined to bounces:

| Five-step test population | Windows | Predicted ball x/y, paddle x MAE | Copy MAE |
|---|---:|---:|---:|
| Any collision in source-to-target prefix | 435 | 6.21 / 4.26 / 4.77 | 12.55 / 10.31 / 5.56 |
| No collision | 589 | 5.58 / 3.89 / 4.82 | 18.84 / 11.64 / 5.53 |
| Ceiling reflection | 52 | 5.63 / 5.98 / 4.65 | 17.03 / 5.91 / 4.97 |
| Paddle reflection | 47 | 5.18 / 3.99 / 5.36 | 15.37 / 6.03 / 4.70 |

Individual collision types overlap and must not be added together. The predictor
loses to copy for ball-y in ceiling windows and paddle-x in paddle-reflection
windows. Real-memory velocity errors also rise around reflections: vx/vy MAE is
1.44 / 1.04 with a collision versus 0.82 / 0.59 without one.

The exact terminal boundary needs separate attention. Of 35 five-step windows
whose target ball-y reaches 61, 34 predicted readouts remain below 61. Even actual
H readouts fall below that threshold in 342 of 496 terminal observations after
warmup, despite their low mean coordinate error. These are recorded threshold
false negatives, not a newly trained terminal classifier or a causal attribution
of the control failures. The boundary rule and model remain unchanged.

[Memory heatmaps, PCA, velocities and attention](../runs/curriculum_2026-09-07/paddle_history/inspection/memory_states.png) ·
[Actual versus decoded imagined rollouts](../runs/curriculum_2026-09-07/paddle_history/inspection/paddle_rollouts.png) ·
[Matched prediction curves](../runs/curriculum_2026-09-07/paddle_history/inspection/paddle_prediction.png)

The first four test episodes are shown for memory; its PCA/standardization fits
only the first eight training episodes. The two earliest eligible test episodes
show actual and imagined RGB under the same recorded actions. In those examples,
the imagined ball fades or disappears even though actual-frame E/D reconstruction
is accurate. This is qualitative evidence from fixed examples, not a prevalence
estimate. Figures were rendered on CPU with two threads; quantitative prediction
metrics reuse the hash-verified completed GPU cache. Checkpoint tensors remain
unchanged. Brief CPU figure rendering, software checks and dashboard verification overlapped controller evaluation;
recorded decision timings include whatever host scheduling occurred.

The full 500-start/100-pair, five-controller comparison is still running. No
aggregate control conclusion is drawn from its partial cases.

## Evidence, limitations and verification

All selected/final task snapshots, A's exact update-2,000 snapshot, both image-only
warmups, and both historical reference snapshots were evaluated. There are
16 detailed perception inspections including the diagnostic and full-validation
checks. Raw per-frame errors, group summaries, region denominators, probe
predictions, PCA bases, attention weights and full PNG/SVG figures are under
[the inspection directory](../runs/curriculum_2026-09-07/inspection).

This is one training seed. It establishes the outcome of the declared bounded
screen, not a general claim that pretraining cannot help. Test evaluations cannot
select a checkpoint or reopen the training budget. Existing historical test
evidence is development context.

Canonical HTML publication passed data/package/browser verification after the
evaluations. Two reporting failures were repaired without repeating completed
training: absolute source paths and the 3 MB portable payload limit. The compact
reader embeds the primary panels; all full-resolution figures remain alongside
their raw evidence. Earlier historical dashboards and raw ledgers are preserved.
