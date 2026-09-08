# Training curriculum results — 7–8 September 2026

The bounded curriculum and all scheduled evaluations are complete, including
all 3,500 Paddle controller episodes. The Paddle follow-up improves learned
first-interception success from 36.2% to 69.0% on ordinary starts and from 20.5%
to 57.5% on opposite-history cases. It still misses the 90% control targets and
the declared memory/prediction accuracy targets.

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
benefit at equal total updates or satisfy the nonworsening adoption rule.

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

The separate identical-current-frame probe contains 100 opposite-direction pairs.
A linear probe fitted on 4,096 training frames gets horizontal direction right
for 100 / 200 members and both members right in 0 / 100 pairs. Frozen U/R gets
171 / 200 directions right and both members right in 72 / 100 pairs. Its vx/vy
MAE is nevertheless 4.640 / 0.880, versus the frame probe's 6.000 / 1.448.
History retains useful direction information, but horizontal speed is
underestimated in these near-interception cases. This is a readout diagnostic,
not a controller-success measurement.

[All paired memory readouts](../runs/curriculum_2026-09-07/paired_memory_inspection/paired_memory_readout.png) ·
[Exact paired readout metrics](../runs/curriculum_2026-09-07/paired_memory_inspection/metrics.json)

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
warmup, despite their low mean coordinate error. On those terminal targets,
median ball-y absolute error is only 0.115 pixels for actual-frame H, versus
9.36 pixels for the five-step prediction. Thus the actual-frame count is
particularly sensitive to the exact cutoff, while the forecast error is much
larger. These are recorded threshold false negatives, not a newly trained
terminal classifier or a causal attribution of the control failures. The
boundary rule and model remain unchanged.

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

## Full controller comparison

All **500 ordinary starts and 100 opposite-direction pairs**, under all five
controllers, completed with no evaluation errors, planning failures or invalid
candidates. The primary outcome is catching the first return before a miss;
it does not imply indefinite survival.

[Complete comparison figure](../runs/curriculum_2026-09-07/control_comparison/control_comparison.png) ·
[Exact matched-case statistics](../runs/curriculum_2026-09-07/control_comparison/comparison.json) ·
[Full evaluation ledger](../runs/curriculum_2026-09-07/paddle_history/evaluation/metrics.json)

| Controller | Ordinary reference → new | Paired reference → new | New paired first action correct |
|---|---:|---:|---:|
| Learned | 181/500 → **345/500 (69.0%)** | 41/200 → **115/200 (57.5%)** | 155/200 (77.5%) |
| Reset memory | 136/500 → 183/500 (36.6%) | 40/200 → 28/200 (14.0%) | 67/200 (33.5%) |
| Random | 115/500 → 115/500 (23.0%) | 14/200 → 14/200 (7.0%) | 62/200 (31.0%) |
| Current-frame tracker | 369/500 → 369/500 (73.8%) | 0/200 → 0/200 (0.0%) | 0/200 |
| Privileged simulator planner | 444/500 → 444/500 (88.8%) | 200/200 → 200/200 (100%) | 200/200 |

Learned improvement over the historical reference is **+32.8 percentage points**
on ordinary starts, with a 95% matched-start bootstrap interval of **[27.4, 38.2]**.
On paired histories it is **+37.0 points**, with a whole-pair interval of
**[28.5, 44.5]**. Paired first-action accuracy rises from 79/200 to 155/200:
+38.0 points, interval [30.5, 46.5].

Each interval uses 2,000 bootstrap draws, seed 93500 for ordinary starts and
93501 for pairs. Both members of a pair are resampled together. These are
post-training descriptive intervals over the recorded cases, not uncertainty
across training seeds or a new adoption gate. Their coverage is approximate;
degenerate intervals for unchanged/all-success controls are not guarantees
about future cases.

The memory ablation is substantial in the new pipeline: learned versus reset
success is 69.0% versus 36.6% ordinary and 57.5% versus 14.0% paired. Direction
information measured in U/R therefore accompanies a useful closed-loop benefit
on these cases. The learned controller remains behind the tracker on ordinary
starts, while it succeeds on many identical-frame pairs where the tracker fails.
Even the exact-simulator planner reaches only 88.8% ordinary success under this
fixed horizon/scoring rule; this is not a proof that the environment itself has
an 88.8% success ceiling.

The new learned policy makes 778 total hits over ordinary cases and 272 over
paired cases, with mean episode lengths of 85.98 and 54.35 intervals. All
controllers retain the two neutral warmup steps and the 200-step episode limit.
Every learned/reset decision evaluates all 243 five-action sequences.

The controllers are:

- **Learned:** five-step exhaustive planning with P5 and the accumulated real-observation memory.
- **Reset:** the same planner, with memory zeroed and rebuilt from only the current frame before each decision.
- **Random:** seeded uniform choices among the three actions.
- **Tracker:** follows the current H ball-x estimate with a two-pixel deadband.
- **Privileged:** exhaustive planning through exact simulator dynamics with the same horizon and scoring rule.

Historical/current case IDs, initial states, seeds and action labels match
exactly. The data, perception checkpoint and evaluation settings also match;
U/P checkpoints differ as intended. Random and privileged ordinary action
trajectories match exactly. The tracker differs on one ordinary trajectory
(`ordinary_488_tracker`) despite identical aggregate outcomes. The RTX 4090 /
Torch 2.14 historical runtime and RTX 3050 / Torch 2.9 home runtime differ, so
these gains cannot be attributed solely to U through a controlled runtime-matched
intervention. Only one trained pipeline per condition was evaluated.

Current learned decision latency is **195.8 ms median / 201.3 ms p95** ordinary,
and **205.1 / 219.1 ms** paired. The historical ordinary median was 35.6 ms on the
RTX 4090. These synchronized timings include planning/scoring and exclude real
frame rendering, real E/U assimilation, decoding and progress writes. Brief CPU
analysis/browser tasks overlapped the benchmark; timing comparisons across these
machines are descriptive, not a model-speed experiment.

[First successful case: actual/reconstructed/imagined futures](../runs/curriculum_2026-09-07/paddle_history/evaluation/visuals/ordinary_1_learned.png) ·
[First failed case](../runs/curriculum_2026-09-07/paddle_history/evaluation/visuals/ordinary_0_learned.png) ·
[Animated evaluation page](../runs/curriculum_2026-09-07/paddle_history/evaluation/index.html)

These are the first eligible success and failure in the fixed case order,
selected by outcome for illustration. Their near-contact predictions use the
same actually executed actions as the real futures. The imagined ball fades in
both, including the successful case; predicted paddle position can also differ
substantially. Repeated replanning can yield useful control despite poor decoded
imagery. These two illustrations do not estimate how frequently each visual
failure occurs.

## Declared engineering targets

| Target | Completed measurement | Outcome |
|---|---|---|
| PushT pose: each XY MAE ≤8 world units and angle ≤10° on selection validation | A/B/C q = 3.009 / 3.546 / 5.159 | All fail; no PushT U/P expansion |
| Paddle actual H: each coordinate MAE <1 pixel | 0.07075 / 0.07611 / 0.09235 over all 17,831 test frames | Pass |
| Paddle real-memory velocity: each MAE <0.5 | 0.86914 / 0.62626 after warmup | Fail |
| Paddle five-step H: each coordinate MAE <2 pixels | 5.84882 / 4.04752 / 4.80097 | Fail |
| Learned ordinary first interception ≥90% | 69.0% | Fail |
| Learned paired first interception ≥90% | 57.5% | Fail |

P1's original copy-comparison gate passed and correctly enabled P5. That
conditional training gate is separate from these final engineering targets.
The new Paddle pipeline is a useful control improvement on the frozen cases,
but the requested overall quality targets remain unmet. No additional training
or evaluation jobs remain queued.

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
evaluations. Reporting failures were repaired without repeating completed training or control:
initial scope exceeded the dataset limit, absolute source paths and full embedded
figures exceeded packaging constraints, and the final nested-scope resolver could
not locate existing rollout PNGs. Each repair preserved raw evidence and restored
browser verification. The compact
reader embeds the primary panels; all full-resolution figures remain alongside
their raw evidence. Earlier historical dashboards and raw ledgers are preserved.

Final software verification: **355 tests pass**, including all three opt-in
installed-browser integration checks. The final canonical artifact is 2,585,987
bytes and passes package/data/source interaction plus 1440/390-pixel browser QA.
See the [test log](../runs/curriculum_2026-09-07/final-software-tests.log) and
[browser receipt](../runs/experiment_dashboard.receipt.json).
