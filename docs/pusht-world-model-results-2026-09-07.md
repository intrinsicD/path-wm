# CCHI PushT E/U/P results — 2026-09-07

The separate PushT implementation runs end to end. The first bounded GPU
experiment **failed the one-step predictive gate**; no five-step training or
full held-out control was performed for that reference. This is measured
negative evidence, not a working learned-control claim.

## Source and protocol

Official CCHI source:206episodes,25650frames,25444valid transitions, RGB96 with
absolute target-XY commands at10Hz. Prepared RGB64 INTER_AREA memory map and
all labels/actions were source-verified. Split164/20/22 by disjoint initial
configuration groups; no exact cross-split frame/trajectory duplicates.
Data fingerprint:
`e6ff0cad101c15ba3e2b839df28bc0bd75dca4063a7d6293141976ab44b4fa9c`.
Training-only motion RMS scales are
`[8.1456785245,8.0708712973,2.0288295915,2.1618275627,0.02969219039]`;
action-offset RMS is`[0.03958002446,0.03922926990]` in normalized XY units.
See the [design](pusht-world-model-design.md),
[source audit](pusht-world-model-data-audit.md), and
[commands](pusht-world-model-usage.md).

## Bounded GPU reference

Command: `python run.py -m world_model.pusht run-all --config
configs/pusht_world_model/baseline.yaml --data data/pusht_world_model/cchi_v1
--run runs/pusht_world_model/baseline`. RTX4090 FP32, seed3107,
constant AdamW3e-4, original declared objectives. The full process log is
`runs/pusht_world_model/baseline.log`; each stage records dependency identities,
normalization, selected/last checkpoints, optimizer/RNG and exact raw metrics.

| Stage | Completed / selected update | Presented examples | Recorded stage seconds |
| --- | ---: | ---: | ---: |
| E/D/H |1000 /200|128000frames|51.941|
| U/R |1000 /1000|2000episodes|110.191|
| P K1 |2000 /2000|128000windows|61.873|
| P K5 |Not started|0|—|

Stage seconds cover optimization and validation, excluding preliminary latent
statistics/observer cache construction and HTML generation. U presented247727
observations and2704997supervised scalars. Recorded peak CUDA allocation is
537006080bytes for the process; this counter is cumulative across its stages.

Selected perception validation on2048fixed frames: pusherXY/blockXY MAE
`[48.6178,57.7777,19.1579,22.0184]` in the512-unit world, angle45.194°,
image MSE0.00688356 and pose6MSE0.107162. The selected reconstruction is largely
blank background. At update1000, training angle MAE is21.053° while validation
remains45.255°; reconstruction MSE improves to0.001614 on validation, but total
validation objective worsens. Initialization is eligible for selection; no
unrecorded normalization or checkpoint-buffer recalibration was used.

Selected memory validation: position MAE
`[24.1909,24.1956,34.6083,40.9290]`, angle45.706°, and causal motion MAE
`[5.03009,5.27395,1.11568,1.34879,0.0177542]` (four world displacements and
wrapped radians per0.1-second interval). The fixed16episodes contain2140frames,
2108post-warm-up observations and23380supervised scalars. These are observable
motion targets, not instantaneous simulator velocities.

| K1 selected validation on256windows | Predictor | Matched copy | Strict improvement |
| --- | ---: | ---: | --- |
| Variance-normalized latent MSE |0.05043577|0.05282639|Yes|
| Pusher-position MSE, world units squared |4164.0863|4090.4309|**No**|
| Block-position MSE, world units squared |783.5225|797.3671|Yes|
| Circular angle MSE, radians squared |1.2464985|1.2482304|Yes|

All four conditions are required. The gate is false, with no smoke bypass.
Perception generalization and reconstruction are being diagnosed before choosing
another training-distribution or objective experiment. No test-set result has
selected these models or their next intervention.

## Corrected CPU smoke

`runs/pusht_world_model/smoke_v2` uses2updates per stage,2training/1validation
episode with explicitly declared32-frame prefixes. Best E/P checkpoints remain
at initialization, and the predictive gate is false. Its explicit smoke override
exercises K5 and export without claiming learning. `inference.pt` contains all
six modules plus latent statistics and normalization.

Validation control uses2groups with initially unsolved five-interval goals,
selected before outcomes at twice either task tolerance. Both replay cases reach
their constructed goal in5actions. Learned/reset/hold/random/repeat-last each
finish0/2 after10actions. Any-time successes differ (hold1/2, random2/2),
demonstrating why final and transient success are reported separately. Replay's
five-action reachability check has its own budget and is excluded from the
primary controller-success chart. No planning failures occurred.

The first smoke remains preserved. Its control run exposed a float64 hold action
entering the FP32 GRU; the corrected adapter normalizes executable/model inputs
to float32 and has a real-observer regression. Independent audits also fixed
goal-dependent reset physics, initial-checkpoint selection, train-only latent
statistics, selected-checkpoint gate recovery and evaluator dependency checks.

## Reporting and collaboration

The [canonical dashboard](../runs/experiment_dashboard.html) passes package,
data and browser verification at1440/390px, with exact raw values available.
Dataset-sharing repaired a50-dataset packaging failure without dropping older
experiments. An isolated prospective full-capacity fixture also passed; it was
never entered into scientific ledgers. Representative source/reconstruction and
imagined-rollout panels were inspected. Flat early-smoke decoder images remain
visible as negative evidence.

Independent Codex agents handled data, models/planning, evaluation/reporting and
cross-module audits. Claude's completed design review is retained in
`runs/pusht_claude_design_review.json`; its reported API-equivalent cost is
$2.65936075. An additional perception review was rejected by automatic approval
review because its specific local payload would go to an external service.
That call did not execute; user approval is pending and local diagnosis continues.


## Matched perception coverage experiment

The prospective [coverage plan](pusht-perception-coverage-plan.md) completed in
`runs/pusht_world_model/perception_coverage_v1/perception`. It keeps the source-only
reference immutable and changes the training population: the same 20,493 source
training frames plus 20,493 independently sampled simulator observations, with
actual post-reset labels and no held-out pose selection. The supplement identity
is `34e6e5a352a1355fd6ac7a1a6471bb77f5e5a466e18bf9882a8d9215afb9b7f3`.

Both arms use seed 3107, 1,000 updates, batch 128, the original architecture and
RGB-plus-pose objective, and exactly the same 2,048 validation indices. The
initial E/D/H tensors and initial validation metrics are bitwise identical;
`runs/pusht_world_model/collaboration/coverage_matched_initialization_receipt.json`
records the check. The new arm presented 63,954 source and 64,046 supplement
frames, rather than the reference's 128,000 source presentations. This change in
source exposure is part of the intervention. Stage optimization/validation took
30.260 seconds; data verification, preparation and HTML are excluded.

| Selected fixed-validation metric | Source only, update 200 | Mixed coverage, update 500 |
| --- | ---: | ---: |
| Pusher x MAE, world units | 48.6178 | 53.0009 |
| Pusher y MAE, world units | 57.7777 | 61.8601 |
| Block x MAE, world units | 19.1579 | 23.7561 |
| Block y MAE, world units | 22.0184 | 24.8967 |
| Wrapped angle MAE, degrees | 45.1940 | 46.2130 |
| RGB MSE | 0.00688356 | 0.00281697 |
| Pose MSE | 0.10716243 | 0.11744663 |
| Combined validation objective | 0.11404598 | 0.12026360 |

The coverage arm reconstructs images better at its selected checkpoint but
worsens every selected pose metric above. It is not adopted, and no new U/P
training follows from this comparison. This is one seed at a fixed budget;
it does not show that synthetic coverage can never help with another budget or
objective. Selected-checkpoint comparison, matched final-update comparison,
group-balanced frame diagnostics, and the synthetic stress grid remain distinct
populations and claims.


At the matched final update 1,000 on the original 2,048 validation draws, source
versus coverage RGB MSE is 0.00161414 versus 0.00142271 and pose MSE is
0.14816116 versus 0.12182466. However, all four position MAEs worsen
(`[47.3805, 53.4498, 19.9759, 21.7634]` versus
`[53.8278, 57.7354, 21.4296, 29.5209]`) and angle MAE worsens from 45.2549 to
47.3376 degrees. Mean orientation-vector norm changes from 0.8895 to 0.6115.
A lower six-coordinate pose objective therefore does not establish better
physical localization or orientation.

Independent CPU diagnostics reuse the exact earlier 256 train and 256 validation
frames, plus the unchanged 64-frame synthetic stress grid. Source-checkpoint
predictions reproduce the earlier diagnosis bit for bit. There are 2,304
checkpoint/frame records across four saved models; the 512 source frames and 64
stress frames remain separate populations. No test frames are used, and every
protected file hash and loaded model tensor stays unchanged.

On the group-balanced 256 validation frames at update 1,000, coverage changes
angle MAE from 46.6131 to 46.1951 degrees, but all four position MAEs worsen.
Dynamic-foreground reconstruction MSE improves from 0.0218476 to 0.0139326
(36.2%). On the separate synthetic stress grid, angle MAE improves from 89.4462
to 59.2611 degrees at matched final updates; selected snapshots improve from
92.9315 to 49.7811 degrees. These support foreground/stress-coverage benefits,
not a source pose or control-quality claim. Differing angle conclusions across
the original frame-sampled and group-balanced validation populations are shown
explicitly rather than pooled.

The exact inputs, per-frame predictions, weighted image-region numerators and
denominators, norms, source comparisons and checkpoint identities are in
`runs/pusht_world_model/collaboration/perception_coverage_comparison/raw.json`.
Adjacent `analysis.md`, `script.py`, `report.py`, execution log and the inspected
`examples.png` are included in the portable archive and canonical dashboard.
