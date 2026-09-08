# Current project state

This file owns the implementation objective, active experiments, evidence and next
steps. Standing development and experiment rules live in
[experiment-workflow.md](experiment-workflow.md). Changing a model, dataset or
research direction does not replace that workflow.

## Objective and current outcome

The user requested the fixed paddle E/U/P architecture from
`world_model_codex_implementation_brief.md`, using
`world_model_design_notes.md` as historical context, plus training on PushT.
Independent Codex and Claude agents contributed implementation and review. The
user explicitly authorized committing, integrating into `main`, and pushing the
complete work so it can continue on another computer.

Both implementations execute through data, perception, recurrent memory,
prediction, planning, evaluation and export. **The requested learned-control
quality targets have not been achieved.** Keep this distinction when continuing:
passing software tests, falling losses and a smoke export do not establish useful
learned control. The original LeWM implementations and results remain preserved.

Read the [paddle results](paddle-world-model-results-2026-09-07.md),
[PushT results](pusht-world-model-results-2026-09-07.md),
[canonical dashboard](../runs/experiment_dashboard.html), and
[portable continuation handoff](session-handoff-2026-09-07.md).

## Architecture reassessment (8 September; proposal)

After the completed first diagnostic, the user requested a literature-based
reassessment of explicit pose versus a more general, extensible encoder/decoder.
The [design and implementation proposal](versatile-perception-architecture-2026-09-08.md)
recommends a shared spatial representation, independent output readouts and
explicit temporal state. Pose remains one diagnostic/output; a proposed first
comparison tests current frozen E against a small pretrained E with RGB and
mask/extent readouts. Local COCO instance annotations match the prepared images;
their transformed masks still need preparation and verification. Shared query
decoding and object grouping are later candidates, not adopted components.
This review launched no training and changed no model or existing readiness gate.

The user then broadened the discussion to software use, generated outputs as
actions, internal computation and driving. The [general-agent clarification](general-agent-world-model-2026-09-08.md)
separates visual cross-scale fusion from temporal/task hierarchy and distinguishes
action generation from observation decoding. No trained cross-scale ablation exists
in the inspected implementation/results. A bounded fusion comparison and a small
software action/outcome benchmark are proposed, not launched; historical fixed
multimodal token counts remain unadopted.

## Complete: frozen PushT pose-head comparison (8 September)

User authorized the [bottleneck protocol](world-model-next-experiments-2026-09-08.md),
then narrowed execution to its first experiment followed by reassessment. Frozen
selected task-only A encoder, fresh linear/nonlinear/spatial pose heads, matched
2,000-update budgets and validation selection; original head is the reference.
All three fits and held-out evaluation are complete under
`runs/bottlenecks_2026-09-08/pose/experiment`; see the
[results and figures](pusht-pose-accessibility-results-2026-09-08.md). Spatial H
reduces test angle MAE 27.31°→12.98° and pusher errors, but worsens object-position
MAE to 25.67/28.08 world units. Fresh linear/nonlinear heads improve validation q
by at least 10%; all heads fail q≤1 readiness. Encoder unchanged, matched draws,
362 tests and dashboard verification pass. No jobs remain active.
Paddle and decoder-conditioning experiments are deferred. No encoder adaptation,
PushT dynamics, capacity expansion or register implementation is authorized here.

## Paddle reference and follow-up

Exactly replay-verified data contain 5,000 / 500 / 500 train / validation / test
episodes. Perception completed 10,000 updates and selected update 9,750; test
H position MAE is about 0.071 / 0.076 / 0.092 pixels. Memory completed 10,000
updates, but validation velocity MAE 0.809 / 0.545 exceeds the 0.5 target.
The original 10,000-update P1 failed its ball-y gate. A separate budget-only
continuation reached 20,000 total updates and passed the unchanged gate. P5
completed 10,000 updates. Its test five-step position MAE is
4.582 / 4.043 / 4.671 pixels, above the 2-pixel target.

The full 3,500-case comparison is complete under
`runs/paddle/continuation_v1/evaluation`. Learned control caught the first return
in 181 / 500 ordinary cases and 41 / 200 paired cases, versus random
115 / 500 and 14 / 200; reset memory achieved 136 / 500 and 40 / 200.
The learned ordinary decision median was 35.6 ms and p95 was 40.2 ms, excluding
real-frame encoding, observer update and rendering. The exported inference bundle
was verified. All original control-quality targets remain unmet.

The separate [history-start experiment](paddle-history-start-plan.md) completed
its fixed 10,000 memory updates, selected update 9,750, and preserved the original
architecture and loss. Exactly 40,000 full and 40,000 suffix sequences presented
2,809,923 observations and 13,729,615 supervised scalars. Selection equally
weights ordinary and suffix validation objectives; paired cases are diagnostic
only. Selected ordinary velocity MAE is 0.859 / 0.634, and paired velocity MAE is
4.732 / 0.810. Independent replay confirms improved cold-start position estimates, but at
matched final updates paired vx/vy error and ordinary validation loss worsen.
All 80,000 sampler draws and cumulative counters match the declared schedule.
The intended memory fix is not established; fresh P1/P5 with this U are not
trained before home continuation. Existing P checkpoints depend on the original U and cannot be
silently reused with the new observer.

## PushT reference and coverage experiment

The official compact CCHI source is prepared at
`data/pusht_world_model/cchi_v1`: 206 episodes, 25,650 frames, and disjoint
164 / 20 / 22 initial-configuration groups. All prepared pixels, actions and
labels match the source; no exact cross-split image or trajectory duplicates were
found. Train-only motion and action-offset RMS scales are fingerprinted. The
separate model uses H6 / R11 and absolute target-XY actions. Corrected two-update
CPU smoke exercises every stage, all six controllers/oracles and export. Its
learned final success is 0 / 2; replay reachability is 2 / 2.

The bounded GPU reference completed perception 1,000 updates (selected 200),
U 1,000, and P1 2,000. It failed the strict P1 pusher-position gate: MSE 4,164.09
versus copy 4,090.43. P5 and full test control were correctly not run. Selected
perception validation angle MAE is 45.19 degrees, and its reconstruction is
largely background. Read-only diagnosis found a pose-generalization gap and no
demonstrated source alignment or circular-angle arithmetic bug.

The separate [coverage experiment](pusht-perception-coverage-plan.md) completed
1,000 perception updates with unchanged initialization, model, objective,
optimizer and fixed 2,048 validation indices. It mixed the same 20,493 source
training frames with 20,493 independent simulator poses. Actual presentations
were 63,954 source and 64,046 supplement frames. It selected update 500: image
MSE improved from 0.006884 to 0.002817, but all four position MAEs worsened and
angle MAE rose to 46.21 degrees. The coverage intervention is not adopted; no U/P
expansion is justified by this selected observer comparison. Matched final and
fixed-frame diagnostics remain separate from the primary validation population.

## Collaboration, provenance and continuation

Claude completed independent physics, learning, code, memory-diagnostic and
PushT-design reviews. Their receipts report $14.63338225 in API-equivalent
usage, not a subscription billing claim. Exact reviews are under
`runs/paddle/collaboration` and `runs/pusht_claude_design_review.json`.
A further read-only Claude perception review did not execute: automatic approval
review rejected the additional external transfer pending specific payload
approval. The scoped prompt and rejected-call record are preserved; do not retry
that transfer without the requested approval. Local Codex diagnosis continued.

The handoff includes exact datasets, source HDF5, supplement, selected/resume
checkpoints, raw evidence and offline HTML in verified split archives. Rebuild
only omitted caches. Preserve prepared manifests verbatim after relocation and
pass the new source HDF5 path explicitly for source verification. The final
handoff document and archive manifest record completed stages, environment,
verification and remaining choices; do not infer success from checkpoint names.

Final software verification: **336 tests pass**, including all three installed-browser
checks. The canonical dashboard passes package/data/source-interaction and
1440 / 390-pixel browser verification, with 49 datasets, 55 charts and five
embedded image blocks. See the final raw test log and receipt under
`runs/paddle/collaboration`. The actual separate-directory restoration passed all 13,870 file hashes, full
paddle/PushT/supplement verification and four CPU inference bundles. The package
is 212,389,903 compressed bytes in five parts. Its receipt establishes continuity,
not model quality. All experiment and packaging writers are closed.

## Home curriculum execution (7–8 September 2026, complete)

The user accepted the [curriculum](training-curriculum-2026-09-07.md) and requested
execution, thorough evaluation and internal-state figures. All scheduled training
and evaluation are complete, including all 3,500 Paddle controller episodes.
Read the [results](training-curriculum-results-2026-09-07.md),
[execution record](training-curriculum-execution-2026-09-07.md),
[artifact navigation](../runs/curriculum_2026-09-07/README.md) and
[verified dashboard](../runs/experiment_dashboard.html).

Full archive/source restoration passed. COCO preparation decoded 82,783 RGB64
images and grouped splits of 74,501 / 4,136 / 4,146. CCHI retains all verified
20,493 / 2,651 / 2,506 frames in the original episode groups. The 100-update
profile fit batch 128 in 1.39 GB; no common budget reduction was necessary.

Seed 4107 completed the matched 4,000-update A/B/C screen. Selected validation
q is 3.009 / 3.546 / 5.159; all fail the q ≤1 readiness threshold. Task-only A
wins at equal total updates. Both warmups worsen every selected physical
validation error relative to A. As predeclared, no confirmation seeds or PushT
U/P expansion were triggered. A's full-test angle MAE is 27.31° versus 5.51°
on training frames. Sixteen selected/final/validation/diagnostic inspections,
train-only PCA/probes, attention, region controls and all raw errors are preserved.
COCO warmup reconstruction MSE is 0.00607 on its 4,146 internal test images;
selected task adaptation raises it to 0.27220. The generic checkpoint is retained.
The [8 September interpretation](curriculum-interpretation-2026-09-08.md) explains
the combined E/D forgetting and proposes domain/pose controls, frozen-feature
readouts, replay and application adapters. During the following voice discussion,
the user authorized the [frozen-encoder decoder recovery](decoder-recovery-plan-2026-09-08.md)
diagnostic. All three arms and full paired evaluation are complete; see the
[recovery results](decoder-recovery-results-2026-09-08.md). Frozen-encoder decoder
refitting restores COCO MSE to 0.006104667 (0.53% above original warmup), while
PushT reconstruction worsens 8.72 times. E/H remain exactly unchanged. Matched
fresh decoders leave a 26.33% COCO gap between adapted and original encoders;
this is one-seed, fixed-budget evidence, not proof of information loss.
All 359 tests and the canonical dashboard browser checks pass. Claude's
[public literature review](world-model-literature-2026-09-08.md) is complete.
Task conditioning, skips, geometry changes and registers remain proposals.
No further training is running or queued.

Paddle uses frozen E/D/H and the preserved mixed-history U/R. New P1 completed
20,000 updates, selected 19,750 and passed the original copy gate, with only a
0.00105-pixel ball-y margin. New P5 completed 10,000 updates and selected 9,250.
Five-step test position MAE [5.849, 4.048, 4.801] misses the two-pixel target;
real-memory velocity MAE [0.869, 0.626] misses the 0.5 target. All-frame H passes
its one-pixel mean-error target over 17,831 test observations.

Full learned control improves from 181/500 to **345/500 ordinary** and from
41/200 to **115/200 paired** first interceptions. Matched-case improvements
are +32.8 pp [27.4, 38.2] and +37.0 pp [28.5, 44.5] in descriptive bootstrap
intervals. New reset-memory control achieves 183/500 and 28/200; tracker achieves
369/500 and 0/200; privileged control achieves 444/500 and 200/200. Both learned
90% targets remain unmet. Case identities match exactly, while the historical
RTX 4090/Torch 2.14 and home RTX 3050/Torch 2.9 runtimes differ. One tracker
trajectory changes despite identical aggregate outcomes. These are not
training-seed intervals or a pure runtime-matched estimate of U's causal effect.

On 100 identical-current-frame history pairs, U/R gets direction right in
171/200 members and both directions right in 72/100 pairs, versus 100/200 and
0/100 for a training-fitted frame probe. Velocity magnitude remains inaccurate.
Memory heatmaps/PCA, attention, actual/imagined rollouts and all paired readouts
are visually inspected. Imagined balls can fade even in a successful control case.

All **355 tests pass**, including the three installed-browser integration checks.
The canonical dashboard passes package/source-interaction and 1440/390-pixel
verification. A final nested-scope path failure was repaired with a regression;
all 3,500 completed cases were reused. The dashboard is explicitly scoped to
this curriculum; earlier ledgers and reports remain preserved. All writers are
closed. No further training, external publishing or private Claude transfer was
launched. Further experiments need a new bounded protocol; no current result
reopens the completed training budgets.

## Preserved LeWM objective and evidence

Build a fresh modular implementation of the published LeWM baseline. Demonstrate
learning and control on PushT and support multiple explicit dataset protocols.
The working baseline is preserved after the user-authorized sample-efficiency research and paired training screen; the full reproduction schedule remains incomplete.
The earlier reset removed the previous implementation and results; the retained
ideas and downloaded source datasets carried forward. The reusable development
harness is now recovered separately from that discarded implementation.

Reuse the pinned authors' baseline components. Document differences from the
reference recipe and distinguish smoke checks, subset training, reproduction
training and benchmark evaluation. Baseline-specific tests cover episode
alignment, causal action timing, reference computations and gradients,
normalization, rollout and checkpoint integrity.

## Completed: paired projection-count training screen with Claude (2026-09-07)

All 12 full-source runs completed 1500 updates, and all 48 planned control outcomes are recorded under `runs/projection_training_2026-09-06`. No training, evaluations or finalization jobs remain queued. See the [final report](projection-training-2026-09-06.md), [frozen plan](projection-training-plan-2026-09-06.md), and [verified dashboard](../runs/experiment_dashboard.html). The short cosine schedule is not full reproduction or an independent-data generalization test.

Increasing resampled SIGReg directions from 1024 to 4096 produced modest final calibrated gains: +2 percentage points on each dataset, equivalent to one additional success out of 50 on average. PushT final mean success is 2.67%/4.67%; TwoRoom is 89.33%/91.33% (1024/4096). Early calibrated effects are −0.67 pp on PushT and +2.67 pp on TwoRoom, with TwoRoom paired differences ranging from −8 to +12 pp. Early saved-buffer TwoRoom control is worse with 4096 in every seed (mean −19.33 pp); final saved-buffer effects are 0 pp on PushT and +3.33 pp on TwoRoom. No large learning-speed or independent-data reduction is demonstrated; 4096 remains optional rather than an adopted sample-efficiency fix.

At 750 updates, the same fixed training-only BN calibration adds 13–33 successful TwoRoom cases out of 50 across all six checkpoints without changing parameters. PushT calibration effects remain mixed. Layer attribution and a prospective normalization correction remain open. All 732 logged pre-clip training gradient samples exceed the threshold, but neither every-update clipping frequency nor early-training conditional gradient variance was measured. Final saved-buffer prediction/copy improves in all three TwoRoom and two PushT pairs. These diagnostics do not substitute for control.

Each arm processes 96,000/192,000 distinct optimizer windows at 750/1500 updates, already covering nearly every source episode by the final checkpoint. Exact frame-row reuse at 1500 is about 1.14× on PushT and 1.42× on TwoRoom. Calibration and full-source action normalization add population access; the random-window protocol can share source episodes/frames across splits. The source cases do not establish unseen-configuration or long-goal generalization.

Claude supplied private-RNG implementation/test design and the paired-statistics implementation through two neutral MCP calls ($1.132410 reported API-equivalent). Integration retained the exact pinned loss expression and legacy behavior. All six pairs match initial weights, populations and global CPU/CUDA RNG at both checkpoints; all 24 calibrated clones change only BN buffers. The 13 protected scientific-code/reference/case inputs remain unchanged. All four preselected qualitative panels are verified and inspected.

Final verification: 103 tests pass, including three browser checks. The canonical dashboard passes source interaction and desktop/mobile verification, with all 12 points in each paired chart at both widths. A browser-output failure was repaired by bounding repeated provenance text while preserving every exact value and full source identity; completed science was reused. The final payload is 2,516,589 bytes. Fixed-scale control and discrete gradient/prediction PNG/SVG figures and actual native charts were visually inspected. Original checkpoints and reproduction configurations remain preserved.

## Earlier: sample-efficiency investigation complete (2026-09-06)

The user requested literature research with Claude, batch/gradient inspection and a plausible route to fewer examples. See [the evidence and proposed experiment](sample-efficiency-2026-09-06.md), [predeclared plan](sample-efficiency-plan-2026-09-06.md), and [verified dashboard](../runs/experiment_dashboard.html). No training or diagnostic work remains active. All original checkpoint hashes and model parameters/buffers are preserved.

Across the first 100 batches, PushT/TwoRoom average 127.50/127.23 distinct episodes and 511.93/511.89 frame rows per 128-window batch: within-batch duplication is negligible. Across the processed prefix, encoded source frames repeat 3.18×/2.49×. PushT has 185 exact first-frame state[:5] configuration groups; this is not full-trajectory deduplication.

Four frozen training batches per dataset show more consistent total gradients at batch128 than32. SIGReg's mean gradient norm exceeds prediction's on the projector by 2.84×/1.94×; encoder terms are mostly near-orthogonal, without severe input/target branch cancellation. Float32 controls support that conclusion. Classical critical-batch estimates are invalid for this coupled BatchNorm/SIGReg objective.

On fixed data, random SIGReg directions produce about11× the gradient variance of dropout alone. A predeclared4096-vs1024 projection probe reduces conditional variance **3.96× on PushT /4.13× on TwoRoom**, with zero new examples or optimizer updates. This is a measured mechanism, **not evidence of faster training or higher control success**. The first proposed experiment keeps latent192/batch128/λ0.09 and tests lower-variance sketches on paired learning curves; a separate configuration-disjoint, coverage-preserving subset experiment tests independent-data efficiency. Literature argues against assuming a32/64-dimensional latent is a free improvement.

Claude contributed mathematical analysis, generic gradient-code implementation and experiment design through two neutral-directory MCP calls ($0.649057 reported API-equivalent total); no repository transfer was needed. Eight primary papers informed the final report. One stochastic diagnostic was rejected and preserved after replicate-count and numerical-reduction defects; corrected results are used. Dashboard payload failures were repaired without changing raw experiment outputs. Final verification:84 tests passed including3 browser checks; canonical HTML verified at1440/390 with31 charts,8 tables and5 image blocks.

## Completed: diagnostic follow-up and one-epoch PushT continuation (2026-09-06)

The accepted work with Claude implementation co-work has completed. See the
[results report](tworoom-followup-2026-09-06.md), its preserved plan/execution log,
and [verified dashboard](../runs/experiment_dashboard.html). No training or
experiments remain active or queued. Completed parents and references are immutable.

TwoRoom 4074's training-only calibrated diagnostic clone reaches **48/50** primary
goals (original 14/50, released 42/50), or **44/46** initially unsolved goals
(original 10/46, released 38/46). It reaches **13/50** longer-goal CEM 10 cases
(original 0/50, released 5/50). Only six BN buffer tensors change, with no parameter
updates. This establishes a large buffer intervention effect on these cases; the
responsible layer and training-time mechanism remain unresolved.

PushT continues from 8404 to **13933 updates** in a separate directory, adding 5529
updates in 6507.80s within the 7800s cap. It preserves optimizer/RNG state and the
full139330-update schedule. Frozen control improves **17/50→30/50**, versus 45/50
released. Effective rank improves 38.71→48.42, mean probe R²0.5462→0.6005, and
h8 rollout/copy0.2906→0.2106. The final saved checkpoint SHA256 is
151b356addea1a9bc7c939fcd102986ed1e7dfca693212463308b8456b3ca4f0.
Validation/status match13933. One full-batch epoch processes1783424 windows;
the ten-epoch reproduction remains incomplete and unscheduled.

PushT calibration gives11/50 at 8404 (original 17/50) and 29/50 at 13933 (original 30/50),
despite better prediction error. Preserve both negative control comparisons;
calibration is not adopted into the training baseline. The first frozen final
PushT case still fails for both local variants, while released weights succeed.

Claude authored the layerwise calibration utility and three essential tests via
a restricted MCP task with no repository access ($0.478356 reported usage).
Broader repository-sharing tasks remain blocked by automatic approval review
pending the specific transfer approval already requested. Codex integrated and
validated the implementation, ranking, continuation and reporting. A supervisor
exit was recovered without restarting the active trainer; missing original exit
status is recorded explicitly, with final checkpoint/status/HTML verified.

**76 tests pass**, including three browser checks. Final canonical QA passes with
31 charts, 8 tables and 4 image blocks. Actual390px/1440px captures, all four new
rollout panels and four final internals panels were inspected. Mobile control bars
are visible, and forked runs retain released panels only on matched recorded
inspection populations. Every numeric record remains indexed.

## Deferred LeWM follow-up

Proposed follow-up: isolate projector versus prediction-projector BN effects on preserved TwoRoom clones using a predeclared paired protocol, then test the identified normalization correction prospectively. Keep saved-buffer results and PushT failures visible. A smaller coverage-preserving dataset must be compared with a size-matched random subset under a new configuration-disjoint holdout, with normalization fitted only on training groups. Do not interpret source-window counts as independent-data efficiency. No follow-up is launched or queued.

Preserve the PushT 13933 saved-buffer checkpoint as the continuation reference. Further training requires a separate bounded experiment decision. Full paper reproduction and passive TAU/Charades extensions remain deferred.

## Earlier: overnight implementation and evaluation complete (2026-09-06)

The authorized work with Claude via MCP, correctness/performance repairs, bounded
training on both full source datasets, control, internals and dashboard review
has completed. See [the results report](overnight-2026-09-06.md) and its preserved
plan/execution log. All completed checkpoints and references remain immutable.

PushT completed **8,404 updates** (60.31% of one epoch), reaching **17/50** frozen
goals versus 0/50 at step 375 and 45/50 for released weights. Its prediction/copy
ratio is 0.2274, full-window rank 38.71/192 and mean state-probe R² 0.5462.
TwoRoom completed **4,074 updates** (79.28% of one epoch): **14/50** primary goals
versus 42/50 released; conditional on initially unsolved goals this is **10/46
versus 38/46**. Longer goals yield **0/50 versus 5/50**. TwoRoom position probes
are strong (R² 0.9939), but prediction/copy is 3.9480 and eight-step rollout/copy
is 3.2835. Representation readability does not establish useful dynamics.

The paper specifies TwoRoom history one and 10 CEM iterations, whereas released
weights/configuration use history three and 30 iterations. Our 100/150 goal/budget
check retains released history and is not full paper reproduction. A matched
10-iteration check kept exactly the same successes (local 0/50, released 5/50)
with 2.96–2.97× less case execution time. Existing frozen evidence is unchanged;
no solver default or scientific pass threshold was adopted.

Validation RNG/resume identity, final-step evidence, TwoRoom simulator/evaluation,
source-loader reads, dataset-aware inspection and dashboard row/publication
integrity are repaired. **64 tests pass**, including two browser checks. Every
completed experiment refreshed the canonical HTML. Final canonical QA passes;
desktop and mobile captures and all three frozen rollout panels were inspected.
Claude completed one review and contributed three essential tests; an additional
review was blocked before launch by automatic approval review pending specific
data-transfer authorization.

## Earlier: requested ten-minute training and recheck complete (2026-09-06)

After inspecting the user's updated dashboard, a fresh source-data run completed
375 updates in 578.48 seconds (9 min 38 sec), using seed 3072, batch 128,
full-batch activation checkpointing, bf16, float32 validation and internals logging.
This was a separate short cosine schedule with three warmup updates, not the
139,330-update reproduction. It processed 48,000 windows (2.7% of one epoch).

Matched control: new checkpoint **0/50**, released **45/50**, replay **50/50**,
stationary **0/50**, with no initial successes. Prediction beats copying by 12.18%
and shuffled actions by only 1.38%. On the same 512 validation windows, effective
rank is 11.51/192 versus 88.72 for released weights; mean probe R² is 0.0437 versus
0.720; eight-step rollout/copy ratio is 0.684 versus 0.114. The first frozen
simulator rollout moves away from the block. Learned control remains unestablished.
See [the full short-run report](pusht-source-10min.md).

The inspector now restores and verifies random-window Subset indices, and records
standalone prediction alongside internals. Dashboard PNG accumulation exceeded
the canonical payload limit; a tested repair selects the focus run's earliest/latest
inspections and a released inspection on the same population. All numeric records
remain indexed. Final canonical desktop/mobile/source QA passes (29 charts,
6 tables), and 49 essential tests pass, including two browser checks. The user's
per-scalar curves and internals instrumentation are preserved. All checkpoints
remain unchanged by evaluation, including old pilot/reference artifacts.

## Earlier diagnosis and source-scale preparation

The dashboard browser path is repaired. Real-time CDP transport preserves the
canonical probes; two reader CSS fixes handle scrollbar width and narrow-screen
legends. Canonical desktop/mobile rendering, source interaction and exact payload
verification pass. The full essential suite passes 32 tests, including two
explicit browser checks. See `viewer/DESIGN.md` and the dashboard receipt.

The saved broader-pilot checkpoint reaches 0/20 frozen training-set goals;
recorded replay reaches 20/20 and stationary actions 0/20, with no initial
successes. The earlier held-out result remains 0/20. On eight matched rollout
cases the pilot selects unsuccessful plans in 8/8, while released weights select
successful candidates in 7/8. Raw 320 model/candidate records, exact values and
five-step rollout curves are indexed in the dashboard. All checkpoints remain
unchanged. See [the diagnosis report](pusht-control-diagnosis.md).

A discarded real-batch native comparison isolates bf16 gradient drift from
encoder batch slicing (6.124% relative L2 on the four-sequence probe). Full-batch
activation checkpointing reduces that measured difference to 1.36e-7 without
changing the objective or full-batch statistics. A single batch-128 backward
fits the local GPU (3.14 GB peak allocated), with finite gradients and zero
optimizer steps. This does not establish the cause of the pilot control failure.

The [source-scale reproduction configuration](../configs/reproduction/pusht_source_scale.yaml)
and [protocol](pusht-reproduction.md) are prepared and data-only validation has
completed: 1,783,548 train / 198,173 validation windows, full-source unbiased
normalization, ten epochs / 139,330 updates, 1,393 warmup updates, and 50 frozen
source control goals. The candidate uses full-batch activation checkpointing,
not the original pilot's encoder slicing. Ordinary episode-split configurations
remain supported. The raw preparation manifest and frozen indices are under
`runs/reproduction/pusht_source_scale_preparation/`.

## Dashboard repaired and model internals measured (2026-09-06)

The canonical reader applied only filters that target every dataset, so all
per-section selectors were silently inert and every chart mixed all runs on a
categorical axis. Charts now show one named selection fixed at build time
(`python -m viewer.dashboard --focus <training run>`), with one panel per
training scalar, a log10 validation-ratio chart, rollout error in the absolute
view, checkpoint-internals charts, an all-checkpoint spectrum, error-versus-
horizon ratios, embedded PNG panels and exact tables. Browser QA passes; the
no-JavaScript fallback shows charts beyond the first eleven as tables (builder
SVG budget). See `viewer/DESIGN.md`.

Read-only inspection of the four saved pilot checkpoints and the released
weights on the pilot's 512 held-out windows is recorded under
`runs/diagnostics/pusht_internals/` and summarized in
[the internals report](internals-report.md). The pilot latent is dimensionally
collapsed (effective rank 14 of 192 versus 66 for released weights), physical
state is weakly linearly readable from it (mean held-out probe R² −0.06, some
position targets weakly positive, versus 0.78–0.97 for released object pose), the predictor is less
sensitive to actions than to state (ratio 0.91 versus 2.6), and multi-step
prediction only modestly beats copying (0.65 of copy error at horizon 8 versus
0.10). Gradient norm is dominated by the encoder at every trained checkpoint.
These are descriptive measurements, not gates, and do not identify a cause.
Checkpoint hashes are unchanged. An opt-in `introspect: true` training key
records the scalar subset at every validation step; existing configs are unchanged.

## Earlier next-work assessment (superseded by the active follow-up above)

At the overnight handoff, no further training was running or queued.
Preserve the 8,404-update PushT, 4,074-update TwoRoom and earlier checkpoints.
The full ten-epoch reproduction remains unscheduled. The overnight prefixes use
the full learning-rate schedules, but neither reaches one complete epoch.

The next experiment should distinguish TwoRoom dynamics/evaluation calibration
from planner behavior using matched, discarded diagnostic clones and preserved
checkpoints. Position is already strongly recoverable; do not attribute failure
to missing position information or the Gaussian prior without causal evidence.
Any new comparison needs a declared case population, history, goal/budget and
CEM iteration count. Random-window validation measures source interpolation;
unseen-configuration generalization requires a separate group-held-out protocol.
Keep internals enabled. Passive TAU/Charades extensions remain deferred.

## Earlier validation
At broader-pilot completion, the modular LeWM implementation passed 13 CPU tests.
Local and upstream control evaluators agree on ten paired subset cases with
released weights. Cached random-initialized PushT training completed all 400
updates in 665.98 seconds on the same 8-episode prefix (7 train, 1 held out).
Its untouched checkpoint beats copy/shuffled-action prediction controls in
float32 and reaches 1/5 held-out control goals; replay reaches 5/5 and stationary
actions 0/5. The checkpoint SHA256 still matches its pre-crash control manifest.
BatchNorm recalibration still improves predictions on discarded diagnostic
clones; it is not adopted as a baseline change. TwoRoom completed 400 updates on
its earlier 32-episode prefix; its full source is now verified and extracted
(10,000 episodes, 920,809 frames). Full PushT source is verified and extracted
(18,685 episodes, 2,336,736 frames). The released checkpoint reaches 45/50 goals
(90%) with full-source normalization through the pinned upstream evaluator.
The wrapper records unseeded reset arguments explicitly; it does not alter
upstream randomness. All recovery/evaluation processes have completed.
Do not repeat the completed training run.
See docs/reference-validation.md and runs/diagnostics/reference_full_source_status.json
for current evidence and recovery status. Broader learned control and full
training reproduction remain unestablished. A full reproduction training run is not scheduled. Research extensions remain closed.

## Completed approved pilot
The user-approved broader PushT pilot completed 1,000 updates in 1,506.89 seconds
on 128 train / 32 held-out initial-configuration groups, batch 128, seed 3072.
The full source has 185 distinct initial configurations; one trajectory per
selected group prevents initial-configuration variants from crossing the split.
The final untouched float32 checkpoint has prediction MSE 0.187671 versus copy
0.213567 and shuffled actions 0.203553, on 512 fixed held-out windows. All three
trained checkpoints (250/500/1000) reach 0/20 on identical held-out control goals;
released weights reach 17/20, replay 19/20, stationary 0/20, with no initial successes.
Final precision and discarded BatchNorm-clone differences are small (about 3–4%);
no baseline change was adopted. All checkpoint hashes remain unchanged.
See docs/pusht-broader-pilot.md and the run's pilot_summary.json. All pilot
training/evaluation processes completed. Do not repeat the completed run. Targeted rollout/control diagnosis is completed above;
research extensions remain deferred. The bounded pilot is complete, while the
learned-control baseline gate remains unmet.
