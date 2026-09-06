# Current project state

This file owns the implementation objective, active experiments, evidence and next
steps. Standing development and experiment rules live in
[experiment-workflow.md](experiment-workflow.md). Changing a model, dataset or
research direction does not replace that workflow.

## Objective

Build a fresh modular implementation of the published LeWM baseline. Demonstrate
learning and control on PushT and support multiple explicit dataset protocols.
Research extensions remain deferred until the reference baseline works.
The earlier reset removed the previous implementation and results; the retained
ideas and downloaded source datasets carried forward. The reusable development
harness is now recovered separately from that discarded implementation.

Reuse the pinned authors' baseline components. Document differences from the
reference recipe and distinguish smoke checks, subset training, reproduction
training and benchmark evaluation. Baseline-specific tests cover episode
alignment, causal action timing, reference computations and gradients,
normalization, rollout and checkpoint integrity.

## Latest: overnight implementation and evaluation complete (2026-09-06)

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

## Next work

The bounded overnight work is complete; no further training is running or queued.
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
