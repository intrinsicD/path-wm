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

## Approved diagnosis complete; source-scale candidate prepared

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

## Next work

Schedule the prepared source-scale run separately before launching it. The
pilot-rate extrapolation is about 58.3 hours and excludes full-source I/O and
changed recomputation cost. No long training has been scheduled or launched.
The protocol documents scheduler/release-history ambiguity and distinguishes
source interpolation from unseen-configuration generalization. Preserve the
completed pilot and diagnostic artifacts. Learned control is still unestablished;
research extensions remain deferred. The three previously approved diagnostics
and preparation items are complete; do not repeat them.

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
