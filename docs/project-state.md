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

## Harness restored; diagnostics remain next

The standing workflow and offline dashboard were recovered from `eca742a` and
adapted to current ledgers. `CLAUDE.md` now links the workflow and this state file.
The harness adds nine integrity checks; all 22 CPU tests pass. No new training or
evaluation was run during this repair.

The current HTML includes 30 result records (6 training, 20 control and 4
standalone prediction records). Canonical artifact, desktop/mobile browser,
source interaction and exact embedded-payload checks now pass. The local browser
transport uses an installed Chromium with real time and an explicit viewport;
it preserves the canonical probes and their failure results. Two narrow runtime
CSS corrections fix scrollbar-gutter header overflow and long mobile legends.
All 22 CPU checks and two explicit browser integration checks passed during the
repair. See `runs/experiment_dashboard.receipt.json` and `viewer/DESIGN.md`.

## Approved work pending after this repair

The user accepted all three recommendations before interrupting that turn for
this harness repair. That turn performed reads only; it launched no runs.

1. Evaluate the saved broader-pilot checkpoint on 20 training-set goals to
   distinguish fitting failure from held-out generalization failure.
2. Inspect multi-step predictions and action ranking against simulator outcomes,
   using saved checkpoints and explicit matched cases and planning budgets.
3. Prepare a larger faithful reproduction configuration and protocol, including
   an explicit schedule and matching evaluation. Preparation is authorized;
   a long reproduction training run has not been scheduled or launched.

These diagnostics are already authorized; do not ask for the same permission
again. Preserve the completed pilot and checkpoint hashes. The learned-control
baseline gate is still unmet, and research extensions remain deferred.

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
training/evaluation processes completed. Do not repeat the completed run. Targeted rollout/control diagnosis is approved above;
research extensions remain deferred. The bounded pilot is complete, while the
learned-control baseline gate remains unmet.
