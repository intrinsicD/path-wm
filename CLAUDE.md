# Working instructions

## Current objective
Build a fresh modular implementation of the published LeWM baseline. Demonstrate
learning and control on PushT and support multiple explicit dataset protocols.
Research extensions remain deferred until the reference baseline works.
The user explicitly requested removal of the previous code and results; only
ideas in docs/ideas.md and downloaded source datasets carry forward.

## Workflow
Read this file first. State the active implementation/validation task. Use small
end-to-end steps: define interfaces and configuration, write essential behavioural
tests, implement, then measure learning and planning. Commit completed steps.
Do not build speculative frameworks, registries, or deferred research modules.
Use plain PyTorch modules, functions and explicit configuration. Components must
be replaceable without changing the training loop or dataset adapters.

Reuse the pinned authors' baseline components before inventing replacements.
Copied/adapted code must carry source URL, commit, file and upstream license.
Document deviations from the reference recipe and distinguish smoke checks,
subset training, reproduction training and benchmark evaluation. Never present
falling loss alone as evidence of a useful world model. Never hide failed gates.

Keep source data under data/ and all new checkpoints/logs under runs/. Never mix
source data, training episodes and held-out evaluation episodes. Record data and
code revisions, seed, configuration, normalization and sample/step counts.
No external experiment tracking or uploads unless requested.

Tests cover silent scientific failures: episode alignment, causal action timing,
reference computation/gradients, normalization, rollout and checkpoint integrity.
Keep ordinary tests fast and CPU-based; substantive GPU runs are validation, not
unit tests. Do not remove a failing test to conceal an implementation error.

## Status
The modular LeWM implementation and 13 CPU tests pass after the broader pilot.
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
training reproduction remain unestablished. Do not start or resume long
training until the user asks. Research extensions remain closed.

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
training/evaluation processes completed. Do not repeat the run or launch more
training without a new request. Recommend targeted rollout/control diagnosis;
research extensions remain deferred. The bounded pilot is complete, while the
learned-control baseline gate remains unmet.
