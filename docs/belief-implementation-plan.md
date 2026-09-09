# Categorical belief and bounded session memory

Status: implementation authorized by Alex on 9 September 2026. This supersedes
the architecture-only boundary for this slice; existing results remain unchanged.

Implement a selectable `belief` model in the existing multimodal recipe, preserving
the Gaussian reference. Reuse modality/task/output modules, optimizer, run/resume
and report pipeline. The new model uses domain-neutral recurrent tokens, categorical
prior/posterior heads and a separate workspace. Ordered event transactions advance
once, accumulate deduplicated partial packets from an immutable prior, then commit.
Live and hypothetical transitions share weights and physical conditioning.

Memory is caller-owned and bounded: exact recent belief envelopes and independent
source features, staging, compressed blocks, protected detail and consolidated
tokens. Source and inferred views remain separate through compression. Consumer
reads have independent gates with an explicit no-memory choice. Inference writes
detach; training replay may keep bounded compressor graphs for grounded losses.
New sessions clear every state store; individual resets remain withdrawn.

Initial library defaults: width 32, 16 context tokens, 8 categorical groups of 8
codes, 8 workspace tokens; recent 32, staging groups 8, compressed 16 blocks of
8 tokens per view, protected 8 records, consolidated 8 tokens per view. Evidence
records use 8 source-only learned tokens. Count actual tensors and metadata rather
than reuse the earlier single-view byte estimate. Numerical choices are development
defaults, not measured optima.

Learning uses observable Gaussian image/audio and categorical byte likelihoods,
separate dynamics/representation KL routes, masked-before-encoding partial views
and detached full-view supervision from the same event prior. Prior-only future
rollouts supply predictive targets. Straight-through categorical draws train the
encoder/dynamics; Monte Carlo log-mean likelihood is an explicitly biased marginal
NLL estimate, not an exact likelihood. Memory replay adds delayed observable recall
and fixed-reader compression distillation. Marking uses paired retained/omitted
detail utility with an explicit storage cost and user priority.

Essential red tests precede implementation: advance/correct/commit identity,
packet permutation and masked-value invariance, no-evidence intervals, source-only
independence, hierarchy eviction/bounds/gradient routes, state roundtrip, fresh
session and shared initial samples across planning candidates. Add recipe gradient
and exact pause/resume checks once the usable path exists.

Budget: CPU tests; one tiny real PushT batch; at most 16 synthetic/instruction
development updates plus 2 real updates and a short exact resume comparison. Small
memory capacities in development force eviction/consolidation within short windows.
No quality threshold is declared: these runs establish workflow and numerical
contracts only, not learned world understanding. Each completed run must retain raw
metrics/checkpoint and a structurally verified standalone report. Reuse the unchanged
renderer and disclose its existing visual-QA limitation.

Claude receives abstract implementation questions only, through the established
isolated CLI. Independently derive failure cases before reading its reply. Record
adopted corrections and verification here when complete.

Initial check receipt: existing 51 tests pass; `tests/test_belief.py` fails during
collection on the absent `pathwm.models.belief_state` module (expected red state).
