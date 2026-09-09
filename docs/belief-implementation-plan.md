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

## Completed implementation and review

The current API and numerical/training choices are documented in
[belief-model.md](belief-model.md). The CLI selects the categorical model by default;
the Python constructor's Gaussian default and existing reference tests are preserved.
The recipe explicitly constructs the new correction, dynamics and memory modules.
The categorical schema is `pathwm-belief-v1`; it never silently loads Gaussian state.

Two actual Claude CLI exchanges completed (`implementation` and
`implementation-reconcile`). Independent notes preceded the reply. Adopted failure
cases cover stable sampling on event retries, frozen pre-event memory, masked-value
invariance, source-only forward independence, finite replay graphs and planning RNG.
Claude withdrew mandatory parameter-disjointness, hard-gate replay, mask-token and
particular RNG-mechanism claims. Our planner has a fixed rectangular horizon, so
all candidates consume identically shaped noise; variable-length branching remains
outside this interface. A scoped finally restores RNG on errors. Sealing accepts
only an open transaction and has no hidden side effects. Weighted composite losses
and feature-distribution distillation are not claimed to be an ELBO or calibrated
world probabilities. No private implementation or measurements entered the review.

Additional red/green regression checks caught missing audio/text diagnostics needed
by the existing extra-update gate, and future memory hidden behind equal timestamps
or old evidence. Review also corrected evidence timestamps at write time, missing
batch members at consolidation and full-logit access in memory readers/compressors.
Source and inferred clocks are now explicit; older summaries retain bounded support.

**Validation:** 69 CPU tests pass; Ruff and diff whitespace checks pass. Exact
pause/resume includes the teacher, replay scores, optional proposal gate, optimizer,
RNG and metric rows. A final tiny real batch exercises forward/backward through all
memory scales. Three preserved development runs consumed 8 synthetic + 8 instruction
+ 2 real updates, with no GPU use or extra proposals. Their short histories use
recent=2, block=2 and compressed-block capacity=1 to force consolidation. These are
workflow checks; no scientific pass threshold or capability claim was introduced.

The eight-update synthetic model has validation image MSE 0.237670 versus copying
0.007451. The two-update real model has 0.236499 versus 0.000050895. Predictions remain
poor. The four-example instruction evaluation has 75% operation error and is not a
meaningful general instruction benchmark. Earlier negative results remain intact.

All three run reports are structurally verified and visually checked in the in-app
browser at its 1280×720 viewport; images load and no horizontal overflow was found.
A local server required sandbox approval and was restricted to localhost/the new run
directory. The modified renderer correctly permits an unapplied world intervention
and labels categorical observation inputs. Two completed reports were re-rendered
from preserved metrics after a caption correction; their training metrics/checkpoints
were not rewritten. New guard-only source changes are covered by the final suite and
real batch; each development run retains its own exact source snapshot.

Receipts and reports: `runs/belief_v1/verification.json`, the three per-run
`report.qa.json`/`report.browser.qa.json` files, and `pusht-final-check.json`.
No individual reset, raw archival store, indexed large-memory retrieval, variable
length planner or claim of broadly useful trained memory was added. Those are
separate future choices rather than hidden default behavior.
