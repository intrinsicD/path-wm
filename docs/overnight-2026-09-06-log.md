# Overnight implementation and evaluation, 6 September 2026

The user authorized collaboration with Claude via MCP, implementation repairs,
training on the local datasets and review of `runs/experiment_dashboard.html`.
The working deadline is 08:00 Europe/Berlin today. This authorizes the bounded
work below; it does not launch or promise completion of a full reproduction.

## Plan and interfaces

First repair operational correctness without changing the baseline objective:
give validation its own random generator, allow documented operational resume
overrides while freezing scientific configuration, and validate the final saved
checkpoint when a time limit stops training. Record loader wait separately from
compute. Preserve the existing fingerprint contract for legacy checkpoints.
Tests must demonstrate identical CPU optimization with diagnostics enabled or
disabled, exact stopped/resumed weights with dropout, and rejected scientific
changes. Correct the misleading held-out-episode label for random-window runs.

Then implement TwoRoom closed-loop evaluation using the pinned SWM simulator,
shared CEM/model interfaces, frozen source goals and replay/stationary controls.
Verify source state/action/render alignment before interpreting learned control.
Use explicit, versioned dataset/evaluation settings; old configurations and runs
remain immutable. The paper uses history one for TwoRoom (Appendix D), while the
generic existing dataset configuration uses three. A new configuration will make
that choice visible. The paper and released evaluation config also use different
goal budgets; report the chosen protocol rather than mixing their scores.

## Compute and scientific budget

One RTX 3050 with 8 GiB; serialized GPU work. Reserve approximately three hours
for a fresh PushT prefix of the prepared 139,330-update schedule, approximately
one hour for a fresh full-source TwoRoom run, and the remaining time for essential
tests, environment checks, matched control, internals and browser QA. Refine step
estimates using measured throughput before launch. Stop training by 07:00 and
complete evaluation/reporting by 08:00. No external tracking or publishing.

The hypothesis is that substantially longer faithful training improves action
use and control compared with the preserved 375-update development run. The
different learning-rate schedules prevent attributing improvement to update count
alone. Compare prediction to copy/shuffled-action controls within each encoder,
and use fixed simulator cases as the control outcome. Effective rank and probes
are diagnostics, not success gates. No new numerical success threshold is set;
the baseline gate remains unestablished until measured useful control is present.

PushT uses seed 3072, the existing random-window source split, batch 128,
learning rate 5e-5, SIGReg weight .09, bf16 and full-batch encoder computation.
The schedule stays 139,330 updates with 1,393 warmup updates; the wall-time ceiling
only stops its execution. Preserve initial/intermediate/final checkpoint identity.
Use the existing 50 frozen goals and released/replay/stationary results after
verifying their protocol hashes. Inspect the final checkpoint on matched windows.
TwoRoom settings and case receipts will be recorded before its training launch.

## Claude collaboration and sources

The registered native MCP server exposes tools but no configured Agent types.
Claude's first actual review ran through its MCP Bash tool as a bounded `claude
-p` process. It completed successfully, using 20 tool calls and reporting
$2.48802025 in API-equivalent cost. Raw prompt, result, usage and session identity
are under `runs/overnight_2026-09-06/claude/`. Follow-up reviews will be bounded;
usage is recorded rather than assuming access to either account's remaining quota.

Claude found no model/objective mismatch. Its resume and CPU random-stream
findings will receive behavioral tests. Its suggestion to change SIGReg precision
is treated as an unproven numerical hypothesis: the pinned mixed-precision recipe
is authoritative, and a discrepancy alone does not justify changing it.

Primary sources: [LeWorldModel paper, Appendices D and F](https://arxiv.org/html/2603.19312v1),
[pinned LeWM implementation](https://github.com/lucas-maes/le-wm/tree/8edfeb336732b5f3ce7b8b210d0ba370a09e2cac),
and the [authors' LeJEPA implementation](https://github.com/galilai-group/lejepa).
The TwoRoom simulator will use the already pinned SWM revision
`6f1e499e9cc0c898d326112f485c1062c3d20f24` with its license and source receipt.

## Implementation evidence

Three new behavioral checks first failed as intended: instrumentation changed
CPU weights, extending the time budget rejected resume, and a timed stop saved
step one with only step-zero validation. They now pass. The new fingerprint
version freezes all configuration except a short declared operational allowlist;
legacy checkpoints keep their original fingerprint and random-stream behavior.
Resume receipts record overrides without rewriting the original manifest.

Native batch-128 bf16 exhausted GPU memory before its first update (6.49 GB peak
allocated, desktop GPU usage also present). Full-batch activation checkpointing
therefore remains selected. This is a capacity measurement on a discarded clone,
not a failed learning run. Both raw measurement and failure text are preserved.
The checkpointed probe completed seven discarded updates with finite gradients:
five timed updates had median 1.284 s and peak allocation 3.282 GB, including
optimizer state but excluding HDF5 loading. Allowing for I/O and validation, the
three-hour PushT prefix should reach approximately 7,500–8,300 updates. The
repaired two-update GPU slice and its canonical desktop/mobile HTML QA passed.
All 50 CPU checks passed; two explicitly opt-in browser tests were not enabled
in that command, while the actual dashboard browser verification did run.

## TwoRoom thin slice

Claude independently wrote three essential simulator/controller tests (12 tool
calls; reported cost $1.67608375). They fail at the missing import before
implementation. Its assumptions about default geometry and action scale were
then checked against eight fixed source episodes: reset renders and all first
25 recorded transitions match exactly. A first ad hoc read lacked HDF5 plugin
registration; the corrected check imports the same compression plugin as the
normal dataset loader. No source file was changed.

History-one diagnostics expose a separate divide-by-zero: normalized attention
entropy divides by log(1). The single-key case will report zero entropy, with a
finite, read-only scalar-summary regression before training. TwoRoom control uses
the same one-current-observation CEM planning interface as PushT, restoring only
agent and goal positions in the verified default-geometry simulator.

The downloaded released TwoRoom checkpoint is pinned to model revision
`77adaae0bc31deab21c93740d1f8bb947cd0bdec`, with weights SHA256
`566f223624ea4bfb39dbfe6ae731198dd6ea73b7b8919fed6b1ecafca810f7dd`.
Its actual positional embedding has three frames, matching its mirror config
and differing from the paper's history-one statement. Before any TwoRoom training,
the plan is amended to retain the existing history-three dataset configuration
for the released-baseline comparison. The entropy repair remains a supported
history-one correctness fix, not an overnight architecture change.

TwoRoom will use full-source random windows, seed 3072, batch 128 and the same
optimizer/objective/precision as PushT, with a 4,800-second ceiling on its own
full ten-epoch schedule. Primary control uses the released config's 25-step goal
and 50-action budget; a separate secondary protocol uses the paper's 100/150
settings. Each freezes 50 source cases with sampling seed 42 and reset/CEM seeds
1234–1283 before evaluating either checkpoint. Both retain initially successful
cases and report their count. Replay and stationary controls are measured on
both sets. These are source-population compatibility checks, not unseen-episode
generalization or a claim of reproducing the paper's multi-seed benchmark.

The primary 25/50 preparation passed exact source-frame and transition checks on
all 50 cases (1,250 transitions): replay 50/50, stationary 4/50, initial successes
4/50. The separate 100/150 preparation also passed exact checks (5,000
transitions): replay 50/50, stationary 0/50, no initial successes. Both canonical
HTML refreshes passed. A one-case CPU integration smoke with the actual released
weights, only six candidates and two CEM iterations, completed with unchanged
checkpoint bytes and passing HTML QA; its score is not a control-quality result.
The full CPU suite now passes 55 checks, with two opt-in browser tests skipped in
that command. Actual per-artifact browser QA did run and pass.

## Measured loader optimization

A fixed small probe on both real sources found HDF5 chunks of
`100 × 224 × 224 × 3`. Four strided frames took median 57.48 ms on PushT and
55.47 ms on TwoRoom; four individual frame reads from one open dataset handle
returned identical bytes in 9.06 and 8.93 ms. Three passes alternated read order.
This is a small warm-cache microbenchmark under shared-machine CPU load, not a
claim of sixfold training speedup. The active PushT run's first 400 updates
averaged about 1.94 s/update, slower than its compute-only preflight.

Add a focused pixel-window reader that uses scalar frame selections for chunked
HDF5 striding and preserves native slicing for contiguous/cached arrays. Essential
checks cover chunk boundaries, episode edges, channels-first input and exact
cached/streamed equality. Then benchmark complete dataset items on fixed real
windows against the original reader. Accept only exact outputs and a measured
read improvement; do not change training inputs, batch size or objective.

The complete-item paired benchmark passed exact pixels, action blocks (including
unused terminal NaNs), episode IDs and local starts on 48 fixed random windows
per source. Median original/optimized time was 55.94/3.58 ms for PushT and
58.02/4.83 ms for TwoRoom; total paired read times improved about tenfold. The
chunk-boundary/cache checks and CPU optimization/resume regressions passed, and
the benchmark's canonical dashboard refresh passed. These are loader timings,
not end-to-end training speedups.

After verification, recover the active PushT run from its next immutable saved
checkpoint to adopt the loader change. Record the stopped process, retained
checkpoint hash, any discarded unsaved work, new code revision and resume receipt.
The original manifest and numbered snapshots stay unchanged. This is an explicit
operational amendment to an in-progress time-bounded run, not a new learning
comparison or a silent resume of a completed reference. Keep the cumulative
training ceiling and the morning evaluation reserve.

## Dashboard control integrity

The primary TwoRoom set contains four already-satisfied goals, so raw success
alone can overstate learned progress. Add exact counts and a separate rate for
cases not initially successful; retain raw success alongside it and leave the
conditional rate unavailable when initial-state evidence is missing. Action
baselines must inherit the same dataset, goal offset and budget context as their
case manifest. Include source identity and goal offset in case navigation keys
so identical episode/row numbers from different tasks cannot merge. Essential
fixtures will first expose the lost context and missing denominator.

Before the loader recovery, add a graceful operational stop marker (`run/STOP`)
for future interruptions: finish the current update, validate and save a numbered
checkpoint, record `stop_requested`, and require removing the marker to resume.
Also aggregate compute and loader-wait timings over every logged interval; a
single sampled update every 100 steps hid occasional loader stalls. Retain the
old single-step field and label new interval means and their update denominator.
CPU tests will verify the checkpoint boundary and timing-count conservation.
The first recovery still uses the old process's already-saved checkpoint because
that process cannot acquire new control code without a restart.

PushT recovery retained checkpoint 1000 at 1,936.336 seconds, SHA256
`f441e5a2cf71eb4f9474a695966423adc3bc0b77a7e4a9ab5773875e86ddc105`.
Only the verified original training PID was terminated. The checkpoint was about
170.52 wall seconds old and the last logged update was still 1000, so an unknown
number of fewer than 100 unsaved updates was discarded. The cumulative resumed
ceiling is reduced from 10,800 to 10,620 seconds, conservatively charging 180
seconds for that work. The 139,330-update LR schedule and optimizer/data state
are unchanged. The raw interruption event and `loader_restart.json` preserve
this deviation. Resume records expose both the new code revision and overrides.

The resumed process restored the same fingerprint with a clean code revision
and only the declared time-ceiling override. Updates 1001–1100 averaged 1.344 s
of measured update processing plus 0.0155 s of loader wait, including worker
startup, compared with about 1.9 s/update before recovery. This is an observed
end-to-end improvement under shared-machine load, not the microbenchmark's
12–16× loader speedup.

## Dataset-aware checkpoint inspection

Extend the existing inspector rather than create a separate analysis pipeline.
TwoRoom probes use source agent x/y (`proprio`); PushT retains its eight existing
pose/velocity targets. Read only requested label rows while preserving duplicate
and out-of-order indices. Released TwoRoom inspection accepts an explicit frozen
reference case manifest for its checkpoint identity, model configuration and
normalization. Render labels, next-frame retrieval and preprocessing follow the
actual history and image size. Essential tests cover label identity/order and
history-one next-frame retrieval without querying future frames for prediction.

The bounded CPU inspection completed with unchanged checkpoint bytes and all
panels, but its wrapper initially failed canonical publication with a desktop
`reader_timeout` at the default five-second readiness budget. Raw output and the
last verified HTML survived. The installed canonical builder explicitly supports
`readyTimeoutMs`, `actionTimeoutMs` and `timeoutMs`; use bounded 10/5/25-second
budgets under this shared CPU load while keeping every verification check.

Inspection also revealed that a failed dashboard build replaced the data
companion before HTML verification. Extend the existing failure regression to
require preservation of the old HTML, artifact and receipt together, then stage
the new artifact until canonical publication succeeds. This repairs consistency
on a measured failure path; no verification result is relaxed or fabricated.

The publication repair passed canonical desktop/mobile verification with all checks retained.
The old HTML, artifact and receipt now survive a failed build together. The
inspector regression suite passed 15 checks and the harness suite passed 16.

## Dashboard growth

The current ledger already has 1,536 spectrum rows; the four planned matched
inspections would exceed the reader’s 2,000-row dataset limit. Partition oversized
native datasets without losing rows, keep complete chart series together, and
show explicitly numbered chart/table parts. Preserve the canonical payload limit
and browser checks. A growth regression must prove row conservation, bounded
parts and unsplit spectrum series before this implementation.

## Qualitative replay

Generalize the existing saved-action renderer to both supported simulators and
manifest goal offsets/budgets. Use actual checkpoint labels, freeze case zero
before outcomes, and require matching source/cases/seeds/planning protocols.
A small CPU fixture must reproduce the TwoRoom result and reject changed final
state evidence and mismatched goal protocols. Preserve all older panels.

The TwoRoom replay fixture now passes and rejects both altered final-state
evidence and changed goal protocols. Re-rendering the preserved first PushT case
reproduced all saved outcomes exactly, with its actual step-375 label and passing
canonical HTML. Released checkpoint hashes and the frozen PushT case-manifest
hash were rechecked before scheduling the matched overnight evaluation.

## Execution queue and verification

All 64 tests pass with `PATH_WM_BROWSER_TESTS=1`, including the two browser
checks (`runs/overnight_2026-09-06/full_tests.log`). A one-shot serial driver
records exact commands and code revisions, waits for the identified PushT
wrapper, checks matching final checkpoint/validation steps and numbered snapshot
hashes, then runs TwoRoom, matched control, saved-action panels and inspections.
Every experiment command uses `run.py` and the queue stops on a failed command
or dashboard. The first queue launch stopped before starting work because this
Python build lacks `os.pidfd_open`; the corrected wait checks Linux process start
ticks as well as its PID. Both attempts remain in the raw execution ledger.

Measured prior PushT control took 265.72 seconds for 50 failed cases and 167.52
seconds for released weights. Before the first run finished, reorder its new
control check ahead of TwoRoom training so those outcomes can be assessed during
TwoRoom compute. Cases and all compute/solver budgets stay fixed. The idle
controller is restarted; active training is untouched. Its handoff also checks
that the dashboard inventory contains the final run status and step and that its
passing receipt postdates that terminal status.

## Outcomes

### PushT training and control complete

The fresh prefix stopped at **8,404 updates**, close to the 8,400 estimate after
the loader improvement. It processed 1,075,712 windows, **60.31% of one epoch**;
the learning-rate schedule remains 139,330 updates. Recorded terminal elapsed
time is 10,625.407 seconds, including final validation, plus the 180 seconds
explicitly charged for discarded work during recovery. Final checkpoint and
validation both identify step 8404. The immutable final checkpoint SHA256 is
`bb611899c74047552d2f93bb974004ed5fa49f401809e0e1590a832a1a186af9`.

On the fixed 512 validation windows, prediction MSE is 0.0400887, copy MSE
0.176300, shuffled-action MSE 0.219683 and short-rollout MSE 0.0819370.
Prediction/copy is **0.227389**, prediction/shuffled **0.182484**, and short
rollout/copy **0.464759**. These are within-encoder comparisons. Validation
fluctuated at intermediate steps; final control uses the predeclared terminal
checkpoint rather than selecting the best intermediate prediction metric.

All 50 frozen source goals were evaluated with unchanged solver settings:

| Checkpoint or action baseline | Successes | Initially satisfied |
| --- | ---: | ---: |
| Overnight, step 8404 | **17/50** | 0 |
| Preserved ten-minute run, step 375 | 0/50 | 0 |
| Released weights | 45/50 | 0 |
| Recorded replay | 50/50 | 0 |
| Stationary actions | 0/50 | 0 |

There are 16 cases both models solve, one only the local model solves, 29 only
released weights solve, and four neither solves. Local control took 215.20
seconds of measured case execution (mean 41.86 actions). Checkpoint bytes are
unchanged. Canonical HTML passed after both training and control. These results
show a substantial improvement over the short run and a large remaining gap to
the released model. The different schedules and operational changes prevent
attributing the gain to update count alone; no formal pass threshold was set.

### Remaining evaluation

TwoRoom started at 05:32:53 Europe/Berlin with clean code revision
`7d7a49e94e8c2d6055bce751dfffa5aacea4d1b2`, 657,728 training / 73,081 validation
windows, and its fixed 51,380-update schedule under a 4,800-second execution
ceiling. Its training/control and both datasets’ full matched inspections and
qualitative panels remain in progress.


## Late protocol clarification and bounded solver check

A final source check confirmed that Appendix D specifies **10 CEM iterations for
TwoRoom**, whereas the [pinned released CEM configuration](https://github.com/lucas-maes/le-wm/blob/8edfeb336732b5f3ce7b8b210d0ba370a09e2cac/config/eval/solver/cem.yaml)
uses 30. The frozen overnight comparisons use 30. Therefore the 100/150 result
is a **longer-goal test with the released settings**, not a full paper-protocol
reproduction; history also remains three rather than the paper’s one. Existing
`tworoom_paper_*` paths and all frozen data are retained. This clarification is
recorded in `protocol_clarification.json`. The planning horizon, action block and
replanning block of five do agree with [Appendix D](https://arxiv.org/html/2603.19312v1).

Before further measurements, prepare a paired exploratory check changing only
CEM iterations from 30 to 10 on the same 50 longer-goal cases and both unchanged
checkpoints. Keep samples 300, elites 30, seeds, normalization, horizon/action
blocks, goal offset 100 and budget 150. The new case file records its parent
SHA256 and asserts all other experimental fields match. Compare successful cases,
actions used and measured runtime; no new passing threshold or automatic recipe
adoption. This tests solver sensitivity, not the history discrepancy or a full
training reproduction. Launch only if the originally queued inspections finish
by 07:32, reserving approximately ten minutes for this pair and the remaining
morning time for final QA/reporting. An unrun check remains explicitly unrun.
