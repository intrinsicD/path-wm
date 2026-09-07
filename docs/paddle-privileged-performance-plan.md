# Exact privileged controller throughput adjustment

The existing exhaustive privileged reference spends approximately 71% of its CPU
decision time rendering unused candidate frames. Twenty fixed validation states
profile at a median 21.086 ms per decision. An isolated prototype using unchanged
physics without rendering and exact scalar prefix scoring measures 2.823 ms.
All 4,860 leaf score tuples and 7,260 child states/events agree exactly. See
`runs/paddle/collaboration/privileged_plan_profile/analysis.md` and its raw records.

This is an implementation throughput adjustment, not a scientific intervention.
Preserve all 243 five-action sequences, stay-left-right enumeration, scalar
arithmetic order, terminal-prefix scoring, time-limit behavior, caller state,
RNG, and public environment observations. The learned planner, simulator physics,
model inputs, controller protocol, datasets and evaluation populations do not
change. Reject the NumPy batch-score prototype: eight candidate scores differ by
one or two ULP despite matching winners in the diagnostic sample.

First add CPU tests comparing state-only and ordinary stepping, full candidate
scores against independent original-style exhaustive simulation, and absence of
rendering/RNG/source mutation during privileged planning. Include random states,
wall/ceiling contacts, catch/miss, terminal states, ties and time limits. Record
the informative missing-interface failures and commit the plan/test slice before
implementation.

Then extract the current event-integrator body into
`PaddleEnv.advance(action) -> (terminated, truncated, info)`. Public `step(action)`
calls `advance` followed by the identical renderer and retains its four-tuple.
The privileged traversal uses `advance` and accumulates the exact scalar score
along shared prefixes. An internal `_privileged_candidate_scores(env)` exposes
all 243 `(score, sequence)` tuples for direct parity audit; `privileged_plan`
selects their minimum. Record the implementation identifier in evaluation hardware
metadata so resume identities and latency evidence distinguish the new path.

After implementation, run the focused environment/planner/evaluation tests and
reprofile the same 20 states against preserved original-style traversal. Confirm
exact scores/states/flags/events and report measured speed, with no timing
threshold in unit tests. No production held-out evaluation has run yet. Preserve
old smoke/reference latency evidence and identify its original implementation.

## Implementation evidence

The 31 essential checks first failed on the missing state-only/candidate-score
interfaces and the existing candidate render calls. The plan/tests were committed
as `23429ec` before production edits. After the refactor all 31 pass, including
exact equality of every candidate score on all 13 declared fixtures.

The separate production benchmark reuses preserved copies of the original
integrator and privileged traversal, and the same 20 validation states. All
4,860 candidate score tuples and 7,260 child states, rendered frames, flags,
counters and events agree exactly. All selected outputs and caller states agree.
Across 60 calls per version, original median/p95 latency is 21.502/22.725 ms;
production is 2.860/3.836 ms, a 7.52× median improvement. Raw timing samples,
source hashes, outputs and parity counts are in
`runs/paddle/collaboration/privileged_plan_profile/production_benchmark.json`.

Evaluation hardware and resume identity now record
`privileged_planner_implementation: exhaustive-5-state-advance-scalar-prefix-v2`.
Existing artifacts retain their recorded timings and identities. The coordinator
is performing full immutable source-episode replay before held-out control.
