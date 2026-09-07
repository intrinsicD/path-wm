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
