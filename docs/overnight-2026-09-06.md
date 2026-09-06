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

## Outcomes

Training/evaluation pending. This section records measured results, failures and deviations at closeout.
