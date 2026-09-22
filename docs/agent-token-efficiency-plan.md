# Reduce coding-agent context and token overhead

Status: complete, 22 September 2026; all four slices implemented and verified. Created at
Alex's request after reviewing IntrinsicEngine PROC-034 (through `e156d3a15`).
This task concerns Codex/Claude workflow costs. The model's
[token-budget plan](token-budget-plan.md) and
[encoder plan](encoder-token-budget-plan.md) retain their separate scope.

## Diagnosis and baseline

The leading candidate is the mandatory full read of accumulated project history.
[AGENTS.md](../AGENTS.md) redirects to [CLAUDE.md](../CLAUDE.md), which requires
[workflow](experiment-workflow.md), [project state](project-state.md), and the
linked active plan. The workflow already calls project state a short page, but
that page has one top-level heading, 2,057 lines and 244 links, combining current
work with many dated decisions and results.

Working-tree snapshot on 22 September, HEAD `60d0eaa9f`:

| Startup source | Bytes |
| --- | ---: |
| `AGENTS.md` | 14 |
| `CLAUDE.md` | 2,366 |
| `docs/experiment-workflow.md` | 8,988 |
| `docs/project-state.md` | 150,043 |
| First linked active plan, `docs/encoder-token-budget-plan.md` | 11,548 |

The first four sources total 161,411 bytes; including that plan totals 172,959.
Project state is 93% of the four-source total. This excludes other linked goals,
skill metadata, tool schemas and conversation history. Source bytes are not token
or credit measurements. Project state grew during this inspection; refresh the
baseline before implementation. Its measured SHA-256 was
`3dd4d57edbe249d29f0f9c5b308cf3d840bc5da6f7b1a575c3fcc168485d2897`.

No repository `.codex/` or `.claude/` configuration was present. Codex's effective
`config/read` for this directory returned `gpt-6-astra`, reasoning `xhigh`, requested
service tier `default`, and `tool_output_token_limit: null`. Null means no explicit
limit reported there; it does not establish unlimited output or actual tool caps.
The inspected repository instructions have no output/wait procedure or explicit
research-manager trigger. The user-scope research-manager relevance gate and an
opt-in Astra/medium profile already exist from PROC-034; do not reinstall or fork
them merely because this is another repository.

## Constraints

- Preserve scientific decisions, negative results, evidence links, provenance,
  run/report status, restart guarantees, and the integrated research goal.
- Preserve the small-library/recipe boundary and meaningful verification. Keep
  raw run artifacts and standalone reports complete; this is not model token
  pruning, fewer training checks, or loss of retained evidence.
- Preserve PATH-WM's public-only boundary for external Claude review. Do not
  transplant IntrinsicEngine's private-source delegation authorization.
- Coordinate with current work before editing shared workflow/state. This audit
  observed concurrent edits to project state, encoder results and the architecture
  atlas. Do not overwrite those edits or change the current research priority.
- Reuse existing documents and Codex settings. No new task framework, mirror
  generator, trainer, monitoring service or mandatory research ledger.

## Implementation slices and acceptance

1. **Separate current state from history — first priority.** Keep
   `docs/project-state.md` as a short index of current goals, active plans, blockers
   and next actions; target at most 8 KiB. Preserve historical records in their
   existing plans/evidence or a linked `docs/project-history.md` when no owner
   exists. Record the mapping and repair relative links/anchors after moves.
   Adjust CLAUDE's startup route to read the plan relevant to the user's task,
   with named workflow sections selected by scope. Complete instructions already
   supplied in context count as read. Do not replace the large state page with
   another mandatory history read.
   - [x] Current state meets the size target; each displaced decision/result has
     an accessible owner, and required contracts remain reachable and mandatory.
   - [x] Ordinary docs work, a code repair, research planning and experiment
     continuation load only their relevant sources; compare startup bytes.

2. **Bound presented output and redundant calls.** Add a short workflow procedure
   for focused excerpts, retained full logs, per-command exit status, independent
   read batching and completion/change-aware waits. Evaluate a project
   `.codex/config.toml` output limit, with 3,000 as a candidate rather than an
   assumed optimum. Preserve progress updates and full training/report evidence.
   - [x] A diagnostic omitted from an excerpt is recoverable from its full log;
     failures remain visible and exact verification commands still execute.
   - [x] Any setting is confirmed through effective configuration; record output
     sizes and calls for a representative run without claiming credit savings.

3. **Reuse the research applicability gate.** Check the installed skill and any
   client-specific consumers before changing repository policy. Ordinary tooling,
   docs maintenance and CI without a research event skip ledger loading. Research
   hypotheses, architecture decisions, experiments, findings, contextual research
   confirmations and explicit ARA work still qualify. PATH-WM is research-heavy;
   do not assume the skip rate seen in ordinary engine maintenance applies here.
   - [x] Isolated engineering, real research-event, brief-confirmation and explicit
     ARA-inspection cases preserve appropriate reads, writes and evidence rules.
     No real scientific records are created by the workflow test itself.

4. **Choose effort using PATH-WM task evidence.** Compare appropriate effort levels
   on matched, verified tasks after the context changes. Declare prompts/tasks,
   quality checks and attempt budgets first; cap the initial pilot at eight model
   calls including repairs. Include input/cache/output tokens, retries, detected
   defects, completion time, model and requested/observed service tier. Reuse the
   existing opt-in profile or explicit overrides; do not infer a research-design
   default from PROC-034's two synthetic Python tasks.
   - [x] Record a scoped effort decision with all attempts and uncertainty. A
     decision to retain defaults is valid when evidence is insufficient. Validate
     any changed setting in the actual client; no blanket downgrade without data.

## Verification and completion

Record measurements and scope in this plan as each slice completes. For prose-only
changes, review the fixed diff, check moved/local link targets and anchors against
the baseline, and run `git diff --check`. Use `wc -c AGENTS.md CLAUDE.md
docs/experiment-workflow.md docs/project-state.md` for source-size comparisons.
Do not add prose-only tests. If shared Python code changes, run the relevant
tests and `python -m pytest` as required by the repository workflow.

For history extraction, compare every displaced block and its evidence references
against the captured pre-edit files. For configuration, use `config/read` with
this repository's cwd; profile selection must be checked in the actual CLI/app.
Retain full local trial logs and compact verified measurements. No training run,
model download or GPU job is authorized merely by creating this plan.

Register this planned workflow follow-up in the compact current-state index when
coordinating its implementation with the current writer. Complete all four slices,
update status/date/commit references here, and remove it from active-work entries
when done. Preserve this record and the measured limitations.

## Implementation record — 22 September 2026

Fresh baseline: `919320d2d90b1214fa38bf8877e67bfb3eabda65`.
The four startup documents total 161,964 bytes; project state is 150,596 bytes.
The original state is retained verbatim after a short preface in
[project history](project-history.md), in the same directory to preserve relative
evidence links. The compact state links the current owners. Concurrent architecture
work remains outside this workflow change.

Pilot declared before execution: at most eight model calls, no repairs or repeats,
`gpt-6-astra`, requested tier `default`, 240 seconds per call. Four isolated gate
cases at medium cover ordinary tooling, a synthetic research event with real
recording behavior, contextual confirmation and explicit read-only inspection.
The remaining four calls compare medium/xhigh once each on regression-test
authoring for the existing `pathwm.io.atomic_json` and `resume_arguments` helpers.
Each suite must pass the extracted current helper and detect four predeclared
faulty variants; syntax/import failures or timeouts are not successful detection.
Prompts, variants, sources, order and checks are fixed in the local manifest before
the first call. Count all attempts; report defects and incomplete calls. This is
workflow calibration, with no training, model download or real research records.
One sample per condition cannot establish a research-design default.

Full local evidence: `/tmp/pathwm-agent-efficiency-frrez0nn/` (manifest, runner,
unabridged CLI traces, generated suites, fixtures, checks and snapshots). Durable
measurements and the final disposition will be recorded below.

### Slices 1–3 verified

- State is 5,903 bytes at this checkpoint (target 8,192), including the concurrent
  attention-review pointer. Archive payload SHA-256 matches the fresh baseline:
  `263f9404805e26629d665ee1c2e617f5276aaaf865facb6538b77ebaf2189f11`.
  All 245 historical Markdown links retain their original destination and resolve;
  current links/anchors pass. The standing CLAUDE contracts and all pre-existing
  workflow sections before completion are unchanged. The completion route now
  distinguishes docs/config checks from shared-code and training verification.
- Static startup-route bytes (same baseline task-plan content in each comparison):
  ordinary docs 169,576 → 20,941; model code repair 177,936 → 34,425;
  research planning 176,644 → 34,556; experiment continuation 177,936 → 32,521.
  These are source-byte budgets, not observed model compliance, tokens or credits.
  Each route retains the named applicable contracts and reads no history by default.
- Effective `config/read` for this cwd reports project output limit 3,000,
  `gpt-6-astra`, user effort `xhigh` and requested tier `default`.
  Retain 3,000 as a practical bounded starting value, not a measured optimum.
  A deliberate failure retained 72,082 bytes while presenting 2,190 bytes/61 lines;
  exit 23 remained visible under `set -e`, and one focused read recovered the early
  diagnostic omitted from the excerpt. Four real structural/config/gate checks
  executed in one independent batch, all exit 0: 5,973 bytes retained, 425 presented,
  with no repeat status polls. The pilot uses one existing process handle.
- Codex exposes the existing user-scope research-manager `2.1.0-local.1`; its entry
  hash remains `bee9ac45a3ee2ad8d5aa16c5d0937f3572ec241517a3e6effd6414fb4e47872b`.
  No Claude-user or repository-local copy was found, and none was installed.
  Other clients follow the repository applicability procedure; client integration
  beyond Codex was not exercised.
- All four fresh gate cases pass trace, digest and record-semantic checks. Ordinary
  engineering: zero ledger/reference reads and only the requested typo changed.
  Research: one experiment, staged observation and affirmed evidence-bound claim.
  Brief confirmation: existing experiment unchanged, observation promoted with
  user-revised provenance. Inspection: staging only, no writes. Evidence fixtures
  are unchanged; these tests write only their isolated synthetic ARA directories.
  One unnecessary `git diff` in the non-Git engineering fixture returned 129;
  the retained trace exposes it and the requested edit still verifies. No retries.

### Slice 4 and completion

Implementation: `c4552de`; this completion record and the compact machine-readable
results are committed with the final verification. The concurrent attention-review
commit `4d1d836` preceded the pilot; `pathwm/io.py` supplied both extracted helpers.
The active workflow entry has been removed from current priorities.

All eight declared calls completed, with no retries, repairs or unreported attempts.
All four generated suites pass their original helper and reject all four predefined
mutants (16/16 fault detections across four suites; four distinct faults per helper).
There were no generated-suite defects detected by these checks. The tests stayed
in isolated fixtures; no shared Python implementation, training run or real research
record was changed by this calibration.

| Extracted helper | Effort | Input / cached input | Output (reasoning subset) | Model + CLI seconds | Through checks, approx. seconds |
| --- | --- | ---: | ---: | ---: | ---: |
| Atomic JSON | medium | 14,200 / 11,520 | 1,268 (213) | 42.187 | 45.246 |
| Atomic JSON | xhigh | 14,200 / 11,520 | 4,043 (2,588) | 128.479 | 131.480 |
| Resume arguments | medium | 14,353 / 11,520 | 1,448 (133) | 49.367 | 52.090 |
| Resume arguments | xhigh | 14,353 / 11,520 | 4,521 (2,667) | 140.531 | 145.056 |

Helper totals: medium 2,716 output tokens and approximately 97.336 seconds through
checks; xhigh 8,564 output tokens and approximately 276.536 seconds. Input and
cached input match for each pair. Cache-write tokens were zero. Reasoning tokens
are included in output, not added again. Generation/CLI duration was timed directly;
subsequent verification time is estimated from the original trace and final check-log
file timestamps. Gate semantic-review time was not recorded.

**Decision:** retain the user's xhigh base setting. Medium is an opt-in candidate
for similarly bounded helper/test authoring with independent acceptance checks.
Reuse the existing user profile or the explicit command in the workflow; no new
profile or automatic effort switch is added. One sample per task/effort, extracted
fixtures, narrow fault coverage and uncontrolled latency variation cannot establish
an optimum or support a blanket downgrade for research, architecture or numerical
work. Requested tier was `default`; delivered tier and credit charges are unavailable.

Full traces, exact prompts, sources, mutants, generated suites, fixture before/after
records and verification logs are retained in `runs/agent_efficiency_v1/evidence/`.
The existing report renderer supplies the [local report](../runs/agent_efficiency_v1/report.html) from
raw JSON/JSONL, with result completion distinct from report status. Report checks
are structural only; no renderer change or visual-browser claim is made. The
versioned [measurement record](agent-token-efficiency-results.json) records every call and check.
Source-byte accounting is not a token/credit saving claim. No repository Python
source changed, so the docs/config verification route applies.

Final current-state size: **5,758 bytes**, down from 150,596 (96.2% fewer bytes).
All four startup sources, even counting the entire workflow, total 23,183 bytes
versus 161,964 before. Scoped routes at closure are 20,796 bytes for ordinary docs,
34,280 for model code repair, 34,411 for research planning and 32,376 for experiment
continuation, holding the task-plan content constant. Actual reads can vary with
the request; scientific requirements and the historical evidence remain available.

Final checks pass: byte-identical historical payload and preserved links, new
links/anchors, unchanged standing scientific contracts, effective project config,
four gate fixtures, four mutation-checked helper suites, and whitespace. All eight
raw metric rows appear in the self-contained report; `report.qa.json` records the
renderer/report hashes and `structural_verified`, with `browser_checked: false`.
Rebuild only the report with
`.venv/bin/python -m pathwm.evaluation.report runs/agent_efficiency_v1`.
Calibration runners are retained as local evidence, not a new repository framework
or a required model-call check for future changes.
