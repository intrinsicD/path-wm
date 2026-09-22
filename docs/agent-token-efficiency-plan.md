# Reduce coding-agent context and token overhead

Status: planned; implementation has not started. Created 22 September 2026 at
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
   - [ ] Current state meets the size target; each displaced decision/result has
     an accessible owner, and required contracts remain reachable and mandatory.
   - [ ] Ordinary docs work, a code repair, research planning and experiment
     continuation load only their relevant sources; compare startup bytes.

2. **Bound presented output and redundant calls.** Add a short workflow procedure
   for focused excerpts, retained full logs, per-command exit status, independent
   read batching and completion/change-aware waits. Evaluate a project
   `.codex/config.toml` output limit, with 3,000 as a candidate rather than an
   assumed optimum. Preserve progress updates and full training/report evidence.
   - [ ] A diagnostic omitted from an excerpt is recoverable from its full log;
     failures remain visible and exact verification commands still execute.
   - [ ] Any setting is confirmed through effective configuration; record output
     sizes and calls for a representative run without claiming credit savings.

3. **Reuse the research applicability gate.** Check the installed skill and any
   client-specific consumers before changing repository policy. Ordinary tooling,
   docs maintenance and CI without a research event skip ledger loading. Research
   hypotheses, architecture decisions, experiments, findings, contextual research
   confirmations and explicit ARA work still qualify. PATH-WM is research-heavy;
   do not assume the skip rate seen in ordinary engine maintenance applies here.
   - [ ] Isolated engineering, real research-event, brief-confirmation and explicit
     ARA-inspection cases preserve appropriate reads, writes and evidence rules.
     No real scientific records are created by the workflow test itself.

4. **Choose effort using PATH-WM task evidence.** Compare appropriate effort levels
   on matched, verified tasks after the context changes. Declare prompts/tasks,
   quality checks and attempt budgets first; cap the initial pilot at eight model
   calls including repairs. Include input/cache/output tokens, retries, detected
   defects, completion time, model and requested/observed service tier. Reuse the
   existing opt-in profile or explicit overrides; do not infer a research-design
   default from PROC-034's two synthetic Python tasks.
   - [ ] Record a scoped effort decision with all attempts and uncertainty. A
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
