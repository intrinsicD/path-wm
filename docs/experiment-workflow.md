# Experiment and development workflow

The user-facing entry point is a readable file in `experiments/`. Shared code is
in `pathwm/`. This workflow replaces the retired dated scripts and aggregate
report wrapper while preserving scientific integrity and the four-step process.

## Test the actual model

User requirements (24 September 2026), mandatory for planning, implementation,
scientific evaluation and completion:

1. Prefer testing the actual model, existing modules and real configured path.
   Record the builder, modules, dimensions, checkpoint (when relevant), inputs and
   exercised consumer. A passing substitute-model test is not model validation.
2. If a smaller preliminary test is useful, use a downscaled configuration of the
   same architecture. Preserve the relevant modules, connections and semantics;
   declare every reduction. Do not introduce an unrelated toy codec/model to make
   the test easier.
3. After every downscaled preliminary check, rerun the corresponding check on the
   actual full configuration before claiming completion. Record both receipts and
   their differences. Unit tests of helpers may isolate arithmetic/contracts, but
   do not discharge the real-model rerun requirement. Do not call an arbitrary
   default or a random untrained fixture the full trained model. Missing reference
   settings, weights, resources or paths are explicit unresolved items.
4. If the desired behavior is missing, identify the exact gap and prepare a concrete
   implementation plan for Alex to discuss before implementing it. Do not invent a
   surrogate, silently add a new component, or count a test of missing behavior as
   evidence. Await agreement for that implementation while continuing independent
   checks of existing behavior. Alex explicitly requested this joint planning step.
5. When correcting previous substitute tests, inventory each affected test/claim,
   identify its real-model counterpart, rerun what exists, and separately list
   missing implementations requiring discussion. Preserve previous runs and failed
   results. A broad pytest pass alone does not close outstanding real-model reruns.

Controlled diagnostics must also declare changes to the actual input domain,
including positions, scales and augmentations. Check those bounds against the real
generator or task before using a diagnostic to justify a repair. An intentional
out-of-domain stress test is useful evidence only for that scope; follow it with
the actual task conditions before claiming an in-task defect.

## Work in small complete slices

1. **Plan.** Update the active plan linked from project-state.md: concrete problem,
   affected interfaces, success criteria and compute budget. Prefer one real user
   path. Apply the standing design principles below and record the relevant choices.
   Do not create another framework or speculative module catalog.
2. **Essential checks.** Write the few tests that catch meaningful failure. For new
   behavior, demonstrate an informative failure before implementation. Tests cover
   data alignment, frozen gradients/buffers, temporal causality, numeric references
   and resume where relevant. Commit the completed plan/check step; label red state.
3. **Implement.** Make the recipe work with a tiny real batch and short explicit
   development run. Verify its local report. A smoke run proves the workflow, not
   model quality. Commit the working slice.
4. **Compare.** Change one scientific factor per comparison with matched populations
   and budgets. Preserve reference settings/results. Broaden only after the first
   user path works. Commit completed steps and update project state.

A scientific experiment separately declares hypotheses, seeds, populations,
metrics, thresholds and budgets before execution. An unset threshold cannot pass.
Record deviations and stopped/failed runs; never choose gates after seeing results.
Do not add tests for prose or trivial glue, or conceal failures by deleting tests.

## Standing design principles

Alex adopts the [DeepSeek-derived abstract principles](deepseek-v41-transfer-proposal.md#abstract-principles-behind-the-mechanisms)
as standing guidance for every model-related design: architecture, modules and
interfaces, memory, training/data, inference, planning and supporting systems.
Actively look for opportunities to integrate them whenever designing or revising
any part of PATH-WM. Choose concrete applications according to the workload,
learning needs and evidence, while preserving the small-library/recipe boundary.

- **Prepare once, use many ways.** Separate reusable source preparation from
  consumer-specific queries, computation and outputs. Share stable representations
  while allowing specialized projections and processing.
- **Reuse invariants; refresh changing work.** Distinguish source values, projected
  K/V, queries and selected addresses. Scope reuse to valid input/weight/state
  identities and preserve gradient paths during training. Evaluate learned sharing
  and reused selections separately from identical-computation reuse.
- **Keep multiple resolutions and select access deliberately.** Combine broad
  context with fine evidence; consider staged retrieval and conditional experts.
  Account for initial search, missed candidates and recovery/full-access controls.
  Separate retained information from the subset read by a particular computation.
- **Organize before discarding.** Use geometry and reversible rearrangement where
  useful; make compression and information loss explicit. Consider multiple learned
  information paths without assuming that mixing provides invertibility or that
  streams acquire predetermined meanings.
- **Give state explicit ownership and lifetimes.** Separate authoritative evidence,
  exact restart state, learned pattern knowledge and disposable derived caches.
  Balance storage against reconstruction cost and reuse. Label approximate replay
  and preserve source provenance, temporal causality and declared exactness contracts.
- **Spend capacity and precision where useful.** Consider lookup versus computation,
  specialist routing and sensitivity-based precision. Check per-modality/task
  behavior as well as aggregate balance; measure decision/ranking effects of
  approximation, not only tensor error.
- **Separate proposing from verification.** Consider cheaper candidate generation
  followed by stronger checking. State what the checker establishes; preserve
  independent evidence for factual or environmental success. Distillation transfers
  behavior across architectures but teacher agreement remains a learning signal.
- **Allocate computation to useful progress.** Consider effort-conditioned reads,
  thinking and search after establishing useful fixed-budget behavior. Keep hard
  resource limits; distinguish total work from critical-path latency and measure
  quality across budgets rather than assuming more computation helps.
- **Design for actual execution.** Account for memory traffic, redundant transfers,
  dependency structure and opportunities to fuse or overlap work. Give shared
  parameters/state clear owners. Match optimizer treatment to parameter structure
  where justified; track sample bias and policy versions in asynchronous work.
- **Train and evaluate the system that will run.** Include adopted access/precision
  restrictions during training, compare restricted and unrestricted behavior, and
  use informative tasks with independently checkable outcomes, fresh compositions
  and preservation controls. Prioritize trustworthy learning signals.

In each substantive design note, briefly state which principles apply, how they
are integrated or why a relevant option is deferred, the expected benefit and
trade-offs, and the smallest meaningful check or comparison. Account separately
for retained/accessed information, numerical fidelity, parameters, compute, memory
traffic and latency where relevant. Keep exact optimizations, learned architecture
changes and lossy approximations distinguishable. The standing commitment is to
consider and apply these ideas thoughtfully; individual mechanisms and performance
claims still require their own scoped evidence.

## Keep it understandable

Use ordinary Python/PyTorch modules, functions, explicit arguments and small data
records. Recipes show construction, data/splits, losses, freeze rules and budgets.
Keep model inputs/outputs documented; readers must be able to trace their tensors
and gradients. No module registry, base Experiment class, universal trainer,
recursive configuration hierarchy, import-time training/downloads or dated paths
inside reusable code. New abstractions require a concrete consumer.

Do not edit shared source during an active run; finish it or use an isolated copy.
Dependencies point from recipes to the library. New experiments may copy a short
recipe, not import another recipe. Helpers and model files cannot import the
historical package, old run manifests or reporting services. Dataset/weight paths
are explicit inputs. Copied upstream code retains its source, revision and license.

## Trustworthy runs and reports

Keep source data/assets under `data/`; each new run owns `runs/<name>/`. Never
overwrite an existing run. Explicit resume must check compatible code, modules,
data/splits, objective, optimizer and precision. Checkpoints include model buffers,
optimizer/scheduler, progress and RNG/sampler state. Document exact replay limits.

Record resolved settings, code identity, source and split identities, initialization,
seed, sample/update counts, precision and package/device information. Raw JSON and
JSONL are authoritative. Result completion must be written before report generation.
A reporting failure preserves raw results and has a distinct visible status.

Every completed run generates a self-contained local `report.html` using repository
code and declared dependencies. Include curves, exact metric values, representative
examples and source/settings context. Structural checks always run. Browser QA is
required for a new or changed report renderer; its receipt applies only to the
checked content/version. Disclose structural-only validation when browser QA is
unavailable. No AI-app or private plugin is needed to train or read a report.

Do not confuse reconstruction quality, latent loss, prediction and closed-loop
control. Avoid comparing absolute losses across differently scaled latent spaces.
Preserve negative results and distinguish observed results from explanations.

For consequential scientific choices use the adopted
[Claude review](claude-collaboration-workflow.md): independent criticism, verify
claims, reconcile disagreements, preserve receipts. Respect the current public-only
export boundary; no private repository content or measurements in external briefs.
Peer agreement cannot authorize scope changes or replace empirical checks.

## Finish a slice

For shared code, run the relevant CPU tests and the documented check/train/resume
path; `python -m pytest` remains required before committing shared-code changes.
For documentation/configuration-only work, check the fixed diff, affected local
links/anchors and effective configuration, and run `git diff --check`; do not add
prose-only tests or start training solely to validate documentation.
Update the task's plan and the short project-state page with what works, evidence and limits.
Report the usable commands and report location. Commit completed work. Do not let
new experiment history grow into the normal user interface again.
Keep project state at most 8 KiB, with short current statuses and links. Detailed
results belong in their existing plans/run evidence; the frozen project-history
snapshot is retrieved only for the older decisions relevant to the task.

## Tool output and waits

Present bounded results while retaining complete evidence. Prefer file names,
headings and focused `rg`/line excerpts over entire large files. Save noisy command
stdout/stderr before presenting a summary or tail; report each command's actual
exit status and log path. A later successful command must not mask an earlier
failure. Read omitted diagnostics from the retained log when the excerpt is
insufficient. Do not truncate raw metrics, checkpoints or standalone reports.

For a noisy shell command, use a fresh log and preserve the producer's status:

```bash
run_log=$(mktemp /tmp/pathwm-command.XXXXXX.log)
run_status=0
python -m pytest >"$run_log" 2>&1 || run_status=$?
tail -n 60 "$run_log"
printf 'exit=%s full_log=%s\n' "$run_status" "$run_log"
exit "$run_status"
```

Batch independent reads/checks and retain every result. Keep dependent steps,
writes and shared-source training sequential or isolated. Wait on an existing
process/job handle until completion or a meaningful change; do not repeatedly
read unchanged logs or launch another job to check status. Keep user progress
updates and apply bounded waits so new input can steer the work.

Codex's project `tool_output_token_limit = 3000` bounds routine tool history;
explicit per-call limits may override it. Full logs remain authoritative. Other
clients use their available output controls and the same evidence procedure.
Check effective configuration for this cwd; cached thread settings may require
a fresh session. This setting changes presentation, not the tests or run budget.

## Research applicability

Before automatic bookkeeping, decide from the current request and conversation.
Do not load ARA ledgers, the research-manager recording references, or old
observations just to decide whether they are relevant. An existing `ara/` or an
edit to a research-named file is not sufficient by itself.

Ordinary tooling, docs/task/config maintenance and routine correctness checks
without a research event exit silently. Research hypotheses, architecture
decisions, method work, experiments, findings, evidence corrections and explicit
research affirmations/refutations qualify. A short confirmation with a clear
research referent qualifies; mixed turns record their research portion only.
Explicit ARA inspection, initialization, maintenance or research resumption uses
its requested scope. Read-only inspection does not itself require new records.

Reuse the installed research-manager gate and recording procedure where available;
do not create a repository copy. Eligible work retains provenance, evidence
bindings, staged observations and closure signals. Missing records from skipped
engineering turns cannot establish topic abandonment. Required evidence must
exist before publication; end-of-turn bookkeeping never postpones that duty or
suppresses an explicit ARA request. This workflow calibration creates no real
scientific records. Research-heavy sessions should expect fewer skipped turns.

## Reasoning effort

Choose effort using verified completed-task evidence for the actual task family.
Keep model, requested service tier, prompts and acceptance checks fixed when
comparing effort; record input/cache/output tokens, all repairs, detected defects
and time through completion. Report unavailable delivered-tier/credit information
and unequal context/cache conditions rather than treating them as zero.
Predeclare the trial budget. A small fixture or smoke run cannot set a default for
research design, numerical diagnosis, architecture or long-running experiments.
Existing base effort stays in force until a scoped comparison supports a change;
verification and scientific review requirements apply at every effort level.

The [22 September PATH-WM pilot](agent-token-efficiency-plan.md#slice-4-and-completion)
passed both extracted-helper test tasks at medium and xhigh. Medium used less
output and time in those four calls; the sample does not justify a new default.
Retain xhigh for the base configuration. For similarly bounded helper/test work
with predeclared independent checks, medium is an opt-in candidate: from this
repository use `codex --cd . -c 'model_reasoning_effort="medium"'`, or select the
existing `intrinsic-routine` user profile. Return to the base setting when scope
requires research design, architecture or difficult numerical diagnosis.
