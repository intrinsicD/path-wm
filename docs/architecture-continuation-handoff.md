# Architecture continuation with Claude Opus 5.5 medium

Updated 24 September 2026 for a new Codex session in
`/home/alex/Documents/path-wm`, branch `main`. Latest completed implementation:
**`98b379a`**, bounded context/retrieval; its plan/red-test commit is `e442d92`.
Earlier evidence-loop implementation: `b51d949` (plan/red checks `3371600`).
The working tree was clean before this documentation update. All selected runs and
the full suite finished; no job from this continuation remains to resume.

## Paste into the new Codex session

> Continue PATH-WM from `docs/architecture-continuation-handoff.md`. Read CLAUDE.md
> and the current project state. Work with actual Claude Opus 5.5 at medium effort
> through multiple substantive design, implementation and review rounds. Preserve
> the public-only external review boundary. Inspect the existing implementation,
> plan the remaining points in small measurable slices, implement and test each
> selected slice, review and repair it, and continue until its declared gates pass
> or evidence identifies a concrete blocker. Preserve failed results. Do not claim
> the overall architecture is complete from the bounded RGB16 or lexical experiments.
> The latest slice passed two seeds and 848 tests, but counting matched learned
> retrieval and context reset/swap changed no answers. Start with Claude by designing
> a falsifiable scan-limited candidate-discovery/retention experiment against a strong
> indexed baseline. Reuse the implemented contracts, predeclare fresh test populations,
> gates and local compute budgets, then implement, review and test measurable slices.
> Keep reconstruction, evidence correction, useful representations and actual total
> compute cost explicit. Use each result to select the next remaining priority.

This handoff records the existing continuation authorization. Do not start a new
architecture from scratch or ask again whether ordinary implementation/review is
wanted. A new costly external training service or expansion of the export boundary
is not authorized by this document.

## Read first, then only the relevant owners

1. [Current work](project-state.md) and applicable sections of
   [experiment workflow](experiment-workflow.md).
2. [Context/retrieval plan and results](context-retrieval-plan.md): latest implementation,
   registered gates, controls, negative findings, commands and verification. Then
   [evidence-loop results](evidence-loop-plan.md) for the frozen visual baseline.
3. [Architecture walkthrough](architecture-walkthrough.md), especially “Completion
   priorities after the concept and conversation discussion” and context ownership.
4. [Integrated latent-agent goal](integrated-latent-agent-goal.md) and
   [Claude collaboration workflow](claude-collaboration-workflow.md).
5. [Model guide](models.md), [experiment guide](experiments.md), and current source.
   Consult [integrated architecture plan](integrated-architecture-plan.md) and
   [concept-learning review](latent-concept-learning-review.md) before claiming a
   concept-learning solution; the earlier transfer/retention failures still stand.

## What the user wants

- Reusable structure learned across instances plus detailed memory of each entity,
  usable for recognition, reconstruction, prediction and other tasks. Human pose
  and viewpoint change is an example, not a mandate to start with human rendering.
- Accumulate evidence; revise uncertain estimates when new evidence arrives. Use
  the best estimate when necessary, without promoting generated guesses to facts.
- Better reconstruction alongside useful representations and compact computation.
- Consistent text/audio interaction. A conversation is a persistent episode in the
  external World State; selected values and references enter active context.
- Local and Global Context are **both internal**. Global is not the persistent
  knowledge graph, and conversation does not require a third peer memory system.
- Reactive receive/update/retrieve/respond-or-wait is sufficient initially. Planning
  is optional where a task actually needs multi-step prediction or action search.
- Reuse learned computation when evidence supports it. Internal refinement index
  `k` must not advance world time `t`. Keep parameter count and execution cost separate.

## Completed implementation: reuse, do not duplicate

| File | Responsibility |
| --- | --- |
| `pathwm/models/detail_memory.py` | Shared part encoder, learned finite-pose decoder, variance, explicit code replacement and validated stored-code consumption. |
| `pathwm/data/detail_views.py` | Procedural RGB16 textures, four RGB8 parts; target geometry only in the data generator. |
| `pathwm/world_state/episodes.py` | EpisodeClient over existing WorldSession/WorldStore: evidence, inferred state, bounded reads, utterance boundaries, interruption, checked emissions, explicit unknown/abort recovery. |
| `experiments/evidence_loop.py` | Ordinary train/calibrate/evaluate/resume recipe, persistence demonstration, controls and standalone report. |
| `tests/test_detail_memory.py`, `tests/test_episodes.py`, `tests/test_evidence_recipe.py` | 18 focused checks, including exact resume and stale/interrupted output rejection. |
| `pathwm/models/tasks.py` | Optional `ContextSelector` under TaskPolicy: supervised metadata ranking with explicit null; existing defaults unchanged. |
| `pathwm/world_state/context.py` | WorkingContext: bounded flat/Local–Global component pins, task ownership, epoch/version checks, eviction/reset and owner-bound portable restart. |
| `pathwm/data/episode_facts.py` | Controlled two-slot synonym task with supplied entity matching; no answer payload in ranker inputs. |
| `experiments/context_retrieval.py` | One ordinary train/evaluate/resume recipe; counted, canonical, rule, random and no-read controls; context interventions and separate live-episode integration. |
| `tests/test_working_context.py`, `tests/test_context_recipe.py` | 10 new checks; includes exact training resume, source races, context budgets and preserving result completion when rendering fails. |

State reads pin sources, component heads and model versions. Invalidated or
budget-omitted details cannot silently become unknown priors. Superseded evidence
invalidates dependent state; unrelated detail survives. Generated output is an
internal authorization record, not observation or proof of physical playback.
All final population codes pass through portable store serialization; a distinct
live-session demonstration checks actual acquisition, correction and restart.

The selected codec has936,768 parameters. It uses learned affine heads for four
familiar quarter-turns, not an analytic rotation inside the model. Part identities,
visibility and canonical input alignment are supplied. Independent hidden textures
use a learned population estimate; this is not relational concept induction.

## Latest results: context/retrieval

- **848 tests passed in622.19 s**, exit0, source unchanged throughout the suite.
  Log/receipt: `runs/reviews/architecture_continuation_20260924/full-suite.log` and
  `full-suite-exit.json`. An earlier attempt was stopped after38.38 s for report
  completion; `full-suite-attempt1-exit.json` records exit-15, not a pass.
- `runs/context_final_s17_v1/` and `runs/context_final_s29_v1/`: 1200 updates/seed,
  64 worlds ×40 questions per split, 512 unknowns/split. Learned, counted-table and
  generator-rule readers all reach **100%** agreement with current source assertions
  on IID and held-out two-alias pairings, in both layouts. Copying is templated.
- **Negative results:** learned-minus-counted =0; Local/Global-minus-flat =0.
  Each split's1024 reset and1024 task-swap interventions leave resumed answers
  unchanged. Full32-key search makes retained context unnecessary. Removing supplied
  entity eligibility reduces accuracy to39.49%. Do not call this learned identity,
  useful retention, autonomous context control, concept induction or language.
- The optional selector has705 trainable /1213 total parameters. Actual mean step
  time1.39 ms, total train/evaluation invocation23.20 s/seed. Flat held-out-pairing
  selection + validated-copy p95 is227.31/227.89 µs versus rule152.32/152.47 µs.
  This excludes ingestion, candidate preparation and the episode protocol; it is
  not full-agent latency. Optimizer/autograd/process-memory details are in the plan.
- Each split rejects512 injected stale references. Nine checks in a separate actual
  WorldSession demonstrate incomplete-input waiting, interruption, correction,
  checked emission and exact session/context restart. Population store/context
  tests and this small session demonstration are separate evidence scopes.
- Frozen-codec regressions: `runs/context_codec_regression_s17_v1/` and
  `runs/context_codec_regression_s29_v1/`. Direct/recalled outputs remain bit-exact
  on512 original test textures ×4 familiar poses; mean MSE0.00024258/0.00023893,
  unchanged weights and zero training updates. No new visual-learning claim.
- Each run owns checkpoint, raw metrics/predictions, settings/source identities,
  result and standalone `report.html`. Reports use the unchanged renderer with
  structural verification; no browser-QA claim. Saved reports were enriched from
  existing predictions by `enrich_reports.py`; metrics and models did not change.
  The current recipe embeds examples directly. Its only difference from the saved
  final-run recipe is that report-example addition; exact diffs and audits are saved.
- `raw-audit.json` independently recomputes metrics/budgets; `artifact-audit.json`
  verifies report/result/checkpoint hashes and unchanged shared source. Both are in
  `runs/reviews/architecture_continuation_20260924/`.
- **Consumed population:** lexical final seed624927 has been inspected. Use it only
  as regression data after new tuning; preregister fresh untouched final populations.
  Development seed624925 used train-type phrases only, never the final alias pairings.
- Tracked evidence: `ara/evidence/tables/context_retrieval_2026-09-24.json`.
  N574–N576 and staged O400 already record this continuation. Do not duplicate them
  or promote staged interpretations merely because a new session begins.

### Integration traps already found and repaired

`WorldSession` replaces its store atomically on commits. Pass the **live session**
to WorkingContext when that session owns updates; retaining its old `session.store`
object silently disconnects later reads. The failed check is preserved in
`focused-v3.log`; the repair is covered by the full suite.

Epoch validation supports synchronous consumption, including null decisions. It
is conservative about unrelated writes and is **not an atomic external emitter**.
Use EpisodeClient's transaction/pinned-dependency path for checked session outputs.
Snapshot checksums detect accidental corruption, not forgery; task tags are trusted
caller metadata. Restore must validate owner, layout, representation and capacity,
and reject stale pins without rebinding. Generated responses remain derived records,
never independent source evidence or proof of playback.

## Earlier evidence-loop evidence and failures

- Earlier full suite: **838 passed in660.49 seconds**. Log and exit receipt:
  `runs/reviews/architecture_completion_20260924/full-suite-detached.log` and
  `full-suite-exit.json`. Earlier interrupted/terminated attempts are not passes.
- Final runs: `runs/evidence_loop_final_s17/` and `runs/evidence_loop_final_s29/`.
  Both pass7 quality gates and11 live-session checks. PSNR36.151/36.217dB;
  newly revealed detail error decreases by over99%; nominal90% hidden interval
  coverage91.849/91.897% on the synthetic population.
- Comparison: `runs/evidence_loop_summary_v1/report.html`. Reports were structurally
  verified; reconstruction contact sheet was visually inspected. No browser QA claim.
- Failed first FiLM fit: `runs/evidence_loop_film_s17_v1/`, MSE0.01822. Preserve it.
  Linear repair: `runs/evidence_loop_linear_s17_dev/`, MSE0.0002413 but hidden
  coverage73.94%. Variance-only calibration repaired coverage while leaving all
  mean/encoder weights unchanged: `runs/evidence_loop_calibrated_s17_dev/`.
- An initial split bug exposed seed240924; it was reclassified as development
  before inspecting the first trained result. Final seed240927 was predeclared and
  evaluated only after selection. **It is now consumed**: if tuning the next model,
  predeclare a new untouched test population and keep this one as regression data.
- Profile: `runs/evidence_loop_final_s17/profile.json`. RTX3050, FP32 batch64,
  diagnostic step median2.63ms. Supported-op FLOPs are incomplete; timing is not a
  matched efficiency claim. Retained source evidence costs storage beyond the codes.
- Tracked evidence: `ara/evidence/tables/evidence_loop_2026-09-24.json`.
  Research records N570–N573 and staged O399 already exist; do not duplicate or
  promote them into a general capability claim. Broader O398 remains unresolved.

`runs/` artifacts are local and ignored by Git. A checkout elsewhere needs these
files copied explicitly for result inspection. Do not invent missing results.

## Remaining points and suggested order

This is a continuation agenda, not an assertion that every item needs a new module.
First inspect what already exists and have Claude challenge the next experiment.

1. **Useful retrieval, retention and context control.** The fixed/learned lexical
   baseline and pin lifecycle contracts now exist. Keep the learned head optional:
   counting solves that task. Next, exceed the candidate scan budget and make later
   queries genuinely depend on discovering or retaining earlier evidence. Include a
   strong indexed/count-based baseline rather than deliberately handicapping fixed
   retrieval. Compare discovery, retention and readout separately, with missed-candidate
   recovery, delayed queries, distractors, corrections, interruptions and restart.
   Keep exact strings/numbers accessible. Test one learned decision under TaskPolicy
   at a time; do not add allocation, summaries and routing simultaneously. Flat and
   Local/Global views need matched storage/scan/read budgets, measured compute and
   causal reset/swap/eviction interventions. A useful-context claim requires a
   measurable dependency on retained context, absent from the previous task.
   Grounded learned responses remain a separate step beyond template copying.
2. **Concept/instance/state learning.** Establish reusable structure beyond four
   familiar transforms, with held-out instances and a separately declared novel
   relation/concept test. Distinguish recognition/association from oracle identities.
   Compare remembered examples with inferred concept codes before adding machinery.
   Test correction, counterexamples, transfer to another use and frozen-weight runtime
   acquisition. Earlier R1 transfer failures must guide task selection. This is a
   separate scientific question from successful graph writes or pose reconstruction.
3. **Evidence and uncertainty beyond independent clean parts.** Test noisy and
   correlated observations, uncertain association, contradiction versus real state
   change, and detail-specific uncertainty. Duplicate evidence must not spuriously
   increase confidence. Evaluate calibration by region and under declared shifts;
   current population coverage is not per-entity or out-of-distribution calibration.
4. **Reconstruction and prediction on harder observations.** Increment difficulty
   from supplied canonical parts to partial views/occlusion and learned alignment.
   Preserve multiscale information and make compression explicit. Benchmark detail
   retention and downstream prediction separately. If adding frozen-feature loss,
   freeze the reference parameters while retaining gradients through reconstructed
   inputs. Add objectives one at a time; do not force pixels to carry every goal.
5. **Shared depth and persistent working state.** Inspect existing Thinker/recurrent
   support first. Compare fixed K=1,2,4 (then8 if justified) against unique depth with
   declared parameter/compute controls. Plot quality versus K and measured latency,
   including reconstruction and prediction. Add one tiny persistent state only after
   a baseline; declare reset/detach and external-time semantics. Then test randomized
   K, then stochastic execution. Each mechanism needs a disable/ablation setting.
6. **Smooth multimodal conversation.** After grounded text behavior, add audio input/
   output through the same episode and learned response state. Define speech timing,
   partial utterances, cancellation, completion and playback acknowledgment. Existing
   audio references are not ASR, speech generation, semantic grounding or fluent
   dialogue. Text bridges versus direct latent speech remain alternatives to compare.
7. **Execution efficiency and optional planning.** Profile actual bottlenecks before
   sparse masks, occasional attention or modality fusion changes. Share mask-independent
   work and test exact semantics where sparsity permits. Learned halting/routing,
   Mamba replacement and complex fusion remain later hypotheses. Add bounded planning
   only for a task with a demonstrated need and measurable benefit over reaction.

Do not implement all seven simultaneously. Select one end-to-end learning path,
record its scope and budget, finish its evaluation, and use that result to choose
what follows. Fluent general dialogue and natural-human synthesis need their own
training/data strategy; do not promise them from a tiny synthetic fit.

## Concrete starting slices for the next session

1. **Select and register the experiment with Claude.** Inspect current code and the
   negative context-intervention result first. Compare a strong exact/indexed reader,
   counted reader and one learned selector/retention decision. Establish why full
   search is unavailable in the proposed task and account for index construction,
   key preparation, initial search and recovery. If a simple index already solves
   it cheaply, keep that result and reconsider the next scientific priority.
2. **Implement one end-to-end path and its essential red checks.** Fix the external
   store and episode owner; specify query/candidate eligibility, read and retention
   budgets, reset/suspend/resume, stale and omitted-detail behavior. Extend existing
   modules/recipe where practical. Test causal inputs, source correction, true context
   dependence and exact restart before claiming a learned result.
3. **Run a bounded development comparison, review and repair.** Before execution,
   register seeds, new train/dev/test populations, numeric quality/retention/latency
   gates, local wall-time budget and allowed repair count. Use independent criticism
   from Claude on public abstractions; inspect private source/results locally.
4. **Freeze and verify the selected path.** Tiny train/report/resume, controlled
   multi-seed final evaluation, saved failures, full repository tests on unchanged
   source, standalone reports and scoped docs/atlas/evidence updates. A null result
   is not a reason to invent a learned benefit. Choose the next item above from the
   evidence; do not declare the entire architecture complete.

Useful existing commands (use fresh output directories; full settings must match
for resume):

```bash
.venv/bin/python -m experiments.context_retrieval --output runs/my_context_check --steps 6 --worlds 1 --stop-after 3
.venv/bin/python -m experiments.context_retrieval --output runs/my_context_check --steps 6 --worlds 1 --resume
.venv/bin/python -m pytest tests/test_working_context.py tests/test_context_recipe.py tests/test_episodes.py
.venv/bin/python -m pytest
```

Existing final runs are regression evidence; do not rerun them unnecessarily before
selecting the next experiment. `--final` in the old recipe uses consumed seed624927,
so it does not automatically create a fresh final test for a changed method.

## Actual Claude collaboration

Use **`claude-opus-5-5` with `--effort medium`**, not the default model or a Codex
subagent renamed Claude. The previous installed CLI was `/home/alex/.local/bin/claude`
(version2.1.280). Verify availability and record the actual returned model usage.
If unavailable, report the specific problem rather than silently substituting.

Latest collaboration: **four actual Opus5.5 medium rounds**, session
`5af6c291-64b3-40fc-b95b-1d9019a806e5`. Exact briefs, replies, `invoke.py`, per-round
receipts and aggregate `claude-receipt.json` are in
`runs/reviews/architecture_continuation_20260924/`. Earlier evidence-loop work used
five calls in session `3827870d-78ce-4c88-be5d-ebf8b1c8cf26`, under
`runs/reviews/architecture_completion_20260924/`. Use the latest runner as a pattern,
not permission to send this private handoff externally. Retain returned usage fields
verbatim; resumed-session totals may overlap, so do not blindly sum them.

Start a fresh review directory and preferably a fresh Claude session with a public,
hypothetical brief. Keep subsequent rounds in that session. The existing runner
uses an isolated temporary cwd, no tools, empty strict MCP configuration, disabled
hooks/slash commands and no imported settings. Preserve those isolation properties.
Only public abstractions and Claude-authored generic code cross the boundary:
**no private source, local measurements, datasets, this handoff or unrelated history**.
Claude can supply generic implementation proposals; Codex inspects and integrates
against actual local code. Be explicit that private implementation review is local.

Useful rounds: (1) alternatives and smallest falsifiable task; (2) objections and
reconciliation, with explicit withdrawals; (3) generic implementation/contracts;
(4) adversarial review of those contracts and failure scenarios; (5) focused repair
if evidence requires it. Do not manufacture rounds after the decision is resolved.
Save exact inputs, outputs, session/model/effort receipts and failures. Agreement
is not experimental evidence. API-equivalent cost fields are not subscription bills.

## Completion discipline

- Keep this a small library plus readable recipes. No extra trainer, registry or
  general execution framework. Reusable modules must not import experiment recipes.
- Define splits, seeds, controls, thresholds and local compute budget before runs.
  Measure parameters, optimizer/activation memory, real step time and quality.
- Run focused meaningful checks, tiny train/report/resume, then controlled comparisons.
  Preserve raw metrics, checkpoint, settings/source identities and each run's report.
- Freeze source while the full test suite runs. Required before shared-code commits:
  `.venv/bin/python -m pytest`; the last complete run passed848 tests in10m22s. Use a durable job
  with saved exit status if a tool session can terminate long processes. Do not count
  partial logs or a dead process as a pass; do not run duplicate suites unnecessarily.
- Update the owning plan, compact project state and scoped architecture atlas.
  Record research-significant findings using the existing research-manager workflow.
  Preserve unresolved failures; green labels require specifically scoped evidence.
- Commit completed work. Final delivery names what works, what failed, limitations,
  exact checks and report/reproduction locations. Do not silently narrow “complete
  architecture” to another protocol-only demonstration.
