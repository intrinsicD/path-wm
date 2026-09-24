# Architecture continuation with Claude Opus 5.5 medium

Updated 24 September 2026 for a new Codex session in
`/home/alex/Documents/path-wm`, branch `main`.

**Current continuation:** [Real visual-memory connection](real-visual-memory-plan.md).
Actual Claude Opus 5.5 at medium effort is the planning, implementation, review and
test partner. Native memory/training support is committed through `54117d7`; the
latest complete CPU suite passes 942 tests in 790.60 seconds with unchanged source.

Native memory contracts pass. Learned identity remains open: the fixed W100 key
candidate scored 59/64 and 63/64 on fresh cross-layout populations, failing the
requirement that both pass. Its stricter cosine diagnostic also remains failed;
the prospective stop-rule correction is fully disclosed. Joint perception/key
continuation passed matched retention but failed calibration on a similar-palette,
different-pattern new machine. No richer representation or alternate codec is used.

The duration-only continuation completed3,000 updates and passed retention and
calibration3405, but failed both primary2407/2408 at60/64 relocated matches.
All persistence contracts pass; J control scores57/64 and56/64. The separately
registered next intervention increases existing identity-loss weight .2→1 at
1,000 updates from J, with unchanged architecture/data. Its run
`runs/real_visual_joint_repair_3501_iw1_v1` passes retention but fails calibration:
even .95 still merges1/64 novel arrivals. Claude is implementing the jointly planned
optional confusable training pairs in the existing recipe. Architecture unchanged;
retention and all known/fixed-fresh populations remain required. See the owning plan.
Earlier failed comparison: `runs/real_visual_memory_comparison_v1/report.html`.
Other rerun families remain open. Original pose diagnostics included out-of-domain
positions; corrected actual-domain evidence and Claude's interpretation correction
are preserved in `runs/reviews/real_visual_memory_20260924/`.

**Current request:** [Actual-model reruns and joint gap plan](actual-model-rerun-plan.md)
owns the audit. Prefer the real model; any downscaled check must be followed by the
actual configuration. Missing functionality must be planned with Alex before adding
it. Full-model reruns remain incomplete until the ledger and missing-path items close.

**User correction: work on the existing architecture.** The separate `DetailCodec`
was introduced for a small memory demonstration and then reused for transfer and
fidelity screens. It bypasses the main multiscale representation. Continuing to
repair that surrogate is not the selected architecture agenda.

First trace the existing multiscale representation, persistent memory, active
context and intended output consumers. Identify a concrete defect, missing
connection or unimplemented contract in that actual path. Reuse existing modules;
add code only where the identified gap requires it. The prior suggestion to choose
a new decoder or residual path for `DetailCodec` is withdrawn as the default next
step. Preserve its scoped evidence; do not repeat or expand those screens. Only bounded training of the existing perception/key in the owning plan is selected;
no alternate architecture is selected by this handoff.

## Completed surrogate diagnostic: nonlinear fidelity (24 September)

[Registered diagnostic and results](nonlinear-fidelity-plan.md), implementation
`04650b8`, registration/red checks `f751a69`. Two codecs × two fresh populations
show optimal numerical affine-decoder MSE0.0308–0.0344 against gate0.002. Raw
support-fitted pixel threshold passes; target encode/decode and latent inference
add error. Old outputs/weights and exact map/threshold reload are preserved.
No neural training; the overall architecture remains unproven.

The surrogate path stops at this concrete blocker. It does not diagnose the main
multiscale architecture: that representation was not used. Do not repair or expand
the surrogate by default; follow the user correction above.
The previous target encode/decode “floor” is a reference, not a general lower bound.
The new range conclusion requires full rank and independent QR/SVD/least-squares
agreement; it is numerical evidence, not an interval-arithmetic proof.

- Final artifacts: `runs/nonlinear_fidelity_final_s{17,29}_{926117,926129}_v1/`.
- Audits/reviews: `runs/reviews/nonlinear_fidelity_20260924/`;3,584 prediction
  images independently audited, separate-process target poisoning/order checks pass.
- Portable results: `ara/evidence/tables/nonlinear_fidelity_2026-09-24.json`.
- Four actual Claude Opus5.5 medium rounds preserve the public-only boundary.
- Final populations926117/926129 are consumed;926101 is development.
- Full CPU suite: **866 tests passed in 647.72 seconds, exit0; source unchanged.** Review directory owns log/exit receipt.

## Continuation result: frozen affine transfer (24 September)

The [representation transfer screen](representation-transfer-plan.md) now passes
its registered affine gates for two frozen codecs and two fresh populations.
Pixel-channel ridge is stronger; this is not learned induction or completion of
the architecture. The nonlinear diagnostic has poor output reconstruction.
The later nonlinear-fidelity diagnostic above localizes an output-range blocker;
continue from that result while preserving old detail and behavior. Do not repeat the completed affine or indexed screens.
Earlier component results below are historical evidence to reuse, not work to
repeat. Final seeds925117/925129 are consumed. The frozen full suite passed
862 tests in620.35s, exit0; source hashes unchanged.

### What the new session can rely on

- Two frozen codec seeds17/29 × fresh populations925117/925129 passed both affine
  operator gates. Latent MSE0.000209716–0.000256037; discrimination100%; error
  18.4–116.3× lower than the better copying/displacement baseline.
- Channel-shared pixel ridge reaches about1e-13 MSE and has a stronger task-matched
  prior. No latent advantage or learned induction is established. The operators
  are unseen within an affine hypothesis class, not unseen mathematical families.
- Nonlinear threshold diagnostic: latent MSE0.08308–0.08375, true-output codec
  reconstruction floor0.05093–0.05189, pixel ridge0.07596–0.07598. Discrimination
  nevertheless reaches100%; do not use it as a substitute for faithful outputs.
  Binary outputs differ from the smooth training distribution. No unique cause
  of the failure was isolated, and no nonlinear gate was declared or passed.
- Neural weights unchanged; familiar-pose reconstruction, correction, stale-source
  rejection, unrelated state and exact float64 map/session/context restart pass.
  These are integrity checks, not learned retention. Population tensor inference
  and one live-session demonstration per invocation have separate scopes.
- Independent audit:9,216 prediction/choice rows; separate-process target poisoning
  left all12 relation cells' maps/predictions unchanged. Four final invocations
  took10.98s total including imports/reporting; peak RSS579–582MiB. Existing codec
  training is a sunk cost, not free learning. Reports are structurally verified
  using the unchanged renderer; a contact sheet was inspected, no browser-QA claim.

### Artifacts to inspect, not rerun

- Protocol and exact results: [representation transfer plan](representation-transfer-plan.md).
- Four run directories: `runs/representation_transfer_final_s{17,29}_{925117,925129}_v1/`.
  Each owns `raw.pt`, `result.json`, frozen `last.pt`, source snapshot, `report.html`,
  timings and `memory/session.pt` plus `memory/context.json`.
- Existing codec inputs: `runs/evidence_loop_final_s17/last.pt` and
  `runs/evidence_loop_final_s29/last.pt`. Neither was updated in this slice.
- Review/audit directory: `runs/reviews/representation_transfer_20260924/`.
  Start with `raw-artifact-audit.json`, `target-isolation-audit.json`,
  `full-suite-exit.json`, `source-freeze.json`, `final-invocations.json` and
  `claude-summary.json`. Exact briefs, replies and runnable audit scripts remain.
- Preserved development: `runs/representation_transfer_smoke_v1/`,
  `runs/representation_transfer_dev_v1/`, `runs/representation_transfer_dev_v2/`.
  The single development repair changed the shuffle to whole-image derangement
  and added validation/report checks; thresholds and model settings stayed fixed.
- Portable evidence: `ara/evidence/tables/representation_transfer_2026-09-24.json`.
  N578 records the experiment; O402 is staged. Do not duplicate or promote them
  into a general capability claim. Broader O398 and earlier failures remain open.

`runs/` is ignored/local. A different checkout needs these artifacts copied before
using them; if absent, report the missing prerequisites rather than invent results.
`last.pt` here is an evaluation audit artifact, not a training-resume interface.

## Paste into the new Codex session

> Continue PATH-WM from `docs/architecture-continuation-handoff.md`. Read CLAUDE.md
> and the current project state. Collaborate with actual Claude Opus 5.5 using
> `claude-opus-5-5` at medium effort through substantive design, implementation,
> review and repair rounds. Preserve the public-only external review boundary.
> The indexed discovery and frozen affine-transfer screens are complete. Do not
> repeat them or add learned complexity to their solved baseline workloads.
> Work on the existing architecture. Trace its actual representation/memory/output
> path and identify a necessary fix or implementation gap before adding components.
> Do not continue developing the separate DetailCodec surrogate. Earlier,
> two frozen codecs × two fresh populations passed the affine gates, but pixel
> ridge is stronger and nonlinear threshold output reconstruction remains poor.
> Use the transfer plan's audits and earlier R1/CI1 failures to choose a smallest
> falsifiable non-affine/output-fidelity question with old-task preservation.
> Compare remembered examples with inferred shared structure; separate supplied
> identities, familiar transformations and genuinely new instances/relations.
> Reuse existing store, episode, context and model contracts. Plan, implement,
> review and test the remaining work in small measurable slices. Predeclare fresh populations,
> numeric quality/retention/reconstruction gates, local compute budgets and repair
> limits before running. Preserve failed results; freeze and verify each selected
> slice, then use its result to choose the next remaining priority. Continue until
> the selected gates pass or evidence identifies a concrete blocker. Keep runtime
> acquisition/correction without weight retraining and actual total compute cost
> explicit. Do not claim the overall architecture is complete from narrow tests.

This handoff records the existing continuation authorization. Do not start a new
architecture from scratch or ask again whether ordinary implementation/review is
wanted. A new costly external training service or expansion of the export boundary
is not authorized by this document.

## Read first, then only the relevant owners

1. [Current work](project-state.md) and applicable sections of
   [experiment workflow](experiment-workflow.md).
2. [Nonlinear output fidelity](nonlinear-fidelity-plan.md): completed decoder-range
   diagnostic and concrete blocker. Then [representation transfer](representation-transfer-plan.md): completed affine
   screen, nonlinear floor, controls, consumed populations and exact audit receipts.
   Read [indexed discovery](scan-discovery-plan.md) only if revisiting retrieval.
   Then the relevant portions of
   [context retrieval](context-retrieval-plan.md) and
   [evidence loop](evidence-loop-plan.md) for implemented contracts and visual limits.
3. [Architecture walkthrough](architecture-walkthrough.md), especially “Completion
   priorities after the concept and conversation discussion” and context ownership.
4. [Integrated latent-agent goal](integrated-latent-agent-goal.md) and
   [Claude collaboration workflow](claude-collaboration-workflow.md).
5. [Model guide](models.md), [experiment guide](experiments.md), and current source.
6. Before selecting the next experiment, read the relevant failure/decision sections
   of the [integrated architecture plan](integrated-architecture-plan.md), especially
   R1 transfer and CI1 (§22), and the [concept-learning review](latent-concept-learning-review.md).
   Consult [shared-abstraction specification](shared-abstraction-spec.md) if proposing
   concept codes. These are prior designs/failures, not a proven concept solution.

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
| `experiments/nonlinear_fidelity.py` | Frozen affine decoder-range oracle, support-only threshold/nearest/ridge controls, numeric cross-checks, exact reload and reports; completed negative fidelity diagnostic. |
| `tests/test_nonlinear_fidelity.py` | Four numeric, support-dependence, frozen-state and report-failure checks. |
| `experiments/representation_transfer.py` | Frozen-codec analytic map, copying/displacement/pixel controls, affine/nonlinear evaluation, separate live map correction/restart and report. No neural training. |
| `tests/test_representation_transfer.py` | Six numerical, query-isolation and source/context integrity checks. |
| `experiments/evidence_loop.py` | Ordinary train/calibrate/evaluate/resume recipe, persistence demonstration, controls and standalone report. |
| `tests/test_detail_memory.py`, `tests/test_episodes.py`, `tests/test_evidence_recipe.py` | 18 focused checks, including exact resume and stale/interrupted output rejection. |
| `pathwm/models/tasks.py` | Optional `ContextSelector` under TaskPolicy: supervised metadata ranking with explicit null; existing defaults unchanged. |
| `pathwm/world_state/context.py` | WorkingContext: bounded flat/Local–Global component pins, task ownership, epoch/version checks, eviction/reset and owner-bound portable restart. |
| `pathwm/data/episode_facts.py` | Controlled two-slot synonym task with supplied entity matching; no answer payload in ranker inputs. |
| `experiments/context_retrieval.py` | One ordinary train/evaluate/resume recipe; counted, canonical, rule, random and no-read controls; context interventions and separate live-episode integration. |
| `tests/test_working_context.py`, `tests/test_context_recipe.py` | 10 new checks; includes exact training resume, source races, context budgets and preserving result completion when rendering fails. |
| `experiments/scan_discovery.py` | Recipe-local exact index over existing store/context contracts; causal corrections, logical restart, scan/cache controls and explicit cost accounting. No trained model or library default change. |
| `tests/test_scan_discovery.py` | Eight checks: discovery, poisoned/stale references, inactive-head behavior, byte-exact keys, independent metering, replay/permutation equivalence and report-failure preservation. |

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

## Latest result: indexed discovery screen

[Registered plan and results](scan-discovery-plan.md). Fresh final seeds924317 and
924329 each have128 queries ×5 arms,256 initial records plus64 distractors,
32 source corrections and16 logical restarts.

- Index, cached index and reset index:100% exact status/text/source agreement.
  Every stale cached reference is rejected; all logical restart checks pass.
- Resetting cache changes0/128 answers per seed. No model was trained. This result
  supports stopping learned complexity for this exact-descriptor workload only.
- Full scan also answers100% and remains physically feasible. The first16 scan
  omits all queries in this deliberately tail-targeted diagnostic. Neither control
  establishes a learned advantage or the physical necessity of the candidate cap.
- Indexed p95 about0.950ms; cached1.340/1.293ms. Complete invocations7.884/7.845s,
  including report rendering. Peak RSS about523MiB; serialized index about20KiB.
- Candidate probes <=1 (index/reset) or2 (stale cache recovery), but authoritative
  validation scans **2H or3H historical headers**. This is not globally bounded or
  sublinear storage work. Internal transaction/replay loops are included in CPU/
  wall time, not public-read API counters. A head-index optimization is an optional
  engineering follow-up if larger workloads demonstrate a need.
- No reconstruction, concept induction, entity association, learned retention,
  language or natural-data capability is established by this screen.

Artifacts:

- `runs/scan_discovery_final_924317_v1/` and
  `runs/scan_discovery_final_924329_v1/`: raw rows, causal `last.json`, restart
  snapshots, source identities/snapshot, metrics, timing and standalone `report.html`.
- `runs/scan_discovery_dev_924301_v1/` and `_v2/`: preserved development/recheck;
  v2 namespaces source IDs by population too. No gate changed after test exposure.
- `runs/reviews/scan_discovery_20260924/`: public Claude rounds/receipts,
  `raw-artifact-audit.json`, independent causal replay auditor, focused/full test
  logs, `full-suite-exit.json`, source freeze receipt and documentation checks.
- Committed portable evidence: `ara/evidence/tables/scan_discovery_2026-09-24.json`.
  N577 records the experiment; O401 is staged, not a promoted general claim.

All1,280 final raw rows were independently audited against source-event replay;
seed populations are disjoint. Final source hashes stayed unchanged during the
856-test suite. Reports use the unchanged renderer with structural verification;
there is no new browser-QA claim. `last.json` is store/context state, not a model
checkpoint; this index-only recipe has no training/resume CLI.

## Earlier results: context/retrieval

- **848 tests passed in622.19 s**, exit 0, source unchanged throughout the suite.
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

1. **Useful representations and concept/instance/state learning — next priority.**
   Go beyond the completed analytic affine screen. Diagnose novel-output fidelity
   and preservation, then test useful shared non-affine structure with held-out
   instances and a separately declared novel relation/concept family. Distinguish
   recognition/association from supplied identities and exact descriptors. Compare
   remembered examples with inferred concept codes before adding machinery. Test
   counterexamples, correction, transfer to another use and frozen-weight runtime
   acquisition. Earlier R1 transfer and CI1 retention failures must guide task
   selection; successful graph writes or familiar-pose reconstruction do not solve it.
2. **Useful retrieval, retention and context control — still open beyond exact keys.**
   Counting solves the earlier lexical task; indexing solves the new exact-key
   screen. Keep the learned head optional. Reopen this question only for a declared
   workload requiring non-exact representations or a measurable dependency on
   retained context. Include strong indexed/count-based controls and separately
   assess discovery, retention and readout. Preserve exact strings/numbers,
   correction, delayed queries, interruption and restart. Test one learned decision
   under TaskPolicy at a time. Flat and Local/Global comparisons need matched
   storage/scan/read budgets and causal reset/swap/eviction interventions. Count
   preparation, validation, index construction, recovery and training costs.
   Grounded learned responses remain separate from validated template copying.
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

1. **Locate a concrete gap in the existing implementation with Claude.** Trace
   actual modules and tensor paths for multiscale features, memory and consumers.
   Separate implemented-but-unwired components from missing behavior and measured
   defects. Choose a smallest end-to-end check of the intended architecture, then
   repair only what it demonstrates is needed. The `DetailCodec` screen bypassed
   that architecture; its output failure is not evidence to replace or enrich the
   existing multiscale representation. Public review uses hypothetical contracts;
   private code and measurements remain local. No replacement toy model is needed
   merely to make a small measurable experiment.
2. **Register the slice, then add essential red checks.** Record a concrete user
   path, training signal, train/dev/final populations, held-out instances and relation
   splits, negative/shortcut controls, numeric reconstruction/task/retention gates,
   local wall/CPU/GPU budget and repair limit. Freeze final populations before
   exposure. Do not invent thresholds after seeing results. Extend existing recipes
   and ordinary modules; preserve one authoritative store and episode/clock owner.
3. **Implement and review the smallest complete path.** Test causal support use,
   source correction, representation/version compatibility, unchanged frozen
   parameters/buffers, retention and exact restart where applicable. Run tiny
   train/report/resume checks for trained paths, then a bounded development
   comparison. Obtain actual Claude criticism of public generic contracts; inspect
   private implementation and measurements locally. Reconcile withdrawn claims
   explicitly. Preserve failures and stop at the preregistered repair/budget limit.
4. **Freeze, evaluate and select the next priority.** Run untouched multi-seed
   evaluation and the relevant full suite on unchanged source; independently audit
   raw metrics and source identities. Every completed run owns its report. Record
   what passed and failed, update scoped evidence/atlas, commit completed work,
   and use the result to select the next remaining item. A null result is useful;
   it is not permission to manufacture benefit or call the whole architecture done.

Useful regression commands, only when relevant (new output directories):

```bash
.venv/bin/python -m experiments.scan_discovery --output runs/my_scan_check --seed 924300 --records 32 --queries 16
.venv/bin/python -m pytest tests/test_scan_discovery.py tests/test_working_context.py
.venv/bin/python -m experiments.context_retrieval --output runs/my_context_check --steps 6 --worlds 1 --stop-after 3
.venv/bin/python -m experiments.context_retrieval --output runs/my_context_check --steps 6 --worlds 1 --resume
.venv/bin/python -m pytest
```

Do not rerun completed experiments as a startup ritual. Their final populations
are consumed: nonlinear fidelity926117/926129; transfer925117/925129; indexed924317/924329; lexical624927 (`--final`);
RGB16 textures240927. Development fidelity926101, transfer925101, indexed924301 and tiny924300
are also known. A changed method needs
fresh preregistered populations. Exact resume requires compatible source/settings;
old runs are not permission to resume training under changed code. Register the
new slice's local compute budget; no future training run or mechanism was selected
by the completed screens. The previous transfer budget is exhausted as a study,
not a standing authorization to sweep new variants.

## Actual Claude collaboration

Use **`claude-opus-5-5` with `--effort medium`**, not the default model or a Codex
subagent renamed Claude. The last successful runner used `/home/alex/.local/bin/claude`.
Verify current availability and record the actual returned model usage.
If unavailable, report the specific problem rather than silently substituting.

Latest collaboration: **four actual Opus5.5 medium rounds**, session
`17aacb48-425e-40e7-8cc6-ce3abfde95f2`. Exact briefs, replies, `invoke.py`, per-round
receipts and `claude-summary.json` are in
`runs/reviews/representation_transfer_20260924/`. Actual returned `modelUsage`
names `claude-opus-5-5`; each request explicitly sets medium effort. Total wall
202.40s; cumulative API-equivalent estimate$0.5413478, not subscription billing.
Earlier index reviews are under `runs/reviews/scan_discovery_20260924/`, lexical
reviews under `runs/reviews/architecture_continuation_20260924/`, and evidence-loop
reviews under `runs/reviews/architecture_completion_20260924/`.
Use the latest runner as an isolation pattern, not permission to send this private
handoff externally. Retain returned usage fields verbatim; resumed-session totals
may overlap, so do not blindly sum them.

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
  `.venv/bin/python -m pytest`; the last complete run passed 862 tests in620.35s. Use a durable job
  with saved exit status if a tool session can terminate long processes. Do not count
  partial logs or a dead process as a pass; do not run duplicate suites unnecessarily.
- Update the owning plan, compact project state and scoped architecture atlas.
  Record research-significant findings using the existing research-manager workflow.
  Preserve unresolved failures; green labels require specifically scoped evidence.
- Commit completed work. Final delivery names what works, what failed, limitations,
  exact checks and report/reproduction locations. Do not silently narrow “complete
  architecture” to another protocol-only demonstration.
