# Architecture continuation with Claude Opus 5.5 medium

Prepared 24 September 2026 for a new Codex session in
`/home/alex/Documents/path-wm`, branch `main`. Completed implementation commit:
`b51d949`; preceding plan/red-test commit: `3371600`. The working tree was clean
before this documentation handoff. No training or test job remains running.

## Paste into the new Codex session

> Continue PATH-WM from `docs/architecture-continuation-handoff.md`. Read CLAUDE.md
> and the current project state. Work with actual Claude Opus 5.5 at medium effort
> through multiple substantive design, implementation and review rounds. Preserve
> the public-only external review boundary. Inspect the existing implementation,
> plan the remaining points in small measurable slices, implement and test each
> selected slice, review and repair it, and continue until its declared gates pass
> or evidence identifies a concrete blocker. Preserve failed results. Do not claim
> the overall architecture is complete from the bounded RGB16/episode experiment.
> Start by selecting the next learning experiment with Claude; keep reconstruction,
> evidence correction, useful latent representations and real compute cost explicit.

This handoff records the existing continuation authorization. Do not start a new
architecture from scratch or ask again whether ordinary implementation/review is
wanted. A new costly external training service or expansion of the export boundary
is not authorized by this document.

## Read first, then only the relevant owners

1. [Current work](project-state.md) and applicable sections of
   [experiment workflow](experiment-workflow.md).
2. [Evidence-loop plan and results](evidence-loop-plan.md): completed slice,
   registered gates, failures, repairs, commands and limitations.
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

## Saved evidence and failures

- Full suite: **838 passed in660.49 seconds**. Log and exit receipt:
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

1. **Memory → active context → useful learned response.** Close selection, capacity,
   eviction, reset and resumption under the existing TaskPolicy ownership. Start
   with a fixed bounded retrieval baseline, then one learned selection mechanism.
   A good next candidate is a small grounded text task over episode/entity evidence:
   delayed questions, distractors, corrected details, interruption and resumption.
   Exact phrasing/numbers must remain retrievable. Compare fixed versus learned
   selection and flat versus Local/Global views at matched storage/read/compute
   budgets. Measure answer correctness, stale-source use, omitted-detail errors,
   retention and latency. Templates may supply a baseline or training labels;
   distinguish their performance from learned language. Predeclare numeric gates.
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

## Actual Claude collaboration

Use **`claude-opus-5-5` with `--effort medium`**, not the default model or a Codex
subagent renamed Claude. The previous installed CLI was `/home/alex/.local/bin/claude`
(version2.1.280). Verify availability and record the actual returned model usage.
If unavailable, report the specific problem rather than silently substituting.

Five prior calls used session `3827870d-78ce-4c88-be5d-ebf8b1c8cf26`.
Exact briefs, replies, receipts and `invoke.py` are in
`runs/reviews/architecture_completion_20260924/`; aggregate `claude-receipt.json`.
Use these as a pattern, not as permission to send this private handoff externally.

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
  `python -m pytest`; the last complete run took about11 minutes. Use a durable job
  with saved exit status if a tool session can terminate long processes. Do not count
  partial logs or a dead process as a pass; do not run duplicate suites unnecessarily.
- Update the owning plan, compact project state and scoped architecture atlas.
  Record research-significant findings using the existing research-manager workflow.
  Preserve unresolved failures; green labels require specifically scoped evidence.
- Commit completed work. Final delivery names what works, what failed, limitations,
  exact checks and report/reproduction locations. Do not silently narrow “complete
  architecture” to another protocol-only demonstration.
