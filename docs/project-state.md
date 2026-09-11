# Current work

**Shared entity reader succeeds, 11 September:** the Claude-reviewed
`--entity-reader shared --entity-association observed` path passes all declared
development gates after 256 updates: 100% identity/state/effect accuracy on 128
identifiable cases and correct bounded ambiguity handling. All 96 CPU tests pass.
Predicted probabilities are unchanged under all eight per-frame reorderings after
undoing output order. It uses 30,021 parameters and 128 persistent state floats.
[Report](../runs/entity_shared_v1/reference/report.html) and
[task evidence](entity-learning-task.md) retain limitations: association, two object
slots and the finite hypothesis set are supplied; graph learning remains unimplemented.
Next proposed: learn association while preserving this working reference.

**Supplied association complete, 11 September:** the Claude-reviewed
`--entity-association observed` diagnostic is implemented; 94 CPU tests pass.
Matched 256-update development identity accuracy improves to 88.3%, but state/effect
remain 64.8%/57.8% and combined gates fail. The trained recurrent reader changes its
probabilities under candidate reordering. This is a useful diagnostic gain, not
reliable binding or learned graph structure. See the
[updated task record](entity-learning-task.md) and
[report](../runs/entity_alignment_v1/observed/report.html). Next proposed diagnostic:
shared per-entity updates and permutation-consistent readout.

**Entity baseline built and evaluated, 11 September:** `--dataset entities` now runs
the [controlled two-object task](entity-learning-task.md) through the existing recipe.
All 92 CPU tests pass. The fixed 256-update baseline reaches development identity/state/
effect accuracy of 60.2%/62.5%/54.7%; every combined development gate fails. Training
state accuracy reaches 100%, but identity is only 66.8%. The graph comparison is
deferred under the declared stop rule. Source, cached resume, oracle, raw scores and
browser report checks are recorded in `runs/entity_learning_v1/verification.json`.
[Open the report](../runs/entity_learning_v1/reference/report.html). Next diagnostic:
separate descriptor association from state updating; no extra run is authorized by
this result, and the earlier entity-reader bottleneck remains unresolved.

**Task definition complete, 11 September:** [the first entity-learning task](entity-learning-task.md)
defines two-object identity persistence, state updates and action-effect prediction.
Both graph structure and latent values are intended to be learned; readable labels
are inspection aids, not imposed semantics. Two actual Claude reviews are reconciled.
A finite 512-case specification check confirms an exact history oracle and final-view-only
bounds of 50% identity and 25% state-pair accuracy. These are contract checks, not model
results. Next implementation step is the task generator/reference path and existing-reader
diagnosis, followed by a bounded recurrent baseline before any graph comparison.
No new neural training or graph implementation was started in this definition slice.

**Entity-design discussion, 11 September:** the user accepted diagnosing accessible
identity first, then testing controlled two-object binding. Subsequent discussion
proposes per-entity learned beliefs and external retrieval. Two actual Claude
exchanges are reconciled in [the design note](entity-memory-design.md): candidate
extraction, uncertain association and persistent keys are separate mechanisms.
Explicit entity storage remains a proposal; no new architecture or run was started.
The follow-up runtime/payload review adds four reconciled Claude exchanges: small
read/propose/commit interfaces, explicit gradient boundaries, and optional raw,
latent or readable payloads. Recalled media may reuse modality encoders but cannot
silently enter as new observations. Source IDs, historical time and typed revisions
remain distinct; a memory-origin label alone is insufficient.

**Current slice complete, 11 September:** the
[trainable encoder initialization comparison](warm-encoder-plan.md) is implemented
as `--fact-encoder-weights` for the event fact reader. Two brief actual Claude
reviews are reconciled; all 88 CPU tests pass. Exact transfer, unchanged remaining
initialization/RNG, encoder updates and donor-bound resume checks pass. Both new
512-update recipients finished in 103.5041 active CPU seconds total. At lr0.0003,
held-out location accuracy improves from the saved cold run's 8/32 to 31/32; at
lr0.001 it improves from 24/32 to 32/32. Held-out entity and joint accuracy remain
0/32 in both warm runs. Training entity accuracy is only 3/96 in each. Both
extraction gates fail; binding is skipped and the declared two-run slice is complete.

The [reference report](../runs/warm_encoder_v1/reference/report.html) and
[learning-rate comparison](../runs/warm_encoder_v1/lr_control/report.html) passed
structural and 1280x720 browser QA. `runs/warm_encoder_v1/verification.json` binds
raw-score, source, donor, matching cold settings/sampler, cached-resume, test and
browser checks. Development combinations are reused; the donor adds 512 upstream
updates / 8192 presentations. The shared encoder also reads the fixed instruction.
These results leave entity learning unresolved without isolating its cause.
Next proposed: freeze the donor encoder in one otherwise matched diagnostic to
test whether preserving its features changes entity learning.

**Prior single-event slice, 11 September:** the
[single-event agent-reader control](event-fact-plan.md) is implemented as
`--dataset facts --fact-reader event`. Two brief actual Claude reviews are
reconciled; all 86 CPU tests pass. A strengthened gradient check confirms the loss
reaches the observed fact, and exact resume preserves cached outputs. Two matched
512-update runs took 109.4034 active CPU seconds total. At lr0.0003, held-out entity/
location accuracy was 3.125% / 25%; at lr0.001 it was 0% / 75%. Joint accuracy was
0/32 in both runs. Both extraction gates fail and binding evaluation is skipped.
No third training run, new objective or memory change was started.

The [reference report](../runs/event_fact_v1/reference/report.html) and
[learning-rate comparison](../runs/event_fact_v1/lr_control/report.html) passed
structural and 1280x720 browser QA. `runs/event_fact_v1/verification.json` binds
source snapshots, settings/data/init/sampler matching, raw-score checks, exact
resume, tests and screenshots. The existing whole path does not learn the task
under this budget; the failing component is not isolated. Recent records are
detached on storage by the existing memory policy, while the live categorical
path still carries gradients. The trainable encoder initialization follow-up is
now complete above.

**Prior direct-control slice, 11 September:** the
[direct fact extraction and binding controls](fact-learning-plan.md) run in the
same recipe (`--dataset facts`). Two short actual Claude reviews are reconciled.
All 83 CPU tests passed; 12 focused fact/run checks passed after the report revision.
The first 512-update reference passed every declared gate in 10.3502 active CPU
seconds: entity, location and joint accuracy are 100% on 96 training and 32 held-out
combinations. Mean held-out NLL is 0.211649. The fixed selector answers every
enumerated two-record query correctly, including 768 queries / 384 pairs whose
constituent facts are both held out. Coherent location swaps also pass. These
reused fact combinations are not independent samples or natural-language evidence.
No second learning-rate run was needed.

The [standalone fact report](../runs/fact_grounding_v1/reference/report.html) has
verified tables, curves and expandable examples at 1280x720, with no broken images
or horizontal overflow. `runs/fact_grounding_v1/verification.json` binds the intact
training snapshot, separate final renderer, checkpoint/results, tests and browser
receipts. CLI resume reused cached predictions and unchanged result/metric files;
an independent probability-space calculation matches every saved binding score.
Next proposed: test the same factual task through the existing agent event/reader
path. Success of this freshly trained encoder and explicit selector does not prove
the world model's learned binding, recurrent retention or memory compression.

**Prior diagnostic slice, 11 September:** the
[current/recent factual recall diagnostic](recall-learning-plan.md) is implemented
in the same recipe (`--dataset recall --recall-mode current-recent`). Actual Claude
completed one review and one reconciliation after Alex approved the export. All 79
CPU tests pass; the actual diagnostic forward/backward check has finite losses and
the expected gradients. The 256-update CPU pilot finished in 144.58 active seconds.
The final checkpoint gets 71.875% seen-location accuracy on its 40 training episodes,
but only 17.5% on the 80 seen cases among 100 fresh development episodes: current
12.5%, recent 22.5%. Both declared gates fail. Overall development task loss is 0.31
at 12% coverage, worse than always abstaining (0.25). No calibration/test data was
loaded and no new memory loss or budget extension was started.

The [standalone report](../runs/recall_diagnostic_v1/pilot/report.html) passed
structural and 1280x720 browser checks. `runs/recall_diagnostic_v1/verification.json`
binds source/results/checkpoint/report hashes, tests and screenshots. Next focus is
basic entity/location binding and generalization; this pilot does not isolate the
input encoder, query binding, readout or insufficient optimization budget.

**Prior recall slice:** historical recall is implemented in the same recipe. Read the
[usable guide](recall-task.md) and [implementation record](recall-implementation-plan.md).
All 76 CPU tests pass, including exact pause/resume, held-out split isolation and
report-failure recovery. A 256-event default-memory forward/backward check completed.
The eight-update CPU development run selected update 7. On its 15 test episodes,
factual accuracy is 20%, every decision abstains, and task loss is 0.25. The model
does not yet demonstrate useful recall. Calibration selected its upper bound T=20;
this small sample establishes no calibration guarantee.

**Overnight work complete:** all seven topic reviews and their combined reconciliation
with actual Claude are complete. The [decision agenda](remaining-decisions-2026-09-10.md)
starts with five choices and the full design, followed by the detailed alternatives,
interfaces, evidence and remaining empirical questions. Recommended next: grounded
local memory learning with current/recent recall controls, then learned marking if
its marginal signal is useful. Preserve fixed capacity, exact historical records and
separate live, hypothetical and offline replay state. Freeze the whole selected
procedure before final calibration; each later task configuration needs its own evaluation.

The final consistency check completed on 10 September at 08:31 Berlin, before the
09:00 deadline. All 21 overnight Claude responses succeeded; saved evidence and source
identity are unchanged. The heartbeat `overnight-agent-design-proposals` is paused
with its persisted status verified. These proposals remain unadopted. Receipts are
`overnight-integration-local-checks.json` and `morning-handoff-check.json` under
`runs/reviews/state_memory_design_2026-09-09/`. No model run or test suite was repeated
for the morning handoff.

**Implemented:** the categorical belief and bounded session-memory design authorized
by Alex on 9 September 2026. Read [the model guide](belief-model.md) and the
[implementation plan/record](belief-implementation-plan.md). One editable
[recipe](../experiments/multimodal.py) still owns construction, targets, training,
evaluation, checkpoint/resume and reporting. No second trainer or runtime LLM was added.

The CLI defaults to `--state-model belief`: recurrent context, grouped categorical
prior/posterior, source-only evidence and a separate task workspace. Ordered event
transactions advance executed action/time once, merge partial packets against a
fixed prior and memory snapshot, and commit once. Shared dynamics power imagination.
Thinking/reflection leave the physical belief and observational history unchanged.
Both evidence age and inferred-state age/ordinal constrain causal memory reads.

Memory contains exact recent latent envelopes, chronological compression staging,
compressed history, protected user/agent marks and gated consolidation. Separate
perception/prediction/thinking readers have learned scale gates and a null choice.
Full categorical probabilities reach readers and compression. Source-only features
remain independently encoded and compressed. The default tensor payload is bounded
at 318,040 bytes per FP32 stream, plus bounded metadata and temporary computation.
Fresh `initial_state()` starts empty memory and workspace with the same model weights;
callers discard prior task progress and plans. Individual memory resets remain withdrawn.

Learning now includes observable reconstruction/prediction likelihoods, split
categorical KL, isolated full/partial teacher targets, delayed recall, frozen-reader
compression distillation and delayed marginal mark utility. These mechanisms are
implemented; effective long-horizon memory and calibrated uncertainty are unproven.
The ordinary two-step default history is too short to train delayed recall across
the default 32-record recent store. The guide gives explicit small-memory settings
that exercise all memory scales within an eight-step development history.

**Earlier belief-slice verification:** 69 CPU tests passed at that stage. Exact
pause/resume reproduces model, optimizer, sampler, RNG and training rows, including the optional extra-update gate. New tests
cover event retry/order, masked inputs, source/belief separation, mixed batches,
full-distribution reads, memory bounds/consolidation, provenance, snapshot loading,
common planning samples/RNG restoration and future memory at equal timestamps.
Final real PushT forward/backward check is saved with the run receipts.

Three short CPU development runs completed: eight synthetic updates, eight
instruction updates and two real PushT updates; no extra proposals or GPU runs.
Every run has raw metrics, checkpoint, source snapshot, media and an offline report.
The current report renderer has now passed browser visual checks at a 1280×720
viewport, with no broken images or horizontal overflow on the three reports.
See [verification](../runs/belief_v1/verification.json),
[synthetic report](../runs/belief_v1/synthetic/report.html),
[instruction report](../runs/belief_v1/instructions/report.html), and
[real report](../runs/belief_v1/pusht/report.html).

**Learning remains weak:** synthetic held-out image MSE is 0.237670 versus 0.007451
for copying the last image. The tiny real check gives 0.236499 versus 0.000050895.
The four-example instruction evaluation has 75% operation error. These development
populations are too small and training too short for capability conclusions. They
confirm execution and expose poor current predictions, not successful world learning.
The earlier negative instruction result is unchanged and preserved in the historical
[task implementation record](multimodal-plan.md); its old run paths are unavailable
in this local checkout and were not reverified or reconstructed.

**Claude collaboration:** two actual isolated CLI exchanges reviewed abstract
implementation invariants. No private source, dimensions or results were exported.
Claude withdrew overbroad demands for parameter-disjoint encoders, hard gates,
mask tokens and a particular RNG mechanism. The adopted properties are forward
information separation, frozen event inputs, bounded replay graphs and side-effect
free planning. Remaining fixed/variable-rollout and sealing concerns were resolved
by the concrete fixed-horizon interface, open-event type checks and local tests.
Receipts: `runs/reviews/state_memory_design_2026-09-09/implementation*`.

The Gaussian reference, existing task controls/output attribution and multiscale
adapters remain usable. Python `build_model()` retains its Gaussian default; use
`state_model="belief"` explicitly in code. Its SVG diagram exporter still describes
the Gaussian reference; the new guide has the categorical flow. Checkpoint schemas
are distinct, with no implicit Gaussian conversion. Completed runs retain their
own source snapshots and must be resumed with compatible source/settings.

**Active design proposal:** [decisions from belief and memory](decision-design.md)
connects the task workspace and planner through an exact objective/cost contract,
a learned observable-outcome head, and verification of actual results. This is the
Claude-reviewed discussion requested on 9 September, completed on 10 September;
it is not implemented or a new capability claim. Historical last-observed recall
with a separate factual not-observed answer and operational abstention is the
recommended first slice. Active current-location inspection is a subsequent task
requiring observation-conditioned continuations and isolated hypothetical updates.
Existing memory bounds, reset decisions and negative results are unchanged.

**Implemented first-task contract:** [selective historical recall](recall-task-design.md)
specifies four locations plus factual not-observed, separate abstention, loss 0/1/0.25,
two retrieval rounds, complete text-observation episodes and independent calibration.
Two further abstract Claude reviews reconcile cost/calibration and data-split claims.
Report seen/old-history performance separately: recognizing only unseen entities can
beat all-abstain without remembering any locations. The subsequent implementation
and development evidence are recorded in the guide linked above.

**Next step:** declare one bounded input/binding/readout diagnostic before any new
memory-learning objective; preserve the failed current/recent pilot. Numerical
costs are explicit research defaults, not inferred application preferences. Deployment-length replay,
general mark selection, calibrated probabilities and closed-loop behavior remain open.
