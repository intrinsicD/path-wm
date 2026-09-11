# Current work

**Active slice, 11 September:** implement and test the
[direct fact extraction and binding controls](fact-learning-plan.md) in the same
recipe (`--dataset facts`). Alex authorized implementation, actual Claude review,
testing, fixes and bounded iteration. Two brief conceptual reviews are complete;
local checks and the predeclared CPU pilots follow.

**Current slice complete, 11 September:** the
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
