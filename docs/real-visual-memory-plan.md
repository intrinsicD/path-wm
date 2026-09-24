# Real visual-memory connection

24 September 2026. Alex authorizes Codex and actual Claude Opus 5.5 at medium
effort to plan, implement, review, test and fix the agreed first actual-model slice.
Claude has the same repository permissions for this task; this explicit instruction
supersedes the older public-only review boundary. Preserve existing architecture,
checkpoints and historical evidence. Other rerun families remain separately open.

## Objective and working plan

Make the existing native R2 visual path rerunnable through persistent memory:
actual RGB64 -> shared MultiScaleImageEncoder -> existing SlotPerception ->
persistent native visual slot -> existing BroadcastDecoder. Keep lossless frame
evidence and expose full-pyramid rederivation under the recorded perception version.
No duplicate pyramid storage, new codec, encoder or learned representation. A slot
is the existing compressed consumer input, not a replacement for full source detail.

First establish exact persistence, ownership/provenance, source correction,
unaffected-record preservation, restart and output equivalence to the live path.
Use the existing J perception/identity checkpoint and native width64, seven slots,
three iterations and decoder width32. Do not count agreement with a lossy live
reconstruction as proof of pixel fidelity; report source reconstruction error
separately. Train existing exercised components only if the declared quality task
requires it, preserving existing abilities and reporting failures.

Trace and reconcile the exact persistence/correction contract with Claude before
implementation. Granularity must follow actual feature dependencies: global
attention makes arbitrary spatial token replacement unsafe. Frame/source-level
correction must not be reported as object-part correction or unseen-operator
transfer. Pin model/source versions and reject invalidated retained reads.

## Measurable slices

1. Joint code-grounded plan, exact APIs and red regression tests.
2. Small missing connection in the existing library/session path, full-size tests.
3. Existing-checkpoint evaluation of actual live versus recalled output, source
   correction, independent unrelated source, restart and negative controls.
4. If needed, bounded training of existing modules and held-out reevaluation.
5. Independent Claude review, fixes, focused and full CPU suite, retained evidence,
   standalone report and updated continuation state.

## Initial budget and gates

Planning/software checks: native CPU configuration, two threads. One initial
checkpoint evaluation on fresh synthetic scenes, <=5 minutes CPU or GPU. No
scientific tuning on held-out examples. Training, if needed, gets its exact
objective, disjoint streams, preservation gates and bounded update count registered
before execution. Maximum initial training allocation: one GPU, <=6GiB, <=20 minutes;
no automatic sweep or architecture addition. Source stays fixed during every run.

Persistence gates: all scales and metadata survive exactly; same consumer output
within declared execution precision; stale/retracted/model-mismatched reads fail;
independent records remain unchanged; restart retains the same results; no weight
updates during memory acquisition/correction. These establish the connection,
not fine-detail quality or the full agent. Quality gates will be specified before
the first new quality evaluation after tracing the existing task and checkpoint.

## Status

Claude round1 completed on actual `claude-opus-5-5`, explicit medium effort.
Implementation round underway; no new quality result yet.
Exact briefs/replies: `runs/reviews/real_visual_memory_20260924/`.

## Reconciled implementation and evaluation protocol

The code trace found that per-instance `appearance` is a32-value identity key,
not decoder input. `_restore_appearance` will additionally commit the existing
64-value slot with the same recognition parent and source frame evidence.
`visual_memory(instance)` and `render_memory(instance)` validate the newest record,
source/lineage and perception version. Invalid newest records return unknown/error;
**no automatic fallback** to an older active sighting. Claude proposed such fallback;
Codex rejected it because it violates the existing source-correction contract.
`source_pyramid(instance)` explicitly recomputes the full native pyramid from the
verified retained frame; it is not latent-only recall or duplicate persistence.

Whole-pyramid token splicing is excluded because of global attention. Identity
reassignment/retraction follows existing WorldStore parents; frame withdrawal
invalidates all records depending on that frame. A new sighting updates the current
record while old stored tensors remain immutable. Single-slot alpha is a logit,
not a normalized scene mask. Output retains the remembered position. No pose transfer,
fine-detail preservation or nonlinear-operator generalization claim is made.

Evaluation before observing results: native R2/J, CPU FP32, two independent seeds
(2401,2402),32 fresh validation scenes each (64 total), validation machine kinds,
no ground-truth identity supplied to the agent. Rendering masks and lamp targets
are evaluator-only. For each scene observe, recall both learned machine identities,
flip one actual lamp, observe again, check the latest record and frozen old records,
save/restart and withdraw corrected-frame evidence. Missing identity is a failed
case, never silently omitted. Recipes keep raw per-scene rows and initial/final
weight hashes. Reuse the report renderer with actual source/reconstruction examples.

Hard software gates: acquisition for both inferred machine instances; exact stored
slot/live-slot and independently recomputed pyramid equality on CPU; exact
same-shape decoder output and restart, with cross-batch differences diagnostic; record immutability;
restart equality; retracted current reads rejected; no weights changed. Report
learned acquisition/matching and lamp accuracy across ALL scenes, with a development
screen of >=95% for each. Report source reconstruction MSE, machine-region MSE and
wrong-memory control separately, without inventing a new pixel-quality threshold.
The earlier0.002 surrogate threshold remains unqualified for this path, not relaxed.
A prior J MSE of0.0105 is an observed error, not a ceiling or a lower bound.

No initial training: the missing connection has no learned parameters, and the
exercised perception/decoder already have J weights. If a declared learned screen
fails, diagnose using saved development evidence, preregister a bounded repair on
disjoint training scenes, then evaluate fresh held-out scenes. Other untrained R2
modules are outside this path and cannot be credited as trained by these tests.

## First evaluation failure and bounded repair protocol

The default-binder seed2401 run completed but failed (acquisition/matching82.8125%).
Preserve the raw run/report. Two distinct problems must not be conflated:

- Initial output parity used absolute1e-6 across convolution batch sizes1 and7.
  Independent same-source diagnosis shows exact stored slots and exact same-shape
  decoder outputs; changing only batch shape produces RGB differences1.8e-6–4.0e-6
  and alpha-logit differences6.1e-5–3.1e-4. Fix the correctness comparison to use the
  **same decoder operation and same shape on the live slot**, with exact equality.
  Continue recording cross-batch maximum differences diagnostically. This is a
  correction of the numerical test, not a relaxed pixel-quality gate.
- J's identity key is machine-trained, while the existing candidate adapter scores
  background and object slots in that same key space. The binder also has generic,
  uncalibrated thresholds. Review candidate scope with Claude; do not silently force
  supplied keys, change the evaluation population or lower the95% learned gates.

Proposed bounded calibration (registered before collecting calibration scores):
128 fresh TRAIN-kind scenes, seed3401, native J perception/key, two lamp states per
scene, no network updates. Use true machine pixel labels only to score same/different
pairs in calibration, never to choose runtime candidates. If the minimum same-instance
score exceeds the maximum different-instance score, let gap be their difference;
choose new_threshold=max_different+gap/3, match_threshold=min_same-gap/3 and
margin=gap/3. Otherwise stop this calibration with a failed result; do not search
thresholds on evaluation data. Save all scores, settings, checkpoint hash and the
resulting policy manifest. Runtime eligibility, if corrected, uses the existing
learned kind head, not hidden labels. Freeze settings before evaluating fresh
validation seeds2403 and2404 (32 scenes each), unchanged95% gates. Budget<=5 minutes
CPU calibration plus<=5 minutes for the two evaluations. Baseline2402 also remains
part of the original two-seed baseline. Other policy/training changes need a new
specific diagnosis and registered bounded step.

Claude round3 found and reproduced a stale-sighting bug: re-deriving an old
recognition could give its slot a newer commit timestamp than a later observation.
The reader now orders by the recognition source's occurrence/availability times,
then derivation revision and numeric operation index, including inactive records.
It still refuses an invalid newest sighting. A red/green regression exercises the
real reassignment-away-and-back path. The default generic all-slot candidate path
is unchanged; J identity-run candidates now require the existing learned kind
head to predict machine. No hidden labels or forced two-slot selection is used.

Before calibration execution, extend the paired training view to a fresh layout
of the same two machines plus the lamp change (128 scenes still). Fresh validation
2403/2404 additionally includes a third new-layout observation, with machine side
swaps on alternate cases. Match is scored against the original corresponding
instance, not membership in an unordered ID set; missing acquisition stays in the
denominator. Add unchanged95% cross-layout match/lamp gates. Retraction now acts on
a restored **live view**, not a session whose view has never been resumed. Full
pyramid parity uses an independent encoder forward, including after restart.

Claude's proposed shared-session pre-scoring exclusion and broad threshold grid
are deferred: neither is required to implement this key-domain correction, and
changing duplicate-candidate semantics would expand scope. First try the registered
closed-form calibration; preserve a failure rather than tuning on validation.

The strict min/max calibration failed on TRAIN data (same minimum0.7665 versus
different maximum0.9333; all machines detected). No manifest was exported, and no
validation threshold was chosen. Adopt Claude's finite-policy calibration suggestion
as the next bounded repair, preserving that failed run: eight existing binder
settings, match in{0.80,0.85,0.90,0.95}, new=match−0.05, margin in{0.05,0.10}.
Use seed3402,64 fresh TRAIN scene pairs with layout/lamp changes. Run every setting
through the **actual native R2 WorldSession/UnifiedAgent**, not a substitute score
simulator. Candidate scope remains the learned machine classifier. Choose the
setting maximizing min(acquisition, correct identity match) among settings with
false matches<=0.5%; ties prefer fewer false matches then larger match/margin.
Export only if both acquisition/matching>=95%, false matches<=0.5%, and weights
remain unchanged. Ground truth is used only for scoring. No neural optimization,
no threshold adjustment to validation. Budget<=10min CPU, two threads. The fresh
validation2403/2404 protocol and gates stay fixed.

Round4 review required three further corrections before crediting identity:
explicit candidate scope (default `all`, optional `predicted-machine`, recorded and
restored with models), nonempty generic key-adapter assertions, and a novel-machine
arrival to detect false merges. The3402 grid completed (99.21875% acquisition/match,
no swaps on its two-machine task), but its false-match check did not test novelty;
that manifest is retained as historical and is not the final policy.

Final calibration protocol: same eight settings,64 fresh TRAIN pairs seed3403;
add a third observation replacing one machine with a train kind absent from the
first pair. False merge means any old instance assigned to the new machine.
Require novel slot detection>=95% and false merge<=0.5% in addition to prior gates.
Fresh validation2403/2404 gets the same novel-arrival probe on a separate continuation
from the saved session; restart/retraction checks still use the pre-arrival snapshot.
No validation outputs have been inspected for this policy selection. Budget<=10min
CPU. The v2 manifest binds completed passing result/policy hashes, recipe/library
source hash, checkpoint and actual perception/key state hashes; loading verifies all.
Candidate selection is now explicitly configured; the earlier integrated rerun and
defaults baselines used all seven slots. Generic defaults remain unchanged.

Known remaining scope: withdrawing a frame invalidates visual/appearance records
and the live view, but the older candidate->frame identity lineage is incomplete;
this slice does not establish retraction of all historical recognition/state claims.
No claim of complete agent-wide correction is made.

The frame->identity gap from round4 is repaired in the R2 `agent.correct` path,
not left as a claimed complete correction: a camera-frame withdrawal now atomically
withdraws the slot-candidate evidence emitted from that frame/event, so existing
WorldStore invalidation reaches recognition and descendants. Generic WorldSession
semantics are unchanged. A red/green test and evaluation check verify identity
withdrawal too; unrelated earlier evidence stays unchanged. Direct low-level store
transactions do not infer this R2-specific dependency and are not this API.

Round5 found a retry bug in the new frame-withdrawal expansion. Fresh retries
now reproduce candidate retractions from that same correction event and skip only
ones withdrawn by a different event. Two red/green cases cover fresh-transaction
and same-object retries, including earlier independent candidate withdrawal.
Novel-slot detection is a perception gate; novel identity acquisition is reported,
not gated (abstention allowed). After a withdrawn newest frame, affected instances
are not automatically re-identified; later sightings create new entities. No
identity replay/reconciliation claim is made; this fail-closed behavior is tested.

The pre-fix full suite was intentionally interrupted after review found the retry
bug (no test failure before interruption); it will be rerun completely. Calibration
3403_v1 passed with100% acquisition,98.4375% matching and0 observed false merges,
but its source-bound manifest is now historical. Reproduce the SAME registered
training population/seed3403 and eight policies under final source in3403_v2;
this avoids selecting a more favorable population. No validation scenes have yet
been used for selection. Then run2403/2404 as planned.

## Native validation and bounded key repair

The final memory software passes the complete CPU suite (925 tests; 689.17s,
source unchanged; `runs/reviews/real_visual_memory_20260924/full-suite-final.xml`).
Calibration3403_v2 reproduces the registered policy. Fresh validation2403 passes
(62/64 cross-layout identities);2404 fails (57/64), while every persistence,
correction, restart and frozen-weight contract passes in both. No quality gate
is relaxed; the learned-quality slice remains open. Reports and failed runs
are preserved. Normal R2 integration also passes4 scenes with exact restart.

Claude round7 traced the misses to absolute key similarities below thresholds;
the correct instance still ranks first. An absolute-cosine loss is a proposed
repair, not proof that a new loss is uniquely necessary. Keep J perception and
its decoder frozen; warm-start and train only its existing key head in the
existing perception recipe. Add optional frozen-perception/key warm-start and
cosine-margin training controls; no new model or trainer. Retain InfoNCE and
add squared hinges (positive .95, negative .50, weight1), masking duplicate
textures, including within-frame negatives. Defaults remain unchanged.

Preregister1000 updates, seed3501, native full configuration, lr3e-4, batch32,
J's texture randomization, <=20min GPU and6GiB. Use last checkpoint, no
validation-based selection. Train-kind monitor seed3502: positive q01>=.93,
negative q99<=.60, within-frame q99<=.60 and bit-identical perception; failed
screen is recorded before further planning. Calibration3404 uses64 train-kind
episodes including alternate side swaps, same eight policies and selection.
Fresh acceptance seeds2405/2406,32 scenes each, unchanged gates. Diagnosed
2403/2404 are secondary checks only. Claude owns recipe implementation/tests;
Codex owns calibration parity and independent review. Every downscaled new
check must also run with the actual full configuration.

Round8 pre-training review:72 native-training/adjacent regression checks pass.
Untrained J monitor fails both declared populations (positive q01 .888/.893;
negative q99 .712/.763). No new checkpoint is selected. Calibration now also
scores its same-layout lamp-change frame: selection maximizes the minimum of
acquisition, same-layout and relocated matching, with unchanged8 settings and
false-match/novel-merge constraints. This is fixed before calibration3404; hidden
labels remain evaluator-only. Round9 fixes empty-negative margin loss, monitor
violation metrics, objective disclosure and monitor resource checks before training.

Key repair3501_v1 completed1000 updates in79.44s, reserved1.3125GiB; perception
is bit-identical, source unchanged, report structurally verified. Its training
screen FAILED: procedural positive q01 .92468 / negative q99 .65804 / within
q99 .52226; kind-table .90057 / .69952 / .61469. No calibration or validation
was launched. This is evidence about one fit, not proof that the existing key
or representation cannot learn the task. Claude round10 reviews loss balance
and proposes the next bounded repair; full CPU suite remains source-frozen.

Round10 training-only gradient diagnostic (fresh seeds7777+, no updates) found
NCE key-gradient norms53–94 times larger than the W=1 hinge gradient. This
corrects the earlier claim that NCE gradients were negligible: hard negatives
keep them substantial. Preregister a matched single-factor W=100 run from J,
same seed3501/batches/1000 updates/lr/margins/monitor/ceilings, new output
`runs/real_visual_key_repair_3501_w100_v1`. No source or representation change,
no checkpoint selection, no gate changes. The W=1 run is a failed weakly
weighted intervention, not proof of impossibility. Claude's full diagnostic and
conditional next steps are in `claude-key-replan.md`; procedural-only training
versus kind-table evaluation is a separate hypothesis if W=100 fails.

W100 matched run completed105.39s, reserved1.3125GiB, unchanged perception/source;
training screen still FAILED: procedural(.92771,.66483,.47281), kind-table
(.90175,.72127,.62084), ordered positive q01/negative q99/within q99. Report
structurally verified. No calibration/evaluation. Weight-only repair is insufficient
in this budget. Joint round11 plans the next existing-key training-data repair,
separating procedural/fixed-kind distribution mismatch from optimization limits.

Round11 fresh train-only diagnostics show broader positive tails than the fixed
128-pair monitor, and weight100 took effect without solving that screen. No
fixed-texture-pool option, new holdout mode or further training is adopted. Codex
challenged whether the newly imposed cosine quantile cutoffs are prerequisites
for the original runtime task: they are not the binder thresholds or original
acceptance criteria, and optimizing them may add unnecessary work. Round12
reviews that distinction before any further run. Original failed screens remain
failed; original real-task acceptance thresholds are unchanged.

## Prospective protocol amendment: measure the actual task

Before any3404 calibration or2405/2406 measurement, Codex and Claude round12
agree that the added train-monitor cutoff is an unvalidated proxy, neither a
necessary nor sufficient condition for the binder's real task. The negative
cutoff.60 differs materially from actual match/new thresholds. Keeping it as a
prerequisite would encourage unnecessary training to satisfy an invented target.
The W1/W100 training screens remain FAILED and unchanged. Prospectively the
monitor is disclosed diagnostic evidence, not a prerequisite. This departure
from the earlier stop rule is explicit; it is not based on unmeasured runtime
success and does not lower any original real-task gate. No extra data mode or
model is added.

Fixed candidate: W100 final checkpoint; matched control: original J (not a
selectable alternative). Calibrate each once on identical seed3404,64 TRAIN
episodes, same8 policies/selection/false-match constraints. Only calibration
passes export a usable manifest. Evaluate passing arms on untouched2405/2406,
32 scenes each, unchanged runtime gates; W100 needs BOTH populations to pass.
No seed, checkpoint or policy shopping after measurement. If J also passes,
these populations do not establish that retraining was needed. If W100 fails,
its actual decisions guide the next bounded repair. Every result must disclose
the failed parent monitor and this amendment; do not say all gates passed.

The key-training and calibration-parity implementation passes the complete CPU
suite:942 tests,812.70s, zero failures/skips, source unchanged. The two completed
key-training runs retain failed diagnostic screens. Actual-task comparison is
next under the prospective amendment; no trained-task success claimed yet.
