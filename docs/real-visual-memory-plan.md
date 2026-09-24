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

## Current status

Native memory and optional training support are committed through `65f4380`;
956 CPU tests pass on that version. Persistence, correction and restart contracts
pass on actual native runs. Learned relocation is not yet qualified.

The first confusable-pair construction also failed calibration. Its corrected
mean-matched construction targets the observed border-pattern coverage gap without
changing the model or losses. Sixteen focused and 73 adjacent checks pass; the
full suite and the registered 1,000-update native fit are running. Acceptance still
requires retention, calibration and all fixed runtime populations below.

No architecture replacement or surrogate is used. Actual Claude Opus 5.5 at medium
effort is the planning, implementation and review partner. Exact receipts and
preserved failures: `runs/reviews/real_visual_memory_20260924/`.

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

## Actual-task comparison result (candidate not qualified)

Matched calibration3404 passes: W100 selects(.85,.80,.10), acquisition/same-layout/
relocated matching100%; Jselects(.90,.85,.10), acquisition100%, bothmatching99.22%.
All256 inputframes are byte-identical acrossarms. Reports include parent diagnostic
failure/amendment. Frozen-source snapshots retained for both calibrations and all
four evaluations. Raw-row audit independently reproduces all published metrics.

| Fixed arm | Seed2405 cross-layout | Seed2406 cross-layout | Overall requirement |
| --- | --- | --- | --- |
| W100 candidate |59/64 (92.19%), FAIL|63/64 (98.44%), pass|FAILED: both required|
| Jcontrol |55/64 (85.94%), FAIL|61/64 (95.31%), pass|not selectable|

All memory contracts, weight freezing and novel detection pass; falsemerges0.
W100 same-layout/acquisition/lamp metrics100%. Original gate95% unchanged.
Reports: `runs/real_visual_memory_comparison_v1/report.html` and individualrun
reports; existingrenderer, structuralQAonly. Improvement does not close the
slice. Round14 diagnoses the actualmisses and plans the next necessary bounded
training repair. No thresholds, checkpoints or seeds are reselected for this
failed comparison. No extra datasetmode/representation is adopted.

Final source including diagnostic disclosure passes942 CPU tests again
(790.60s, unchanged source). Metadata-only change reviewed clean by Claude
round13; calibration manifests and allreports retain exact parent hashes.
This software pass does not change the failed W100 two-population task result.

## Joint continuation for observed stripe-phase sensitivity

Round14 identifies every remaining W100 miss as a below-threshold correct
identity (not a wrong match), concentrated on validation stripe kinds49/50.
Controlled1–3px shifts alter keycosine strongly while slotcoverage stays>=.95.
This is evidence for learned phase sensitivity, not proof that the architecture
cannot work. More key-only fits or a new data framework are not adopted.

Preregister `real_visual_joint_repair_3501_v1`: existing J perception+key, native
fullconfig, seed3501,1000updates,lr3e-4,pairedproceduralrate1, identity joint,
margin(.95,.50,W100),<=20minGPU/6GiB. Lastcheckpointonly. Single intervention
relativeW100: permit existingperception weights to learn under its existing
RGB/mask/kind/attribute/lamp objectives plus identity. No modules added, no
encoder/decoder replacement. Existingrecipe supports this without sourceedits.

The earlier bit-identical-perception guarantee ends for this separate candidate;
J/olderruns remain unchanged and new stores use the new perceptionversion.
Retention beforecalibration: identical256 TRAIN-kind scenes,seed3506, J vsjoint:
RGBMSE<=1.10xJ; pixel/mask, eachattribute, machine detection, lamp and both
pointer accuracies>=J-.005; kindaccuracy>=J-.001; existing C1screen true.
Use actual `ev.perception_metrics` and `perception_loss`, report all scores.
Jointrecipe's usual validation-kind C1monitoring is disclosed, never used to
select checkpoints. Phase robustness remains diagnostic, not an added gate.

If retentionpasses, calibrateonce seed3405/64TRAINscenes, samegrid/selection.
Primary: untouched2407/2408,32sceneseach, originalgates, bothrequired. Jcontrol
usesits3404manifest, notselectable. Secondaryknown2405/2406 reportedseparately.
Mechanism diagnosis used validationkinds; no unseen-kind or sealed-test claim.
All failures remain visible; further repairs require a separate concrete plan.

Joint3501 completed1,000 updates in180.11s, reserved3.416GiB, source unchanged;
C1validation passes (allfour attributes, lamps, machine detection and both
pointers100%; RGBMSE.010272). Matched retention also PASSES all12 checks:
J→joint RGBMSE.009126→.008898, pixelaccuracy.98406→.98617; kind/lamp/detection/
pointers remain100%; attribute1 improves.99219→1, othersremain1. Identical
256TRAIN scenes are hash-checked across both evaluators/models; standalone
report `runs/real_visual_retention_joint_3501_v1/report.html`, structuralQA.
The secondary known-kind phase diagnostic is mixed: kind49 minimum cosine
.412(W100)→.852(joint), kind50.799→.682. It is not a gate or selection tool,
and does not establish the mechanism is solved. Calibration3405 is underway.

Joint calibration3405 FAILED after431.73s: threshold.85 gives100%matching but
one false novel merge/64; threshold.90 gives99.22%matching with the same merge;
threshold.95 removesfalsemerges but matchingfalls to89.06%. No manifest exported,
no2407/2408 evaluations launched; those primary scenes remain untouched.
Retention remains a pass, identity calibration remains a fail. Round16 diagnoses
the actual TRAIN false merge before choosing one further existing-training-control
change. No threshold relaxation or new representation is adopted.

## Duration-only continuation

Round16 reproduced the TRAIN falsemerge: episode20 replaces kind17 with33;
the newmachine matches17 at cosine.909. These have similarpalettes and different
patterns. Controlled scores are also high in J(.93) and W100(.94), so this
confusion predates jointtraining;3404 simply did not contain that pair.
No mixture/dataoption is adopted: the fixedtable and proceduralgenerator share
their parameter family, and repeatedly showing48fixedidentities would not
by itself establish unseen-pattern discrimination.

Preregister fresh `real_visual_joint_repair_3501_u3000_v1` from J: only the
update budget changes1000→3000; same seed3501/batches/lr3e-4/pairedprocedural1,
joint identityweight.2, margins.95/.50/W100. Max20min/6GiB, lastcheckpointonly.
The first1000 updates repeat the same data stream; no altered resume identity.
Training curves were still improving and retention had headroom. This is a
bounded optimization test, not a claim that longer training guarantees success.

Repeat identical retention3506/256scenes and all12checks. Ifpass, repeat SAME
calibration3405 episodes/grid/selection so the identified17/33 failure remains
included; label this calibration as development data that motivated the repair.
Only afterbothpass: untouched2407/2408 primary, same runtime gates, bothrequired;
Jcontrol uses3404manifest, known2405/2406 secondary. No extra seeds/threshold
shopping. If similar-palette failure persists, round16 proposes separately
testing identityweight1.0 at1000updates; not combined with this duration test.

The3,000-update run completed475.25s at3.418GiB reserved; source unchanged.
All first1,000 training metric rows exactly match the shorter run (retained
prefix audit). C1 passes with all reported attributes/lamps/detection/pointers100%.
Identical retention3506 passes all12checks: J→candidate RGBMSE.009126→.008639;
pixelaccuracy.984057→.984651; allattributes now1 and otherheads/pointers1.
The new checkpoint is now repeating calibration3405, including17/33.

Diagnostic correction: the original phase probe added0–3px to already-jittered
centers and therefore included positions outside the real generator bounds
(machine0 x14–18,y12–15). Its extreme cosines are preserved but are not evidence
for a defect restricted to the actual task domain, nor proof that encoder stride
is the unique cause. Codex is repeating the controlled comparison at all20 valid
positions, same real models and fixed scenes. Actual runtime/calibration failures
use the normal generator and remain valid. The standing workflow now requires
explicit input-domain disclosure and an actual-domain follow-up for such probes.

The corrected actual-domain phase check covers all20validpositions on each of
8known validation kinds. Minimum same-identity cosines for kinds49/50 are
J .751/.825, W100 .722/.748, joint1000 .898/.876, joint3000 .863/.926.
This supports residual learned position sensitivity inside the task domain,
without identifying a unique architectural cause. Weights remained unchanged.

Duration-only joint3000 calibration3405 PASSED: selected .95/.90/.10 gives
acquisition1, same-layout1, relocated123/128=.9609375, no false matches or novel
merges, novel detection1. The two reserved primary populations2407/2408 and
fixed J control have now started. No identity-weight branch is adopted.
Claude round17's pattern diagnostic suggests weak pattern discrimination, but
near-unit slot cosine and weak centroid decoding do not prove information is
absent or unrecoverable. Its shifted-position subset also needs domain correction;
its base pattern comparisons are reviewed separately.

## Existing identity-weight intervention

The duration candidate fails BOTH primary populations2407/2408 at60/64 relocated
matches (.9375); all other task gates and all persistence contracts pass. This
failed qualification is preserved. Calibration's strict .95 match cutoff was
needed to avoid the TRAIN novel merge; increasing duration alone did not resolve
the separation/position tradeoff sufficiently.

Adopt Claude rounds16/17's already-planned next single-factor branch: native J
initialization, seed3501,1000updates, all joint1000 settings identical except
existing `--identity-weight 1.0` (previously .2). No new modules or data options.
Output `real_visual_joint_repair_3501_iw1_v1`; <=20min/6GiB, lastcheckpoint only.
Compare against joint1000 for the scientific intervention, not against3000 as a
matched-budget claim. Repeat same retention3506/256 and calibration3405/64.
If both pass, require unchanged task gates on ALL known2405–2408 regression
populations plus untouched2409/2410,32scenes each; fixed J remains the control.
Do not discard failures by moving to new seeds. Fresh scenes remain development
validation, not unseen-kind or sealed-test evidence. If retention/calibration
fails, diagnose before another intervention; no automatic sweep.

Claude round19 independently reviewed the identity-weight registration and actual
loss/pairing/pointer/gradient code: no concrete bug found. Global gradient clipping
couples effective step sizes, so increasing identity weight changes the objective
balance but is not a clean claim about gradient magnitude alone. Joint held-out
slot geometry was not separately measured; no stronger attribution is made.
Failed duration comparison has its own aggregate report:
`runs/real_visual_memory_comparison_joint_u3000_v1/report.html` (structuralQA).

Identity-weight1 training completed1,000updates in160.71s,3.416GiB reserved,
source unchanged; C1passes. Retention3506 passesall12: J→candidate RGBMSE
.009126→.009371 (+2.68%); pixelaccuracy .984057→.980046 (-.00401),
objectpointer1→.999023, attribute3 1→.998047, attribute1 .992188→.996094;
otherheads unchanged. These are within registered tolerances, not no-degradation
claims. Calibration3405 is running. Training already has a valid101-file source
snapshot; its original index/hashes verify (the generic evaluation sealer stopped
on an ordering collision without modifying indexed files).

Round20 rejects a proposed positive target .99 continuation: existing .95 target
is not saturated (8–14% training positives still violate), while confusable
negatives force the high matching cutoff. No such fit is launched. Corrected
TRAIN-palette position probes improve with iw1 (q01 .933→.958); pattern-only
negative similarity barely changes. These are diagnostics, not new gates or proof
of absent slot information. Conditional planning considers targeted confusable
training pairs in the existing recipe only if the actual task still fails.

## Conditional confusable-pair data repair (joint plan, round21)

If the current iw1calibration completes with no admissible policy, adopt one
bounded training-data repair in the existing perception recipe. Runtime failure
involves similar-colour different-pattern machines; independently drawn palettes
rarely force pattern discrimination. Add optional `--confusable-twins .25`:
a selected scene's second body shares the first's ordered colours and period,
but uses a different pattern among0–3. Check every proposed twin against the
existing held-out-texture exclusion; rejected twins retain original data and
are counted. Base scenes/lamps/labels remain identical; default0 consumes no
additional randomness and keeps existing default batches and settings. New-code pause/resume is exact;
old source-locked training runs still require their retained original source.
No model, decoder, trainer or runtime labels are added.

Native-full red checks cover default equivalence, rendered distinctness, held-out
rejection, pairing/negative masks, finite gradients into real J perception/key,
invalid CLI and exact pause/resume. Claude implements; Codex independently reviews
and runs checks, then full CPU suite. Source stays frozen for each run. Only the
existing recipe should change; determine calibration hash validity from actual
source dependencies rather than assuming every repository edit invalidates it.

Single factor vs joint1000: Jinit,seed3501,1000updates,lr3e-4,procedural1,
identityweight.2,margins.95/.5/W100,plus twins.25; output
`real_visual_joint_twins_3501_v1`,<=20min/6GiB,lastcheckpointonly. Retention3506
(all12), calibration3405, allknown2405–2408 plus fixed2409/2410 with unchanged
gates. Existing J controls stay historical matched-input controls, never selected.
Report actual twin counts and diagnostics; no proxy becomes a new gate. If this
fails, inspect the real failure before further changes; no automatic sweep or
representation replacement. This necessary training repair is within Alex's
explicit delegated planning/implementation with Claude and same permissions.

IW1calibration completed404.79s, source unchanged: FAILED. Every policy has at
least one false novel merge/64; .95matching122/128=.953125 stillmerges1/64.
No manifest exported, no candidate validation launched. The conditional data
repair is now adopted; paired native training examples are the only new behavior.

25September: Claude implemented the optional data repair;14focused native checks
and73adjacent regressions pass. Codex independently reviewed the diff: no blocker.
Only latent_agent recipe changes (57added/3removed lines) and one testfile; runtime
source hash is unaffected, existing manifests still validate. To avoid idle compute,
freeze this reviewed source now and run the full CPU suite alongside the registered
GPU fit; no source edits until both finish. Commit only after the suite passes.
This scheduling change does not alter scientific settings/gates; concurrent-load
wall times are descriptive, not a performance comparison.

Independent actual-runtime replay of iw1calibration at .95 reproduces exactly one
falsemerge: episode20, novel33→stored17, cosine.952681 (otherknown45:.348918).
This confirms the same confusable-pair failure after higher identityweight.
`diag_novel_iw1/novel_merges.json` records the actual novelkind from each episode,
correcting the stale-variable field in the earlier historical replay.

Twin1000 training completed170.01s,3.416GiB reserved,source unchanged,C1pass.
Applied7931/32000 scene proposals (24.784%);7933attempts,twoheldoutrejections.
Retention passesall12: J→twin RGBMSE .009126→.009051,pixel .984057→.982681;
allattributes/detection/lamps/pointers1. Corrected TRAINpalette diagnostic shows
position-positive q01 .933→.967 but pattern-only negative share>.90 only
.483→.452. No strong pattern discrimination claim follows. Calibration3405
and whole CPU suite are still running; no qualified identity result yet.

Round24 actual-native diagnostic (TRAIN32pairs,seed3510,unchangedweights) finds
weighted identity/perception gradient norms1.308/1.412 and globalcosine+.208 at
twins1000. Twinned-anchor ranks9/12 versusJ8/12; too few for an improvement claim.
Full objective delivers a nonzero gradient; this does not isolate each twin's
gradient or prove no conflict in other batches/parameters. Clipping is measured
at these diagnostic points only, not every historical training step.
Claude withdraws a proposed linear-probe prerequisite: no substitute readout or
proxy gate is added, and a failed probe would not establish absent information.

If the current twin1000 actual calibration/qualification fails, preregister one
duration-only repeat `real_visual_joint_twins_3501_u3000_v1`: sameJinit,seed3501,
all twin1000 settings, updates3000 onlychange,<=20min/6GiB,lastcheckpointonly.
Motivation: actual training curves still improve and retention has headroom;
no architectural inference follows from number of failed attempts. Compare exact
first1000metric rows to twin1000. Same retention3506,calibration3405,ALLknown
2405–2408 and fixed2409/2410 runtime gates. No seed/threshold changes or automatic
sweep. If curves flatten and actualconfusions persist, reassess that concrete
limitation with Claude before any further repair; no speculativearchitecture.

Twin1000 calibration completed487.78s,source unchanged: FAILED. Strict .95/.90/.10
has zero falsemerges but only121/128=.9453125 relocatedmatches (requires122).
Lower .90 retains one falsemerge. No manifest or candidate validation is produced.
Adopt the already registered3000-update duration-only follow-up; all original
gates/knownregressions/fixedfreshscenes remain unchanged.

Full CPU suite: 956tests, 0failures, 0errors, 0skipped, 834.253s. Source unchanged throughout.
Native training-data option is software-verified; learned identity remains open.

Twin3000 completed521.89s,source unchanged,C1passes; all first1000training rows
exactly match twin1000. Retention3506 passesall12: RGBMSE .009126→.008415,
pixelaccuracy .984057→.989279; attribute1 .992188→.994141, others/lamps/pointers1.
TRAINpattern diagnostic continues modestly: same-palette negatives>.90
.483(J)→.452(twins1000)→.425(twins3000); no broad pattern-discrimination claim.
Actual calibration3405 is running; no candidate runtime qualification yet.

Twin3000 calibration completed425.06s,source unchanged: FAILED. At.95 there
are no novelmerges but only116/128=.90625 relocatedmatches; at.90 two novel
merges remain despite127/128matching. TRAIN replay identifies33→17(.92596)
and11→6(.91761). No candidate validation is launched. Duration did not fix the
tradeoff; stop duration fits. Next diagnosis checks whether same-palette patterns
actually remove the rendered-mean shortcut (finite body/panel changes proportions).
No new representation or training run is selected by that diagnostic.

## Repair the confusable-pair construction (round26)

Actual renderer measurements partly refute the broad finite-mean hypothesis:
most existing twins already have close means (median RGB distance .022), but
border/dots versus other patterns form a separated-mean tail. Both actual false
merges are border versus checker/diagonal across different palettes. The helper
excluded border targets and did not match visible means for those pairs. That is
the concrete coverage gap selected for repair, not an encoder redesign.

Replace the existing helper, keep the same rate flag and default0. Generate a
different feasible pattern from all six; shift both proposed colors equally to
match the source's visible-body mean, preserving contrast and period. Obtain
visible pattern fractions once from the actual renderer's white/black body pixels
(excluding nonbinary panel/lamp pixels); cache immutable values. No duplicated
renderer equations, no clipping: require new colors within [.2,.95], the existing
uniform sampler branch, and pass heldout_like. Choose among feasible targets;
reject/count when none. Disclose per-scene versus per-target counters. Version
the rule in run settings; preserve all v1 source snapshots and failed results.
Rendered means must agree within1/255 per channel (quantization bound), with
visibly distinct bodies, at every valid machine position. Default batches/RNG,
pairing/masks, native gradient and exact native resume checks remain mandatory.

Codex also measured current hard-negative gradient norms on the actual full
model: a Euclidean penalty at the same violation boundary could amplify them.
That alternative is NOT adopted; changing loss simultaneously would confound
this targeted data correction. No substitute model or new objective is added.

Native J,seed3501,1000updates,lr3e-4,procedural1,identityweight.2,cosine margins
.95/.5/W100,rate.25; only pair construction changes. Output
`real_visual_joint_twins_meanmatched_3501_v1`,<=20min/6GiB,lastcheckpointonly.
Require retention3506,calibration3405,ALLknown2405–2408 and fixed2409/2410,
unchanged gates. First red/full-native tests, independent review, frozen source
for full CPU suite and GPU fit; commit only on suite pass. No automatic sweep.

Implementation review note: white/black palettes are renderer instrumentation
used only to count visible pattern pixels. They are never neural training or
model-test inputs. Actual new neural inputs retain the generator's scene bounds
and accepted color/contrast/texture exclusions. A 40-batch data audit finds
228accepted/336proposed pairs across1280scenes (17.81%effective), including31
border targets. This acceptance loss is disclosed; the registered rate is not
changed after seeing it. All16focused tests pass; adjacent regressions pending.

Mean-matched v2 fit completed175.11s,source unchanged,C1pass. Applied5255/32000
scenes (16.422%);7933proposals,2678scene rejections;21097out-of-range andone
heldout-like candidate-pattern rejection. Retention3506 passesall12: RGBMSE
.009126→.008941,pixel .984057→.984556; attribute1 .992188→.999023,
attribute3 1→.999023, otherheads/pointers1. Calibration3405 is running; full
CPU suite is also still running. No new identity qualification yet.

Mean-matched v2 calibration3405 completed457.47s, source unchanged: FAILED.
At.90 relocated125/128 with1/64 novel merge; at.95 relocated115/128 with
the same merge count. No candidate validation. Full native CPU suite958passes,
0skips,769.68s wall;16focused and73adjacent checks also pass. This verifies
the corrected data construction, not learned identity qualification.

## Calibration resolution correction (round29–30)

The hardcoded match grid .80/.85/.90/.95 can miss feasible policies. A failed
grid does not establish that the representation cannot separate identities.
Add an explicit optional threshold list to the existing calibrator, preserving
the default grid, margins, new=match-.05, selection and acceptance gates. Reject
invalid values before output creation; record the grid and selected policy with
checkpoint/source integrity. Source changes invalidate old manifests for new
runs; historical results remain valid as recorded. No new architecture or fit.

TRAIN replay is diagnostic only: threshold-dependent memory histories require
actual sequential calibration. A narrow estimated TRAIN interval is not proof
of poor transfer and introduces no new robustness gate. Threshold fitting on
TRAIN is legitimate; all six fixed development populations2405–2410 still
decide qualification. Claude is implementing and testing this gap; the next
TRAIN-only checkpoint/search protocol will be fixed before running it.

Round30 selects only the existing joint3000 checkpoint for the next calibration
repair, from its TRAIN pass at.95 versus novel merges at.90. Other checkpoints
remain nonselectable; J is a recorded control. Before running, use a bounded
TRAIN3405 score replay only to place additional grid points between.90 and.95;
then actual sequential calibration, original policy selection/gates, all six
fixed development populations. Replay histories are approximate and cannot
prove infeasibility. No validation score determines the threshold or checkpoint.
Budget: replay<=2min, calibration<=25min/6GiB; full CPU suite and read-only
calibration may overlap after source review/freeze. No timing comparison claim.

TRAIN replay of fixed joint3000 completed54.51s, frozen source. Highest known
novel score .924886; low true scores .881076,.912729,.936687,.939693,.948219.
These are policy-history-dependent diagnostics, not task predictions. Freeze
actual calibration grid now: [.80,.85,.90,.925,.93,.935,.94,.95], margins
[.05,.10], TRAIN3405/64episodes, original selection, output
`real_visual_binding_calibration_3405_joint_fine_v1`,25min cap. Retained original
J control runs keep their original source/policy labels; no control checkpoint
selection. Default native two-episode parity is byte-identical for policies,
results, metrics and settings versus saved pre-change recipe.

Fine calibration completed877.43s, source unchanged: PASS. Selected
match.935/new.885/margin.10 by the original rule; acquisition/same-layout1,
relocation126/128=.984375, false matches/novel merges0. All eight legacy
policies exactly reproduce the earlier coarse run. Full suite972passes,0skips,
845.66s; code committed2449864. Source snapshot101files verified; all256TRAIN
frames match every prior3405calibration; ten actual-model manifest/fault checks
pass without altering originals. All six fixed task populations now follow.

All six fine-policy task runs completed on frozen source: relocatedmatches
2405=60/64 FAIL,2406=64/64,2407=61/64,2408=62/64,2409=61/64,2410=62/64.
Other task metrics are1, novelmerges0, contracts/frozenweights allpass. Thus
five pass but required six-population qualification FAILS. Independent raw-row
audits and101-file source snapshots pass. Aggregate report:
`runs/real_visual_memory_comparison_joint_fine_v1/report.html`. No conditional
cleanup or promotion is activated. Next review considers actual TRAIN decision
margins and the existing highest-threshold tie-break; no new rule adopted yet.
