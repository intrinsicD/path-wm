# Frozen-writer recall repair

13 September 2026. Alex approved repairing memory reading and workspace formation
while preserving the encoder and stored-state writer. This is a bounded native
facts/image-output experiment, not evidence of general entity understanding.

## Comparison declared before training

Use both completed balanced-relocation source checkpoints7801/7802 unchanged.
For each source compare identity against fixed channel normalization of retrieved
memory tokens. Both arms continue the SAME existing weights, with the same sampler
seed8001, optimizer, update count and trainable parameters. Freeze the entire model
except Thinker, factual readout and image feature producer; the image decoder head,
all visual encoders, initial state, state updater, dynamics, direct diagnostic and
all prior calibration buffers stay fixed. No auxiliary write loss or direct-reader
training. Alternate ordinary and reset query losses, as in the original recipe:
factor CE + foreground-weighted RGB MSE +0.1 standardized target feature MSE.
Targets enter losses only. There is no target/encoder-detail input at recall.

Preserve raw snapshots, keys, top-k selection, and state context. The adapter only
subtracts/divides retrieved values using channel moments over BOTH stored snapshots
and all tokens from TRAIN histories, float64 accumulation, std floor1e-4 (fixed from
the probe protocol; not tuned). Identity has zero/unit buffers and the same class.
No timestamp or positional features are added. Lack of explicit snapshot time is
an unresolved alternative, not changed simultaneously. Empty memory remains empty.
This fixed-budget comparison can show useful improved fitting or output accuracy;
it cannot establish asymptotic superiority or recover information proven absent.

Train128 pairs seed7701, validation32 pairs7722, test64 pairs7723, relocation
curriculum. All16 tuples/motions familiar; fresh noisy backgrounds only. Exact
frame hashes must be disjoint. Source checkpoints are evaluated on this same new
test population. A separately reported direct readout applied to the same stored
values (the existing native factual head and the previously fitted probes) is a
conditional access control, never a perfect information ceiling. No probe retuning.

Four runs1536 updates, batch16, AdamW lr0.001 wd0.0001, global trainable gradient
clip1, FP32, max300 training-loop seconds each,1200s total. Final checkpoint only;
validation128 updates. Same seeds/samples within each source pair. GPU cap4GiB with
1GiB headroom. Development is a separate16-update run, seed18001/data17701/17722/
17723; no development accuracy gate or architecture tuning. One formal768+768
resume; CPU exact resume/standalone reload regression. Commit source before formal
runs; do not edit source while any test/training/evaluation runs.

## Gates and causal checks

Use existing relocation screen: ordinary AND reset facts/images>=90%, complete
selection and relocation pairs>=80%, +30 points against erased history, reset bank
erasure, cue erasure and later-view erasure; swapped bank alternate answers>=80%.
Weighted pixel error must beat background and pair-mean baselines. Both sources
must pass to promote a repair. Separately call calibration beneficial at this budget
only if reset factual AND image accuracy each improve>=10 points over identity
retraining in BOTH source pairs. Report native baselines, absolute scores and all
failures even if neither gate passes. Do not select a winner by test outcomes.

Audit unchanged all frozen parameters/buffers, source file hashes, raw bank values,
train-only calibration, detached storage, no cross-episode mixing, erased/swapped
causality, independent pixel classifications and saved-logit metrics. Strict export
loads without donor data; GPU replay max error<=1e-4 and categorical equality.
CPU replay uses the existing1e-4 tolerance but preserves known raw-encoder device
limits as a separate result. Every run keeps raw arrays, weights, optimizer/RNG/
sampler, curves, timing/resources and a verified standalone report. Renderer is
unchanged; structural QA only because browser QA remains unavailable. Inspect PNGs.

Claude's public-only review accepts the bounded comparison with caveats: report
learning curves and fix the std floor before evaluation; improved conditioning or
faster fitting is a valid bounded outcome, not proof of erased-information recovery.
No private code, data or measurements were exported.

## Implementation checks

Added a read-only calibration adapter and an opt-in `--repair identity|calibrated`
path in the existing recipe. Older checkpoints and training defaults remain valid.
The complete frozen parameter/buffer set is checked in addition to prior codec
hashes.41 targeted tests and lint pass, including calibrated exact CPU training
resume/reload, unchanged stored banks after optimization, positive gradients into
all three intended trainable parts, and bank erasure/swapping causality. The RED
checks were committed before implementation. The16-update development run takes
2.801s training plus1.544s calibration,424MiB reserved; report structurally verified
and comparison PNG inspected. No development accuracy used to tune the design.
Claude explicitly accepts fixed-budget practical improvement without requiring
either arm to converge asymptotically. Both public-only receipts are retained.

Example formal command (change source7801 to7802 and arm identity to calibrated):
`OMP_NUM_THREADS=2 .venv/bin/python -m experiments.memory_output --weights runs/memory_relocation_v1/seed_7801/weights.pt --repair identity --seed 8001 --output runs/recall_repair_v1/agent_7801_identity --device cuda:0`

## Results

Sourcecfe472b. All four runs complete1536 updates; no checkpoint selection or
budget extension. Two public-only Claude exchanges reconcile the fixed-budget
interpretation. Both conditions have76,824 trainable parameters and identical
initial learned weights/samples within each source pair. Calibration adds only
fixed buffers, fitted from the declared training banks. All frozen tensors equal
their originals after fitting; raw bank values and original checkpoint files match.

Scores on the SAME128 fresh-background test histories:

| Frozen writer | Condition | Reset facts | Reset images | Complete relocation pairs, facts | Complete relocation pairs, images |
| --- | --- | ---: | ---: | ---: | ---: |
|7801|Unchanged source|48.4375%|47.65625%|0%|0%|
|7801|Retrain only|66.40625%|63.28125%|32.8125%|29.6875%|
|7801|Retrain + calibration|58.59375%|57.03125%|17.1875%|15.625%|
|7802|Unchanged source|50%|51.5625%|0%|3.125%|
|7802|Retrain only|75%|75%|50%|50%|
|7802|Retrain + calibration|73.4375%|73.4375%|46.875%|46.875%|

Every run fails the complete predeclared screen. Calibration's separate10-point
benefit gate fails too: its reset facts/images are lower than identity preprocessing
in both matched pairs. Keep it opt-in; do not promote either condition as reliable
recall. Focused retraining is a partial native-output improvement with unchanged
codecs/writer, not proof that freezing alone caused the gain. More updates and a
changed learning objective are also part of this continuation versus its source.
This says nothing about asymptotic superiority or general image generation.

Ordinary facts/images:65.625/65.625%,66.40625/66.40625%,74.21875/74.21875%,
72.65625/72.65625% in writer7801 identity/calibrated, then7802 identity/calibrated
order. Reset color is100%; shape is99.21875% for writer7801 calibrated and100% in
the other factual readouts. Location remains the dominant error. Removing the later
view reduces identity-arm facts to35.9375/
43.75%, versus66.40625/75% with it. Erasing memory reduces them to7.03125/8.59375%;
cue erasure6.25% for both. Bank swapping produces the alternate answer with the
same factual/image accuracies as ordinary reset, supporting actual bank dependence.
Paired failures and double-position images remain visible in the saved panels.

Unchanged direct encoder readers and frozen teacher decoding both score100%.
Previously fitted readers applied to these SAME stored working tokens recover
location98.4375/85.9375% for writer7801,76.5625/75% for7802. These are conditional
access controls, not information ceilings. The weaker writer and probe dependence
still limit interpretation; failure after the repair cannot prove erased identity.

Training seconds in the order above:241.590,176.859,171.921,170.862; total761.232
(12.69min),424MiB peak reserved. Calibration and evaluation are separate; the split
run's first-chunk timing/resources are preserved in its receipt. Fixed update
budgets are matched; run-order/timing variation is not an efficiency claim.

41 targeted tests and lint pass. Independent NumPy checks verify696 fitted-run
metrics,348 unchanged-source metrics and32 stored-probe metrics (1,076 total;
worst error2.50e-8). Split frame hashes are disjoint. The actual768+768 GPU resume
preserves all774 prior ledger rows; the CPU test proves exact optimizer/RNG/sampler
replay. All four standalone GPU exports reproduce logits/pixels exactly with the
training IEEE/deterministic setup. A first audit script omitted that setup, failed
GPU replay and was corrected; its unmatched control/CPU artifacts and repair note
are retained. Training source and results were unchanged.

CPU replay of raw observations still fails1e-4: max logit differences0.224920,
0.156007,0.178067,0.092645; max pixel differences0.019403,0.005649,0.012029,0.005180.
All categorical factual/image predictions agree on all128 episodes per run. This
is the existing cross-device numerical issue, not fixed by memory training. Strict
inference reproduction requires the same device and explicit precision setup:
`from pathwm.io import seed_everything; seed_everything(8001)` before loading/running.

The existing report renderer is unchanged. Each completed run and the combined
overview are self-contained and structurally verified; browser QA unavailable.
All four comparison panels and the exported score/learning-curve figure inspected.
Raw results, source controls, fixed probe predictions, replay environments and
audit receipts are in `runs/recall_repair_v1/`.

Next proposed step: isolate temporal/snapshot routing. The current reader flattens
both memories into one attention context and does not supply their timestamps.
Compare an explicitly time-aware reader against this continuation, retaining raw
stored-state controls and checking the weaker writer separately. This is an open
hypothesis and needs a fresh bounded protocol; no extra training or larger model
was launched after these failed gates.

## Snapshot timing comparison

13 September: Alex approved the proposed temporal-memory test. Continue from BOTH
previous identity-repair checkpoints (writer7801 and7802), preserving them. For each,
compare another untimed continuation with a temporal continuation from identical
weights. All freeze rules, losses and trainable parameters remain as above. Temporal
retrieval adds `position(snapshot_time - query_time) - position(0)` to each token in
that selected snapshot, using the existing sinusoidal function with unit amplitude.
No normalization, learned time embedding, extra parameters or raw-storage changes.
Gather values AND timestamps with the same top-k indices. Query time is state.time;
thinking steps do not advance it. Existing future-observation checks remain. Common
clock-origin shifts cancel. Defaults and old standalone exports stay untimed.

Fresh validation32 pairs seed7732 and test64 pairs7733, training128 pairs7701. All
appearance/movement support remains familiar; timestamps are0/1 at query2 for every
quartet. They cannot alone distinguish the correct side. No absolute-time or unseen-
delay generalization claim. Evaluate both frozen source continuations on the SAME
new test; apply both unchanged pre-storage working probes to these histories as
conditional access controls, with no retuning. Timestamp metadata comes from the
harness observation clock, not labels or image targets.

Four1536-update fits, continuation seed8101 for all arms, batch16, AdamW lr0.001
wd0.0001, gradient clip1, FP32/IEEE, final checkpoint only. Validation every128 steps;
max300 training-loop seconds per fit (1200s total), GPU cap4GiB with1GiB headroom.
Development16 steps with seed18101, train17701/validation17832/test17833, at most60s.
One actual768+768 resume plus exact CPU regression. No source changes during a run.
Commit RED checks/plan before implementation, then working source before formal fits.

Retain the complete90% facts/images,80% complete-pair and intervention screen above.
Call the time cue beneficial at this budget only if reset facts AND images improve
by>=10 percentage points over matched untimed continuation in BOTH source pairs.
Additionally report test-only reset-time-erasure (all snapshots assigned query time,
zero additive code) and reset-time-swap (reverse timestamps, keep values/keys fixed).
An attachment-sensitivity screen requires each intervention to reduce reset facts
AND images by>=10 points in BOTH temporal runs. This is an OOD sensitivity check,
not a proof of causal temporal understanding or an alternate valid-world answer.
Untimed runs must be exact no-ops under both time interventions. Full reliability,
benefit and sensitivity gates are separate; failures cannot be hidden by averages.

Essential checks: timestamp/value gather alignment including a top-k subset and
batch-specific choices, common clock-shift invariance, bank permutation equivariance,
zero-age identity, erased/future/nonfinite memory, unchanged raw banks/gradients,
full-query time interventions, frozen tensors, standalone reload and resume. Save
new intervention logits/images and independent metrics, raw source/probe controls,
resource/precision metadata and standalone reports. Use existing renderer; browser
QA remains unavailable, so structural QA plus inspection of exported PNGs. Prior
CPU/GPU numeric failures remain open; match precision explicitly in replay scripts.

Claude's two public-only reviews identify an important interpretation limit: an
arbitrary fixed snapshot tag could also improve pooling. This comparison tests the
implemented age cue at two fixed ages (effectively snapshot differentiation), not
its superiority over such tags or learned elapsed-time semantics. That distinction
remains open rather than adding another training arm. Include bank permutation
checks through the actual workspace, including trained exports. Image scoring is
the unchanged shared deterministic template comparison, not a changing classifier.

Implementation checks:46 targeted tests and lint pass, including temporal CPU
training/resume/standalone replay, live workspace permutation equivariance and
untimed timestamp-intervention no-ops. The16-update temporal development run takes
2.910s with424MiB reserved; frozen tensors unchanged, report structural QA passed
and comparison PNG inspected. No quality tuning from this development run.

Formal command (use both writer7801/7802 and repair identity/temporal):
`OMP_NUM_THREADS=2 .venv/bin/python -m experiments.memory_output --weights runs/recall_repair_v1/agent_7801_identity/weights.pt --repair temporal --validation-seed 7732 --test-seed 7733 --seed 8101 --output runs/memory_time_v1/agent_7801_temporal --device cuda:0`

## Snapshot timing results

Sourceca42e39. All four1536-update fits complete under their fixed300s caps. Both
arms have76,824 trainable parameters and identical initial tensor hashes within
each source pair; only the explicit age-cue setting differs. Frozen weights, raw
stored states and original source checkpoints remain unchanged. Old untimed defaults
are preserved. No larger encoder/decoder, tag-control arm or further training added.

Fresh-background test128 histories, identical across all rows:

| Writer | Condition | Reset facts | Reset images | Complete relocation pairs, facts | Complete relocation pairs, images |
| --- | --- | ---: | ---: | ---: | ---: |
|7801|Frozen prior repair|69.53125%|66.40625%|40.625%|34.375%|
|7801|More untimed training|72.65625%|75%|45.3125%|50%|
|7801|Temporal continuation|77.34375%|75.78125%|54.6875%|51.5625%|
|7802|Frozen prior repair|74.21875%|74.21875%|48.4375%|48.4375%|
|7802|More untimed training|74.21875%|74.21875%|48.4375%|48.4375%|
|7802|Temporal continuation|74.21875%|74.21875%|48.4375%|48.4375%|

All four full reliability gates fail, as does the replicated10-point age-cue benefit
gate. The first temporal model gains six correct factual answers and one correct
image versus equal untimed training; the second gains none. No robust benefit is
established. Color and shape factual accuracy are100% in all four final models;
remaining factual errors concern final location. Ordinary facts/images are78.90625/
75.78125%,78.90625/78.125%,74.21875/74.21875%,74.21875/74.21875% in writer7801
identity/temporal then7802 identity/temporal order. No90% ordinary result either.

Temporal-model test interventions, facts/images:

| Writer | Correct ages | Ages collapsed to query time | Ages misassigned between snapshots |
| --- | ---: | ---: | ---: |
|7801|77.34375/75.78125%|57.8125/55.46875%|64.84375/64.84375%|
|7802|74.21875/74.21875%|74.21875/74.21875%|69.53125/69.53125%|

The replicated sensitivity screen fails because the second writer is unaffected by
age erasure and loses less than10 points under misalignment. Both interventions are
exact no-ops in both untimed models. The first model's functional sensitivity does
not establish elapsed-time semantics: these are OOD metadata changes with fixed
training ages. Arbitrary snapshot tags remain an untested alternative mechanism.

Unchanged direct encoder and teacher-image controls score100%. The previously fitted
stored-working readers score98.4375/82.8125% location on writer7801,75.78125/74.21875%
on writer7802. This retains the gap between accessible stored information and native
recall in the stronger source, while the weaker source/probe uncertainty remains.
It does not identify a perfect information ceiling or prove the writer lost location.

Training seconds:161.488,161.303,159.149,160.714; total642.654 (10.71min),424MiB
reserved per run. Final checkpoint only; learning curves and failed gates retained.
76 relevant tests pass:46 focused checks plus30 tests of shared-memory consumers
(belief, multimodal training and task contracts). Independent NumPy audits verify
864 fitted-run metrics,432 unchanged-source metrics and32 fixed-probe metrics,1,328
total; worst error7.67e-9. Exact split frame hashes are disjoint. The real768+768
resume preserves774 ledger rows; exact CPU optimizer/RNG resume is covered by tests.

All four strict standalone GPU exports reproduce logits/pixels exactly under matched
IEEE/deterministic settings. Reordering complete stored records also leaves the
actual post-think workspace exactly unchanged on all128 cases per model, on GPU and
CPU; timestamps remain attached to the selected values. CPU raw-observation replay
still fails1e-4: maximum logit error0.475857, pixel error0.015305. Factual labels agree
throughout, but writer7801 temporal produces different image-side labels on two
episodes (indices68 and100). Do not claim numerical portability fixed or silently
relax the tolerance. A stale calibrated-arm path in the reused array audit was
corrected and that audit rerun; model training and reported metrics were untouched.

Each run and the overview have standalone structurally verified reports, with raw
arrays, source/probe controls, resume receipt, resources and independent audits.
All four comparison panels and score/learning-curve plot inspected; browser QA
unavailable. Two actual public-only Claude reviews, no private repository export.

Next proposed work: compare the successful direct stored-state readout with the
native workspace path on exactly the same values, using it as a concrete reference
for a reader repair. Check weaker stored-state accessibility separately. Another
timestamp feature or unbounded continuation is not supported by this comparison;
the exact next interface and budget still need declaration. No new run launched.

## Frozen-reader workspace supervision protocol

User approved the next repair on13 September. Test a task-specific training signal:
apply the existing frozen stored-working TokenProbe to post-think working tokens
and add mean factor cross-entropy against TRAIN labels, coefficient1. The matched
control has the same attached frozen probe and coefficient0. Native factual/image
outputs still come exclusively from their existing working-token heads; neither
probe answers nor stored-state targets enter inference. This is auxiliary supervised
learning, not teacher-logit distillation or exact latent reconstruction.

Start from both untimed continuations in runs/memory_time_v1/agent_780{1,2}_identity.
Keep writer, initial state, input encoder/calibration, raw memory, decoder head and
probe weights/statistics frozen. Only the same thinker, native factual head and
image feature producer learn. Use old stored-working probe seed7901 for each writer
(the first of exactly two existing seeds, not a new search); evaluate7902 equally
prominently as an unoptimized reader sensitivity check. The convention uses a
previously observed reader, so this is hypothesis generation, not independent
selection validation. Reuse the normalized reader in the library; no recipe imports.

Four fixed1536-update fits, batch16, AdamW lr0.001/wd0.0001, clip1, continuation
seed8201, same sample sequence,300s training cap each,4GiB GPU cap/1GiB headroom.
Training128pairs seed7701; validation32pairs7742; test64pairs7743, final checkpoint
only. Same trained tuples/motions; held-out backgrounds, not new concepts. Separate
16-update development seed18201/train17701/validation17842/test17843,60s cap.
One formal supervised fit uses768+768 explicit resume; no budget extensions/tuning.

Primary existing reliability gate remains unchanged: ordinary/reset native factual
and image accuracy>=90%, complete selection and relocation pairs>=80%, required
memory/cue/later-view erasure drops>=30points, swapped-bank alternate>=80%, image
error beating fixed background and pair-mean controls. Auxiliary benefit requires
>=10percentage-point improvement in BOTH reset native outputs for BOTH writers
over matched coefficient0. Also report both frozen readers on stored and recalled
working tokens, per-factor accuracy and paired histories. Reader-only improvement
does not pass the repair gate, and disagreement between readers limits any claim
about general accessibility. Failed probes do not prove absent information.

Essential checks: isolated auxiliary gradients through frozen reader into thinker;
zero-weight exact baseline; frozen buffers/writer/raw banks; strict self-contained
export/resume; invalid weight/missing or mismatched reader rejection; unchanged
native inference if inspection reader is modified. Verify all native raw-array
metrics independently, frozen source/probe hashes, training identities, GPU reload,
CPU differences and memory permutations. Existing report renderer, structural QA
and inspected PNGs; browser QA unavailable. Preserve previous artifacts.

Claude's public-only review identifies reader-boundary exploitation and shifted
normalization as real interpretation limits. We retain two-reader diagnostics and
native causal output gates. A shuffled-label auxiliary arm and exact normalized
token reconstruction remain untested alternatives; no claim that this signal is
uniquely effective or restores general latent fidelity. No more arms in this budget.

Implementation validation:49 relevant tests and lint pass, including exact zero-weight
baseline, isolated auxiliary gradient, unchanged inference after reader mutation,
strict standalone reader reload and4-update exact optimizer/RNG resume. The16-update
GPU development check completes in2.6975s,424MiB reserved, all frozen tensors fixed;
standalone report structurally verified and comparison PNG inspected. Claude accepted
the narrowed diagnostic scope; its gradient-flow blocker is covered by the test.
No development quality tuning. Formal source is committed before training.

Command (both writers; reference-weight0 baseline,1 supervised):
`OMP_NUM_THREADS=2 .venv/bin/python -m experiments.memory_output --weights runs/memory_time_v1/agent_7801_identity/weights.pt --repair identity --workspace-reference runs/memory_probes_v1/agent_7801_probe_7901/weights.pt --reference-weight 1 --seed 8201 --validation-seed 7742 --test-seed 7743 --output runs/reader_supervision_v1/agent_7801_supervised --device cuda:0`


## Frozen-reader supervision results

Source4bc44f4; all four predeclared1536-update fits complete under their300s caps.
Both arms attach the same frozen17,512-parameter reference and train the same76,824
native parameters. Within each pair, initial model tensors, data/source identities
and final sampler states match exactly. Frozen writer, codecs, calibration, probe
and raw banks stay unchanged. No new inference input or larger model was introduced.

Test128 fresh-background histories, all appearances/motions supported in training:

| Writer | Condition | Reset facts | Reset images | Complete relocation pairs, facts | Complete relocation pairs, images |
| --- | --- | ---: | ---: | ---: | ---: |
|7801|Frozen source|74.21875%|71.09375%|48.4375%|42.1875%|
|7801|Equal training, weight0|86.71875%|89.0625%|73.4375%|79.6875%|
|7801|Reader supervision, weight1|84.375%|86.71875%|68.75%|75%|
|7802|Frozen source|72.65625%|74.21875%|45.3125%|48.4375%|
|7802|Equal training, weight0|74.21875%|74.21875%|48.4375%|48.4375%|
|7802|Reader supervision, weight1|74.21875%|74.21875%|48.4375%|48.4375%|

The added loss loses three correct native factual answers and three correct images
in writer7801; writer7802 is unchanged. All four full reliability gates and the
replicated10-point native-benefit gate fail. Ordinary factual/image accuracy is
96.09375/95.3125% for7801 baseline,93.75/92.1875% supervised, and74.21875/74.21875%
for both7802 runs. Final checkpoints only; late validation fluctuations did not
trigger checkpoint selection, coefficient tuning or budget extensions.

Both frozen readers on the same held-out stored/recalled working tokens:

| Writer | Reader seed | Stored joint, all conditions | Recall joint, equal training | Recall joint, supervision |
| --- | --- | ---: | ---: | ---: |
|7801|7901 (auxiliary gradient)|96.09375%|25%|89.0625%|
|7801|7902 (no auxiliary gradient)|80.46875%|22.65625%|63.28125%|
|7802|7901 (auxiliary gradient)|77.34375%|32.8125%|74.21875%|
|7802|7902 (no auxiliary gradient)|74.21875%|22.65625%|50.78125%|

This demonstrates improved compatibility with the diagnostic readers, not better
native recall. Even the second reader's joint gains do not imply improved location:
its side accuracy drops66.40625→64.84375% in7801 and74.21875→67.1875% in7802.
It shares the same probe family and is not a general semantic interpreter. Different
readout coordinates, representation shifts and optimization effects remain possible;
these results do not prove loss of stored information or unique reader exploitation.
Original encoder/teacher-image controls remain100%. Raw erasure/swap/cue/later-view
interventions, both readers' per-factor/pair scores and normalized workspace statistics
are retained in the report. No shuffled-label auxiliary or direct token reconstruction
arm was run, so no comparative claim about those alternatives.

Training seconds:164.016562,163.478603,163.892945,217.121054; total708.509163
(11.81min),424MiB reserved per fit. The16-update development check took2.6975s.
49 relevant tests and lint pass. Independent NumPy checks verify948 fitted metrics,
432 unchanged-source metrics and192 two-reader metrics:1,572 total, maximum error
3.88e-7. Split frame hashes are disjoint. Real768+768 supervised7801 resume preserves
774 ledger-prefix rows; exact optimizer/RNG resume is covered in the CPU test.

All native and auxiliary GPU logits/pixels reload exactly. Actual post-think workspace
is exactly invariant to complete-bank reordering across128 cases per model on both
GPU and CPU. Frozen source and original-writer tensors, reader weights, raw banks
and input checkpoint hashes match. The optional probe can be mutated without changing
native inference, and the isolated auxiliary gradient reaches the thinker without
updating the frozen probe, writer or output heads.

CPU raw-observation replay still fails1e-4. Maximum native logit discrepancy0.460575,
pixel discrepancy0.025750. Writer7801 supervised changes factual-side labels on cases90
and122 and an image-side label on case27; other runs' native categorical outputs agree.
This is not harmless numerical portability. It remains a separate unresolved issue;
GPU results were not replaced by CPU results or tolerance-relaxed. A focused replay
recorded the newly observed factual mismatch indices in addition to image differences.

Two actual public-only Claude reviews; conceptual claims narrowed before training,
and the requested gradient-through-frozen-reader check passes. Every run and the
overview have standalone structurally verified reports, with all four example panels
and the score/learning-curve plot inspected. Browser QA unavailable. All historical
weights/results preserved. No checkpoint promoted as a complete repair.

Next proposed diagnostic: feed the stored working tokens directly to native factual
and image output heads in a controlled frozen-writer comparison. This would separate
workspace formation from output learning without treating a successful auxiliary
classifier as a repaired agent. It would be a supplied routing control for this task,
not a general learned retrieval architecture; its exact protocol remains to be set.

## Direct native-output readout protocol

User approved the direct stored-token diagnostic on13 September. Freeze the complete
writer AND thinker; train only the same native factual head and image feature producer.
Two input stages: native reset-recall working tokens, or working tokens from the actual
latest timestamped bank snapshot with thinker bypassed. Latest selection uses bank
metadata only, handles each batch independently, and is invariant to bank ordering.
Absent memory supplies zero working tokens; tied latest times average candidates.
These are explicit diagnostic policies, not learned retrieval or calibrated uncertainty.
Keep raw storage unchanged and reuse existing causal bank validation.

Cross each stage with raw versus fixed per-channel standardization fitted separately
on THAT stage's training tokens (std floor1e-4). Thus eight fits across two frozen
writer sources from runs/reader_supervision_v1/agent_780{1,2}_baseline/weights.pt.
Same native head initialization in every arm per source. Normalization buffers may
differ by stage; no test calibration. Existing frozen probes remain diagnostics,
reference loss weight0. No extra targets or probe answers enter inference.

Predeclare1536 updates, batch16, AdamW lr0.001/wd0.0001, clip1, seed8301,180s
training cap per fit,4GiB GPU cap with1GiB headroom. Train128pairs7701, validation
32pairs7752, test64pairs7753, final checkpoint only. Cache frozen TRAIN reset-route
tokens and teacher features once; loss remains native factor CE + weighted pixel MSE
+0.1 standardized teacher-feature MSE. No ordinary/reset alternation in this diagnostic.
Evaluate both modes live, plus training-fit scores, erasures/swaps and timestamp
controls. Save cache identities/timing separately. Existing recipes/renderer suffice.
One stored-standardized7801 fit resumes768+768. Separate16-update development uses
seed18301, train17701/validation17852/test17853,60s cap. No coefficient/epoch tuning.

Primary conditional output-sufficiency screen: reset native facts/images>=90%,
complete selection and relocation pairs>=80%, erasure/cue/later-view drops>=30points,
swapped-bank alternate>=80%, image error beats background and pair-mean controls.
Retain the full existing gate (including ordinary) separately. Report each condition;
replicated sufficiency requires both sources. Route-advantage and normalization-benefit
screens each require>=10point reset factual AND image gains in BOTH sources for each
matched contrast. Failure does not prove absent information or asymptotic incapacity.
This diagnoses frozen source representations and head optimization, not a repaired
learned retrieval system. Actual time interventions on supplied latest routing can
change outputs; untimed native controls should remain exact no-ops.

Essential RED checks: actual bank selection, per-batch latest/ties/order/future times,
erased-bank zero output input, untouched raw bank, frozen thinker/writer/reference,
train-only per-stage statistics, detached cache/live input/loss/gradient equivalence,
head-only updates, standalone loading and exact resume. Audit saved native metrics
independently, source/buffer hashes, both pre-existing readers, GPU replay and CPU
mismatches; retain exact caches and resource/source records. All reports standalone,
structural QA plus inspected PNGs; browser QA unavailable.

Claude's public-only review flags per-stage fitting and actual cache/live equivalence
as prerequisites; both are explicit above and will be checked. Zero fallback and
mean ties are supplied policies. Equal updates need not equal optimization difficulty,
so training and validation curves remain visible; no general workspace-loss claim.

Implementation checks:58 relevant tests and lint pass, including six cache/live
comparisons spanning both stages and normal/erased/tied-time queries, head-only
gradients, fixed writer/thinker, and exact optimizer/RNG resume plus standalone
exports for both standardized stages. Development16 updates complete in1.1753s;
460MiB reserved, structural report QA passes and comparison PNG inspected. Claude
accepted the clarified per-route statistics and empirical cache-equivalence contract.
Source committed before formal comparison; no development quality tuning.

Command (both sources, stages native/stored, with/without --standardize-output):
`OMP_NUM_THREADS=2 .venv/bin/python -m experiments.memory_output --weights runs/reader_supervision_v1/agent_7801_baseline/weights.pt --repair identity --readout-stage stored --standardize-output --seed 8301 --validation-seed 7752 --test-seed 7753 --output runs/direct_readout_v1/agent_7801_stored_standardized --device cuda:0`


## Direct native-output readout results

Source504a587. Eight1536-update fits complete under180s training caps. Only67,032
native factual/image-producer parameters learn; encoder, snapshot writer, thinker,
initial state, reconstruction head and reference probes stay frozen. Same initial
head tensor hash and final sampler state in all four arms per writer. Standardizer
buffers are independently verified from each stage's training cache. No held-out
statistics, probe-answer inputs or learned retrieval claimed.

Held-out128 histories, fresh backgrounds within trained tuple/motion support:

| Writer | Input route | Scaling | Reset facts | Reset images | Ordinary facts | Ordinary images | Reset-only gate | Full gate |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
|7801|native|raw|96.87500%|98.43750%|81.25000%|59.37500%|True|False|
|7801|native|standardized|98.43750%|96.87500%|68.75000%|50.78125%|True|False|
|7801|stored|raw|78.90625%|95.31250%|78.90625%|95.31250%|False|False|
|7801|stored|standardized|75.00000%|94.53125%|75.00000%|94.53125%|False|False|
|7802|native|raw|75.00000%|75.78125%|75.00000%|75.00000%|False|False|
|7802|native|standardized|75.00000%|75.00000%|75.00000%|75.00000%|False|False|
|7802|stored|raw|76.56250%|75.78125%|76.56250%|75.78125%|False|False|
|7802|stored|standardized|77.34375%|76.56250%|77.34375%|76.56250%|False|False|

The first writer's raw native reset workspace supports96.875% factual and98.4375%
image accuracy after head-only training, versus frozen-source90.625/89.84375% on
this SAME test population. Raw native complete relocation pairs score93.75/96.875%.
Standardized native reset scores98.4375/96.875%, with pairs96.875/93.75%. Both pass
all reset-only causal screens: bank/cue erasure reduces accuracy strongly, later-view
erasure loses more than30points, swapped answers follow the alternate bank, and
pixel error beats background and pair-mean controls. This positively establishes
conditional usability of these current frozen workspace tokens.

However, ordinary running-state output was not trained in this diagnostic and
regresses severely. All eight full gates fail; no model is promoted as a complete
repair. The raw first native run's ordinary facts/images fall to81.25/59.375%; its
standardized counterpart68.75/50.78125%. The failure is exposed rather than redefining
ordinary success. Reset-only sufficiency was declared separately before the runs.

Direct stored input does not beat native input at this budget. The first writer's
stored raw image output reaches95.3125% while its factual head reaches78.90625%;
standardization yields75/94.53125%. This difference between output heads accessing
the same stored tokens is not evidence that those tokens lack location information.
Prior native head weights are already adapted to native workspaces; identical
initial weights/budgets do not equalize optimization difficulty across input formats.
Neither stage's normalization contrast passes the replicated10-point benefit gate,
and neither scaling choice passes replicated direct-route advantage. No superiority
of raw input or inevitable failure of direct routing is established in general.

The weaker writer remains limited, including on training histories. Its native raw
training facts/images are75/76.5625%, standardized75/75%, stored raw76.171875/76.5625%,
stored standardized79.296875/78.125%. First-writer native training is100/99.21875%
raw and100/100% standardized; stored training83.59375/96.875% raw and80.859375/98.828125%
standardized. Thus failure is not solely unseen-background generalization. These
fixed heads/readers still do not define a perfect information ceiling.

Unchanged old stored-working readers score98.4375/82.8125% joint for writer7801,
77.34375/75% for7802. Both are evaluated on all routes; no auxiliary loss is used.
Frozen raw native workspaces themselves remain unchanged despite much better new
head readout in writer7801. Original encoder and teacher-image controls stay100%.
All causal/time interventions and ordinary/reset examples remain in raw arrays.
Stored latest selection, zero fallback before standardization and averaging of tied
latest timestamps are supplied policies, not learned temporal understanding.

Training seconds by7801 native raw/standardized, stored raw/standardized, then7802:
35.311589,34.458273,34.650228,34.877306,38.434092,40.226550,39.996134,38.809122;
total296.763295s (4.95min),460MiB reserved. Eight first cache preparations total
14.330486s; the resume reconstructs its cache, but that second preparation duration
was not separately recorded. Training caps exclude cache preparation and final test
reporting. The16-update development fit took1.175333s plus1.941675s cache preparation.

58 relevant tests and lint pass. Independent NumPy audits verify1,896 fitted test
metrics,240 training-fit metrics,474 unchanged-source metrics and320 frozen-reader
metrics:2,930 total; maximum error1.86e-7. Actual inputs/teacher features match the
saved GPU training caches EXACTLY across all256 cases per fit in shuffled16-example
batches. Labels/targets and cache hashes also match. CPU loss/gradient equivalence tests
cover native/stored, normal/erased/tied-time inputs. Cache identity participates in
strict resume; real768+768 stored-standardized7801 resume preserves774 ledger rows,
and exact optimizer/RNG resume is tested for both standardized routes.

Every exported GPU native/reference logit and pixel matches saved predictions exactly.
Full post-query workspace is exactly invariant to complete-bank reordering across128
cases per model on both devices. All frozen source tensors, reader weights, raw banks
and original file hashes stay unchanged. Standardizers match independently calculated
training-only channel means/stds; raw arms retain exact zero/one buffers.

CPU raw-observation replay still fails1e-4: max native logit discrepancy0.815565,
max pixel discrepancy0.0159254. Five factual-side predictions differ across three
writer7801 runs: native raw case111, stored raw13/109, stored standardized45/74.
All native image classifications agree across devices in this comparison. Exact GPU
replay and within-device bank equality do not fix cross-device portability.

Two actual public-only Claude reviews resolve the per-route calibration and empirical
cache-equivalence contracts. Every run and the overview has a self-contained report
with structural QA, all eight example panels and score/learning-curve plot inspected.
Browser QA unavailable. Original runs/weights retained; no further fit or promotion.

Next proposed repair: keep upstream modules frozen and train native output heads on
both ordinary and reset working states, rather than optimizing only reset readout.
Use the weaker writer as a separate conditional-accessibility check, and retain CPU
numerical gates. The exact mixed-context sampling/budget remains to be declared;
this result does not justify another memory cue or automatic codec expansion.


Post-comparison review found and fixed a reconfiguration edge case: a raw run started
from an already-standardized export could retain the old route's scaling. A new RED
check demonstrated it. Configuring a route now starts with identity scaling; that
route's training cache may fit new statistics, and standalone loading restores its
saved buffers afterward. Ten focused tests pass (including one new regression;
59 distinct relevant tests across this slice). All eight original GPU exports again
reproduce native/reference logits and pixels exactly after the fix. Formal training
remains source504a587 and its audited raw/standardized buffers were correct; no result
was rerun or replaced. The post-fix replay receipt is retained separately.

## Mixed ordinary/reset output training protocol — 13 September

User approved the proposed next experiment. Hypothesis: exposing the same native
output heads to both running and reset workspaces repairs ordinary output while
retaining recall. This tests a training policy at a fixed total budget, not a unique
mechanism or equal reset exposure. Encoder, writer, thinker, initial state, decoder
reconstruction head and existing reference readers stay frozen. Only the same
67,032 native factual/image-producer parameters train. No added context feature,
memory cue, architecture expansion or learned retrieval claim.

Four fits: sources `runs/reader_supervision_v1/agent_{7801,7802}_baseline/weights.pt`
each get reset-only and mixed training. Native route, identity output scaling.
Both arms start from identical source head values, not the later reset-only export.
Same sampled history IDs at each update, seed8401,1536 updates, batch16,
AdamW lr0.001/wd0.0001, clip1,180s training cap per fit; final checkpoint only.
Train128 pairs seed7701, validation32 pairs7762, test64 pairs7763. Familiar balanced
appearance/motion support with fresh backgrounds; exact frame disjointness checked.
Mixed training replaces eight reset presentations per batch with ordinary ones:
position p uses ordinary iff (p+step)%2==0, one-based step; phase resumes exactly.
Reset-only uses all16 reset workspaces. Histories/total presentations match; mixed
intentionally has half the reset presentations. Heads have no cross-example layers.
Training uses detached cached workspaces/teacher targets; cache/live equality for
BOTH contexts checked on shuffled batches. Record token differences and actual
per-step context counts. Calibration, if exercised in tests, uses the pooled TRAIN
contexts only; formal runs use raw values. No held-out fitting.

Success requires the existing full gate for each mixed fit: ordinary AND reset
facts/images>=90%, selection and relocation complete pairs>=80%, weighted image
error below both fixed-background and paired-target-mean controls. Original-target
accuracy must drop>=30 points after all-history erasure (both modes), and after
bank/cue/later-view erasure (reset); swapped-bank alternate facts/images>=80%.
These>=90% intact baselines are required alongside causal drops. Policy-benefit
gate additionally requires ordinary facts AND images improve>=10 points versus
matched reset-only training, with reset losses<=3 points for both outputs.
Report each writer separately; replicated repair requires both writers pass both
full and benefit gates. No threshold changes, ratio sweep or more training after
seeing failures. CPU/GPU replay tolerance1e-4 is a separate portability gate.

Essential RED checks: two-context cache alignment and live loss/gradient equality,
balanced step mask including odd-step resume, frozen upstream/buffers, standalone
ordinary/reset replay and cache identity participating in strict resume. Record
each cache preparation, including resume. Development16 updates, seed18401,
train17701/validation17962/test17963,60s cap. GPU cap4GiB with1GiB free headroom.
Relevant CPU tests, independent NumPy metrics, source/cache/frozen-state hashes,
actual GPU replay and CPU categorical differences. Existing standalone renderer,
structural QA and inspected figures; browser QA remains unavailable.

Actual Claude reviewed two public-only conceptual briefs. It initially required
a reset-dose-matched third arm, then accepted that this is a policy comparison and
that mechanism identification is a separate study. Adopted checks: actual workspace
differences, context counts/phase, no cross-example leakage and matched intact/erased
accuracy gates. The intact>=90% requirement above already supplies its final caveat.
Receipts: `runs/reviews/continuation_2026-09-11/mixed-context*-receipt.json`.

Implementation check: two RED failures demonstrated the missing mixed cache path;
44 relevant CPU tests and lint now pass, including mixed loss/gradient equality,
TRAIN-pooled normalization, both mask phases, exact odd-step optimizer/RNG resume
and ordinary/reset standalone reload. Development16 updates complete in1.2498s,
cache preparation2.1106s; standalone structural QA and example inspection pass.
Formal writer7801 mixed run will pause after769 updates and resume the remaining767
to check mask phase through a real GPU restart. Source is committed before fitting.

## Mixed ordinary/reset results

Source f0ac167. Four1536-update fits complete under180s caps, all67,032 native output
parameters train; writer, thinker, encoder, initial state, reconstruction head,
reference probes and raw banks remain unchanged. Each matched pair has identical
head initialization, reset caches, targets/features, dataset identity and sampled
history sequence. Actual mixed training uses12,288 ordinary plus12,288 reset
presentations; reset-only uses24,576 reset presentations. No extra context feature.

Held-out128 histories (familiar tuples/motions, fresh backgrounds):

| Writer | Training | Ordinary facts | Ordinary images | Reset facts | Reset images | Full gate | Policy benefit |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
|7801|reset only|82.8125%|58.59375%|97.65625%|96.09375%|False|control|
|7801|mixed|96.875%|96.875%|97.65625%|94.53125%|True|True|
|7802|reset only|75%|75%|75%|75%|False|control|
|7802|mixed|75%|75%|75%|75%|False|False|

Writer7801 mixed improves ordinary facts14.0625 points and images38.28125 points;
reset facts are unchanged, reset images lose1.5625 points, within the declared3-point
allowance. Ordinary complete relocation pairs93.75/93.75%; reset95.3125/89.0625%.
It passes both the full task and policy-benefit gates. This repairs the measured
ordinary-state regression for this source without crossing the recall-loss allowance.
It does not establish a unique mixing mechanism, equal reset dose, or full-agent
reliability. Replicated repair fails because writer7802 does not improve.

Causal screens for the successful mixed model: erased bank and entire history
facts/images6.25/6.25%; cue erasure6.25/4.6875%; later-view erasure43.75/42.1875%;
swapped-bank alternate facts/images97.65625/94.53125%. Ordinary/reset weighted MSE
0.00227671093/0.00263550482 beats background0.06856732070 and pair mean0.01887924969.
Timestamp erasure/reversal remain exact native no-ops. These are controlled-object
image classification/pixel results, not general image synthesis or exact reconstruction.

Unchanged source on this same test: writer7801 ordinary96.875/96.09375%, reset90.625/
89.84375%;7802 all75%. Thus mixed training preserves the source's ordinary output
while improving recall. Old stored-working readers7901/7902 score98.4375/83.59375%
for7801 and75.78125/75% for7802; their frozen native-workspace readouts remain weak.
Head-only training changes native outputs without changing the underlying tokens.
Training histories:7801 mixed ordinary100/100%, reset99.21875/98.046875%; reset-only
ordinary81.25/56.25%, reset100/99.609375%. Both7802 arms score75% in both modes and
outputs even on training data. This bounds this particular policy/budget, not the
information content or best possible reader.

46 relevant CPU tests pass, including exact optimizer/RNG resume at an odd step,
TRAIN-pooled calibration and standalone ordinary/reset replay. All1,826 saved metrics
independently recomputed (948 fit-test,212 training,474 source,192 frozen-reader),
max error7.57e-8. All four GPU ordinary/reset native/reference logits and pixels
replay exactly. Both cached working-state inputs and teacher targets exactly match
shuffled live16-example batches across all256 training histories; labels/targets
align. Ordinary and reset workspaces differ for all256 histories in both sources
(RMS1.63049/1.14463), so the intervention is substantive. Bank reorder is exact on
both devices. Source files, frozen tensors/probes and banks are unchanged.

Actual769+767 GPU restart preserves776 prior ledger rows and the next mask phase;
strict cache identity and optimizer/RNG resume also pass CPU tests. All cache
preparations now have receipts, including resume. Four training times35.310234,
35.840297,34.805949,34.877570s total140.834049s; cache preparation9.209641s including
resume. Peak458MiB reserved. Caps exclude cache preparation and final evaluation.
Development16 updates1.249775s plus2.110640s cache,458MiB.

CPU numerical tolerance1e-4 still fails: max native logits1.10498428 and pixels
0.02662665, though ALL ordinary/reset factual and image labels agree with GPU on
this slice. The successful mixed export's max logits0.82806873/pixels0.02592957.
Earlier device-dependent label failures remain historical evidence; current
categorical agreement is not a portability repair.

Two actual public-only Claude conceptual exchanges; implementation review and
empirical checks performed locally. Every run plus overview has a self-contained,
structurally verified report; development/four example panels and chart inspected.
Browser QA unavailable. Report: `runs/mixed_context_v1/report.html`; full settings,
exports, raw predictions, cache receipts and audits retained alongside it.

Example repeat (use a NEW output directory):
`OMP_NUM_THREADS=2 .venv/bin/python -m experiments.memory_output --weights runs/reader_supervision_v1/agent_7801_baseline/weights.pt --repair identity --readout-stage native --readout-context mixed --seed 8401 --validation-seed 7762 --test-seed 7763 --output runs/mixed_context_repeat --device cuda:0`

Next proposed: target the weaker writer/workspace formation while retaining mixed
output training. A matched comparison allowing the writer to learn is a candidate;
its exact loss/budget and causal claims need declaration before any fit. Current
probe/head failures do not prove missing information or a writer-only cause.
Keep CPU numerical portability separate. No further training launched.

## Shared observer/writer learning protocol — 13 September

User requested the next Claude-reviewed implement/test/fix iteration. Test whether
unfreezing the shared observation updater repairs the weaker source while retaining
mixed ordinary/reset native output training. This updater forms historical writes,
ordinary state and reset-query state; the experiment does not isolate a memory-only
mechanism. Initial state, encoder, thinker, monitor, dynamics, actions, reference
readers, reconstruction head and all calibration buffers stay fixed. Only native
heads plus updater action/attention blocks learn; updater uncertainty scale stays
fixed because this task's monitor feedback is detached.

Runtime episodic writes are detached. During each short supervised training episode,
retain the two actual observed-state tensors and replace ONLY the temporary bank's
values with their stack. Assert exact equality to the normal write values at each
write; preserve keys, times, sources, capacity and causal guards. Keys/selection
remain detached/nondifferentiable. Default inference/export still uses ordinary
detached writes, with no persistent training graph or cross-batch memory. This opens
a task-loss gradient to earlier writes; it does not add inputs or auxiliary labels.

Four fits from `runs/mixed_context_v1/agent_7802_mixed/weights.pt`: training seeds8501
and8502 × frozen/trainable updater. Both arms run the SAME live mixed-context loss;
no working-state caching in either. Same source heads, history samples, optimizer,
updates and16 output presentations per update. Ordinary iff(position+step)%2==0.
Native raw output; zero auxiliary loss. AdamW lr0.001/wd0.0001, clip1,1024 updates,
360s training cap per fit, final checkpoint only. Train128 pairs7701; validation32
pairs7772; held-out64 pairs7773. Familiar tuples/motions, fresh backgrounds. Two
training seeds from one weak source are limited optimization repeats, not independent
writer replication or a precise effect-size estimate. Additional trainable capacity
and gradient paths are the intended treatment, not separately identified causes.

Primary repair: BOTH trainable fits pass existing full ordinary/reset>=90% factual
and image accuracy, complete selection/relocation pairs>=80%, pixel error below
background/pair-mean controls; intact baselines>=90% with>=30-point all-history,
bank/cue/later-view erasure drops as already specified; swapped alternate>=80%.
Benefit additionally requires>=10-point gain in all four ordinary/reset factual/image
metrics against each seed's frozen control. Report individual gates if replication
fails. CPU/GPU numeric1e-4 remains a separate portability gate. No tuning thresholds,
extra writer losses, ratio sweep or budget expansion after formal results.

Essential RED checks: equal runtime/replay bank values/metadata/outputs pre/post
update, isolated earlier-write gradients with query state detached and zero path
after bank erasure, refreshed live inputs after optimizer updates, empty new episode
banks, frozen buffers/modules, no stale cache, exact optimizer/RNG odd-step resume,
and standalone inference using detached banks. Audit current live states and historical
bank provenance; changes in bank content are allowed only in trainable-writer arms.
Compare unchanged source/stronger successful control and two frozen diagnostic readers
on the fresh test. Recompute metrics independently and audit GPU/CPU exports.

Development16 updates: seed18501, train17701/validation18072/test18073,60s cap.
Formal8501 trainable pauses at513 then resumes511 to exercise odd mask phase.
GPU cap4GiB with1GiB headroom. Existing report renderer, structural QA plus inspected
figures; browser QA unavailable. Record source, raw metrics and elapsed/resource
receipts. Commit plan/RED checks, then implementation, before formal fits.

Actual Claude reviewed two public-only briefs. It accepted the shared-updater scope
after initially treating the diagnostic detached query as a required training rule.
Adopted exact replay bookkeeping checks at every training write, gradient-isolation
tests, live recomputation, episode isolation and explicit limited-seed/capacity claims.
Direct stored-state auxiliary supervision is a different future intervention, not
part of this comparison. Receipts: `runs/reviews/continuation_2026-09-11/writer-learning*-receipt.json`.

Implementation verification: two RED failures demonstrated missing live replay.
All50 relevant CPU tests and lint pass; earlier-write gradient isolation uses the
actual fresh hidden-query state detached from the updater. Runtime/replay values
and query tokens agree before and after learning; no stale workspace cache is used.
Both frozen/trainable policies pass exact optimizer/RNG odd-step resume and standalone
ordinary/reset export tests. Frozen control has67,032 trainable parameters; adding
the shared updater action/attention blocks gives75,768 (8,736 extra). Development16
updates complete in3.190238s,402MiB reserved, structural report and inspected panel.
No quality-based tuning. Implementation is committed before formal comparisons.

## Shared observer/writer learning results

Source bc7a7a6. Four1024-update fits finish; all full and replicated10-point benefit
gates fail. On fresh test7773, frozen seed8501 ordinary/reset facts/images all74.21875%;
frozen8502 ordinary75.78125/75.78125%, reset74.21875/74.21875%. Trainable8501 ordinary
83.59375/89.84375%, reset81.25/75%; trainable8502 ordinary86.71875/82.8125%, reset81.25/
75%. Training ordinary scores83.984375/91.796875% and85.9375/83.203125%, reset81.25/
75.390625% and81.25/75.78125%. A reliable repair is not established in either repeat.

The stronger unchanged source on the same test scores97.65625/97.65625% ordinary,
98.4375/96.09375% reset; weak source all74.21875%. Both pre-existing readers remain
frozen and are rescored on changed stored/recalled states; input-coordinate shifts
mean probe failure cannot establish absent information. Shared-updater gradients and
extra trainable capacity, including current/reset-query processing, prevent a uniquely
historical-write interpretation of the partial gains.

50 relevant tests,1,826 independently recomputed metrics, exact four-export GPU replay,
bank-order and replay equality pass. Every live training write checks exact values;
new banks are empty, runtime inference banks detached, permitted writer changes occur,
other frozen tensors unchanged.513+511 restart and odd-phase resume tested. Total
training481.754467s. CPU numeric gate still fails: ordinary factual side case18 changes
for8501 trainable (GPU[0,1,1], CPU[0,1,0]); images and reset labels agree here.

A post-fit plateau diagnostic makes no optimizer updates. On the same first16 TRAIN
histories, updater gradient L2 is0.012015 at the source,0.176185/0.390894 at the final
trainable checkpoints. With the query detached, initial/stored token gradients remain
nonzero and disappear after bank erasure. Weights remain unchanged. This rules out
a disconnected pathway in this diagnostic, not ineffective optimization or missing
information. All evidence/report retained at `runs/writer_learning_v1/`; interpretation
does not promote a complete repair. The next iteration below follows this failure.

## Joint observer and workspace-reader protocol — 13 September

Continue the authorized iteration with one additional factor: unfreeze the thinker
(workspace attention reader and feedback network) alongside the observer and native
heads. Keep encoder, initial state, monitor, dynamics, action head, reconstruction
backend, reference readers, calibration buffers and bank mechanics fixed. Same live
mixed objective, ephemeral value-gradient replay and causal inference boundary; no
new input, curriculum or auxiliary loss. Add an opt-in `--train-thinker` flag requiring
trainable writer/native raw mixed context. Default loading must reproduce prior
exports exactly, including parameter counts and frozen/trainable sets.

Two new fits, seeds8501/8502, SAME original weak checkpoint
`runs/mixed_context_v1/agent_7802_mixed/weights.pt`, train7701/validation7772,1024
updates,batch16,AdamW0.001/wd0.0001,clip1,360s caps. Final-step selection; no schedule or
objective change. Test is fresh64-pair seed7783. Reuse prior writer-only checkpoints
as baselines by evaluating them on THIS SAME fresh test, with their original source,
training/validation/settings/checkpoint hashes explicitly preserved. Old test7773
numbers do not enter the new contrast. Same initialization/history sampling and budget;
two seeds from one original source limit effect-size and generalization claims.

Primary joint repair requires BOTH new fits pass the existing full>=90% intact
ordinary/reset factual/image,>=80% paired/alternate,>=30-point causal-drop and pixel
baseline screens. Benefit additionally requires>=5 points in ALL four ordinary/reset
factual/image metrics over each matched writer-only baseline on7783. Report these gates
separately; five points is declared now for this incremental comparison, not a revision
of the failed prior10-point writer-learning screen. No tuning after formal results.

Essential new RED check: joint writer/thinker gradients and exactly permitted parameter
updates; default-off freeze semantics and invalid configurations. Exact odd-step resume
and standalone detached inference for joint mode. Verify all previous writer-study GPU
exports under the default-off new code before baseline reuse. New fits use no persistent
cache; record both old/new source identities and exact matched optimizer/sample budgets.
Development16 updates seed18601, train17701/validation18172/test18173,60s cap. GPU4GiB
cap with1GiB headroom. New GPU/CPU replay, NumPy scoring and retained individual/overview
reports; structural QA and inspected figures, browser QA unavailable.

Two more public-only Claude exchanges accept this bounded comparison and clarify that
fresh re-evaluation of reused baselines IS valid comparison evidence; stale prior-test
numbers are excluded. They require preserved provenance, default-off regression checks,
freeze/gradient checks and limited attribution. Receipts: `joint-reader*-receipt.json`
under the same ignored reviews directory. No claim of a uniquely isolated memory cause.

Joint implementation check: new gradient/freeze test failed RED before the flag was
added. All52 relevant tests pass, including joint odd-step resume and standalone
inference with detached banks; lint passes. Default-off loading reproduces all four
writer-study GPU ordinary/reset exports exactly and matches original trainable
names/counts. Compatibility receipts live separately under `runs/writer_reader_v1/`;
original writer-study receipts are retained. Joint parameters85,560. Development16
updates3.243214s,404MiB; structural report and inspected panel pass. No quality tuning;
source is committed before the two formal fits and fresh baseline evaluations.

## Joint observer and workspace-reader results

Source 872b4d7. Both final 1024-update joint fits pass the full task gate on fresh
test 7783; both unchanged writer-only baselines fail it. Exact matched scores:

| Training seed / policy | Ordinary facts | Ordinary images | Reset facts | Reset images |
| --- | ---: | ---: | ---: | ---: |
| 8501 writer only | 84.375% | 90.625% | 81.25% | 75% |
| 8501 writer + thinker | 99.21875% | 90.625% | 100% | 95.3125% |
| 8502 writer only | 86.71875% | 83.59375% | 81.25% | 75.78125% |
| 8502 writer + thinker | 93.75% | 92.1875% | 94.53125% | 91.40625% |

All required pair, causal erasure, alternate-target swap and pixel-baseline gates
pass for both joint fits. The separate five-point benefit gate passes only 8502;
8501 ordinary image accuracy is unchanged. Replicated full task success therefore
passes, while the stricter combined task-and-benefit gate fails. Neither threshold
was revised. Joint training ordinary facts/images reach 100/91.015625% and
97.65625/92.578125%; reset 100/97.265625% and 97.265625/91.796875%.

Same train/validation data, initial full model, sampler states, optimizer settings
and update budgets are verified. Baselines retain source bc7a7a6 and original config,
checkpoint and training evidence; their fresh test outputs have separate evaluation
provenance. No baseline retraining or use of old test 7773 scores in this comparison.
Joint changes both permitted updater and thinker; baseline changes only updater.
All frozen parameters/buffers remain equal to the original source. Every actual
write has matching replay values, source and time; detached inference, bank-order
invariance and temporary replay outputs pass on CPU and GPU. GPU exports are exact.

52 relevant CPU tests pass, including new RED-first gradient/freeze and joint exact
odd-step resume/export checks. Independent scoring verifies 1,826 metrics: 948 fresh
checkpoint-test metrics, 212 training-fit metrics (including original baseline
arrays), 474 unchanged-source metrics and 192 frozen-reader metrics. The earlier
513+511 GPU restart belongs to the writer-only baseline; joint restart is tested
by the CPU resume test, not claimed as another formal GPU restart. Default-off
loading reproduces all four old writer-study GPU exports and trainable sets exactly.

CPU numerical portability still fails at tolerance 1e-4. Maximum native logits
0.2179412842 and image pixels 0.0079333782; joint 8501 ordinary factual shape changes
at case 67 (GPU [2,0,0], CPU [2,1,0]). Reused writer 8502 ordinary image side changes
at case 76 (GPU [3,0,0], CPU [3,0,1]). Other ordinary and all reset categories agree
here. Capability scores are GPU results. Frozen readers are rescored on changed
coordinates and cannot establish absence of information from failed predictions.

Joint training takes 369.148984s, peak reserved 423,624,704 bytes (404MiB). Fresh
baseline evaluation takes 18.056552s, zero optimizer updates. Combined formal training
across the two iterations is 850.903451s for six fits; separate developer runs use
3.190238s and 3.243214s. No cache preparation, budget expansion or test-guided tuning.
Every completed run and comparison has a standalone structurally verified report;
development, all eight formal/evaluation panels and both comparison charts inspected.
Browser QA remains unavailable. Raw records: `runs/writer_reader_v1/verification.json`,
`default_off_compatibility.json`, `implementation_checks.json`, per-run `reuse.json`
and [report](../runs/writer_reader_v1/report.html).

Interpretation: joint adaptation is sufficient for the declared task screen in two
optimization seeds from one original model. This does not identify a uniquely broken
writer/reader component, demonstrate independent-source robustness or establish
general entity learning. Image scores measure synthetic color/shape/side correctness;
rendered outputs retain artifacts despite passing pixel controls. Next proposed:
replicate across independently initialized upstream models before extending histories
or distractors; keep CPU portability and broad multimodal capability gaps open.

Reproduce a joint fit with a fresh output directory (preserve completed runs):

```bash
OMP_NUM_THREADS=2 .venv/bin/python -m experiments.memory_output \
  --weights runs/mixed_context_v1/agent_7802_mixed/weights.pt \
  --repair identity --readout-stage native --readout-context mixed \
  --writer-learning trainable --train-thinker --seed 8501 \
  --validation-seed 7772 --test-seed 7783 --device cuda:0 \
  --output runs/joint_writer_reproduction
```

## Existing-source robustness protocol — 13 September

User adopts independent-initialization validation and continued Claude iteration.
Use the two existing agent sources from initialization seeds 7801/7802, sharing a
frozen codec. Both followed the same five preparation stages (relocation1536;
identity repair1536; untimed continuation1536; unsupervised continuation1536;
mixed head training1536). Both were previously explored, and the joint recipe was
selected using source7802. This is robustness across existing sources, not blinded
replication or independently initialized codecs. Exact lineage, prior-exposure and
initial-weight/codec checks go in `runs/cross_source_v1/preparation.json` before fits.

For source7801, four NEW fits: writer-only versus joint, each optimizer seed8501/8502.
Source `runs/mixed_context_v1/agent_7801_mixed/weights.pt`. Reuse the four unchanged
source7802 checkpoints from writer_learning_v1 and writer_reader_v1; re-evaluate all
cells on fresh64-pair test7793. Same train7701/validation7772,1024updates,batch16,
AdamW.001/wd.0001,clip1,live equal ordinary/reset loss, final-step selection,360s caps.
No changes to training mechanics. Primary robustness gate: ALL FOUR joint cells pass
existing full>=90% factual/image ordinary/reset,>=80% pair/alternate,>=30-point erasure
and pixel-baseline gates. Secondary source-retention gate: no joint output accuracy
falls more than5 points below its unchanged source on the same fresh test. Report
writer-only contrasts descriptively for every source/seed; no positive gain required
at ceiling. Preserve earlier failed benefit gates separately. Do not treat cell/test
examples as independent model-initialization replicates or compute a population CI.

Implement an evaluation-only path in the existing memory_output recipe: load exact
export settings, preserve original run/checkpoint provenance separately from current
code/data/environment, no optimizer or copied training ledger, no overwrite. Complete
results survive report failure. Essential RED tests cover these failure modes and
source immutability; existing gradient/freeze/resume tests remain. Tiny development
fit16updates seed18701/train17701/validation18272/test18273; development evaluation
uses the fresh-evaluation path. Do not inspect formal test scores to choose settings.

Preflight verifies source histories, differing seeded agent initialization/tensors,
identical codecs/calibration, within-pair starts and independent sampler states,
fresh test and exact-frame disjointness. Fresh scorer gets no old metrics as inputs.
Audit all8 cells with independent NumPy scoring, GPU replay, frozen tensors, permitted
updates, bank order/value/time/source identity and detached inference. Separate CPU
portability checks stay at1e-4 including categorical differences. Source controls use
the same fresh test. No new fitting of probes. Four fits max1440s training; GPU4GiB
cap with1GiB headroom. Reuse report renderer; structural checks and inspected figures,
browser QA unavailable. Commit plan/RED, implementation/development, then formal runs.

Claude public-only review requested explicit prior-exploration ledger, codec checks,
seed semantics and source/evaluation identity. Adopted; clarified fresh procedural
test is from the same task distribution, so no literal absence of all correlation.
Review receipts `cross-source*-receipt.json` under the existing reviews directory.

Implementation: two RED provenance tests precede the new `--evaluate-only` path.
All54 relevant tests pass. Development exposed a real JSON-list versus checkpoint-
tuple comparison bug in nested calibration settings; a regression reproduced it,
canonical JSON comparison fixed it, and both focused tests pass afterward. Failed
`development_eval` remains preserved; `development_eval_fixed` completes with a
structurally verified report. Training16 updates3.397702s; both development panels
inspected. Training/objective/scoring function ASTs match872b4d7; model code unchanged.
No training-policy change was introduced by the new evaluation entry point.
Claude accepted the final clarifications: reconstructed exposure ledger, every-pair
checks, ties count as success for both without superiority, and partial-render failure
preservation without a power-loss guarantee. Three bounded public-only exchanges.

Post-fit report review found empty evaluation ledgers still rendered a misleading
training-chart placeholder. A RED assertion reproduced it; the existing renderer
now skips curves when no metric rows exist. Two focused evaluation tests and all
8 run/report regression tests pass. Seven evaluation-only reports were rebuilt,
with before-fix HTML/QA receipts retained and raw metrics unchanged. Training and
inference code are unchanged; formal source remains6c9f284, renderer repair recorded
separately. Browser QA unavailable; report validation remains structural plus figures.

## Existing-source robustness results

Formal source6c9f284; post-fit empty-chart renderer fixb1cbc04. Fresh test7793:

| Source / optimizer seed / policy | Ordinary facts | Ordinary images | Reset facts | Reset images | Full gate |
| --- | ---: | ---: | ---: | ---: | --- |
| 7801 / 8501 / writer | 100% | 100% | 100% | 100% | pass |
| 7801 / 8501 / joint | 100% | 100% | 100% | 100% | pass |
| 7801 / 8502 / writer | 100% | 100% | 100% | 100% | pass |
| 7801 / 8502 / joint | 100% | 100% | 100% | 100% | pass |
| 7802 / 8501 / writer, reused | 84.375% | 89.84375% | 81.25% | 76.5625% | fail |
| 7802 / 8501 / joint, reused | 99.21875% | 86.71875% | 100% | 93.75% | fail |
| 7802 / 8502 / writer, reused | 86.71875% | 83.59375% | 81.25% | 75% | fail |
| 7802 / 8502 / joint, reused | 92.1875% | 88.28125% | 92.96875% | 89.0625% | fail |

All-source joint task robustness fails; the separate <=5-point source-retention
screen passes all four joint cells. Both policies succeed in the stronger source,
so that source does not demonstrate joint-training superiority. The earlier weaker
source's test7783 gates remain passed on that population; fresh7793 failures reveal
sample sensitivity, without altering those earlier results or benefit thresholds.
Unchanged source7801 ordinary97.65625/96.09375%, reset96.875/91.40625%; source7802
ordinary75/76.5625%, reset75/75% facts/images.

Weaker joint ordinary image color/side are100% in both seeds; shape86.71875/88.28125%.
There are16/8 ordinary cases with entirely correct factual outputs but incorrect
images, independently confirmed from saved arrays. First eight per checkpoint are
shown as explicitly selected error examples, with original case indices retained.
This establishes access for the factual head on those cases; it does not uniquely
identify which representation/feature producer/decoder component needs repair.

All54 relevant tests pass; focused two-test nested-settings and report regressions
rerun after fixes, plus all8 run/report tests. NumPy verifies2,794 metrics:1,896 fresh
cell-test metrics,424 original training-fit metrics,474 unchanged-source metrics.
All four pairs have exact initial weights and sampler/optimizer budgets; differing
upstream initializations and identical codecs/input calibration confirmed. Fixed
decoder calibration equality also checked after fitting. All source files unchanged.
Every eight-checkpoint GPU ordinary/reset export reproduces exactly; permitted
writer/thinker updates, other frozen tensors, bank value/time/source provenance,
detached inference and bank-order/replay equality pass. CPU audit covered all eight
cells in two disjoint partitions, first three overlapping the last GPU fit.

CPU numerical tolerance1e-4 still fails: maximum native logits0.2586591244 and pixels
0.0092121363. Source7802 joint8501 ordinary factual shape case99 changes from GPU
[2,0,0] to CPU[2,1,0]. All other native categories agree in this sample. Earlier
cross-device factual/image disagreements remain evidence; no portability claim.

Four new fits complete in515.275510s training, peak423,624,704 bytes (404MiB). Reused
cells and source controls perform zero optimizer updates. Twelve completed individual
reports (including development fit/evaluation) and the overview are structurally
verified; ten formal/control panels, two development panels, two error panels and
the comparison chart inspected. Failed original development evaluation is retained;
seven before-fix evaluation report copies preserved. Browser QA unavailable.
Raw results/audits/provenance in `runs/cross_source_v1/`; [report](../runs/cross_source_v1/report.html).

Interpretation remains conditional on two previously explored sources and a common
codec. No new initialization population or unseen-world generalization was tested.
The next proposed comparison is image-output-only learning on frozen final joint
states versus matched continued joint learning, retaining factual and causal gates
and using a new declared test. This is a proposed diagnostic, not a decided repair;
no further training launched after this comparison.


## Image-output continuation protocol — 13 September

User adopted the proposed diagnostic. Hypothesis: further image-producer learning
can repair residual shape output errors from frozen final joint states, without
changing factual answers. Compare against continued joint adaptation as a practical
training-policy contrast. Parameter count, coupled gradients and optimization
cost differ by design; this does not isolate one mechanistic cause.

Sources: writer_reader_v1/seed_8501_joint and seed_8502_joint, both previously explored
and sharing the same upstream initialization and frozen codec. For each source,
image-only versus joint continuation with optimizer seed8601/8602 respectively.
Four NEW fits, identical within-pair source tensors, sampler, data and optimizer.
Train128 pairs seed7701, validation32 pairs19072, fresh test64 pairs19073. Validate
fresh test identity and exact-frame split disjointness before fits. Final1024updates,
batch16, AdamWlr.001/wd.0001,clip1,360s per-fit cap,4GiB GPU cap/1GiB free headroom.
Unchanged sources get evaluation-only reports on19073. Development16updates on
train17701/validation19372/test19373, seed19601, max60s. No test-based selection,
extra budget, capacity sweep or follow-up fit within this comparison.

Only factor is trainability. New opt-in image-only configuration freezes updater,
thinker and factual head; only the image feature producer learns. Joint continues
updater+thinker+facts+image producer. Both retain frozen input encoders, image
reconstruction backend, initial state, probes and calibration buffers. Both run the
same live histories and alternating8 ordinary/8 reset loss examples per batch;
no cache. Factual CE+weighted RGB MSE+.1 standardized feature MSE remains identical;
CE contributes no gradient in image-only. No detached image targets enter inputs.
Fresh optimizer state in both arms, not an exact continuation of the old optimizer.

Primary gate: each arm's BOTH source cases pass the unchanged full task gates
(ordinary/reset>=90% facts/images,>=80% selection/relocation pairs and swapped
alternate accuracy,>=30-point causal erasure drops, pixel baseline gates). Separate
image sufficiency/retention screen: each ordinary/reset image accuracy>=90%, no
image decrease>2 points versus its unchanged source, joint facts decrease<=2 points.
Image-only factual logits and workspace tensors must remain exactly unchanged on
the same GPU; this is an implementation invariant, not evidence of learning.
Report gains, shape errors, factual/image disagreements, feature/pixel loss curves
and trainable counts. No superiority claim from both passing. Failure cannot prove
information absent or rule out larger capacity/budget. Success is conditional on
these two explored optimization trajectories, not independent-init replication.

Essential RED checks: exact freeze/update and factual/workspace invariance, reject
conflicting configuration, standalone trainability restoration and exact odd-step
resume. Retain all default-off tests. Tiny development fit and structural report
inspection before source commit/formal fits. Independently recompute every saved
metric, audit paired initialization/sampling, original source immutability, frozen
parameters AND buffers, native GPU export replay, detached inference bank provenance
and replay/order equivalence. CPU numeric1e-4 and categorical portability audited
separately before conclusions. Reuse renderer; browser QA unavailable, inspect PNGs.

Claude public-only review accepted constant CE and requested explicit conditional
scope, parameter/cost reporting, split-context gates and calibration checks. Adopted.
A post-hoc doubled-budget suggestion is deferred: it cannot strengthen the declared
fixed-budget claim and would require a fresh protocol/test. All frozen tensors and
source workspace comparisons cover silent calibration drift. No private code/results
are sent externally; local implementation review and tests remain necessary.

Implementation completed: opt-in `--image-only` freezes the factual head in the
existing native live recipe, with conflicting flags rejected and export/resume
freeze rules restored. Two RED checks reproduced before implementation; all56
relevant tests pass, including exact odd-step restart and standalone predictions.
Source reconfiguration preserves every source tensor; image-only58,128 versus
joint85,560 trainable parameters. Fresh19073 and disjoint split frames verified.
Development16updates3.098006s completed; structural report and panel inspected.
Final review also corrected image-only report limitation text to identify the
factual head as frozen. Two public Claude exchanges found no remaining material
conceptual contradiction; implementation review remains local.

Post-fit audit found a real inference-mode defect. All four task gates passed,
but the declared bitwise frozen-state invariant failed on GPU, despite identical
frozen weights/buffers. Factual logit deltas versus unchanged-source evaluations
were up to4.2915344e-5, with no categorical change. Explicitly freezing the source's
parameter flags makes its states/facts exactly equal in the diagnostic; CPU had
matched already. Preserve the original comparison and failed invariant. This is
not evidence that image learning altered frozen parameters. Parameter flags affect
GPU numerical execution even under no_grad in this environment; no unverified
claim about a particular backend kernel is needed.

Repair evaluation to temporarily clear requires_grad flags and restore them even
on exceptions. RED checks precede repair. No additional optimization, checkpoint
selection, budget or gate changes. Re-evaluate all four exact checkpoints and both
unchanged sources on the SAME declared19073 test under the corrected inference
contract in distinct evaluation directories. Preserve original raw metrics/arrays
and expose pre/post numeric and categorical differences. Record post-fix inference
and gate results separately from the original bitwise failure. Re-run exact resume,
export and all relevant tests; audit GPU replay and source-equivalent frozen states
with matched inference flags. CPU portability remains an independent gate.

The two inference-mode RED checks reproduced the missing normalization. Repair
uses a try/finally flag restore; all58 relevant tests now pass, including restart
and failure paths. The existing development checkpoint evaluates successfully
under the new contract, with zero optimizer updates and a separate report.
The original development limitation-text error was also corrected with before
copies retained and metrics/gates/checkpoints unchanged.


## Image-output continuation results

Training source1964a33; inference repair090ab5a. Final corrected evaluation uses
unchanged checkpoints and the same test19073; no extra optimization or selection.

| Source / policy | Ordinary facts | Ordinary images | Reset facts | Reset images | Full gate |
| --- | ---: | ---: | ---: | ---: | --- |
| 8501 / unchanged | 100% | 91.40625% | 100% | 97.65625% | pass |
| 8501 / image-only | 100% | 100% | 100% | 100% | pass |
| 8501 / joint | 100% | 100% | 100% | 100% | pass |
| 8502 / unchanged | 97.65625% | 92.1875% | 99.21875% | 90.625% | pass |
| 8502 / image-only | 97.65625% | 100% | 99.21875% | 100% | pass |
| 8502 / joint | 100% | 100% | 100% | 100% | pass |

Both policies pass both full-task cases and all separate image-sufficiency/source-
retention screens. All rendered color/shape/side labels are correct in ordinary
and reset modes. Joint improves the second checkpoint's remaining factual errors;
image-only preserves factual outputs exactly under matched inference. Both unchanged
sources already pass the minimum gate on this sample: the benefit is reduced
residual error, not first full-task passage. Earlier7793 failures remain preserved.
Both sources share one upstream initialization/codec and were previously explored.
No independent-initialization or general-world success claim follows.

The original GPU bitwise invariant failure remains recorded under
`before_inference_fix/`: source/fitted factual differences up to4.2915344e-5, with
unchanged categories. Matching parameter trainability flags gives exact states.
The repaired evaluator temporarily freezes all parameters, then restores flags
including on exceptions. Six evaluation-only runs preserve original manifests,
checkpoints and raw scores; every categorical answer and full gate is unchanged
across inference versions. Maximum numeric difference across all saved outputs is
0.0001435279846. GPU standalone outputs and frozen-source workspaces/factual logits
now reproduce exactly. This repairs the recipe's inference contract; direct model
calls must use matched flags for bitwise comparisons. It does not imply CPU/GPU
numerical identity or identify a specific backend kernel without further evidence.

All58 relevant tests pass after the two new inference RED regressions, alongside
image-only gradient/freeze, exception restore, exact odd-step resume and standalone
checks. Independent NumPy scoring verifies3,056 values:1,422 original evaluations,
1,422 corrected evaluations and212 unchanged training-fit metrics. Both pairs have
exact source starts, sampler streams, optimizer/update budgets and data identities.
All frozen parameters/buffers unchanged; permitted image-producer updates verified.
Banks retain exact source/value/time/order and detached inference behavior. No new
probe fit, cache, image backend, capacity or loss was introduced.

CPU1e-4 numerical gate still fails: maximum native logits0.1853666306 and pixels
0.0020754635. All native factual/image labels agree on this test; earlier categorical
mismatches are not erased. Four1024-update fits cost507.666166s training (8.46min),
peak423,624,704 bytes (404MiB); development16updates3.098006s. Inference correction
adds zero optimizer updates. Fourteen individual reports and overview structurally
verified; original/final comparison panels and score/loss figures inspected.
Browser QA unavailable. Development's erroneous adapting-factual-head limitation
text corrected with before copies and unchanged metrics/gates/checkpoints.

Two actual Claude public-only design exchanges; local implementation/repair review.
[Comparison](../runs/image_continuation_v1/report.html) includes exact results,
loss components, causal controls, source mapping, original failure and corrected
inference receipts. `--image-only` is a targeted option, not a declaration that
frozen-state training is universally preferable. Both completed policies should
remain fixed for a newly declared robustness evaluation before longer histories
or real observations. No further benchmark or fit launched.


## Fixed-checkpoint nuisance robustness protocol — 13 September

User adopted the fixed-checkpoint robustness follow-up. Keep all four final
image_continuation_v1 source8501/8502 image-only/joint checkpoints and both their
unchanged writer_reader_v1 sources fixed. One upstream initialization and codec,
two previously explored optimization trajectories. No new optimizer, selection,
calibration, training, capacity change or test-guided repair of model performance.

Declare three new procedural test seeds20073/20074/20075,64 pairs each (128 histories,
32 semantic quartets). Seed varies background texture/intensity only; semantics,
positions, shapes and label combinations remain familiar. Each seed is evaluated
at input RGB offsets-16,0,+16 in8-bit units, applied to ALL observed pixels and
frames, including the hidden query. Target renderings remain canonical and labels
unchanged. Offset is an operational exposure perturbation, not new scene semantics.
Verify no clipping, invertible shift, nearest canonical foreground color preserved,
exact selection/movement quartet ambiguity and target equality before inference.
Keep default offset0 data/identity exactly backward compatible. Nonzero transform
identity records parent-image hash and offset. Expose `--input-offset` only on the
existing evaluation-only CLI; training rejects this evaluation override.

Six checkpoints x three seeds x three offsets =54 evaluation-only runs, all10 causal
query modes, with standalone reports. GPU4GiB cap/1GiB headroom; zero training budget.
Evaluation/report loop wall cap900s (preserve partial completion if exhausted),
separate verification up to900s; no extra cases selected from failures. One separate
workflow development evaluation at seed21073/offset+16, same128 histories, no fitting.
Preflight records checkpoint hashes, origin/settings, split freshness against prior
runs, disjoint exact frames from original train/validation/test populations and
between current seed/offset populations. No claim of semantic or model independence.

Primary neutral gate per policy: both checkpoints pass existing full task/causal
screen at EVERY new seed/offset0. Primary stress gate: same conjunction at both
nonzero offsets. Separate retention gate: each ordinary/reset factual/image accuracy
loses<=5 percentage points against its own same-seed neutral cell.5pp is a declared
engineering tolerance, not statistical significance. Report each accuracy and exact
continuous margin for all seeds/offsets, never pool to conceal a failure. Source
controls receive all cells but are excluded from trained-policy conjunctions.
At each offset/seed, image-only factual logits and state must equal the unchanged
source under matched inference flags. This is a freeze invariant, not brightness
invariance or learning evidence. Keep GPU numeric and CPU numeric1e-4/categorical
portability separate; no population CI from correlated seeds/checkpoints/quartets.

Essential RED tests cover transformed inputs/unchanged canonical targets, quartet
ambiguity, no-op identity, clipping/type rejection and evaluation-only CLI routing.
Extend existing immutable-export test to a shifted dataset to verify recorded
transform/source provenance. Relevant data/evaluation/freeze/resume tests follow.
Do not add another runner/framework; new reusable code is the small dataset input
transform and existing recipe option. Independent run-local orchestration/scoring
is retained beside raw results.

NumPy re-scores all54 outputs and task gates. Audit unchanged checkpoint/parent
hashes, paired populations, detached memory values/time/source/order and exact GPU
replay for all36 trained-model cells. CPU checks cover the same36 cells, with
separate categorical differences, no silent tolerance relaxation. Reuse existing
renderer; all55 individual reports and overview require structural receipts and
inspected representative figures. Browser QA unavailable.

Claude conceptual review requested color-semantics checks, continuous per-cell
margins, explicit single-lineage scope and tolerance labeling. Adopted. Its extra
post-hoc offset suggestion is deferred to a separately declared protocol. Clarified
that factual/state invariance compares source and image-only at the SAME offset;
images themselves are tested, not assumed invariant. Public-only briefs; private
implementation remains local. No fresh outcomes examined while setting this plan.

Trace coverage is the first complete16-history batch of each population; full
ordinary/reset export predictions and factual invariance cover all128 histories.
This bounds inspection overhead; it is not an exhaustive per-intermediate-tensor
proof for every example. Default-off replay and objective/forward AST equality
against2bb8bb9 are checked before formal evaluation.

Implementation complete: nine RED transformation/CLI cases reproduced, all67
relevant tests pass. Shifted-data export provenance and report-failure preservation
verified. Eight training/evaluation function ASTs and model source remain unchanged
from2bb8bb9; default datasets match original manifests. Preflight verifies all nine
populations, clipping/inverse/color margins, paired ambiguity, unchanged checkpoints
and exact-frame disjointness. Development evaluation completes with zero updates,
structurally checked report and inspected panel. Claude reconciliation finds no
material conceptual contradiction. No formal population scored before this commit.

## Fixed-checkpoint nuisance robustness results

Formal evaluation source f78b8ff; plan/RED commit 74b6bb4. All 54 declared cells
completed with zero optimizer updates. Each trained policy passes all six neutral
full-task cells, fails all twelve shifted full-task cells and all twelve shifted
five-point retention checks. Source controls pass three of six neutral cells and
none of twelve shifted cells. Controls do not enter the trained-policy conjunctions.

| Policy / input | Ordinary facts | Ordinary images | Reset facts | Reset images |
| --- | ---: | ---: | ---: | ---: |
| Image-only / neutral | 96.094–100% | 98.438–100% | 95.313–100% | 96.875–100% |
| Joint / neutral | 100% | 100% | 100% | 100% |
| Image-only / −16 | 39.844–47.656% | 35.938–45.313% | 41.406–51.563% | 39.844–45.313% |
| Joint / −16 | 32.813–43.750% | 35.156–46.094% | 35.156–39.844% | 35.156–42.188% |
| Image-only / +16 | 31.250–38.281% | 31.250–35.938% | 35.938–42.969% | 35.938–42.188% |
| Joint / +16 | 50.781–65.625% | 42.969–57.031% | 57.813–61.719% | 48.438–57.813% |

Ranges describe both checkpoints and three declared background seeds, not
independent-model uncertainty. All six checkpoints share one upstream initialized
agent and codec. Familiar semantic quartets and rendering patterns are unchanged.
New backgrounds confirm scoped neutral retention; the additive input stress screen
fails decisively. No offset, gate or checkpoint was changed after seeing outcomes.

The frozen direct reader drops from 100% neutral to 62.5% under −16 and
69.531–73.438% under +16. Both facts and images lose color/shape/side accuracy.
This broad sensitivity does not support a decoder-only account, but does not locate
where information becomes inaccessible: the direct reader has its own learned head.
No encoder-information-loss or general physical-lighting claim follows. Canonical
color assignment, inverse pixel transform, clipping bounds, unchanged targets/labels
and paired-history ambiguity all pass independently of tested model predictions.

All 67 relevant tests pass after nine transformation/CLI RED cases. Eight existing
recipe function ASTs and model source equal 2bb8bb9; default data identities match
old manifests. NumPy recomputes 12,798 metrics (maximum discrepancy 1.172e-8) and all
54 task gates, including each continuous margin. Origins/checkpoint hashes and
zero-update records pass. Every image-only saved factual logit equals its unchanged
source at the same input condition. All 36 trained-model GPU ordinary/reset replays
are bitwise exact over 128 histories. First 16 histories per cell pass detached bank
source/value/time/order, training-value replay and frozen-workspace checks. Traces
are sampled intermediate checks, not an exhaustive tensor proof.

CPU numerical tolerance 1e-4 fails in all 36 cells: maximum native-logit discrepancy
0.4816207886, pixels 0.0140872598. Native categories differ in 15 stressed cells and
in no neutral cell. Counts are ordinary facts 4, ordinary images 8, reset facts 7,
reset images 7 example/query/checkpoint comparisons; these may overlap and are not
unique-world counts. Detailed indices and both answers are retained. This is a
separate portability failure; earlier numerical/categorical findings remain valid.

The 54-cell evaluation/report loop took 449.777682s (7.50min); recorded evaluation
compute sums to 429.097025s. No training or calibration, peak reservation
310,378,496 bytes (296 MiB). One separate workflow development export also had zero
updates. All 55 individual reports and overview verified structurally; heatmap and
representative panels inspected. Browser QA unavailable. Two actual Claude public-
only design/reconciliation exchanges; private implementation and results reviewed
locally. No private scores or code were sent to Claude.

[Report](../runs/output_robustness_v1/report.html) retains all cells, attribute scores,
continuous gate/retention margins, paired source contrasts, 12 predetermined panels,
provenance and numerical/categorical audit records. Authoritative summaries are
`verification.json`, `array_audit.json`, `gpu_audit.json`, `cpu_audit.json` and
`final_checks.json` beside it; run-local scripts reproduce the audits and report.
The individual recipe remains the public entry point with `--evaluate-only` and
`--input-offset`; the evaluation transform is deliberately rejected for training.

Next proposed, not implemented here: predeclare training-time brightness
augmentation and a normalization diagnostic, preserving neutral retention and
fresh held-out confirmation. First test whether frozen features support a better
reader before deciding to adapt encoders. These examined stress populations are
now diagnostic data. A larger decoder or longer unchanged training is not yet a
supported repair. No additional fit, normalization or test selection occurred.

## Brightness repair protocol — 13 September

User adopted the augmentation/normalization follow-up. First test output-head
adaptation on frozen working states, before changing encoders or state formation.
Use both final joint checkpoints from image_continuation_v1 (source8501/8502).
These share one upstream initialization/codec and are already explored; no
independent-model replication or new semantic support is claimed.

Four matched fits: per source, cached native mixed ordinary/reset output learning
on five neutral copies versus five variants at RGB offsets −16, −8, 0, +8, +16.
All variants use train7701,128 base pairs; expanded set 1,280 histories. Concatenate
complete paired/quartet blocks, retain canonical targets, and record ordered offsets
and parent data identity. Both policies use identical sample indices, initial weights,
AdamW settings, 1,536 updates of batch16, optimizer seeds8701/8702, wall cap180s/fit.
Only factual head and image feature producer learn. Freeze encoders, initial state,
updater, thinker, memory behavior, direct/reference probes, backend and scaling.
No fresh calibration or output standardization. Check cached/live tokens on a
training batch before formal fits. Match training presentations, not presumed
optimization difficulty; failure at this budget cannot locate lost information.

Validation23072 is neutral and diagnostic only. Confirmation seeds23073/23074,
64 pairs each, offsets −12,0,+12: new backgrounds and interpolation within the
training perturbation family. No model/checkpoint/offset/gate selection from them.
Previously examined20073–20075 populations are diagnostic history only. Preserve
train/validation/test hashes and exact-frame disjointness checks. One separate
16-update development fit uses17701/24072/24073 and the augmentation policy.

A separate zero-update normalization diagnostic centers each observed frame per
RGB channel on a fixed median from the original neutral training observations.
Per-frame spatial median subtraction uses no offset metadata, labels, targets,
background mask or later frames. Keep observation validity/time/provenance and a
serialized calibration identity/reference in the evaluation manifest. Reject
out-of-range valid inputs/results instead of silently clipping. Default behavior
unchanged. This is a task-specific evaluation preprocessing option, not a new
trained checkpoint or learned general invariance. Dominant background motivates
the reference; median translation equivariance itself is algebraic. It may discard
useful absolute intensity, so neutral fidelity is tested separately.

Evaluate four fitted exports, two unchanged sources and two centered-source
conditions on all six confirmation populations:48 immutable standalone evaluations,
all10 causal modes, plus the four training-run neutral exports. Existing full task
screen must pass every neutral/stress cell for a policy. Neutral retention allows
at most5 percentage points loss in each ordinary/reset factual/image metric against
its same-source neutral frozen baseline. Augmentation benefit requires at least
5 points in EVERY stress factual/image metric against matched neutral-only fitting;
centering benefit uses the unchanged source at that same stress input. Full passage,
retention and benefit are separate conjunctions. Report all per-cell metrics and
continuous margins; engineering thresholds are not confidence intervals.

Budget: four fits <=720s training, evaluation/report <=900s, verification <=900s,
GPU4GiB cap with1GiB headroom. No extra fit chosen from failures. Essential RED checks:
augmentation alignment/provenance/clipping, cache/live equality and gradients,
odd-step exact augmented resume/export, median equivariance/per-frame causality,
validity/provenance and bounds, immutable centered export and CLI scope. All existing
relevant tests follow. Original defaults must reproduce saved GPU outputs exactly.
All48 outputs independently rescored including task gates; all48 ordinary/reset
GPU and CPU replays, frozen parameter/source checks, first16-history bank/workspace
traces. Preserve CPU numeric1e-4 and categorical failures separately. Reports use
existing renderer, structural receipts and inspected figures; browser QA unavailable.

Claude public review accepted controls and scope. Its stronger negative-result
inferences were corrected: failed head learning does not uniquely justify encoder
replacement; failed centering does not prove deeper feature entanglement. Claude
acknowledged these corrections and median translation algebra. Two public-only
exchanges; implementation/results remain local. No formal confirmation scores seen.

Implementation verified before formal fitting: 78 relevant tests pass, including
augmented exact odd-step resume and centered immutable-export/report-failure checks.
GPU preflight found CUDA median-with-indices incompatible with deterministic mode;
original failure preserved, replaced with deterministic sorting for the same lower
median. All 13 affected tests pass again; GPU cache/live/source equality and centering
bounds/equivariance pass. Fresh confirmation hashes and frame disjointness verified.
Development16-update augmented fit completes with a verified report/inspected panel.
Eight old training/inference/scoring function ASTs unchanged. No formal test scores
used to change the protocol. Centering reference is serialized in evaluation records;
it remains an explicit evaluation-time diagnostic, not a modified weight export.

## Brightness repair results

Plan/RED ebc04b7; implementation/formal source 9ff3be3. Four matched fits completed
at 1,536 updates of batch16, adapting 67,032 factual/image-feature parameters each.
Initial models, sampler/RNG states, presentations and targets match within source;
all frozen parameters/buffers remain exact. Only permitted output parameters change.
No encoder, updater, thinker, memory policy, image backend or calibration is trained.
Original neutral source weights remain unchanged for the centering diagnostic.

| Condition | Neutral full task | Stress full task | Neutral retention | Stress benefit |
| --- | ---: | ---: | ---: | ---: |
| Unchanged source | 4/4 | 0/8 | 4/4 | Not applicable |
| Neutral-only heads | 4/4 | 0/8 | 4/4 | Not applicable |
| Augmented heads | 4/4 | 0/8 | 1/4 | 6/8 |
| Centered source | 4/4 | 8/8 | 4/4 | 8/8 |

Each conjunction requires all its cells; augmentation fails stress task passage,
neutral retention and the separate benefit conjunction. Centering passes them all,
with 100% ordinary/reset factual and image-category accuracy on every population.
This is a zero-optimizer-update preprocessing result, not a newly trained checkpoint.
It is exposed through `--evaluate-only --center-input`, records its training-only
reference, and remains opt-in. Existing default exports still reproduce exactly.

| Augmented input | Ordinary facts | Ordinary images | Reset facts | Reset images |
| --- | ---: | ---: | ---: | ---: |
| −12 | 59.375–69.531% | 64.063–76.563% | 66.406–67.969% | 62.500–69.531% |
| Neutral | 94.531–100% | 93.750–96.094% | 92.969–100% | 93.750–95.313% |
| +12 | 85.938–93.750% | 87.500–92.969% | 79.688–86.719% | 89.844–91.406% |

Ranges describe declared cells, not independent-model uncertainty. Neutral-only
continuation and original sources score100% on neutral inputs but fail every stress
full-task gate. Both sources share one upstream initialized agent and codec; new
backgrounds and intermediate offsets do not establish unseen semantics, realistic
lighting, general generation or perfect pixel reconstruction. Canonical targets and
complete paired/quartet histories stay unchanged; no clipping or test-based choice.

A post-hoc descriptive breakdown of saved training predictions adds80 per-offset
accuracies without a new gate or fit. At the seen −16 training offset, augmented
factual accuracy is46.094–62.891% across sources/query modes. Thus the limitation
already occurs on training variants; it cannot be attributed solely to interpolation
at new offsets. Equal budgets do not equalize optimization difficulty. Optimization,
head capacity and accessibility of frozen working states remain alternatives; no
unique encoder information-loss or replacement conclusion follows.

Centering removes uniform additive shifts algebraically: median(x+c)=median(x)+c
without clipping. Task success is a separate empirical result. A dominant stable
background motivates the chosen reference, which can remove meaningful absolute
intensity elsewhere. Every observed frame is processed independently before the
encoder, including recalled/replayed state formation, while time, validity and
derived provenance are preserved. No labels, target pixels, shift metadata, mask
or future frame enters the operation. GPU preflight initially rejected CUDA median
under deterministic execution; failure log retained, deterministic sort implements
the same lower median and passes all bounds/equivariance checks.

All78 relevant tests pass and13 focused checks pass after that correction, including
augmentation alignment, cached/live gradients, exact odd-step resumed training,
standalone loading and centered immutable-export/report-failure behavior. NumPy
independently reproduces12,536 recorded metrics:11,376 confirmation,948 original
fit-export and212 training-fit metrics. Maximum confirmation metric discrepancy
1.118e-8. All48 ordinary/reset GPU replays over128 histories are bitwise exact, as
are both original default exports. All48 first16-history sampled bank source/value/
time/order/replay traces pass, including source-equivalent frozen workspaces for
trained heads. Sampled traces are not an exhaustive intermediate-tensor proof.

CPU tolerance1e-4 fails all48 cells: maximum native-logit discrepancy4.991394043,
pixels0.035994232. Native categories differ in11 cells (4 augmented,4 control,
3 unchanged,0 centered). Counts: ordinary factual2/image3, reset factual4/image9
example/query/checkpoint comparisons, potentially overlapping. Centered categories
match GPU throughout, but numerical portability is not repaired. Earlier failures
and their original thresholds are retained.

Training139.516520s (2.33min), all-fit loop276.386649s including cache preparation
and reports;48-cell evaluation/report365.402301s (6.09min). Peak training reservation
746,586,112 bytes (712 MiB), within4GiB cap. Development16 updates1.191706s, isolated
from formal confirmation. Two bounded actual Claude public-only design/reconciliation
reviews; private implementation/results reviewed locally. Fifty-three individual
reports plus overview verified structurally; heatmap, loss curves and representative
panels inspected. Browser QA unavailable. No new fit selected from these outcomes.

[Report](../runs/brightness_repair_v1/report.html) includes exact cells, continuous
margins, training curves, per-offset training diagnostics, eight predetermined
comparison panels, calibration provenance and full audit receipts. Run-local scripts
and raw JSON/NPZ/checkpoints are retained under `runs/brightness_repair_v1`.

Next proposed: vary backgrounds, object coverage and nonuniform illumination in a
new declared screen before treating centering as a default input policy. If pursuing
learned invariance, first investigate the head/state optimization shortfall under a
fresh protocol with neutral retention. No automatic deployment or further training
was performed; current test populations are now examined diagnostic evidence.

## Proposed brightness robustness closure checklist

Status review, 13 September. This scopes “this part” to visual observations feeding
the current memory task and its factual/image outputs. It is a proposed definition
of completion, not an approved new experiment protocol or a claim about general
image generation. No additional training or evaluation was launched for this review.

Already demonstrated: centered original weights pass all 12 declared synthetic
conditions, including ordinary/reset factual and image-category accuracy and causal
gates. The 78 relevant tests pass and 48 GPU replays are exact. Augmented head
training remains an unsuccessful repair; CPU numerical portability remains open.

The following checks are required before making centering the normal input policy
for a declared synthetic operating range. Specify populations, ranges, fresh seeds,
budgets and any new thresholds before running; retain the existing gates below.

| Check | What must be established |
| --- | --- |
| Define nuisance versus signal | Specify which lighting changes should leave identity/state unchanged and which brightness/color changes are actual task information. Current image targets are canonical, not copies of the altered camera pixels. Include paired cases with nuisance changes and genuine object/state changes. |
| Challenge the scene assumptions | Vary background color, brightness, texture, clutter and foreground coverage. Include large objects and scenes without a stable background reference. Correct identity, state and output must survive within the declared domain. |
| Challenge illumination | Test fresh additive offsets and temporally varying shifts; separately test gain, contrast/gamma, per-channel color shifts and local shadows. Test selected combinations after single-factor diagnosis. Median centering only guarantees cancellation of uniform additive shifts without clipping; broader task performance needs evidence. |
| Preserve memory and causality | Change lighting between selection, last observation and query; include occlusion/reappearance and distinguish appearance changes from identity changes. Repeat ordinary/reset, selection/relocation pairs, history/cue erasures and memory swaps. Check that preprocessing uses neither future frames nor target/offset labels. |
| Preserve the clean task and image content | Retain factual/image-category gates, neutral-retention bounds and pixel-error baselines. Examine actual colors, shape, location, boundaries and representative reconstruction errors. Category accuracy alone cannot establish high-quality pixels. |
| Fresh confirmation and replication | Separate diagnostic and confirmation scenes; freeze the policy/reference before confirmation. Use independently initialized upstream models as well as new data seeds. The two existing source trajectories share an initialization and codec. Predeclare replication count and uncertainty reporting within budget. |
| Handle invalid and unsupported input | Existing guards reject clipping/out-of-range centered values. Define and test the consumer's fallback for rejection, saturation, missing/corrupt frames and inputs outside the supported range. Do not silently clip or write an invalid observation into memory. If claiming learned reliability, separately test whether predicted confidence tracks actual errors. |
| Integrate and reproduce the selected policy | Package the fixed training reference and transform version with the deployed configuration; test training/evaluation/streaming/recalled-input consistency and reload. Preserve time/source attribution. Resolve CPU/GPU discrepancies or explicitly support the validated GPU path only. Measure complete-path latency, peak memory and sustained memory growth; current 712 MiB measures the tiny training task. |

Existing minimum task gates remain: at least 90% factual and image-category accuracy
for ordinary/reset queries; at least 80% selection/relocation pair performance;
required causal erasure drops of at least 30 percentage points; at least 80%
alternate-target performance after memory swaps; pixel error below the recorded
background/pair-mean baselines. Neutral losses must stay within five percentage
points of the matched original. Retain the declared stress-benefit requirement
where comparing a repair. These are engineering tolerances, not confidence bounds;
new perturbation-specific ranges and criteria are still to be declared.

Learned robustness is a separate question from deploying a fixed normalization
policy. If it is required, first diagnose why augmented heads fail even on seen
training extremes. Use bounded optimization/capacity/readout comparisons with
matched controls and clean retention; probe state accessibility before expanding
the trainable encoder/state path. A failed probe or head fit alone cannot establish
information loss. Successful fixed centering would not close this learning question.

Before claiming webcam readiness, add held-out real recordings with different
sessions, backgrounds, motion, occlusion, exposure/white-balance changes, blur,
sensor noise and compression. Split by recording/session rather than neighboring
frames. Verify real-time latency, sustained memory use, and failure/re-observation
behavior at the intended camera resolution. This is a later boundary beyond the
synthetic closure, not a prerequisite to report the bounded synthetic result.

Efficient next order: (1) freeze original weights and challenge backgrounds,
coverage and frame-varying/local lighting; (2) repair only demonstrated failures,
including any consumer fallback; (3) confirm the selected path on fresh scenes and
independent upstream runs; (4) validate integration and the declared runtime;
(5) run a small held-out webcam pilot. Keep the learned-robustness branch separate
unless fixed preprocessing fails the required task or learned invariance is chosen
as an additional objective. Full graph learning, planning and other modalities are
not prerequisites for closing this bounded visual-input task.

## Centering scene and lighting challenge protocol

User adopted the open-point iteration. First complete this fixed-weight challenge
before choosing learned repair, integration or real-camera claims. No task weights
will change in this screen. The current output contract is a canonical selected
object (color/shape/final side), deliberately invariant to observed size/background
and the declared lighting changes. Actual changes between the existing color/shape
classes must still be distinguished. Absolute scene illumination is not a target;
centering discards it and must not be described as a generally lossless interface.

Two original sources: image_continuation_v1/source_8501_joint and
source_8502_joint. Raw versus training-referenced centered input; two fresh test
seeds25073/25074, 64 selection pairs each, relocation curriculum. Both sources
share an upstream initialization/codec: this is replication across explored
trajectories and fresh scenes, not independent upstream training. Development uses
seed26073 with16 pairs; confirmation is not used for implementation choices.

Ten declared conditions, each applied identically within a complete counterfactual
quartet, canonical targets unchanged:

| Condition | Exact observation change, in 8-bit units unless stated |
| --- | --- |
| neutral | Original observations |
| temporal-offset | Whole-frame offsets[-12,+12,-8] over the three frames |
| channel-offset | Whole-image RGB offset[+8,-8,+4] |
| background-tint | Background-only RGB offset[+12,0,-8], foreground/cue unchanged |
| background-texture | Background-only 8px checkerboard, amplitude8 |
| background-bright | Background-only RGB offset[+48,+48,+48] |
| foreground-large | Object half-size12 instead of8, cross half-width6 instead of4; selection border resized; canonical target size unchanged |
| clutter | Fixed RGB[70,95,110] bands at rows8:18 and46:56; no task objects/cues overwritten |
| gain-dark | Multiply all observed pixels by0.75, round to nearest integer |
| local-shadow | Subtract12 from the left half of every observed frame |

Implement a bounded scene-transform method on existing relocation episodes and
an evaluation-only recipe option. Dataset renderer may use its object metadata;
model inputs remain RGB only. No target/label/foreground mask/offset value enters
preprocessing or model. Transform provenance includes full source identity and
parameters. Reject unsupported arguments, transformations of already-transformed
sources and any uint8 clipping; retain all originals and default identities.

Essential RED checks: exact neutral/default rendering; label/target/source identity
and quartet/shortcut bounds; background-only changes spare object and cue pixels;
large rendering preserves color/shape/side and cue meaning; frame/channel additive
cancellation under per-frame centering; source/timestamp/validity causality;
non-injectivity when absolute brightness is meaningful; bounds/clipping guards;
CLI evaluation-only and provenance through immutable report export.

Pass criteria: unchanged full task gates and five-point retention relative to
matched raw neutral scores, required separately for every attempted condition.
Centering coverage must be100% of histories/frames for a condition to pass. If
centering rejects any history, report exact coverage/reasons and mark that cell
not passed; do not score a selected accepted subset or hide the rejection. For
all-rejected conditions end-to-end correct-answer yield is zero. For partially
rejected conditions no accuracy is claimed until an explicit partial-service
protocol exists. Report raw versus centered changes and pixel errors continuously;
new-scene robustness does not require an arbitrary positive gain over already
successful raw conditions. Prior brightness-repair benefit gates remain unchanged
for the earlier comparison; this screen has a different coverage/retention question.

Budget: zero optimizer updates;80 formal cells, each all10 existing causal modes,
900s evaluation/report cap,4GiB GPU cap with1GiB free headroom. Stop before writing
if free disk falls below3GiB; do not delete prior artifacts. Verify all recorded
metrics independently, GPU replay ordinary/reset, model/checkpoint immutability,
provenance and fresh-frame disjointness. Preserve rejected/failed cells. Reports
use existing renderer plus run-local overview; structural and figure QA, browser
QA if accessible. CPU portability remains an explicitly unresolved separate issue;
no repeated full CPU matrix is needed unless this work changes that path.

Claude public conceptual review supports axis separation and coverage reporting.
Reconciliation requested for overstrong claims: discarded median is non-injective
in-range; arbitrary frame-varying additive shifts cancel algebraically; neutral
retention and causal dependencies need not imply identical raw/centered outputs;
broader calibration alone does not guarantee repair. Private code/results remain
local. Next repair will be predeclared from diagnostic findings, not silently tuned
on these confirmation populations.

Pre-implementation:24 new cases fail for the missing scene API/CLI. Actual Claude
acknowledged all five corrections in the public reconciliation; no outstanding
mathematical disagreement. Both exact exchanges and receipts are preserved under
runs/reviews/continuation_2026-09-11/centering-challenge-*-public*.

Implementation/development:24 new integrity cases pass;72 affected cases including
export/report-failure variants pass. The GPU development export on separate26073
completes; its comparison panel was inspected and report structurally verified.
Before formal inference, population construction exposed a protocol arithmetic
error:10 conditions ×2 seeds ×2 sources ×2 input policies is80 cells, not40. The
declared conditions/seeds remain unchanged; corrected count fits the existing900s
and disk budgets (roughly37MiB per successful export). No formal model scores were
seen. Deterministic pixel preflight predicts partial centering rejection for bright
backgrounds; those cells will be marked failed coverage without subset scoring.

## Coverage reporting repair protocol

The new challenge exposes a user-path gap independently of task scores: generic
`--evaluate-only --scene background-bright --center-input` raises on out-of-range
centering, leaving result failed/report pending. The run-local challenge can report
coverage, but the normal recipe must offer the same inspectable failure outcome.
After the frozen screen and its audits finish, implement a shared per-frame range
inspection on PixelMedianCentering and recipe preflight. Keep strict forward range
rejection unchanged. If any history is unsupported, complete a coverage-failed
result/report with attempted/accepted counts and frame validity; run no model
queries, save no fabricated predictions, and do not score an accepted subset.
No automatic clipping, raw fallback, learned confidence or deployment is implied.
All-valid exports retain their original predictions and task gates.

RED checks: pure range inspection preserves time/validity and does not call the
wrapped encoder; consistency with strict forward guards including invalid masked
frames and bad valid values; rejected export leaves source model/checkpoint exact,
never invokes observe_history or an optimizer, and reports partial/all coverage
without an accuracy claim. Report failure must preserve a completed coverage
result and remain report-failed. Verify accepted export/reload behavior too.

Budget: zero optimization; focused CPU tests and at most four GPU exports (two
accepted regression cases and two rejected cases),150s cap and unchanged4GiB GPU/
3GiB disk limits. Accepted regressions reuse already-scored25073 neutral and
temporal-offset source8501, compared bitwise against this screen, explicitly not
fresh confirmation. Rejection examples use fresh27073, both source trajectories.
Keep the frozen screen's source identity and results intact; record the new source
commit separately. This repairs observability/workflow only, not perception accuracy.

## Centering challenge results

Formal source311c3ca;80 attempted cells,76 task-scored and4 coverage-failed.
Centered input passes12/40 full-task/retention cells: neutral, temporal additive and
RGB-channel additive conditions each4/4 at100% facts/images. Raw passes8/40: neutral
and background texture each4/4 at100%. Centering drops texture scores to76.562–86.719%.
All other condition groups fail every full-task cell. Centered clutter9.375–22.656%,
large objects45.312–58.594%, gain71.875–88.281%, local shadows71.875–91.406% across
ordinary/reset factual/image cells; these ranges do not imply all metrics pass.
Bright-background centering accepts only28/128 histories on25073 and20/128 on25074;
no accepted subset is scored. Raw brightness-background scores0–12.5%. Centered
background-tint direct-reader accuracy100% contrasts with native ordinary facts
73.438–79.688% and reset75.781–81.25%; this does not uniquely localize failure.

All18,012 recorded metrics and task gates independently verified. All76 scored
ordinary/reset GPU replays are bitwise exact on128 histories each. Original model,
checkpoints and recorded source files unchanged;12 original data cases byte-exact.
87 relevant CPU tests pass. Screen571.926s, zero training, peak298MiB. All82 reports
(including development/overview) structurally checked; heatmap and representative
panels inspected; browser QA unavailable. Both reviewed source trajectories still
share an upstream initialization/codec. [Report](../runs/centering_challenge_v1/report.html).

Centering remains opt-in. Broad learned robustness, independent upstream runs and
real recordings remain open. Five coverage-reporting tests now fail informatively:
missing pure range API and rejected exports incorrectly entering model evaluation.
The following implementation is a reporting/validity repair, not a perception fix.

Coverage implementation: pure per-frame range inspection shares the exact centering
computation with strict forward validation. Normal exports save input_coverage.json;
any rejected history yields completed result/gate-false/task-unscored, zero model
queries and no predictions. Report failure preserves that completed result. Five
new tests pass;45 focused checks including accepted exports, prior centering and
scene tests pass. No renderer change or clipping/fallback policy was introduced.

## Coverage repair verification

Source716d661; four GPU exports complete in16.367s with zero training. Both
accepted25073 neutral/temporal-offset source8501 exports reproduce every saved
prediction array and metric exactly against screen311c3ca, including all10 causal
modes. Fresh27073 bright-background exports on both sources accept16/128 histories,
record all per-frame decisions exactly against independent NumPy calculation,
complete coverage-failed reports, and generate no model predictions. All source
checkpoints and the repair's recorded source manifest remain unchanged during checks.

The turn covers92 distinct relevant CPU test cases, with45 focused reruns after the
range/report repair. The initial24 scene RED failures and5 reporting RED failures
remain preserved. No model weights were trained, no raw fallback/clipping was
introduced, and no broader robustness gate was relaxed. The screen and repair have
separate recorded code identities. All86 reports across screen, development,
overview and repair receive structural checks; representative panels/heatmap are
inspected. Browser QA unavailable. [Updated report](../runs/centering_challenge_v1/report.html).

Next learning question: begin a matched clean/scene-variation state/readout
comparison, starting with background tint where the existing direct reader proves
encoder-level accessibility on the tested population. Keep clean retention and
fresh held-out confirmation; do not infer lost encoder information from native
failure. Texture provides a required raw-input control against a harmful centering
prior. Clutter, size, gain, shadows and wider scenes require further training/design
experiments. Independent upstream initializations, real recordings, confidence
calibration, CPU portability and deployed streaming resource tests remain open.

## Centered tint readout learning protocol

User adopted the open-point iteration. Test native factual and image-feature heads
on frozen ordinary/reset working tokens, with fixed centering in both arms.
Sources are source8501/8502 joint image continuations (shared upstream initialization,
not independent replication). Compare three whole-history blocks: neutral/warm
background(+12,0,-8)/cool background(-8,0,+12), against three neutral copies.
Both retain canonical targets and complete selection/movement counterfactuals.
Neutral training7701 supplies the single fixed RGB reference before augmentation;
persist reference/version/calibration identity in checkpoint settings and buffer.
Standalone loads must restore it automatically and reject metadata/buffer mismatch.

Four fits:128 training pairs per block (768 histories),1,536 updates,batch16,
optimizer8901/8902,existing AdamW/loss,67,032 native output parameters only. No output
standardization; encoder, state writer, thinker, memory, reference readers and image
backend fixed. Neutral diagnostic validation28072/32 pairs and terminal test28071/64
pairs do not select checkpoints. Development17701/29072/29071,16 updates is separate.

Fresh confirmation28073/28074,32 pairs each; six conditions: neutral, warm/cool
training tint magnitudes, interpolated warm(+8,0,-4)/cool(-4,0,+8), and texture8.
Evaluate four fits and two unchanged centered sources on all conditions, plus two
raw sources on neutral/texture:80 cells, all10 task/causal modes. Preflight all RGB
ranges; any unsupported history fails coverage without subset scores. Full task
gates unchanged. Repair requires every augmented cell to pass, plus at most5pp loss
in ordinary/reset factual/image accuracy versus matched raw neutral and raw texture
controls. Primary warm-tint benefit requires at least5pp improvement in each of
those four accuracies versus matched centered neutral-only fit. Other differences
are descriptive; do not demand an improvement where a control is at ceiling.

Budgets:180s per fit,900s evaluation loop,4GiB GPU cap with1GiB headroom; keep3GiB
disk free, preserve old runs. Check data alignment, cache/live parity and gradients,
all frozen tensors, source/transform provenance, odd-step resume, standalone reload,
independent metrics and exact GPU replay. Record per-block training fit descriptively.
No model selection, third arm or live-state training after seeing confirmation.

Actual Claude public review requests explicit cache provenance and live consistency;
adopt both. Local corrections: native live evaluation with causal controls can
establish bounded trained-path success; a negative rules out this budget without
uniquely locating failure. Texture is a robustness control, not a leakage proof.
The follow-up reconciliation was rejected by automatic approval review as containing
nonpublic experimental details and was not sent. No bypass or further export.
Receipts: runs/reviews/continuation_2026-09-11/tint-learning-public-*.
Private implementation/results are reviewed locally; reconciliation is outstanding.

Tint implementation/development:9 essential RED cases reproduced missing scene
augmentation and checkpoint centering. Added whole-history scene blocks and bounded
warm/cool interpolation presets; fixed-reference centering now persists with training
exports and is automatically restored/validated before standalone evaluation. All
training/validation/test inputs receive coverage preflight; caches record reference,
version and source weights. Native inference/objective remain unchanged.
Initial101-case suite has98 behavioral passes and3 incorrect new expectations
(counts scale with repeated blocks; JSON normalizes tuples). Corrected expectations;
all16 tint-focused cases pass, including7 CLI/continuation checks. The suite includes
exact odd-step centered-scene resume. Sixteen-update GPU development29071 completed
with a structurally verified report and inspected panel. No formal results yet.

## Centered tint comparison results and resume guard

Formal source5451ce1; all four1,536-update fits and80 evaluations complete. Every
frozen tensor/buffer stays exact; starts, samplers, RNG and presentations match.
Native augmentation passes19/24 full task cells; both neutral-only and unchanged
centered controls pass14/24. Warm tint improves from0/4 to4/4 full task passes
(92.188–100% native accuracy). Mild warm/cool and neutral each pass4/4; clean native
accuracy98.438–100%. Strong cool passes2/4 (recall images as low as81.25%). Texture
passes1/4 and81.25–98.438%, while raw inputs remain100% on all texture cells.

Overall repair fails. All-scene accuracy within5pp of raw-neutral passes14/24;
clean-cell retention itself passes4/4. Texture retention passes0/4. Primary warm
benefit passes1/4; some control image scores exceed95%, leaving less than5pp of
possible improvement. This predeclared threshold is not relaxed. Training already
underfits: source8502 cool recall has100% factual answers and48/256 wrong rendered
shapes. Failure is not uniquely attributable to the encoder.

108 distinct relevant tests pass across the original suite and corrected focused
checks.20,120 metrics independently reproduced;80 ordinary/reset GPU replays exact;
12 original datasets byte-exact and7 inference/objective functions unchanged.
Training315.47s total, peak602MiB; full fit loop500.52s, evaluation458.95s. Concurrent
user GPU processes affected timing; no isolated speed claim. Before confirmation,
recorded timing_amendment.json increased elapsed evaluation allowance900→1,800s;
updates, populations, gates, VRAM and disk limits stayed fixed. Actual loop<900s.

The initially rejected Claude follow-up was subsequently explicitly authorized by
Alex and completed. Claude accepted cache/live provenance, scoped native live-task
success, bounded negative-result and texture-control corrections. One remaining
claim is not adopted: shared initialization limits across-model generalization,
but does not confound a within-model history intervention with weights held fixed.
Both actual review receipts and the initial rejection remain recorded.

Local review then reproduced a resume-only integrity gap using an intentionally
altered checkpoint: standalone loading checks centering buffer against settings,
but Run restoration inside training bypasses that check. Add the same validation
inside the existing failure-handled training block after restore, before updates.
A mismatch must mark failed status, preserve checkpoint/committed metrics and run
no optimizer steps. Regression: existing exact odd-step centered-scene resume stays
valid. Budget: focused CPU checks only, zero new capability training/evaluation.
Keep formal5451ce1 source evidence distinct; no inference/metric change is needed.

Resume repair: the new RED case attempted an optimizer update with an intentionally
mismatched restored reference. The new guard now rejects it, records failed status,
and preserves checkpoint and committed metrics. All18 focused cases pass, including
valid odd-step resume and standalone reload;109 distinct relevant cases pass across
this iteration. No scientific fits, scores or thresholds changed. The final fix
changes only this post-restore validation in train; all inference/objective/metric
functions and the80 verified GPU predictions retain their formal source evidence.

86 reports structurally verified and embedded PNGs decoded successfully; development,
accuracy grid and a labeled post-hoc cool-shape error panel inspected. Browser QA
unavailable. [Report](../runs/tint_readout_v1/report.html). Next: separate balanced
texture/shape fitting comparison with clean/raw retention, plus image-feature
producer diagnostics where factual shape is already correct. Do not conclude the
encoder lost identity. Independent upstream replication, real recordings, confidence,
CPU portability, broader scene/illumination coverage and streaming runtime remain
open. Disk headroom remains limited; another large matrix needs a fresh capacity check
while preserving completed runs and the3GiB free-space floor.

## Texture and shape continuation: locked diagnostic protocol

User requests the next implemented/reviewed comparison. Select the weaker source8502
augmented checkpoint from tint_readout_v1 deliberately for diagnosis, not blinded
selection or independent replication. Two equal continuations train native factual
and image-feature heads only (67,032 parameters), with fixed encoders/state/memory,
image backend, centering reference and independent probes. From identical weights,
seed9101, AdamW/loss unchanged,1,536 additional updates,batch16,180s training cap:
train7701/128 pairs per block; neutral,warm,cool,texture8 versus neutral,warm,cool,
neutral. Each arm has1,024 histories. Neutral validation30072/32 pairs and terminal
30071/64 pairs never select weights. Small separate development31701/31072/31071,
16 pairs each,16 updates checks the workflow before formal comparison.

Fresh confirmation30073/30074,16 pairs=32 histories per condition; complete paired
quartets and canonical tuple support. Six conditions: neutral,warm,cool,mildwarm,
mildcool,texture8. Evaluate both new fits and unchanged source on all conditions;
raw original8502 source on neutral/texture only:40 cells, all10 causal/task modes.
No new model or training adjustment after confirmation. Full task gates unchanged.
Repair requires every texture-arm cell to pass and all four ordinary/reset factual/
image accuracies within5pp of matched raw-neutral; texture additionally within5pp
of raw-texture. Report clean-only retention separately. Added texture-mixture benefit
is separate: pool the four response types across both seeds (256 predictions),
require at least25% relative error reduction versus continuation and no regression
in any of the eight paired response accuracies. Relative benefit is assessable only
when continuation has at least8 errors; below that, report unassessable, not a pass.
Always show absolute errors and exact differences per correlated four-history group;
no independence-based significance, inferred effective sample size or replication
claim. Both arms succeeding without assessable benefit supports extra optimization,
not a necessary texture-data advantage. Training-condition shape fit is descriptive.

Expose ordered scene training scores in the existing recipe/report, including shape
and joint accuracy, with duplicate clean blocks separate and provenance/alignment
checks. Keep aggregate training metrics and held-out gates unchanged. Essential RED
checks cover hidden shape failure, bad alignment, duplicate blocks, report labeling,
and the existing exact centered-scene resume path. CPU checks plus tiny GPU development,
then freeze source; independent NumPy scores, frozen-tensor/source/cache checks and
exact live GPU reloads follow formal runs. Browser QA for the changed table if available.

Budget: two fits,40 evaluation cells,900s evaluation loop,4GiB GPU cap and1GiB free
headroom; preserve all completed runs and3GiB disk floor. Starting disk4.4GiB supports
this smaller matrix, not another80-cell comparison. Actual Claude generic review
requested a matched continuation already present; correction sent for reconciliation.
Adopt condition-level diagnostics, absolute-error reporting and diagnostic-only scope.
Receipts: runs/reviews/continuation_2026-09-11/texture-*-public-*.

Development review:7 new RED cases reproduced missing scene diagnostics. Added the
ordered block artifact and standard report table. Actual Claude acknowledged the
existing continuation arm and withdrew its redundant third-arm request. It accepts
exact quartet differences without unsupported effective sample sizes. Residual wording
is narrowed locally: continuation versus unchanged measures that retained-mixture
continuation, not a data-independent pure optimization effect.

The first tiny GPU development reached16 updates but then exposed a diagnostic-only
CPU/CUDA rounding mismatch in reconstructed target floats. Preserve the failed run;
repair alignment validation to tolerate only float normalization rounding while still
checking original block hashes and labels exactly. Add regression coverage before
retrying development. This changes no training objective or held-out scoring.

Implementation verified:73-case CPU suite had72 behavioral passes and one test API
mistake (renderer returns a Path, not HTML); corrected. New roundoff RED regression
failed exact float comparison as intended. All9 focused checks now pass, including
exact centered-scene resume, all new diagnostics and1/255 target corruption rejection;
74 distinct relevant cases pass across this iteration. Repeated16-update GPU development
completed after the1e-7 absolute normalization tolerance; original uint8 block hashes
remain exact. Original failed development is retained with an explicit failure report.
Standard reports structurally verified; development example panel inspected. Supported
browser runtime initialized but URL policy blocked local-file navigation; no workaround.
Browser QA unavailable. Development does not establish capability. Source is now frozen
for the two fits and40 confirmation cells; thresholds and budgets unchanged.

## Texture continuation results

Formal source38cf923. Both1,536-update fits and all40 confirmation cells completed.
Texture-arm task passes10/12, versus8/12 for matched continuation and unchanged source.
Raw neutral/texture passes4/4 at100%. Added texture benefit passes:8 versus34 errors
across256 correlated responses (26 fewer,76.471% relative reduction), with no regression
in any of the eight paired response accuracies. Texture task gates improve0/2→2/2;
texture-arm accuracy93.75–100%. This is one deliberately selected difficult source,
not independent replication or256 independent trials. Exact quartet differences saved.

Full repair still fails. Cool-tint recall image accuracy81.25%/78.125%; both cool task
cells fail. All-scene raw-neutral retention7/12, clean retention2/2 at100%, and stricter
raw-texture retention1/2 (the other population has93.75%, below95%). Source/control
cool training recall image accuracy remains81.25%; texture training reaches79.297%.
In the texture fit53/256 cool training images have wrong shape (plus classified as
square);46 of those have correct factual answers. Control has48 such errors, all with
correct facts. Ordinary cool image accuracy is99.609% in the texture fit. Additional
mixed-head updates and texture data do not resolve this recall shape shortfall.
Do not infer unique encoder loss or assume more generic data fixes the decoder path.

74 distinct relevant CPU cases pass,9 focused checks after the diagnostic fix;
10,188 independently reproduced metrics;40 full-population ordinary/reset GPU reloads
bitwise exact. Both caches have exact recorded hashes and identical first three
blocks; targets/presentations/starts/RNG match and all frozen tensors stay exact.
Inference, objective, scoring and data generation are unchanged frome837ac7.
Training197.99s total, fit loop338.58s, evaluation189.83s; peak646MiB reserved. All
original runs preserved, disk floor maintained. No fixed-budget or gate amendments.

Two actual Claude conceptual reviews completed; corrections and residual interpretation
are in review_status.json. Private implementation/results reviewed locally.45 standalone
reports structurally checked with valid embedded PNGs; development, corrected overview
and selected error quartet inspected. Browser URL policy rejected local-file navigation;
no alternate route attempted. Browser QA remains unavailable. The first failed development
and its repaired failure report remain preserved alongside the successful rerun.
[Report](../runs/texture_shape_v1/report.html),
[verification](../runs/texture_shape_v1/verification.json).

Next: isolate the recall image feature producer's shape objective/gradients and capacity
while factual shape remains readable. Keep existing comparisons and clean/raw retention;
use fresh confirmation for any later intervention, not repeated selection on30073/30074.
Learned input reliability, general scene invariance, independent upstream replication,
real images, CPU portability and streaming runtime remain separate open points.

## Recall shape diagnosis before the next intervention

Continue the unresolved cool recall shape task. First run a zero-update local loss
and gradient diagnostic on the texture-trained source8502 checkpoint and its verified
frozen training cache. Fixed first32 histories of each of four256-history scene blocks
cover all16 canonical tuples and complete quartets. Ordinary and reset contexts are
separate. Record pixel and standardized-feature losses, producer gradient norms/cosine,
combined-gradient projection onto pixel descent, per-level feature contributions and
output saturation. Keep source/cache hashes exact; no confirmation samples or parameter
updates. Budget60s GPU,4GiB cap/1GiB headroom,3GiB disk floor. These are local diagnostics,
not proof of a unique failure mechanism. Actual Claude generic review is in progress.
Choose and preregister a bounded matched intervention only after this diagnostic.

Zero-update diagnostic completed: all eight scene/context aggregate gradient cosines
positive(.145–.635), no output saturation, and nonzero pixel gradients through every
feature level. On cool reset the weighted feature-gradient norm is2.59x pixel, cosine
.595; this does not support an aggregate opposing-gradient or clamp explanation.
A separate concrete concern is target-dependent foreground weighting: false negatives
cost10x false positives in the differing shape corners. Normalization also differs
by shape. Before changing capacity, verify that asymmetry with canonical target pairs
and test a training-only rectangle mask that treats the shared shape extent equally.
Keep existing evaluation metrics untouched and the teacher-feature coefficient fixed.
A second generic Claude review is pending; no causal mechanism inferred from gradients.

## Recall shape weighting comparison: locked protocol

Test the full training weighting policy (mask plus its own normalization), not mask
support alone. Keep the feature penalty0.1 and producer capacity unchanged. Compare
foreground-only weighting to weight10 throughout the foreground bounding rectangle
and1 outside, derived only from canonical training target pixels. For matched square/
plus targets the rectangles and box denominators must be exactly equal; blank targets
use uniform weights. This training supervision never enters inference. Existing
weighted-error evaluation metrics, nearest-template labels and all task gates remain
unchanged. Measure original loss asymmetry and corrected symmetry independently.

Both arms start from texture_shape_v1/source_8502_augmented, seed9501, frozen native
state, factual head, image backend, centering and calibration. Only58,128 image-producer
parameters train. Extend the existing cached image-only recipe path; do not add model
capacity. Train7701/128 pairs × neutral,warm,cool,texture blocks=1,024 histories;
1,536 updates,batch16,existing AdamW,180s training cap. Neutral validation33072/32 pairs,
terminal33071/64 pairs do not select. Development32701/34072/34071,16 pairs,16 updates.

Fresh confirmation33073/33074,16 pairs=32 histories per cell; same six neutral/warm/
cool/mild-warm/mild-cool/texture conditions. Both fits and unchanged source on all,
raw original8502 on neutral/texture:40 cells,all10 modes. Full repair requires all12
box cells pass unchanged task gates and all four ordinary/reset factual/image responses
within5pp of raw-neutral; texture additionally within5pp of raw-texture. Report clean,
texture and all-scene retention separately. Primary cool image benefit: pool ordinary/
reset images over both seeds (128 correlated responses), require≥25% error reduction
and no regression in any of four paired image accuracies. Assessable only if control
has≥8 errors; otherwise report unassessable. Factual logits must remain exactly equal
to unchanged source throughout every evaluation mode. Report absolute errors, original
weighted MSE and exact quartet differences; do not claim independent trials or unique
mechanism. No extra architecture/fit/threshold change after confirmation.

Budgets:2fits,40cells,900s evaluation,4GiB GPU with1GiB headroom,3GiB disk floor. Essential
REDs: canonical mask/symmetry and blank/error cases; default loss unchanged; box gradients
only into producer; CLI scope; exact odd-step resume/reload including settings; frozen
facts/state/backend. Small GPU development precedes source freeze and formal comparison.
Actual Claude accepts bounded gradient/ablation scope; follow-up correction addresses
its mistaken target-supervision leakage claim and mask-versus-normalization distinction.

Implementation: four informative RED failures covered missing loss, cached image-only
training, resume and CLI path; four invalid CLI cases also remain rejected after the
option exists. All47 focused checks pass:8 new weighting cases,14 direct-readout,
17 tint and8 scene-diagnostic cases. Default loss matches exactly; canonical reciprocal
shape errors become symmetric, blank targets finite, and target masks never enter
inference. Cached image-only training now accepts frozen native state with exact
odd-step resume/reload, settings provenance and frozen facts/backend. Existing live
image-only mode remains intact; box weighting rejects unsupported paths.

Sixteen-update GPU development completed; report structurally checked and panel inspected.
No shared renderer change. Prior browser local-file URL denial remains a QA limitation;
no workaround or repeated blocked navigation. Source now freezes for the comparison.
Three compact actual Claude reviews complete, with target-supervision leakage and
intervention-scope corrections acknowledged. The denominator shorthand in its final
reply is read only for equal box weights, not the foreground10:1 formula; numerical
reference checks remain authoritative. No private source/data/results were exported.

## Recall shape weighting results

Formal source0287a24. Both1,536-update fits and all40 fresh33073/33074 evaluations
completed. Box weighting passes12/12 task cells; matched foreground continuation and
unchanged source each pass10/12. Cool image errors5/128 vs10/128, a50% reduction with
no regression in the four paired image accuracies; the declared benefit passes.
Ordinary cool images are100%; reset cool images90.625%/93.75%, versus84.375%/84.375%
for continuation and81.25%/81.25% unchanged. These are correlated familiar synthetic
responses from one selected source, not independent replication or calibrated beliefs.

Full repair still fails the stricter retention criterion. Clean retention2/2 at100%,
texture retention2/2 at96.875–100%, all-scene retention9/12. Failures are two cool reset
image scores below95% and warm33074 reset factual accuracy93.75%. The latter is frozen
and exactly unchanged; image-only training cannot repair that factual output. All
factual logits, including every causal mode and independent probes, remain bitwise
identical across the24 trained evaluation cells versus their unchanged reference.

Training cool reset shape errors fall31→19/256; correct-facts/wrong-image cases24→12.
All remaining wrong image categories in that block are plus→square, with color/side
correct. The weighting-policy effect is demonstrated without capacity or feature-loss
changes, but neither masks alone nor a unique global mechanism is isolated. Both mask
support and per-image normalization are part of the intervention. The independent
conditional-risk calculation gives foreground threshold.0990099 versus box.5; these
are analytic objective thresholds, not measured model probabilities. Fixed-batch
positive gradient cosines and zero saturation do not rule out conflicts elsewhere.

All47 focused tests pass, including8 new loss/CLI/freeze/resume cases.10,188 metrics
independently reproduced;40 GPU reloads exact; identical full training caches, sampler/
RNG/initialization and all frozen tensors. Inference, scoring, default training behavior
and data generation remain unchanged. Training95.158s, fit loop169.802s, evaluation
92.157s, peak646MiB; resource floors preserved and no budget/threshold amendments.
Three compact actual Claude reviews complete; supervision, geometric equality and
intervention-scope corrections reconciled, with denominator shorthand qualified locally.
44 standalone reports structurally verified; development, overview and matched error
panels inspected. Prior browser URL denial remains a disclosed limitation; no workaround.
[Report](../runs/recall_shape_v1/report.html),
[verification](../runs/recall_shape_v1/verification.json).

Next: preserve the demonstrated loss improvement and separately diagnose the remaining
cool image and warm factual errors. A small producer-refinement/capacity comparison
can target correct-facts/wrong-image cases; factual repair needs its own declared
trainable path. Keep current thresholds and use fresh confirmation rather than selecting
on33073/33074. Broader learned reliability, natural scenes, upstream replication,
CPU portability and streaming remain open; no general generative backend has been trained.

## Producer refinement: locked protocol

Continue from recall_shape_v1/source_8502_box, preserving its rectangle-weighted
loss and frozen factual path. Compare one residual MLP per spatial feature scale
(LayerNorm, width→2width, GELU, 2width→width), inserted after query attention and
before feature projection, against the original producer given identical extra
optimization. Zero the last linear only: initial outputs must be bitwise equal;
its weights receive first-step gradients while earlier residual layers receive
zero gradients until that projection changes. Preserve RNG during construction.
This tests one added parameterization under a fixed optimizer/budget, not an
intrinsic capacity limit. The small pointwise MLP adds no spatial attention.

Both arms train all existing image-producer parameters; only the refinement arm
adds parameters. Facts, state formation, decoder backend, encoder, preprocessing
and all calibration stay frozen. Train7701/128 pairs on neutral,warm,cool,texture
(1,024 histories), seed9701, 1,536 updates, batch16, existing AdamW and box RGB loss
plus 0.1 feature loss. Validation35072/32 pairs and terminal35071/64 pairs never
select checkpoints. Development34701/36072/36071:16 pairs,16 updates. Essential
REDs cover identity/RNG/legacy reload, first and second-step gradient reachability,
frozen state/facts/backend, strict settings/load and exact odd-step resume.

Fresh confirmation35073/35074,16 pairs each, six prior conditions. Two trained
arms and unchanged box source on all conditions, raw original8502 on neutral and
texture:40 cells, all10 causal modes, original metrics/gates. Primary image repair:
all12 task gates plus ordinary/reset image accuracy within5pp of raw-neutral,
and texture additionally within5pp of raw-texture. Report full factual+image
retention separately; frozen factual errors cannot be attributed to refinement.
Added cool-image benefit requires≥25% relative error reduction vs continuation,
no regression in any of four paired cool image scores; assessable if control has
≥4 errors out of128 correlated responses, otherwise explicitly unassessable.
Report unchanged-source comparison, absolute errors, and quartet deltas. No
further fit, architecture change or threshold change after confirmation.

Two fits,40 cells,180s per training fit,900s evaluation,4GiB GPU ceiling with1GiB
headroom,3GiB disk floor. Inspect refinement activations/parameter changes after
training to establish use, not causal necessity. Verify cache/source/RNG/frozen
parameters, independent metrics and full-population GPU reloads. Existing report
renderer unchanged; structural/figure QA with prior browser limitation disclosed.
Actual Claude conceptual review supports the comparison; correct its first-step
gradient wording and retain its recommendation to inspect residual usage. Only
generic methods are sent externally; local code/results review remains separate.

Implementation/development: three informative RED failures now pass; all58 focused
checks pass (six refinement cases plus prior weighting/readout/scene tests and
five general image-output tests). Optional refiners preserve initial outputs and
RNG, legacy checkpoints load without new keys, saved refined checkpoints load
strictly, and odd-step resume is exact. Frozen facts/state/backend remain intact.
A sandbox GPU preflight could not access CUDA (retained development.log); the
approved local GPU retry completed16 updates with a standalone report and inspected
panel. No failed optimizer run was hidden. Ruff check/format pass. Two compact
actual Claude exchanges complete, first-step gradient correction acknowledged;
activation magnitude demonstrates use only, not causal necessity. Freeze source
before formal runs; no shared renderer change or browser-policy workaround.

## Producer refinement results

Formal source c22f1e5. Both 1,536-update fits and all40 fresh35073/35074 evaluations
completed. Each trained arm passes11/12 task cells, versus10/12 unchanged; raw4/4
at100%. Both have zero cool image errors out of128 correlated ordinary/reset
responses. Unchanged cool reset images score93.75%/90.625%; both continuations
reach100%. The primary added-capacity benefit is unassessable because the control
is at ceiling, not a demonstrated refinement win or a proof capacity never helps.

Image repair and full repair remain false. Both arms retain images in8/12 cells
under the fixed95% requirement; warm and texture fail on both fresh seeds. Refined
warm reset images87.5%/93.75%, versus81.25%/93.75% control. Warm35073 also fails the
original reset image/pair/relocation-pair task checks. Texture reset images93.75%
both seeds for both arms. Clean and both mild-tint conditions have100% images.
Factual logits are exactly unchanged across every mode in24 trained cells: warm35073
ordinary facts93.75%, texture35074 ordinary/reset facts93.75%. Image-only training
cannot repair those. The unchanged source itself misses texture retention on these
fresh seeds (reset87.5%/93.75%); this is not evidence that continuation caused the
whole shortfall. Warm35074 does regress from source100% to93.75% after either fit.

Training cool reset images98.828125% control versus100% refined; warm93.75% both;
texture100% control versus99.609375% refined. All refinement scales are active on the
fixed training diagnostic (residual RMS0.003915–0.136701); activity does not establish
causal necessity. Refinement adds17,024 parameters (75,152 vs58,128 total trainable).
No branch, checkpoint or new threshold was selected after seeing confirmation.
Keep the module optional; the primary test does not justify adopting added capacity.

All58 distinct focused tests pass. Original objective, scoring, data and shared
report renderer are unchanged; old default inference is intact. New refined exports
strictly restore their architecture. Exact initial outputs/shared weights/RNG,
identical full training caches and final sampler/RNG, frozen tensors,10,188 independent
metrics and40 exact GPU reloads verified. Training89.427s, fit loop165.991s,
evaluation89.370s, peak646MiB; disk/headroom floors respected. Two actual Claude
conceptual reviews reconciled the first-step gradient correction and the distinction
between activation and usefulness.44 reports structurally checked, development and
overview figures inspected; prior browser URL limitation remains, no workaround.
[Report](../runs/producer_refinement_v1/report.html),
[verification](../runs/producer_refinement_v1/verification.json).

Next: isolate warm/texture readout learning and the factual path under the corrected
loss, rather than assuming more image capacity is necessary. A bounded factual+image
head continuation against matched image-only training could test factual repair while
keeping encoder, memory/state and image backend fixed. Declare fresh confirmation
and unchanged scene retention/causal gates first. Current failures also show that
passing one small rendering sample does not close robustness. Broad real imagery,
learned reliability, independent upstream replication, CPU portability and general
multimodal generation remain open.

## Factual readout repair: locked protocol

Continue from producer_refinement_v1/source_8502_control (no added capacity).
Compare factual+image readout continuation against matched image-only training,
with the corrected box RGB objective and0.1 feature penalty. Both see detached
native mixed-context tokens; their parameters are disjoint. Factual loss has no
image-producer gradient path. Independent norm1 clipping per readout removes the
coupling introduced by global gradient clipping; use it in both arms. In image-only
training this must reproduce original global clipping exactly. This tests factual
learning, not a new cross-modal consistency mechanism or shared representation.

Extend the existing box-weighted cached path to factual+image learning with explicit
separate readout clipping. Keep global clipping/default behavior unchanged. Reject
live writer/thinker, stored route, output standardization and evaluation overrides.
Validate trainable parameter partition, log each pre-clip norm/clipping indicator,
and prove image gradients, weights and optimizer states match across arms while
only joint-arm facts adapt. All encoder/state/memory/backend/calibration tensors
stay fixed; no new architecture. Require initial state/cache/RNG equality and exact
odd-step resume/reload with correct saved trainability/settings.

Two1536-update fits, seed9901, batch16, existing AdamW; same1024 train7701 histories
(neutral,warm,cool,texture,128 pairs each). Validation37072/32 pairs and terminal
37071/64 pairs never select. Development36701/38072/38071,16 pairs,16 updates.
Fresh confirmation37073/37074,32 pairs=64 histories per cell for better diagnostic
resolution; all6 prior scenes. Joint, image-only and unchanged source on all scenes;
raw original8502 on neutral/texture:40 cells, all10 causal modes. No reused confirmation
seed or post-confirmation fitting/threshold change.

Full repair requires all12 joint task gates and all four ordinary/reset factual/image
accuracies within5pp of raw-neutral, texture also within5pp of raw-texture. Report
factual and image retention separately. Primary factual benefit pools warm/texture
ordinary/reset factual errors across both seeds (512 correlated responses):≥25%
relative reduction with no regression in any of eight paired factual scores;
assessable only if image-only has≥8 errors. Otherwise explicitly unassessable.
Require image weights/optimizer state and all evaluated images bitwise identical
between fitted arms; violation invalidates the intended isolated comparison.
Report branch norms/clip counts, absolute errors and quartet deltas. Rendering seeds
are not independent upstream replications. Broader robustness/generation remain open.

Budget:180s per fit,900s evaluation,4GiB GPU ceiling with1GiB headroom,3GiB disk floor.
Essential REDs cover global-clipping confound/reference, nonfinite/unknown trainable
scope guards, image equality and factual changes, CLI path, exact resume and exported
freeze/settings. Small GPU development before source freeze; reports use the existing
renderer with structural/figure checks and the prior browser limitation disclosed.
Actual Claude reviews generic clipping/causal methodology only; private code/results
reviewed locally. No external private export or new architecture is needed.
