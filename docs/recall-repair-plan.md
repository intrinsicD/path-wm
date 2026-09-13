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
