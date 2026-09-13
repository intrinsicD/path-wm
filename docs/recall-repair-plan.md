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
