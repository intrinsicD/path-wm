# Frozen-state location probes

13 September 2026. Alex approved probing the frozen relocation agents before
changing their state updater, readout or training duration. No agent training or
capability promotion in this slice.

## Question and access

Can newly trained readers recover the selected object's final location from the
existing states? Extract from the two balanced checkpoints7801/7802 without
changing any parameters/buffers: (1) both visible frames' direct encoder tokens,
(2) working/reasoning tokens immediately after the selection cue, (3) those tokens
after the later view, immediately before storage, (4) all30 tokens at that point,
(5) working/reasoning tokens after fresh-state reset and memory recall, (6) all30
tokens after that recall. Working/reasoning contains8 tokens. Different token
counts reflect the actual interfaces; probe architecture/parameter count is held
fixed, not total attention computation. No pixels, labels, times, entity IDs or
other metadata are appended to state-probe inputs. Direct encoder access uses
the existing diagnostic route and its existing time encoding.

Initial-cue final-side accuracy must be50%: identical first frames have opposite
future sides within each quartet. Its appearance accuracy is descriptive. It is
an intentional negative control, not evidence of failed memory. Compare the old
frozen factual heads at storage/recall and the old trained direct encoder reader.
Also train an independent recall-working probe on one fixed random permutation
of TRAIN labels; test it against true held-out labels. This checks probe selectivity,
not an assertion that every finite random-label control must be exactly at chance.

## Learning, splits and budget

Reuse the approved balanced relocation generator. Train128 pairs seed7701 (the
agent's training population), validation32 pairs seed7712, test64 pairs seed7713.
Whole quartets remain together and exact frame hashes must be disjoint across
splits. All target tuples and movement types are familiar; this tests fresh noisy
background histories, not new objects or motion rules. New test seed is not used
for any tuning. Training labels supervise probes only; cached tensors are detached.

At every stage use one learned query, two existing Attend layers, an8-logit head
(4color/2shape/2side), mean factor CE. Predeclare fixed per-channel mean/std fitted
over training tokens only, std floor1e-4; subtract/divide at probe input, freeze
and save these buffers. No learned normalization or additional feature channels.
This affine preprocessing does not prove model-native accessibility or exact
information preservation under finite precision. No capacity/preprocessing sweep.

Probe seeds7901/7902 crossed with both frozen agents: four runs,1536 updates each,
batch16, AdamW lr0.001 wd0.0001, gradient clipping1 separately per probe, FP32.
Each probe gets identical sampled episode indices per update; separate parameters
and gradients, no shared optimizer clipping or gradients into the agent. The
random-label mapping is fixed by seed17901 and saved. Initialize matched heads
identically within a run. Final checkpoint only; validation every256 updates,
no early stopping/selection. Cap120 training-loop seconds per run,480 total;
cache extraction/evaluation are recorded separately. CUDA cap4GiB leaving1GiB free.
Development uses separate data/init seeds and at most16 updates. One actual
split/resume run; essential CPU resume test. Commit source before formal runs.

## Interpretation and numeric gates

A stage passes an accessibility screen if final joint and side accuracy>=90%
and both-members-correct relocation side accuracy>=80%, in BOTH probe seeds for
the respective agent. Also report complete joint relocation/selection pairs,
per-attribute accuracy, side CE, training accuracy and original readouts. Call the
probe protocol interpretable only if the matched fresh encoder probe and the old
encoder reader each reach>=90% joint, initial final-side is50%, and the random-label
recall probe has<=60% final-side accuracy. Otherwise mark diagnostic controls
inconclusive; report successful individual recoveries without erasing limitations.

Success shows recoverable task information for this reader and population; it
does not repair the native image/factual output or establish how the agent reasons.
Failure cannot prove absent information, especially if the positive control fails.
Successful all-state but failed working-state probes would motivate workspace
routing; successful recall-working probes would motivate native readout training.
No new model update follows automatically from these diagnostic results.

Integrity: unchanged frozen hashes, cache/state alignment, initial counterfactual
equality, no label/target path into extraction, train-only normalizer, identical
probe initialization, isolated gradients, exact CPU optimizer/sampler replay,
strict standalone probe reload on its cached inputs, independent saved-logit
metrics. Preserve existing CPU/GPU source-encoder numerical limits; formal caches
are extracted on GPU, not silently mixed with CPU versions. New probe replay is
checked separately (max logit error<=1e-4 on the same cached tensors).

Predictions use argmax, lowest index on ties. A relocation side pair requires both
true sides correct; a joint pair requires every attribute correct in both members.
Claude public-only review emphasizes that equal parameters/steps do not equalize
extraction difficulty. We retain conditional positive recoverability claims only;
even a successful encoder control cannot make a failed state probe prove absent
information. Report both seed values and training accuracy. A capacity sweep or
oracle-signal sensitivity study is a later option, not part of this bounded screen.
Follow-up requests acknowledgment of that narrower scope. No private export.

Use the existing Run/report renderer. Save all logits, labels, cache identities,
normalizers, weights, curves and an explicit stage comparison. Reports get
structural QA; browser QA remains unavailable. No external private-data export.

## Implementation checks

38 targeted tests and lint pass. New checks cover exact initial counterfactual
equality, saved-bank alignment, unchanged native output, detached tensors, matched
but independent head initialization, training-only fixed statistics, isolated
gradients, paired scoring, exact CPU optimizer/RNG/sampler resume and standalone
probe replay. Existing model construction/loading moved unchanged into the model
module so recipes do not import each other; old recipe imports remain compatible.

Development16 updates complete in0.919s; peak292MiB, frozen source hash unchanged,
standalone report structurally verified. No formal outcomes used for adjustments.
Claude accepts the restricted positive-recoverability scope; two actual public-only
exchanges. Both seeds will be reported separately, with no broad stability claim.
