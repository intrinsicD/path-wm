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

## Results

Source248eef0. All four runs complete1536 updates within120s each; no checkpoint
selection. Training seconds61.547/51.854/52.051/51.692, total217.144. Peak reserved
GPU90MiB per fit. Cache extraction4.632/4.651s, separate from training. Agent7801,
probe7901 exercises768+768 explicit GPU resume; all771 committed ledger rows are
preserved. Each probe has17,512 learned parameters. Encoder access672 tokens;
state access8 or30. Identical architecture is not equal extraction difficulty.

Final-side test accuracy,128 fresh-background episodes per run:

| Stage | Agent7801, probe7901 | Agent7801, probe7902 | Agent7802, probe7901 | Agent7802, probe7902 |
| --- | ---: | ---: | ---: | ---: |
| Encoder, both views |100%|100%|100%|100%|
| Initial cue, working |50%|50%|50%|50%|
| Before storage, working |96.09375%|82.03125%|77.34375%|74.21875%|
| Before storage, all |87.5%|89.84375%|70.3125%|85.9375%|
| After recall, working |50%|50.78125%|50.78125%|50.78125%|
| After recall, all |50%|49.21875%|51.5625%|51.5625%|
| Randomized-label recall control |47.65625%|50%|46.875%|49.21875%|

All four control screens pass; both old and freshly trained direct readers reach
100% joint. Initial-cue future side is necessarily50%; all its appearance predictions
are correct. Native pre-storage side48.4375%/50%, native recall50% in the two agents.
Pre-storage probe joint accuracy equals its side accuracy. Some after-recall shape
errors make joint slightly lower than side. Full metrics include every attribute,
selection/relocation pair, CE and training fit, without averaging away seed spread.

Only agent7801/probe7901's pre-storage working reader passes the individual stage
screen (123/128 joint,59/64 complete relocation-side pairs). No state stage passes
the prespecified BOTH-probe-seeds requirement. Post-recall training-side accuracy
also stays50–53.91%, so the observed weakness is not just held-out backgrounds.
Training pre-storage working-side98.047%/84.766%/76.953%/75%, showing variation in
optimization as well as generalization. Random-label training is not perfectly fit;
it is a selectivity control, not proof of universal probe capacity.

These results show task information is available in the encoder, and at least
some location information is recoverable from states actually written to memory.
The cache alignment test verifies that the second stored bank entry is exactly
the pre-storage state. No tested reader successfully recovers final side from the
post-recall state. This narrows the next intervention toward memory reading and
working-state formation, but does NOT prove information destruction, identify a
unique internal mechanism, or repair the native factual/image outputs. Pre-storage
location coding/readout remains imperfect too. A different probe family could
change negative outcomes; a capacity/scale-sensitivity study remains an option.

Integrity: all source checkpoint and cache hashes unchanged; exact-frame split
disjointness, initial counterfactual equality, working/all-state alignment and
training-only statistics verified.38 targeted tests pass. Independent NumPy scoring
verifies672 train/test metrics (168 per run), max difference1.25e-7. All GPU probe
replays are exact. CPU replays on the SAME GPU-cached tensors differ by at most
1.55e-5 logits (<1e-4) and agree categorically throughout. This deliberately excludes
raw encoder recaching on CPU; its previously failed numerical portability check
remains open. Individual/combined HTML structural checks pass; stage PNG inspected,
SVG export retained. Browser QA unavailable. No agent checkpoint promoted.

Artifacts: `runs/memory_probes_v1/report.html`, `stages.png`, `stages.svg`, two
`cache_*/` directories and four `agent_*_probe_*/` runs with weights, optimizer/RNG
checkpoints, raw logits, train logits, permutation, calibration and audit receipts.
The export contains probe weights only; it does not replace the agent.

Commands for a fresh cache and run:

```bash
.venv/bin/python -m experiments.memory_probes \
  --checkpoint runs/memory_relocation_v1/seed_7801/weights.pt \
  --cache runs/memory_probes_v1/new_cache --prepare --device cuda:0
.venv/bin/python -m experiments.memory_probes \
  --cache runs/memory_probes_v1/new_cache \
  --output runs/memory_probes_v1/new_probe --seed 7901 --device cuda:0
```

Next proposed slice: hold encoded observations and stored states fixed, train the
memory-reader/thinker and its output readout for location recovery, and compare
with direct access to the same stored states. Predeclare the learning signal and
budget; retain these probes to distinguish changed state accessibility from merely
changed native-head performance. Consider a scale-preserving or higher-capacity
probe as a sensitivity control before making any claim of lost information.
