# Real-photo continuation

User requests real-image training after the synthetic output experiments. Use local
COCO train2014 photographs already prepared in data/curriculum/coco_v1, derived from
/home/alex/Documents/datasets/train2014. Keep own architecture and existing weights;
no model/data download or overwrite of prior checkpoints. Source for the current
conditional generator: runs/decoded_image_v1/sample512/weights.pt.

Task: observe the same64px photograph at times0 and1, then a fixed neutral blank at2;
reproduce the earlier photo from ordinary state or reset+episodic recall. These are
real photographs in a constructed still-image episode, not real motion, novel views,
arbitrary text prompts or photographs generated without earlier observation.
Train only the conditional output generator; visual encoder, state/memory, factual
heads and reconstruction codec stay frozen. The codec teacher sees target photos
only for supervision/reconstruction adequacy. Generator inference sees working tokens
and noise, without raw-photo or encoder-feature shortcuts into the decoder.

Remove the source's synthetic-background median centering explicitly: it can reject
or distort arbitrary photographs. Persist input_centering=null in new exports and
strictly reload without that wrapper. Preserve other upstream normalization/state
weights. Fit generator feature means/scales on training photos only, then freeze
those buffers. Evaluate a pre-optimizer baseline AFTER these identical preparation
steps; changes due to calibration/preprocessing are not counted as optimizer benefit.

Select1024 train,128 validation and256 test photos using seed45001 from the existing
prepared split, one image per duplicate group, with group/row separation verified.
Record selected rows, source filenames/groups/canonical hashes and complete input
manifest/array identities. No validation/test photo enters feature calibration or
updates. Prior codec/agent had other COCO and synthetic exposure; holdout claims are
for this continuation, not a virgin global dataset. Use existing64px crops to preserve
tensor interfaces. Higher resolution requires a separate data/architecture test.

Objective: equal mean standardized feature velocity MSE plus10 times uniform RGB
MSE of the actual eight-step sample through frozen decoder, half progress0 otherwise
uniform. Replace synthetic foreground weights with uniform real-photo pixel error.
Continue current generator parameters. Batch8, AdamW lr.0003, decay.0001, clip1,
2048 terminal updates or600s training budget, reserved GPU<=3GiB/free>=1GiB,
disk free>=3GiB. Save paused/budget-stopped runs visibly; do not extend a stopped
budget or select a checkpoint on test. Development16 updates on disjoint reserved
training groups, with small validation population; no hyperparameter sweep.

Before/after tests use fresh heldout photos with ordinary, reset, erased memory and
adjacent-pair-swapped memory; noise13 with same draw within a swapped pair. Report
RGB MSE, mean per-photo PSNR, finite-difference edge MSE and clipped-output fraction,
also the training-mean image and frozen codec reconstruction. Score swapped outputs
against BOTH original and swapped targets, so a mismatch control is not mistaken
for desired behavior. Plain photos carry no valid synthetic color/shape/entity labels.

Learning screen: test ordinary and reset MSE each>=10% below the prepared initial
model and training-mean image. Context screen: correct reset MSE each>=10% below
erased and wrong-bank/original-target MSE; swapped-target MSE no more than1.1 times
correct reset. Both needed for this pilot's success; quality metrics remain explicit,
not proof of high-fidelity or general generation. All gates fixed before execution.
Test resolution and task scope must be visible with the outputs.

Pre-execution review addition: also evaluate a photo-free history, with all three
observations blank and otherwise identical memory operations/noise. Require correct
reset MSE at least10% below this content-blind baseline. This helps distinguish
retention of a particular photo from a generic photographic prior. Do not require
any ordering between the blind-history and erased-bank controls: they are different
interventions. Claude's prior-exposure and codec-adequacy concerns are incorporated;
the brief does not establish that the generator itself already had a real-photo
prior or set a numeric resolution/parameter fidelity ceiling.

Essential checks: group split isolation and deterministic subset selection; history
causality/blank current input; swap alignment; train-only calibration, frozen gradients,
nonzero optimizer update, exact pause/resume and standalone export/encoder exclusion
at generator-only inference. Development real-batch profile/report, source freeze,
then training, fresh heldout outputs, independent metric and GPU reload verification.
Use the existing Run and report code via an ordinary readable recipe. No new trainer
framework. Claude public-method review was initially denied; user explicitly approved
sending the exact saved brief. No private photographs, code or measurements exported.

## Implementation and development

Plan and three informative failing checks committed as `2311939`; implemented
recipe and fixed protocol as `d97372a`. Fifteen focused tests pass: real-photo
selection/history/metric checks, frozen parameters and buffers, nonzero updates,
exact CPU resume, standalone reload and generator inference without an encoder,
plus the existing conditional-generator and decoded-loss checks.

The separate 16-update GPU development run used the next 16 training and validation
groups after the formal selection. It completed in 3.85 training seconds with
544 MiB peak reserve. Direct codec reconstruction averaged 30.50 dB PSNR on those
development photos. The development learning screen failed; no hyperparameters
or fit budget were changed based on quality. Its report is structurally verified,
and the first six photo/codec/generated examples were visually inspected.

An actual Claude Sonnet methodology review was completed after the user approved
the exact brief. Matched noise and a content-blind history control were adopted.
Unsupported prior-training/capacity assumptions and the requested ordering of
different null interventions remain documented disagreements; no withdrawal or
private code/result review is claimed. The CLI reported $0.012266 API-equivalent
usage, not a subscription charge. Receipts: `runs/reviews/real_photo_v1/`.

Training and evaluation commands are in [experiments](experiments.md). Run-local
`verify.py` and `summarize.py` preserve independent numeric checks and figure/report
reproduction from saved outputs; neither is a new library or experiment framework.

## Formal result

The declared run completed all 2,048 updates in 370.03 training seconds, with
700 MiB peak GPU reserve. It optimized the same 314,576 generator parameters;
all 520 frozen tensors, including feature statistics, stayed bitwise equal to the
prepared initial model. The terminal checkpoint was evaluated without test-driven
selection, further fitting or budget extension.

| Fixed test, 256 photos | Before MSE | After MSE | After mean PSNR |
| --- | ---: | ---: | ---: |
| Ordinary state | 0.097074 | 0.056845 | 12.87 dB |
| Reset + correct memory | 0.096738 | 0.058667 | 12.73 dB |
| Training-mean image | 0.069642 | 0.069642 | 11.94 dB |
| Direct codec reconstruction | 0.000912 | 0.000912 | 30.92 dB |

Reset pixel MSE improves 39.36%; ordinary improves 41.44%. Correct recall is
15.76% below the mean-image baseline, 49.82% below erased memory, 39.01% below a
blank-only history and 26.75% below swapped memory scored against the original
photo. Swapped-target error equals correct recall within numeric precision.
Both declared learning/context screens pass on validation and test.

These screens establish this pilot's relative optimization and context sensitivity,
not photographic fidelity. The first eight predetermined examples lose the old
synthetic square pattern but remain mostly coarse color fields; the actual objects
are not reconstructed. The direct codec looks substantially better. Neither these
metrics nor that visual comparison uniquely identify whether the remaining gap is
encoding, state/memory compression, generator readout or optimization. Only one fit
seed and one fixed noise realization were tested; no broad replication is claimed.

Independent NumPy calculations reproduce 160 saved metric/count values. Fresh GPU
reloads reproduce both full 128-photo validation outputs and both 32-photo test
prefixes exactly, across eight saved tensors each. A separate disjoint development
check reproduces four uninterrupted GPU updates versus one plus three resumed
updates, including parameters, AdamW state, metrics and RNG. Source checkpoint,
60 code/snapshot files, training-mean calculation and split separation are verified.
Seven standalone reports pass structural checks, and the development/final figures
were visually inspected. The unchanged renderer's browser QA remains unavailable.

Use [report](../runs/real_photo_v1/report.html) and
[comparison figure](../runs/real_photo_v1/comparison.png) to inspect the outcome.
The loadable checkpoint is `runs/real_photo_v1/training/weights.pt`; the initial
baseline, optimizer/RNG checkpoint, raw predictions and verification receipt stay
beside it. The default renderer and original weights were not replaced.

Next proposed test: locate recoverable photographic detail at the encoder, stored
state and workspace with matched diagnostic readers before deciding which stage
to train next. This is a proposal, not an implemented repair or a causal conclusion.
