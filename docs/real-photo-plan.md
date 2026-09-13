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
