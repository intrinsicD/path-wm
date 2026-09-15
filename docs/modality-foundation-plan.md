# Non-image modality integration and diagnosis

User request: all existing modalities should work with the modular World State,
not only images. Inspect and repair concrete interfaces, then separately measure
small-task learning and real-input transport. No general language/speech/video
capability inferred from tensor shapes or toy fitting.

## Plan before implementation

1. Reuse current PyTorch multiscale encoders, native text/audio/image decoders,
   video trajectory decoding, BeliefAgent, WorldSession, Run and portable reports.
   Add the missing audio/video inputs and audio/text outputs to the foundation
   constructor. Keep modules directly replaceable; no second framework.
2. Provide a small masked feature-to-candidate adapter shared by text spans,
   audio chunks and video/image tokens. Candidate extent is supplied; whole-window
   pooling is a diagnostic baseline, not object/word discovery or cross-modal
   identity alignment. Version spaces separately until alignment is trained.
3. Test every input alone and mixed, output generation, masks, causal ordering,
   gradients, source attribution, unchanged writes during recall, and exact restore.
   Correct concrete failures before calling the interfaces complete.
4. Run isolated encoder/decoder learning checks for text, audio and video; measure
   direct feature decoding separately from retrieval/context decoding so missing
   training in the latter cannot be hidden by a direct path. Retain image control.
5. Exercise local real video/audio and UTF-8 text using explicit preprocessing and
   source identities. No synthetic speech labels or invented person identity.
   Save raw metrics, output examples, traces, checkpoint and standalone report.

## Fixed development budget and gates

CPU first, one seed 61301; small batch of four examples per modality, 128 updates
per isolated branch, Adam 0.003. Up to one additional 128-update repair run after
a documented failure. Keep native small interfaces: RGB16, audio256 samples,
short byte strings and ordered four-frame clips. These are fitting diagnostics,
not held-out perceptual quality or sample-efficiency tests. Final resource use and
per-modality loss reductions are reported; no architecture superiority claim.

Mechanical checks require exact persistence/retry behavior, finite gradients and
mask exclusion. Development learning requires at least 20% relative loss reduction
per branch; text autoregressive exact match, audio error versus silence, frame
error versus mean/copy and temporal-order sensitivity remain explicit diagnostics,
not substitutes for each other. Real media use the existing local Charades source;
report the resize/resampling/chunk durations and lack of general task labels.

## Initial findings

The current FoundationModel only registers image/text input and image output.
The general agent already contains all four input modalities, but its default
audio output is only 32 samples and text is a tiny byte model. Video output is an
ordered sequence of image decodes from states, not a pretrained temporal generator.
Current joint tests demonstrate gradients/shapes, not isolated modality learning.

## Progress and results

Initial direct fit (128 updates each) passes loss-reduction checks. Text freely
reproduces four words; audio MSE is 0.00001265 versus silence 0.1800. Video MSE is
0.014726, worse than the mean-output control (0.011801), with almost no order
sensitivity. Image control also remains weak. These are tiny fitting diagnostics.

One bounded video repair comparison is now specified before execution: from the
same direct checkpoint, 128 further video-only updates with ordinary MSE versus
normalized foreground-weighted MSE (weight12 on the known synthetic object mask).
Same data, optimizer, parameters and CPU budget. Evaluate unweighted RGB, foreground
and background MSE, wrong-context sensitivity, centroid motion and reversed input.
Repair screen: RGB beats mean output; foreground error improves at least20% against
equal-duration continuation; motion direction correct in at least3/4 development
clips with displacement above1 pixel. This is not a real-video quality standard.
One split-run replica checks exact resume; other modalities' weights must stay fixed.

Actual Claude two-round conceptual review accepts isolation/provenance/null-context
controls. Its fixed shuffled-target-training suggestion was corrected: four fixed
permuted pairs can also be memorized. Evaluation uses wrong-example/zero context and
a unigram comparator instead. Sample-size limits remain explicit; no discovery or
alignment claim. No private source/data/results exported.
