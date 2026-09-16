# Shared image codec for video

16 September. Alex requests actual image-encoder reuse plus video-specific processing,
and asks whether video frames also train image reconstruction. Previously only the
first synthetic video frame was a separate image example; image/video weights were
independent. This slice reuses the existing R/P/M/C **spatial VAE**, not just the old
patch-stem architecture. No silent replacement of the BeliefAgent's token encoder.

## Contract and plan, before implementation

One `VideoVAE` owns an injected existing `SpatialVAE` (v1 or v2), exactly once.
Image calls use that object directly. Video flattens batch/time for that encoder,
then an optional small causal residual convolution adjusts posterior means using
past/current spatial latents, elapsed times and validity. Variance remains local.
Sample the resulting Gaussian; decode each frame with the same image decoder.
Only spatial latents cross the bottleneck; no RGB/encoder-feature decoder bypass.
Zero-initialize temporal output for initial independent-frame equivalence. Expose
detached frame/temporal posterior traces. No persistent streaming cache, optical
flow, future prediction or shared-core direction repair is claimed by this slice.

1. Red tests: actual object/weight reuse, zero-initialized equivalence, learned
   temporal causality including equal timestamps, masked NaNs/gradients, arbitrary
   spatial dimensions, image+video gradients, unique checkpoint ownership.
2. Implement small wrapper and ordinary recipe using existing Run and report.
   Import current image weights explicitly. Verify exact short resume and source
   immutability. Actual Claude gets a public generic design brief only.
3. Fixed bounded development comparison: independent frames vs causal temporal
   residual; same pretrained image codec, two seeds 7301/7302, AdamW lr3e-4,
   batch2 clips x4frames, 128 updates, CPU2threads, max120s training per fit.
   L=0.5*(video VAE loss + independent-frame VAE loss), beta0.01, variance0.5.
   Both arms use the same sampled frames and noise streams; paired image training
   is present in both. This does not isolate auxiliary-loss benefit. All frames
   in a clip remain in their source split. No target future frame in earlier reads.
4. Existing six local real clips are **previously inspected development data**:
   train 0EJAG/0GFE8/0JQ26/0LDP7, validation12VVC, reserved evaluation1KKYX.
   Extract first4seconds at4fps, scale to48x48 explicitly outside model, four
   nonoverlapping4-frame clips per video. Same crops/cadence across arms. Record
   source SHA256 and decoded tensor identity. No natural-video generalization claim.
5. Preregistered narrow benefit screen on reserved development source: >=5% lower
   frame-difference MSE in both seeds, <=2% RGB and mean-color MSE regressions,
   <=2% direct-frame image MSE regression; finite KL. Report raw and clipped RGB,
   edge error, actual KL, parameter/time resources, frozen-source reference,
   zero-latent and wrong-past controls. No threshold changes or extra fits after
   results. Failing benefit keeps independent-frame choice available/default.
6. All runs retain metrics/checkpoints/reconstructions/error maps and standalone
   report; static/structural verification (browser QA unavailable). No changes to
   report renderer. Current disk constrained: <100MiB new artifacts, >=300MiB
   free reserve, no source/historical deletion. Update docs/atlas and commit.

Task completion means usable shared codec + validated training path, not solved
video generation or temporal understanding. Future agent integration requires a
separately evaluated spatial-grid-to-core interface; native agent remains intact.

## Review and interface verification

Actual Claude review and one reconciliation are saved under
`runs/reviews/shared_video_vae_v1/` (public conceptual briefs only). Adopt separate
image/video loss logs, checkpoint ownership checks and future RGB/time/validity
perturbations plus a zero future-gradient check. Claude initially disputed the KL
interpretation incorrectly; corrected explicitly and acknowledged: independent
posterior noise conditional on the observed clip and an iid standard-normal prior
make per-frame KL summation exact. Dataset aggregate correlation is separate.
The additional image loss is an auxiliary objective, not a single joint-video ELBO.
No learned temporal prior or natural-video generation follows from this experiment.
Seven new contract tests and54 relevant regression tests pass (61 total).

## Result, 16 September

Implemented the reusable `VideoVAE` in `pathwm/models/video_vae.py` and a small
`experiments.video_vae` recipe. Both image and video objectives train the **same**
pretrained spatial image encoder and decoder. No duplicated branch/checkpoint
weights. Optional causal residual mixing adds1340 parameters to122979 image
parameters; mean refinement reads current/two previous latent grids and elapsed
seconds/validity. Variance stays per-frame, no persistent streaming state.

Four preregistered128-update real-video development fits completed in24.80s CPU
training (including periodic evaluation), two seeds and two arms. All source clips
stay in their split. Source weights and media hashes are unchanged. The reports are
[comparison](../runs/shared_video_vae_v1/report.html) and each seed/arm child report.

| Seed | Frame RGB MSE | Temporal RGB MSE | Frame-difference reduction | Image-retention MSE, frame → temporal | Benefit gate |
|---|---:|---:|---:|---:|---|
|7301|0.00829246|0.00820123|0.2174%|0.01007547 →0.01006639|fail|
|7302|0.00946337|0.00940493|0.1429%|0.00987705 →0.00987921|fail|

The predeclared5% frame-difference benefit fails in both seeds; RGB/color/retention
regression limits pass. Wrong-past perturbation changes final output only by mean
absolute0.000354/0.000245, demonstrating a small history dependence, not useful
motion understanding. KL is measured rather than assumed matched: frame1381/1348
versus temporal1351/1308 bits/frame. The achieved rates differ slightly; this is
not an equal-rate or equal-compute superiority test.

Both continued arms worsen reserved-source RGB versus the untouched pretrained
codec (0.00794473). Validation RGB improves, and eight previously inspected image
retention examples improve slightly against their frozen reference. Do not select
new weights as a general image/video repair. Original weights are preserved;
`temporal=False` remains the default. Current fully observed reconstruction does
not require temporal inference; next design a separate masked/occluded-frame test
with matched current-frame information before adding a larger temporal module.
Do not interpret this tiny six-source development study as natural-video evaluation.

Verification:61 scoped tests;1724 exact-resume checks (8 vs4+4 CPU updates);
777 independent metric/source/report/Claude-receipt checks. Training loss components,
source-disjoint identities, identical initialization, masks/future gradients and
actual shared weights are checked. All seven reports have embedded-media and HTML
structural QA; full-sequence PNGs inspected. Browser interaction unavailable.
The recipe exports updated image weights and full video/optimizer/RNG Run state.
New artifacts use50MiB; no source or historical run deleted.

This completes the codec reuse slice. It does **not** replace the categorical
agent's old patch encoder. Spatial-grid-to-agent integration, longer temporal
memory, direction retention and learned future-video generation remain explicit
next tasks. Atlas diagram16 shows this boundary; no capability color promoted.

### Follow-up explanation and source audit

[Primary-source literature](video-codec-literature.md) covers SVD,IV-VAE,CogVideoX,
HunyuanVideo and the narrower MSE-prediction precedent. Published weights exist;
none downloaded. IV-VAE's temporal-compression failure mechanism is not established
here because our latent retains all frame positions. Our temporal residual is
currently after the image posterior projection and before sampling, with a per-frame
decoder; early feature scales remain independent.

The untouched source was trained with beta0.1,color_weight6, whereas both new arms
use the preregistered simple beta0.01,color_weight0 objective. Their paired comparison
remains matched, but the frozen-source regression also confounds domain and loss
changes. Preserve original-loss and frozen-codec continuation controls before
attributing that regression to temporal processing. These follow-ups are proposals.
