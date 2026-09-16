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
