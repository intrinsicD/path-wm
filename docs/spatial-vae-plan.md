# Spatial VAE comparison

Authorized 15 September 2026: implement/test the reviewed design, then cross-scale
attention, then a meaningful modification. Preserve the existing agent/codec.
This is a standalone image representation experiment, not end-to-end agent training.

## Ordered variants and controls

A `base`: exactly the reviewed K=3, channels32/64/128, latent8, no pre-block,
one post-block per encoder stage and mirrored decoder; PixelUnshuffle/Shuffle,
bias-free two-3x3-convolution residual mixers, no normalization or encoder skips.
B `attention`: A plus coarsest-grid queries reading all compressed finer encoder
scales (projected to width32,4 heads), one residual cross-attention before posterior.
No fine feature crosses into the decoder. Native variable-length grids, analytic
coordinate features; no resolution-specific learned table or global-vector latent.
C `reversible`: B with local residual mixers replaced by additive reversible
couplings. Half channels x1/x2: y1=x1+F(x2), y2=x2+G(y1). Each F/G uses half->full->half
bias-free 3x3 convolutions; total mixer weight count equals A/B. This tests local
processing preservation before projection. Attention/projections/posterior are
still not invertible, and the entire VAE is not called lossless.

Common initialization: seed56101, same projection/posterior tensors, mixer final
convolutions zero; attention output projection zero. All three initial mu/logvar
and reconstructions must match. C's local subnetworks differ internally; their
identity outputs and parameter counts, not individual hidden weights, match B.
Sampler and noise streams identical; no prior pretrained/old codec weights fit this
new interface. Separate random initialization from identity initialization in exports.

## Data and tasks

Use existing prepared COCO RGB64,1024 training,128 validation,192 test groups,
selection seed56001; one row/group, no cross-split groups. Reserve the next16 train
and16 validation groups exclusively for development. Source manifest hashes and
selected row identities recorded. No additional downloads or webcam recording.

Train reconstruction on real photographs, batch8,512 AdamW updates per arm,
lr3e-4, weight_decay1e-4, gradient clip1, FP32. Standard reference: beta1 and fixed
Gaussian variance0.5; KL and RGB-summed distortion each divided by original area.
Sample posterior for training; logvar bounded[-12,8]. No perceptual/adversarial
loss, labels, attention in A, or hidden raw detail. Evaluate mean and sampled
posterior separately and log achieved KL nats/pixel and active channels.

Final test tasks (one shared population per condition):
- held-out RGB64 reconstruction, color, edges and high-frequency residual errors;
- native photo center crops at64x64,128x128,63x79 (32 sources each),96x160 (16),
  and192x256 (8); verify original JPEG hashes; never upscale RGB64 and call it
  high-resolution reconstruction. Native crop pixel errors cannot be read as a
  pure resolution intervention because field of view/content also changes;
- synthetic diagnostic edges, textures, colored objects, high-frequency patterns
  and1-pixel shifts (32 patterns), with reconstruction and phase sensitivity;
- saved-latent decode using a separately loaded decoder, encoder-free exact replay;
- photo-instance retrieval over192 candidates from mu using mild brightness+shift
  query views; raw-pixel and per-image RGB-mean controls. This is instance matching,
  not category learning, object discovery or persistent identity validation;
- wrong-latent, zero-latent and spatial-latent-shuffle controls; unchanged targets;
- standard-normal prior samples and sample diversity as diagnostics only; no
  invented image-quality gate, natural-language output, physics or planning claim.

Aggregate tasks separately. Save per-image errors and predictions, fixed first8
examples, mean-vs-sampled reconstructions, prior samples and native-resolution panels.

## Predeclared gates and bounded follow-up

Workflow: exact save/reload, same-device pause/resume incl sampler/noise/optimizer,
finite gradients to posterior and all stages, exact shuffle identities and padding,
local coupling inverse close in float64, attention reads fine scales, decoder-only
boundary and no cross-batch influence. Shape checks include1x1 and odd rectangles.
Tiny deterministic beta0/mu-only8-photo overfit: <=25% of initial training error at
128 updates; development only, not a VAE/held-out claim.

Base held-out capability screen: mean and sampled RGB64 MSE <=0.01 (PSNR>=20dB per
aggregate-MSE convention), >=20% MSE reduction versus training-mean image, retrieval
>=80%, native128 and odd crop MSE<=0.015. All must pass for this bounded composite;
prior sample quality, semantics, general generation and other modalities remain open.

B benefit vs A and C benefit vs B: RGB64 MSE lower by>=5%, paired95% bootstrap error
improvement interval above0 (1000 draws,seed56191), no >5% relative regression in
RGB64 edge error, native128/odd MSE or retrieval error (absolute tolerance1/192 for
zero-error reference). Report parameters, time and achieved KL separately; this is
an equal-update comparison, not equal-compute or equal-rate superiority.

A/B/C are run in that order. If all three validation final KL rates <0.01 nats/pixel
and fail the20% validation improvement over training mean, execute a separately
labelled lower-KL control: continue each own final checkpoint for512 more updates
at beta0.001, reset optimizer identically, same new seed56201/data/noise stream.
This trigger uses validation only and is frozen before test scoring. Preserve
beta1 results. It tests objective operating point, not a replacement of the user's
base design or matched-rate architectural benefit. Apply the same gates within that
cohort. No further tuning based on test results.

## Resource and delivery bounds

Local RTX3050 8GiB; GPU inaccessible in sandbox but available with normal approved
hardware access. Max3072 formal updates across6 fits if the validation-only trigger
fires; per-fit wall cap300s, total formal training<=1800s. Development <=180s and
numeric/replay checks<=180s, evaluation/report budget<=300s per full cohort.
Peak GPU reserve<=3GiB, free GPU>=1GiB, remaining disk>=3GiB, new artifacts target<1GiB.
No deletion of prior data/results. Profile before formal source freeze; if the
profile exceeds a cap, reduce batch size consistently for every arm before runs
and record the amendment. Never silently train a subset of arms longer.

Use one readable `experiments/spatial_vae.py` recipe with ordinary model modules,
existing Run/checkpoint and report utilities. Every train/evaluate output retains
raw metrics, loadable weights, settings/source identities and standalone report.html;
comparison report records all failures and validation-only trigger decisions.
Browser QA uses the installed browser tool if available; prior local-file policy
limits must not be bypassed. Inspect static figures if browser access is denied.
