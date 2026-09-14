# Explicit-scale spatial VAE

Status: proposed experiment, 15 September 2026. Alex requested a discussion with
Claude of this design. This note specifies a first version; no VAE implementation,
training run, replacement of the current codec, or GPU budget is claimed here.

The defining separation is **spatial rearrangement → learned processing → explicit
channel projection**, with optional processing before rearrangement. Preserve the
current world-model design and treat this as an independently testable image codec.

## Guarantees and limits

PixelUnshuffle(2) is an exact permutation from B,C,H,W to B,4C,H/2,W/2; PixelShuffle(2)
reverses it. Spatial resolution reduction alone need not lose values.
Ordinary residual convolution blocks are not guaranteed injective, even if shape is
unchanged. The explicit channel reduction is a declared dimensional bottleneck;
it is not a guarantee that preceding learned processing preserves every detail.
For example, x + F(x) with F(x) = -x discards x. No orthogonality or invertibility
constraint is required in v1; practical information access will be measured.

A 2×2 stride-2 convolution can be written exactly as PixelUnshuffle(2) followed by
a 1×1 convolution with rearranged kernel weights. Some strided convolutions can
preserve information. The experimental advantage here is explicit interfaces and
processing before reduction, not a proof that every strided convolution is inferior.
A projection from 12 to 64 channels is expansion; 256 to 128 is reduction. More
coordinates do not create information, and equal coordinate count is no guarantee
of preservation. Tensor size is not a measured encoded bitrate.

These identities were checked independently on small tensors, including an exact
three-stage pad/unshuffle/shuffle/crop roundtrip for 63×65 input. This checks the
operations only; it does not validate a learned VAE.

## Minimal modules and tensor contracts

Use ordinary `nn.Module` composition, no module registry or abstract plugin framework.
`nn.Identity`, `nn.PixelUnshuffle`, `nn.PixelShuffle` and `nn.Conv2d` already implement
three of the requested concepts. Name them explicitly inside each stage.

```text
EncoderStage(C_in, C_out):
  pre_process:   B,C_in,H,W         -> same shape
  downsample:    PixelUnshuffle(2) -> B,4*C_in,H/2,W/2
  post_process:                   -> same shape
  projection:    1x1 convolution   -> B,C_out,H/2,W/2

DecoderStage(C_out, C_in):
  projection:    1x1 convolution   -> B,4*C_in,h,w
  processing:                     -> same shape
  upsample:      PixelShuffle(2)   -> B,C_in,2h,2w
  post_process:                   -> same shape
```

Record each projection as expansion, equal-width mixing or reduction according to
its actual input/output widths. Reversing an encoder expansion can produce a
reducing decoder projection; not every decoder projection is an expansion.
Set projection to Identity to disable that bottleneck; its output width must then
be 4*C_in. Decoder weights are learned independently, not algebraic inverses.

For processing, start with `x + Conv3x3(SiLU(Conv3x3(x)))`, preserving shape and
width. A zero-initialized final residual convolution is an optional explicit
initialization choice. Start without normalization on the identity path or a
mandatory normalization layer. Residuals support learning identity but guarantee
neither invertibility nor nonzero gradients. Pre/post depth counts support
`downsample → process → project` and `process → downsample → process → project`.
Later blocks need only honor the same spatial tensor contract. Global attention
may flatten a variable-length grid internally; no fixed-size global vector or
resolution-specific learned position table is introduced.

A proposed small starting configuration, not a measured optimum:

| Stage | Unshuffle channels | Projected channels | Spatial size for RGB256 |
| --- | ---: | ---: | --- |
| Input | — | 3 | 256×256 |
| 1 | 12 | 32 (expansion) | 128×128 |
| 2 | 128 | 64 (reduction) | 64×64 |
| 3 | 256 | 128 (reduction) | 32×32 |
| Posterior | — | 8 each for mu and log_var | 32×32 |

Start with pre_depth=0, post_depth=1. Expose stage channels, per-stage processing
depths, latent channels and K through constructor arguments in the experiment
recipe. No attention, skip pyramid, pretrained teacher or perceptual/adversarial
loss is required for this first comparison. Residual processing at 4*C_in can be
expensive: measure peak memory, FLOPs/time and parameter count before choosing
batch size. This note does not promise a GPU fit from parameter count alone.

## Resolution and shape policy

Keep B,C,H,W throughout. Pad once at the input on bottom/right to a multiple of
2^K using replicate padding, record original H,W and padded geometry, then crop the
final decoded image to original H,W. Positive odd sizes and rectangles are allowed;
zero or negative sizes are rejected. Bucket batches by original size for v1.
No resizing is hidden inside encode/decode. PixelUnshuffle raises on incompatible
sizes, so boundary validation must be explicit.

A small posterior record holds mu, log_var, original_size and padded_size.
`encode(x)` returns this record; `sample(posterior, generator)` returns z;
`decode(z, output_size)` reconstructs pixels, checking the requested size against
latent geometry. A convenience forward can return reconstruction and posterior.

The same weights accept different dimensions, but receptive fields and object scale
still matter. Shape flexibility does not establish scale-equivariance, trained
quality at unseen resolutions, seamless tiling, or constant compute/memory usage.
A latent grid grows with image area; it is not a fixed-size world-state memory.

## Spatial posterior and training objective

Use separate 1×1 projections for mu and log_var, both B,Z,h,w; prior N(0,I):

```text
std = exp(0.5 * log_var)
z = mu + std * epsilon
KL = 0.5 * sum(mu^2 + exp(log_var) - 1 - log_var)
```

Use the same bounded log_var in both sampling and KL, computed in float32 if mixed
precision is later enabled. Record bounds, likelihood variance and beta schedule
in the recipe/checkpoint; monitor saturation and nonfinite values. A candidate
initial log_var range is [-12, 8], subject to explicit development checks. Neither
clamping, KL warmup nor free bits guarantee against collapse; free bits are deferred.

For image i with original area A_i=H_i*W_i:

```text
D_i = sum_valid_RGB((x - reconstruction_mean)^2 / (2*sigma_x^2)) / A_i
R_i = sum_all_latent_KL / A_i
loss = mean_over_images(D_i + beta * R_i)
```

Use RGB floats in [0,1] and an unconstrained decoder mean; no training-time clipping.
The fixed-variance Gaussian constant may be omitted because it does not change
optimization. Proposed simple reference: sigma_x^2=0.5, beta=1. It is a reference
ELBO convention, not a predicted good operating point for image quality. Any beta
warmup or lower-beta distortion/rate comparison is separately declared before a run.
Report unclipped loss plus consistently labelled clipped display metrics. The same
original-pixel denominator applies to reconstruction and KL; do not average KL over
latent coordinates independently when comparing latent widths or hierarchy depths.

Exclude padding from reconstruction loss. Include all latent cells in KL because
boundary cells mix valid and padded content; document padding fraction and KL
contribution rather than inventing independent latent validity. Compare overlapping
size distributions and account for boundary overhead on small images.

Report RGB MSE/PSNR and edge/detail errors, KL nats per original pixel, scalar counts,
active posterior dimensions and per-stage/coordinate KL. KL/log(2) is a rate proxy
in bits, not achieved file size without a coding scheme. A same-network mu-only,
beta=0 deterministic autoencoder is a distinct diagnostic; beta=0 with stochastic
sampling is not identical to it. Evaluate posterior-mean reconstruction, sampled
reconstruction and unconditional prior samples separately.

The base decoder consumes z and shape metadata only. Encoder-to-decoder feature
skips or raw-detail bypasses would change what must be stored and generated; keep
them out of the base VAE so they cannot mask an inadequate latent.

## What would validate the idea

First establish mechanics: exact shuffle inverse; odd/rectangular/tiny geometry;
independent noise and repeatable sampling; analytical KL reference; finite gradients
to both posterior heads and all stages; exact save/reload; loss denominator checks;
no image-content route to the decoder except z. Identity-processing/no-compression
rearrangement tests must bypass the stochastic posterior and verify only that path.

Then use a tiny training-only overfit check and a separately declared real-photo
holdout experiment. Compare process-before-compression with immediate compression
at a common latent geometry, distortion/rate target and recorded parameter/compute
budget. Include a conventional strided-convolution control as an experiment arm;
it does not become the new default. Exact parameter/FLOP equality may be impossible
across these choices: state discrepancies and compare quality-versus-budget curves.
A common beta or latent shape alone does not mean achieved rates are matched.

Probe before/after processing as well as before/after projections. Use positive
controls and at least one nonlinear readout family; failed probes are limited
accessibility evidence, not a proof of irreversible loss. Vary processing depth,
channel schedule and latent width separately. Test shifted 1-pixel patterns and
subpixel-phase artifacts, non-square resolutions, edges, textures, color fidelity
and held-out photographic scenes. Do not tune on the held-out images or confuse
small-image reconstruction with general concepts, sample efficiency or generation.

This discussion sets no trained-quality threshold, training duration, data split or
GPU allocation. Those belong in a short predeclared experiment plan before fitting.

## Connection to the existing agent

The current `PyramidEncoder` and `FeatureSpec`-based image consumers declare fixed
spatial layouts (`pathwm/models/encoders.py`, `pathwm/models/features.py`). The current
`ConditionalFeatureGenerator` predicts those declared codec features. A dynamic
spatial posterior is therefore a new codec interface, not a weight-compatible
replacement for those components. Validate the codec independently first; add a
small geometry-aware adapter at the consumer boundary when integrating it.

For output without an image, a learned producer must generate a compatible latent
field from workspace/task/memory and requested output geometry. The VAE decoder
accepts any correctly shaped tensor mathematically; quality depends on its learned
input distribution. Random standard-normal samples provide the unconditional VAE
route, whose quality must be tested. Controlled world-state or instruction-based
output needs its own trained conditioning path. This proposed codec does not repair
the separately measured observation/state/recall bottlenecks by itself.

## Review and source record

The isolated Claude CLI received only the user-authorized conceptual design and
review questions. Exact briefs/responses, independent notes and operation checks
are in `runs/reviews/spatial_vae_v1/`. Claude explicitly accepted all six corrections in the follow-up. Its remaining
request was to disclose and hold fixed the Gaussian reconstruction variance; the
objective section above includes this requirement. No unresolved conceptual
disagreement requires a third round. Agreement is not experimental validation.
The follow-up corrected overstatements about
residual gradient guarantees, strided-convolution factorization, runtime errors,
probe interpretation, KL versus file size, and decoder input restrictions.

Primary references: [PixelUnshuffle](https://docs.pytorch.org/docs/2.14/generated/torch.nn.PixelUnshuffle.html),
[PixelShuffle](https://docs.pytorch.org/docs/2.14/generated/torch.nn.PixelShuffle.html),
[Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114).
The exact stride-convolution factorization and residual counterexample above are
independent derivations checked numerically; they are not claimed as new research.
