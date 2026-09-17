# Relevance of multimodal-vae-comparison

Alex supplied a duplicated URL on 18 September 2026. Inspected the intended
[repository](https://github.com/gabinsane/multimodal-vae-comparison), pinned at
`5cfef9a20fae810e716573c7b07b8a504d98d6ad` (commit dated 19 May 2025).
This is a source review, not a reproduction or adoption of its models/results.

**Useful for controlled multimodal evaluation and optional probabilistic fusion.**
Our [input design](multiscale-modality-design.md) still exports each scale after
learnable processing and before compression. A probabilistic fusion/readout module
can consume that representation when its application calls for it.

## What transfers

| Reference | What it provides | Possible use here |
| --- | --- | --- |
| CdSprites+ | Paired images/captions with five attribute-complexity levels; joint and cross-generation evaluation | Test access to shape, color, size and position, compare image-only/text-only/both, and measure attribute errors rather than only reconstruction loss |
| VILANRO-TRIMODAL | Images, language instructions and action trajectories | A later test of conditioning an action-output consumer on multiple inputs; dataset performance alone would not validate closed-loop control |
| MVAE / product of experts (PoE) | Product-based Gaussian latent fusion | A baseline when a consumer needs a posterior combining modality evidence |
| MMVAE / mixture of experts (MoE) | Mixture-based latent inference | An alternative posterior-combination baseline |
| MoPoE | Mixture of products over modality subsets | A way to compare partial-input combinations; the inspected implementation enumerates nonempty subsets, so cost grows exponentially in modality count |
| DMVAE | Shared and modality-private latent variables | An example of reserving capacity for modality-specific information |

Dataset/evaluation descriptions come from the pinned
[README](https://github.com/gabinsane/multimodal-vae-comparison/blob/5cfef9a20fae810e716573c7b07b8a504d98d6ad/README.md);
fusion and private-latent behavior were checked in the
[model implementations](https://github.com/gabinsane/multimodal-vae-comparison/blob/5cfef9a20fae810e716573c7b07b8a504d98d6ad/multimodal_compare/models/mmvae_models.py).
These are candidates for comparison, not a ranking for our workloads.

## Fit to the current architecture

The inspected toolkit's central path is modality encoders → posterior fusion →
latent sampling → modality decoders. Its example uses CNN image and transformer
text encoders with a 24-dimensional latent. It does not implement our repeated
invertible processing, pre-compression scale exports and generic transformer-loop
consumer contract. See the
[example configuration](https://github.com/gabinsane/multimodal-vae-comparison/blob/5cfef9a20fae810e716573c7b07b8a504d98d6ad/multimodal_compare/configs/config_cdspritesplus.yml)
and [base model](https://github.com/gabinsane/multimodal-vae-comparison/blob/5cfef9a20fae810e716573c7b07b8a504d98d6ad/multimodal_compare/models/mmvae_base.py).

Our interpretation: put an optional fused stochastic representation **downstream
of F**, for example in a consumer whose transformer loop reads F and whose own
layers produce posterior parameters or an output. Other consumers retain access
to F. Forcing all information through that new bottleneck would change the chosen
interface and needs its own justification. No VAE/KL objective is required merely
to learn the invertible filter banks.

DMVAE's shared/private split concerns **modalities**. The earlier base-plus-residual
question concerns **applications**. Image-private texture, for example, could be
useful to several applications. Latent factorization does not itself create
additive feature/weight residuals, guarantee disentanglement, or establish equal
application priority. The toolkit's optional dimensionality-based likelihood
scaling is also a different choice from our task-exposure/loss-normalization policy.

A caption generally leaves image detail unspecified. Text-to-image quality should
therefore measure described attributes and the distribution of plausible remaining
details; predicting the exact hidden image is not information preservation.
Conversely, reconstruction from the image's own pre-compression features can test
retained evidence. Synthetic attribute tests cannot establish broad semantics,
real-image quality or invertibility; numerical inverse tests remain separate.

PoE precision should not count several scales of one observation as independent
new evidence. Actual modalities can also have correlated errors, missing content
or contradictory observations. Evaluate modality subsets and corruption/conflict
cases before trusting fused uncertainty. A mixture operator does not inherently
require conditional independence; that is separate from any factorization of
the model's generative likelihood.

## Practical reuse boundary

Use the benchmark ideas and inspect small methods individually. Keep our readable
recipes rather than importing its training framework. The repository declares
GPL-3.0; any copied code needs its provenance/license retained. Its documented
environment uses Python 3.8 and Lightning 1.7.7, with explicit CUDA assumptions in
model paths. The README directs supplied leaderboard weights to an older revision.
Source review does not establish compatibility with our environment or validate
those weights. Sources: [environment](https://github.com/gabinsane/multimodal-vae-comparison/blob/5cfef9a20fae810e716573c7b07b8a504d98d6ad/environment.yml),
[license](https://github.com/gabinsane/multimodal-vae-comparison/blob/5cfef9a20fae810e716573c7b07b8a504d98d6ad/LICENSE), and README above.

The Gaussian parameter contract also needs auditing before code reuse: the base
`product_of_experts` exponentiates its second input as a log variance, returns a
variance, and POE.forward supplies that result directly as `Normal`'s standard
deviation. Encoder paths use positive softmax outputs as distribution scales.
This is a concrete inconsistency in the inspected revision, not a conclusion
about every upstream method or historical published result. See
[base helper](https://github.com/gabinsane/multimodal-vae-comparison/blob/5cfef9a20fae810e716573c7b07b8a504d98d6ad/multimodal_compare/models/mmvae_base.py#L204),
[POE caller](https://github.com/gabinsane/multimodal-vae-comparison/blob/5cfef9a20fae810e716573c7b07b8a504d98d6ad/multimodal_compare/models/mmvae_models.py#L189)
and [encoder output](https://github.com/gabinsane/multimodal-vae-comparison/blob/5cfef9a20fae810e716573c7b07b8a504d98d6ad/multimodal_compare/models/encoders.py#L49).
Executing only the inspected helper on two unit Gaussians with log-variance zero
returns approximately 0.5 as its second value; passing that as a Normal scale gives
variance 0.25, while the correct product variance is 0.5. The isolated check and
source hash are saved in `upstream-gaussian-sanity.json` under the review directory.

First suggested use: a small paired image/text attribute task, with separate
consumer scores for observed-input readout and missing-modality prediction.
Compare the reversible and unconstrained encoders at the accepted scale taps;
declare evidence access, splits, capacity, per-task quality margins and costs.
Posterior-fusion comparisons can follow only if a concrete consumer needs them.
This is a proposed experiment, with no installation, dataset download, new model
or training budget selected.

Inspection snapshots, pinned-source receipt and public-only Claude methodology
review are saved in `runs/reviews/invertible-filter-choice-20260918/`.
