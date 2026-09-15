# Spatial VAE: refined research specification

15 September 2026. **Original refinement; subsequently implemented.**
Alex has now authorized implementation. See the [v2 implementation and measured
results](spatial-vae-v2-plan.md). The paragraphs below preserve the earlier design
decisions; deferred experiment controls are identified in the implementation plan.

Original status at refinement: **Specification review only.** Alex explicitly requested
precision before implementation. This document refines the new R/P/M/C proposal;
it does not change model code, weights, experiment budgets or the current agent.
The [first implemented VAE and its results](spatial-vae-plan.md) remain the reference.

The useful hypothesis is:

> At comparable resources and achieved information-rate proxy, does learned
> processing before an explicit dimensional bottleneck retain more useful image
> information than immediate compression or processing after compression?

Exact spatial rearrangement preserves tensor values by construction. Whether this
factorization improves reconstruction or semantic utility is an empirical question.
Semantic utility requires a separate target/task; image reconstruction alone cannot
settle that part of the research question.

## Repository fit and proposed implementation plan

| Concern | Existing implementation to reuse | Proposed extension |
| --- | --- | --- |
| Language/framework | Python 3.11+, ordinary PyTorch modules; `pyproject.toml` | No new framework |
| Model and spatial posterior | `pathwm/models/spatial_vae.py` | Extend/version the codec, preserve old exports |
| Stage geometry and decoder | Existing shuffle, pad/crop, stages, mu/logvar, independent decoder | Named stem, mixer and instrumentation boundaries |
| Attention | `pathwm/models/blocks.py`; existing VAE cross-scale attention | Coarsest **self-attention**, a distinct ablation |
| Training/configuration | Readable `experiments/spatial_vae.py`, explicit dictionaries/constructors | Extend this recipe; no second trainer/config hierarchy |
| Logging/checkpointing | `pathwm/io.py`: Run, JSON/JSONL, source snapshots, optimizer/RNG/sampler | Additional rate, compute and probe records |
| Data | Grouped COCO selection and immutable image/source hashes | Preserve split rules; no resizing inside the model |
| Diagnostic readers | `pathwm/models/photo_probe.py`: detached linear/RBF ridge readers | Bounded stage-feature samples and normalized probe metrics |
| Tests | `tests/test_spatial_vae.py`, `tests/test_runs.py`, `tests/test_photo_detail.py` | New attention/loop/probe contracts; retain old export tests |
| Visualization | Existing standalone HTML renderer and labelled comparison PNG slot | Error maps, probe tables and actual rate/distortion points |

Proposed order after implementation is requested: (1) stage/stem/shape contracts
and informative failing tests; (2) local processing and explicit bottlenecks;
(3) posterior/decoder compatibility; (4) lowest-grid self-attention; (5) shared
loop and trace; (6) detached probes and rate/compute logging; (7) a small two-arm
sanity comparison. Run focused checks after each addition. Reuse existing shuffle,
posterior, loss, data, checkpoint and report machinery rather than rebuilding them.
The full ablation/rate study follows the sanity check under a separately fixed budget.

## 1. Guarantees and roles

| Operation | Meaning | What is guaranteed |
| --- | --- | --- |
| R | Spatial/channel rearrangement | Exact permutation, invertible with compatible geometry |
| P | Channel transformation and **local** feature processing | Declared input/output shape; no general invertibility guarantee |
| M | Additional **nonlocal** communication | Declared shape; no general invertibility guarantee |
| C | Explicit reduction of channel dimension | A declared dimensional bottleneck, not an exhaustive location of all loss |

A 3×3 convolution in P already mixes neighboring positions; P and M are an
engineering separation of local processing and nonlocal communication, not disjoint
mathematical categories. A same-size residual can erase its input: `x + (-x) = 0`.
A stride-one stem with more output channels is not guaranteed injective either.
Replace “no information loss occurs here” with “no explicit spatial subsampling or
designated dimensional bottleneck occurs here; retention is measured.”

A `C_out < C_in` projection cannot be injective on the whole ambient feature space.
It may still preserve the information needed on the actual data manifold. The
Gaussian posterior projection, sampling and KL constraint are additional potential
limits; instrument the final features, mu/logvar and decoded mean/sample as well.
Do not sum stage probe errors and call the sum total lost information.

## 2. Concrete stage and shape contract

```text
RGB -> overlapping stem -> P0
    -> [P_pre -> R -> P_post -> M -> C] repeated over stages
    -> spatial mu/logvar -> sampled z
    -> learned channel projection
    -> [channel expansion -> P -> PixelShuffle -> P] in reverse order
    -> final 3x3 RGB projection -> crop to original H,W
```

Keep BCHW tensors. The initial stem is Conv3×3, stride1, padding1, followed by one
small residual local block. A proposed P branch is per-position channel LayerNorm
-> Conv1×1 -> SiLU -> Conv3×3 -> SiLU -> Conv1×1, added to the unchanged residual
input. Keep the branch hidden width equal to its input initially; no mandatory
fourfold expansion. Normalization lives only in the residual branch. Record its
choice and initialization, because either can affect reconstruction. No claim of
invertibility is attached to the block.

R is only PixelUnshuffle(r), with no parameters, nonlinearities or pooling. C is
a separate 1×1 reduction. Identity C is legal only when subsequent dimensions use
the uncompressed width. Identity R changes the required channel/geometry contract;
arbitrary replacement without compatible neighboring modules must raise an error.
Derive actual shapes, rather than pretending an Identity module downsamples.

For v1 use one fixed factor r=2 and a configured stage-width list. Derive the number
of stages from that list. Pad bottom/right to a multiple of r^K, retain original and
padded dimensions, and crop only at the decoder output. No pixels are dropped or
resized inside encode/decode. Batch equal-sized images initially. Replication padding
is part of the model definition; report its fraction for small/odd images.

Decoder stages learn reconstructions; they do not invert learned compression.
The final RGB projection is an output projection, not necessarily a channel expansion.
No encoder features, probe outputs or raw pixels bypass the latent into the decoder.

## 3. Attention and shared iterations

The requested v1 M is self-attention on the **lowest spatial grid before that
stage's C**, after P_post. This differs from the current implementation's coarsest
queries reading finer grids after their projections. Do not reuse its name/results
as if these were the same experiment. Keep the decoder convolutional in the first
comparison, so an encoder ablation does not also change decoder attention.

Use the existing standard pre-norm residual Transformer block where appropriate.
For positions, use fixed 2D sine/cosine features generated from normalized cell
centers and an explicit projection/interface. No fixed learned H×W table and no
custom RoPE are needed initially. Normalized coordinates encode relative location
in the image, not fixed physical/pixel distance. Shape flexibility is neither
scale equivariance nor proven quality on unseen resolutions.

Choose one injection policy: when iterations >0, add position features once at
the loop entrance; expose that initial state and subsequent states. Reuse exactly
the same Transformer object for every iteration. Iterations=0 bypasses both position
injection and the block; iterations=1 is ordinary one-pass attention; iterations=2
or4 is the shared-loop ablation. Intermediate traces are optional and detached for
logging; avoid retaining entire graphs by default. Backpropagation through the
active iterations remains intact. No time-step embedding or dynamic stopping in v1.

Shared weights hold registered parameter count constant as iteration count changes,
but do not hold compute or training activation memory constant. With iterations=0,
the self-attention variant still registers the block but does not execute it. A
separate `mixer=none` variant may omit it entirely; do not conflate changing mixer
type with changing iteration count. Inactive block parameters have no gradients:
tests must require gradients for
**executed learned paths**, and report registered versus active parameters separately.
Do not demand different initial outputs from additional iterations if a residual
branch is initialized to zero; verify call count/sharing and test a nondegenerate block.

Global attention still costs O(N²) in tokens. Fixed hierarchy depth means N grows
with input area. Document tested size/resource limits, profile the high-channel
pre-compression block, and fail explicitly above a declared token budget. No silent
pooling, resize or bottleneck insertion to make a large input fit.

## 4. Stage probes: accessible information, not certified loss

Expose optional snapshots around P/M as well as each C, including the posterior
projections. Record stage name, tensor shape, scalar count and source geometry.
Keep diagnostic caches bounded; sample spatial positions only within training
images for probe fitting, preserving independent validation/test image groups.

Freeze the trained encoder, freeze its mode/buffers and detach features. Fit a small
reader from F_after to F_before, first linear, with a predefined nonlinear check if
linear failure matters to the interpretation. Reuse existing ridge machinery with
bounded samples; do not create an unbounded pixel-by-pixel Gram matrix. Probe losses
never update the VAE by default, and test groups never choose probe capacity/ridge.

For each probe report held-out squared error divided by a training-estimated target
variance (with a declared epsilon), and both target variance and error themselves.
Near-zero target variance is marked degenerate, not a preservation success. Include
training-mean and shuffled-input controls and a known recoverable/identity control.
These are empirical baselines, not theoretical error floors.

Also compare before/after accessibility of one **common image-space target**, such
as the corresponding RGB neighborhood, using compatible probe budgets. A feature
space can become easy to reconstruct by discarding image content before the measured
compression. Feature reconstruction alone would miss this. A poor readout can also
reflect its own capacity/optimization limits. No probe score proves exact mutual
information loss or identifies the only irreversible operation.

## 5. Rate, objective and comparisons

Keep the current transparent Gaussian/MSE convention. For image i with original
area A_i and latent grid h_i×w_i:

```text
K_i = 0.5 * sum_Z,h,w(mu² + exp(logvar) - 1 - logvar)
D_i = sum_valid_RGB((prediction - target)² / (2 * sigma_x²)) / A_i
loss = mean_images(D_i + beta * K_i / A_i)

KL_nats_per_sample = mean_images(K_i)
KL_bits_per_sample = mean_images(K_i / ln(2))
KL_bits_per_latent_position = mean_images(K_i / (ln(2) * h_i * w_i))
KL_bits_per_original_pixel = mean_images(K_i / (ln(2) * A_i))
```

Bits per latent position sum over Z channels. Include all latent cells in KL,
exclude padded image pixels from distortion, and retain size-specific results.
Keep sigma_x², beta, sampling and logvar bounds explicit. L1 would be a separate
objective choice, not silently interchangeable with the Gaussian/MSE likelihood.

KL/ln(2) expresses the same KL exactly in bit units. Its interpretation as a coding
rate is a proxy: no quantized codec or entropy-coded file has been implemented.
Under the joint data/posterior distribution, expected KL to a prior equals mutual
information plus aggregated-posterior/prior mismatch, for a fixed image/latent
geometry (or conditioned on geometry). It is not semantic information.

A beta run gives a rate/distortion **point**, not a curve. A curve needs several
predeclared operating points. Use validation to identify overlapping achieved-rate
ranges; compare only in those ranges and do not extrapolate nonexistent matches.
Record total/active parameters, approximate forward MACs or FLOPs with a stated
counting convention, actual train/inference times, peak GPU allocation/reservation,
update/example counts and the exact input/latent geometries. Matching beta, latent
shape or parameter count alone does not match achieved rate or compute.

## 6. Corrected ablation matrix

Hold stem, posterior, decoder, objective, data and evaluation fixed within each pair.
The new stem/P implementation differs from the earlier tested codec; keep that old
codec as a separate historical reference, not one side of an allegedly single-factor pair.

| Arm | Encoder stage | Question |
| --- | --- | --- |
| A_exact | Conv2×2/stride2 | Equivalence control for B with exactly rearranged weights/padding |
| A_local | Conv3×3/stride2 | Practical conventional comparator; receptive-field change disclosed |
| B | R -> C | Immediate-compression reference |
| C | R -> P -> C | Does processing before reduction help versus B? Extra compute is a confound |
| C_after | R -> C -> P | Does processing location matter versus C, at approximately matched budgets? |
| D | R -> P -> M_once -> C at lowest grid | Does nonlocal communication help, with P unchanged? |
| E | R -> P -> M_shared(2 or4) -> C at lowest grid | What is gained from extra shared computation? |

A_exact and B are the same linear family, not evidence of a better architecture.
A_local and B differ in overlap/receptive field. C versus C_after is the important
location comparison; the changed processing width makes exact resource matching
nontrivial. Report discrepancies instead of treating depth equality as compute equality.
C versus D already isolates adding attention when other settings are held fixed.

Before attributing gains specifically to **weight sharing**, add an untied stack
at equal width/depth and report its extra parameters. A separate narrower untied
control can approximately match parameter count while changing width/compute.
Do not claim width, depth, parameters and compute all match simultaneously.
These controls are needed for the corresponding causal claims, not for the first
forward/backward sanity test.

## 7. Scope of the next implementation, not a run authorization

Expose only configuration fields with distinct purposes: stem width, compression
widths (deriving stage count), pre/post block counts, reduction mode/factor, latent
width, mixer enabled at the last stage, iteration count, and beta. Fix the baseline
normalization, activation and Fourier policy in readable construction code. Avoid
duplicated `num_stages`/`stage_channels`/`compression_channels` settings and broad
registries. `Identity` is allowed under the shape contract.

First compare A_local with C as a mechanics/learning sanity test, with comparable
model size and disclosed compute. Also run A_exact/B numerical equivalence and
probe positive controls. Then add C_after, attention and shared-depth comparisons;
the complete matrix is not required in the first training batch. Determine data,
steps, beta points, seeds, stopping rules, success thresholds, time/memory/disk
limits and fresh evaluation groups in the experiment plan **before** training.

Save originals, mean/sample reconstructions and absolute RGB error maps with a
common stated color range, plus per-stage probe error/variance, KL/rate and compute.
Retain failed gates. Use the same model instance at two resolutions for forward
and backward checks, exact shuffle/pad/crop and strict checkpoint roundtrips,
attention position/shape checks, loop sharing/call counts, active gradients,
diagnostic isolation and compatible Identity replacements.

Adaptive regional budgets remain only an extension point after post-processing and
before C. Future masks/routing/shape metadata and their coding costs would be part
of the representation budget. Do not add pruning, flows, MoE, learned stopping,
diffusion, adversarial objectives or pretrained backends to this baseline.

## Review and primary references

Actual Claude review receipts: `runs/reviews/spatial_vae_refinement_v2/`.
Review supplied only the user-provided design and generic methodological questions.
Private repository code, data and measurements were not exported.
The follow-up explicitly corrected overbroad stride equivalence, a mistaken C/D
conflation, probe normalization/baseline wording, a purported mandatory position
injection policy and simultaneous matching of untied/shared resources. No remaining
substantive disagreement was identified; peer agreement is not experimental proof.

- [PyTorch PixelUnshuffle](https://docs.pytorch.org/docs/2.9/generated/torch.nn.PixelUnshuffle.html): exact rearrangement and compatible tensor shapes.
- [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114): Gaussian posterior reparameterization and the variational objective.
- [Fixing a Broken ELBO](https://proceedings.mlr.press/v80/alemi18a.html): representation quality and rate/distortion are distinct from one aggregate objective value.
- [Independent diagnostic probes](https://arxiv.org/abs/1610.01644): a probe tests accessibility to its readout family without updating the original network.
- [Universal Transformers](https://arxiv.org/html/1807.03819v3): shared iterative attention is an established design family; this proposal omits their dynamic stopping and time-step encoding. The one-time spatial-position injection above is our simple convention, not a theorem or a faithful reproduction of that paper.

The exact A_exact/B factorization and the residual erasure counterexample are direct
mathematical observations, also checked in the earlier design review. The proposed
controls and defaults are research-design choices; no new efficacy claim follows.
