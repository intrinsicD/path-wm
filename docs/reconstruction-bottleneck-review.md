# Where reconstruction detail is lost

12 September 2026. Source and saved-checkpoint review, with primary-paper comparison.
No new training, model revision or high-resolution benchmark. These mechanisms are
separate from entity recognition, memory and action prediction.

## Exact local information path

1. Dataset preparation resizes/crops photographs to 64×64. Information outside the
   crop and above that sampling resolution is unavailable to this model.
2. `ImageEncoder.patch` (`pathwm/models/modalities.py:124`) applies a non-overlapping
   4×4 stride-4 convolution, with 3 input and 32 output channels. Each patch contains
   48 RGB scalars; its linear projection has shape 32×48. It can retain at most
   32 independent input directions. That is a continuous linear-algebra restriction,
   not a dataset-specific reconstruction-error floor or an encoded-bit-rate claim.
3. The handwritten assignment (`experiments/hierarchy_fusion.py:265`) is much more
   restrictive: only paired per-channel color means vary with the image. Its matrix
   rank is 3. Other values duplicate signs, add constants or encode position; they
   do not retain the original arrangement of pixels within that patch. Opposite
   checkerboards with identical patch means were already verified to have identical
   features at every scale. The strongest demonstrated loss occurs before attention.
4. Later merges pool features, with local cross-attention. However, the decoder
   receives all three scales, including the finest 16×16 grid. It is not forced to
   reconstruct exclusively from the final 4×4 grid. These later operations cannot
   recover distinctions absent from their inputs.
5. The handwritten decoder initially has center-only convolution kernels, nearest
   upsampling, and zero convolution weights in a residual branch. Its output is
   constant within each 4×4 block. Neighbor-mixing kernels are trainable, but the
   original residual weight gradients are zero until a branch is opened; its final
   bias can still learn. Ordinary initialization produces finer reconstructions with
   the same architecture. Nearest upsampling followed by learned convolution is not
   itself a proof of an unavoidable block-artifact limit.

Read-only checkpoint SVD reconfirmed ranks 3 (handwritten), 11 (repaired hand,
seed7501), and 32 (ordinary trained, seed7501), at relative tolerance 1e-8. Rank
increase alone does not prove useful retained detail. Existing checkerboard probes
and saved validation panels remain the relevant behavioral evidence.

Spatial size does not by itself measure information capacity. Rearranging
64×64×3 values into 16×16×48 values can preserve every value exactly, with an inverse
rearrangement and zero trained parameters. This is a mathematical control, not
compression or learned world understanding. A later channel reduction, quantization
or other non-injective operation can discard information. More model parameters
after such a fixed loss cannot recover which of two indistinguishable inputs occurred.

## What published models add

[DC-AE](https://arxiv.org/html/2410.10733v8) combines shortcuts based on rearranging
spatial values into channels with learned residuals and staged high-resolution
adaptation. Its full downsampling blocks also reduce channels, so they are not
lossless. Its ablation finds that adding capacity at the same latent size can still
hurt reconstruction because optimization becomes harder. Its reported 1024×1024
face reconstructions have finite PSNR/SSIM, not exact pixel recovery. These are
representative published results, not a current leaderboard claim.

[DC-AE 1.5](https://arxiv.org/html/2508.00413v1) demonstrates that increasing latent
channels improves reconstruction, while making the downstream generative problem
harder in its experiments. It trains partial-channel reconstruction to organize
structure and detail. This supports separating reconstruction capacity from useful
latent organization; it does not validate our hierarchy's entity semantics.

[Latent diffusion's autoencoder](https://arxiv.org/html/2112.10752v2) combines
perceptual and adversarial objectives to encourage realistic local detail. A sharp
reconstruction may contain inferred texture rather than the original pixel pattern.
[Blau and Michaeli](https://arxiv.org/abs/1711.06077) formalize a perception/distortion
tradeoff. Pixel fidelity and perceptual realism therefore require separate evaluation.
Under ambiguous encoding, squared error favors an average of possible targets;
sharpness alone does not establish faithful memory of an observation.

## Factors and what our evidence can establish

| Factor | Role and current evidence |
| --- | --- |
| Initialization | The hand stem explicitly loses within-patch arrangements. Its numerical scaling also made decoder updates unstable at the original learning rate. |
| Architecture and latent capacity | The 48→32 stem is structurally compressive; the rank-3 assignment is more severe. Wider retained channels, smaller patches or a retained fine-scale path are candidates to compare. |
| Parameters and compute | More parameters can learn richer transforms and priors. More latent capacity can carry more image-specific evidence. These are different resources; neither is fully used by the hand circuit. |
| Optimization and duration | Only 384 updates × batch4 = 1,536 sampled-image exposures, or three dataset-pass equivalents by count, from512 images. This is a short diagnostic, not a convergence study. Actual dataset coverage need not equal three complete passes. |
| Data and resolution | Generalization needs representative image detail at the desired resolution. Reconstruction uses the image itself as target; segmentation/entity labels are not required for this objective. |
| Objective | Current RGB MSE measures fidelity. Perceptual/adversarial terms can change detail appearance and may trade away exactness. They cannot reveal unobserved details with certainty. |
| Curriculum | A low-resolution learning phase followed by higher-resolution adaptation is a concrete option. It cannot substitute for an information path that retains the desired distinctions. |

Proposed efficient order: establish a small reconstruction control with a reversible
patch rearrangement; compare a modest detail-retaining stem/latent path with current
initialization; first fit a tiny fixed training set, then evaluate held-out images.
Keep optimization/exposure matched when testing architecture. Extend duration on
the same model separately, and only then compare higher-resolution inputs/targets.
Keep direct pixel bypasses explicit: successful copying would not validate compressed
agent state or memory. Preserve the handwritten checkpoint as a reference; no new
constructor, curriculum or run budget is adopted by this review alone.
