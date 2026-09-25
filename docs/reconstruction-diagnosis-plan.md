# Reconstruction diagnosis

Alex asked to establish what is actually needed, with actual Claude Opus 5.5 at
medium effort. This follows the [native decoder comparison](native-pyramid-decoder-plan.md).
No new representation, fitted readout, model pathway or encoder change is adopted.

## Audit and saved curves

The actual full model exports 16×16×64 fine features before coarse pooling. The
48→64 patch projection may be injective; residual attention/MLP processing has no
invertibility guarantee, but its raw residual stream is retained. LayerNorm is
inside the branches, not an overwrite of the exported stream. Coarse pooling is
lossy, but does not remove the fine export. Thus no operation audited so far proves
that the full export loses the observed image detail. The preservation guarantee
in the agreed filter-bank design remains unimplemented; empirical loss and absence
of a structural guarantee are different claims.

The optional fine decoder connection reduces 64 channels to32; each RGB4×4 patch
has48 scalars. That connection is a candidate bottleneck, not a bound on the whole
decoder, which also receives slots, coarse features and spatial neighbours.
Nearest upsampling and the joint mask/RGB objective are additional candidate
limitations. Do not attribute the result to one of these without a comparison.

Exploratory summaries of the saved training curves use disjoint250-update means,
all2000 rows, no selection. Last-block versus preceding-block reconstruction error
changes: slots −0.006%/−0.096%, pyramid −0.470%/−0.438% for seeds3601/3604.
These changing-batch curves have no intermediate fixed-population evaluations;
they do not establish convergence. Saved analysis and Claude receipts:
`runs/reviews/reconstruction_diagnosis_20260925/`.

## Registered actual-encoder input-inversion diagnostic

Registered before execution,25September2026. This measures the existing frozen
export without adding/training a decoder. It optimizes an input image, not model
weights. Scope: constructive retention evidence on a small declared population,
not deployable reconstruction, a global inverse or model-quality qualification.

- Actual `SlotPerception(64,7,3,decoder_width=32)`, native RGB64, patch4, two scales,
  depth1; parent `runs/real_visual_joint_repair_3501_u6000_v1/last.pt` SHA256
  `5b51c448395d8eec9c962212e7b97f913b7cb3b548a816e68432ef41c2cac8ce`.
  Freeze/eval all model weights; verify state and source hashes unchanged.
- Regenerate the original first64-image chunk of each fixed decoder-evaluation
  population, train seed3602 and validation seed3603, kind-table textures, using
  existing `perception_batch`. Require the original chunk hash against all four decoder runs. Use first16 images
  in each population, with the actual full architecture (no model downscaling).
- Optimize each image's pixels by matching **only the exported fine-scale features**
  to the target's detached features. Loss=sum of per-image feature MSEs. Pixel RGB
  targets are used only for scoring, never in the optimization objective.
- Two separately reported starts for every image: gray0.5 and uniform noise with
  seeds3611/3612 for train/validation. Adam initial lr0.03,1000 updates; predeclared cosine decay
  lr(t)=0.03×(1+cos(πt/1000))/2 for t=0…999. Clamp input to[0,1]
  after every update. No early-success stop, best-iterate/start selection,
  or adaptation from scores. Report final iterate for both starts.
- Record feature loss and diagnostic RGB error at0/every50/final. Final raw per-image
  RGB MSE, machine-body MSE and adjacent-gradient error use the previous recipe's
  definitions. Compare the identical16 saved records of both pyramid decoder seeds.
- Positive retention criterion: **all64 inversions (32 images × two starts)** final RGB MSE≤1e-4;
  each population/start mean machine-body and gradient errors≤10% of **each** matched
  saved pyramid decoder mean. Failure is inconclusive, since inversion optimization
  may fail. Success is evidence of recoverable details for these images, not proof
  of global injectivity or that the current decoder can learn the inverse.
- Sanity reference: SVD and pseudoinverse of the actual frozen64×48 stem kernel;
  undo its bias on its actual convolution output, fold patches back toRGB. Require
  full column rank and every reference-image MSE≤1e-8. No fitted readout.
- FP32 CUDA, total10min/6GiB cap; abort and preserve failure if exceeded, no budget
  adaptation. Save raw metrics, final candidate images, parent/source/settings and
  an existing-renderer standalone report with structural QA. No model checkpoint
  needed because no weights change; save final image/optimizer state for inspection.

Claude critiques this protocol before execution; independent result review follows.
Any needed model modification will be planned with Alex before implementation.

Pre-execution Claude review: adopted fixed cosine decay, all-four input-hash checks
and explicit64-inversion count. A near-zero feature residual with high pixel error
would warrant investigating approximate ambiguity; finite-tolerance equality would
not by itself prove exact non-injectivity. No extra collision gate is adopted.
Success is a constructive recovery witness, not proof the features uniquely determine
the image. Claude’s stronger wording in its review is not adopted. Deterministic
algorithms and IEEE fp32 are requested through existing `seed_everything`.

## First result and registered longer diagnostic

`runs/native_encoder_inversion_v1/report.html`: valid, strict retention gate **fails**.
1000updates: train/validation mean RGB MSE gray0.0000600/0.0000701,
noise0.0005025/0.0005054. Both starts substantially outperform matched decoders, but
one gray-validation image exceeds1e-4 and all noise images do; noise gradient guards
also fail. Feature errors continue falling into the cosine tail. Runtime36.3s,
peak reserved0.170GiB. These are partial recovery measurements, not a passing
strict-retention claim; the first result and protocol remain preserved.

Before executing a **separate follow-up**, register exactly10000updates from the
same fresh gray/noise starts (not resume), with cosine denominator10000, lr0.03,
scoring every500updates. Everything else, including targets, full model, all64
individual-image criteria, body/gradient gates and10min/6GiB cap, stays unchanged.
This is an explicitly result-informed optimization-budget diagnostic, not an
independent confirmation population. Run `native_encoder_inversion_long_v1` uses
`runs/reviews/reconstruction_diagnosis_20260925/invert_long.py` and saves its own
protocol snapshot. No second extension or selection is authorized by this protocol.
The question is whether more inversion computation recovers the residual detail;
it does not test whether longer decoder training works.
