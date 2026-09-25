# Reconstruction diagnosis

Alex asked to establish what is actually needed, with actual Claude Opus 5.5 at
medium effort. This follows the [native decoder comparison](native-pyramid-decoder-plan.md).
No new representation, fitted readout, model pathway or encoder change is adopted.

## Outcome

**Keep the existing encoder and representation; diagnose the learned decoder next.**
The actual frozen fine export supports near-exact recovery on these32 known images
from two independent starts. No added readout, model weights, modules or parameters.
This shows available detail that the trained decoder fails to reproduce here;
it does not identify which decoder/training limitation is responsible.

The separate10000-update diagnostic `runs/native_encoder_inversion_long_v2` passes
all64 per-image reconstruction criteria and all body/gradient guards:

| Population | Start | Mean RGB MSE | Worst image MSE |
| --- | --- | ---: | ---: |
| TRAIN | Gray | 4.5235e-12 | 1.6589e-11 |
| TRAIN | Noise | 4.6120e-12 | 1.4609e-11 |
| Validation | Gray | 9.8216e-12 | 9.9777e-11 |
| Validation | Noise | 1.3601e-11 | 1.1443e-10 |

Matched learned pyramid decoders have mean MSE0.00727–0.00782 on these images.
Stem rank48; condition number20.226; numerical stem-inverse worst MSE4.07e-14.
The long run takes347.57s and0.170GiB reserved. Independent audit recomputes pixel,
body and gradient metrics from saved images, verifies all optimizer counters10000,
identical initial rows/targets, parent/source hashes and report status. All pass.
Existing reports receive structural QA; no changed renderer or browser QA claim.
No shared model/recipe/test source changed; no new full regression-suite run claimed.

Preserve the valid1000-update failure and invalid first longer attempt described
below. The longer budget was chosen after the first result, on the same known
images: this is a recovery witness, not independent confirmation or a general
invertibility guarantee. The seven stored slots and fine-detail memory were not
tested by this inversion. The formal preservation guarantee remains a design gap,
but is not established as the cause of this reconstruction shortfall.

Report: [Actual-encoder inversion](../runs/native_encoder_inversion_long_v2/report.html).
Audit: `runs/reviews/reconstruction_diagnosis_20260925/audit.json`.
Portable evidence: `ara/evidence/tables/reconstruction_diagnosis_2026-09-25.json`.
The final section is a proposed next slice for discussion, not authorization or
implementation of new recipe behavior.

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
three gray-validation images exceed1e-4 and all noise images do; noise gradient guards
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

Execution correction: `native_encoder_inversion_long_v1` is **invalid**. A literal
loop bound remained1001 while settings and recorded final step said10000. The
last scored state was at1000, and the serialized state followed1001 updates. Its
original `valid:true` is superseded by `invalidation.json`; no quality inference.
Preserve all outputs. Corrected `invert_long_v2.py` uses10001 loop evaluations,
10000 optimizer updates and asserts the optimizer counter. Rerun as
`native_encoder_inversion_long_v2` under the same registered protocol/gates/budget.

## Proposed next model slice — joint plan, not implemented

Alex subsequently clarified the end-to-end goal: model feature access, faithful
latent-code inspection and image generation. The [image-code contract plan](image-code-contract-plan.md)
owns that broader path. This objective comparison remains a decoder diagnostic
within it, not the complete solution.

Keep the encoder, slots, representation, feature connections and decoder width
fixed. Before changing capacity or synthesis, compare the existing reconstruction
objective against RGB-only training. This tests one candidate cause: interference
between mask supervision and reconstruction. It does not assume that cause is real.

- Actual full pyramid decoder, same u6000 parent and zero-initialized connections.
  Paired seeds3601/3604, identical data and initialization per seed. Arm A:
  RGB MSE+0.5mask CE; arm B: RGB MSE. Same2000 updates, AdamW3e-4, clip1,
 10min/6GiB per run; last checkpoint only. Four runs total.
- Add fixed evaluation at steps0,500,1000,1500,2000 on the existing256+256 images.
  Preserve training RNG and sampler state; A must reproduce the previous A final
  records exactly. Report reconstruction and mask trajectories separately.
- Proposed quality gate: B reduces full-image MSE≥20% against A on both populations
  for both seeds, with no worse mean body or gradient error. A win is evidence for
  this training change, not proof of a unique mechanism.
- Proposed preservation gates: matched alpha pixel accuracy drops≤0.5 percentage
  points and machine-pointer agreement with the frozen parent drops≤1 point against
  A on each population/seed. Report mask CE, raw assignments and failure examples.
  RGB improvement with mask degradation is a trade-off result, not adoption.
- The missing recipe objective option and periodic fixed evaluation/mask/pointer
  metrics require implementation. Discuss this concrete slice with Alex before
  adding them, per the actual-model workflow. No new model module is proposed.
- A null result only fails to show a benefit under this budget; it does not rule out
  objective/optimization effects. Flat curves cannot prove a capacity limit. Choose
  any budget or connection change as a later, separately agreed single-factor slice.

Claude withdrew its initial fitted-readout recommendation and agreed to diagnose
the existing decoder's training before adding width or another architecture.

Metadata disclosure: the longer-run settings string retained `t=0..999`; the
formula denominator, executable loop and actual optimizer counters use10000
updates (t=0…9999). Preserve the original metadata and use those audited counts.
The diagnostic script would conflate result/report status on a rendering exception;
both completed reports rendered successfully, so no such failure occurred.

Final actual Claude review found no material issue and agreed with the bounded
next-slice proposal. Its phrase “exact recovery to fp32 precision” is stronger than
adopted here: the measured MSEs are small but nonzero; no bitwise equality claim.
The audit's `*_gate` booleans mean **consistency of the recorded gate**, not quality
success; the short run remains a failure. See `claude-completion.md` and raw results.
