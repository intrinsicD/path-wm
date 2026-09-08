# Location-distribution supervision with frozen representations

Adaptive geometry-decoder follow-up within the overnight encoder/decoder task;
albedo excluded. This is a declared extension to the program, not a replacement
for the12shared-decoder fits. Existing P1 fresh results show substantial error
tails. An exploratory six-pixel boundary cut finds136/512fresh cases near an
edge, versus3.30% of CCHI training frames. The first paired head's pass fraction
on those cases is41.9%CNN/62.5%ViT versus88.6%/86.7% for the remaining cases.
These observations motivated this test; the boundary diagnostic is post hoc.

Coordinate MSE constrains the mean of each softmax map without uniquely defining
its shape. Test one objective change: add0.001times mean KL(target||predicted)
for the two location distributions. The target places bilinear probability
weights on the four surrounding vertices of the same16x16[0,1] output basis;
its expectation exactly equals the target coordinate, including boundaries.
This distribution describes a point-supervision convention, not an object mask,
spatial feature center, learned uncertainty model or epistemic confidence.

Keep both frozen P1 encoders and exactly the same independent pose-head topology,
initialization, input normalization, AdamW3e-4/wd1e-4, norm1clipping, FP32/TF32off,
FP16feature cache, seeds9107/9108/9109,4,000updates and minimum-validation-q selector
(earliest exact tie). Preserve orientation MSE and its independent learned
pooling. Do not change pooling, grid, depth, backbone, data or augmentations.
All CCHI training/validation/test coordinate targets were checked in[0,1]; reject
out-of-range targets instead of silently clipping them.

Only the pose head trains in this stage. The RGB/mask heads in P1 were independent
and cannot affect its gradients. Preserve the original sampler's discarded COCO
draw before each32-frame PushT draw so the actual pose examples match P1 exactly.
Log the complete paired draw identity but count32 actual training frames/update,
not64. No COCO image is consumed by this fit. Preserve a zero-regularizer
development control for each encoder and verify its pose weights/output behavior
against the corresponding original P1 development run before formal execution.
The full reference remains the completed P1 fit, not a newly selected checkpoint.

Evaluate all2651validation/2506test PushT frames,512training-prefix frames, and
all512fresh simulator cases without adaptation. Preserve physical raw errors,
location distributions, output-map entropy, per-case tolerance rates, and the
fixed six-pixel boundary/interior cut. Report full-population q and case-pass
fractions as primary outcomes, orientation separately, and all three paired
seed differences. The test can reveal a useful decoder objective; a null result
does not prove boundary information absent. It does not repair missing training
coverage, prove calibrated uncertainty, or establish prediction/control quality.

Claude's public conceptual review connected this design to DSNT and integral
regression, and requested explicit manipulation checks. Preserve mean/per-object
map entropy, probability on the target's nonzero support, and target-to-predicted
KL at initialization, validation and held-out evaluation; log both training loss
terms from the first through final update. Compare the same map statistics against
the saved P1 fresh outputs. A null coordinate result without changed map statistics
is inconclusive about useful distribution supervision. The coefficient is fixed
before development and is not claimed comparable to DSNT's different loss scale.
KL(target||prediction) is finite for the sparse target and finite model logits;
the reverse direction would be infinite outside target support. Bilinear targets
vary in sharpness continuously with the coordinate. This convention is not claimed
novel, and no universal boundary benefit follows from it.

Essential checks: normalized nonnegative targets, exact target expectations at
random/end/grid-boundary coordinates, finite KL and gradients, and a zero-weight
objective that reproduces ordinary pose gradients. Then50-update development
prefixes for each encoder with weights0and0.001, verified HTML, commit and source
freeze. Six formal fits, initially15minutes each (90minutes maximum fitting),
one GPU workload at a time after D2. Actual development timing may tighten this
cap prospectively; no extra architecture or regularization sweep. Stop new
training by06:00Berlin and report by07:00Berlin. All failures/stopped runs stay
visible. The complete program now contains33vision/geometry fits and15independent
category probes; new geometry fits have half the per-update frame consumption.

## Development control refinement before formal fitting

The first zero-weight CNN development check failed its blanket2e-5 parameter
tolerance: location biases differed by4.2915e-5 and orientation-pool bias by
2.7228e-5. All other parameters differed by at most1.1921e-7; first-step losses,
gradients and every sample draw matched. These particular per-channel spatial
softmax biases cancel analytically. Tiny reduction roundoff can nevertheless
give their theoretically zero gradients an Adam update. This is not evidence
of a different informative head function.

Preserve the failed control and exact source under `development/localization/`.
Before changing the criterion, add a test that independently shifts both named
softmax bias tensors and verifies unchanged pose, location and pooling outputs.
Development revision2 uses a gauge-aware parameter comparison: preserve/report
the maximum difference over all parameters, but require maximum difference
over all parameters except exactly those two redundant biases≤2e-5. Additionally
require maximum absolute difference≤2e-5 in normalized pose outputs and both
probability maps on every development PushT frame in all three splits. No other
parameter is excluded, and no model/training behavior changes.

Run the four revision2 controls/prefixes in `development/localization_v2/`, after
the now-started D2 formal queue. Only then commit, seal and run the six geometry
fits. The failed check is retained as development evidence, not silently relabeled
as passed. This refinement addresses a proven parameter redundancy, not a relaxed
accuracy gate or a coefficient change.
