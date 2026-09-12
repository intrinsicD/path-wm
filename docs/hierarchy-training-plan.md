# Train from handwritten weights and locate reconstruction limits

12 September2026. User explicitly adopts handwritten hierarchy weights as training
initialization and asks whether reconstruction is limited by encoder, decoder or both.
Use the exact `runs/hierarchy_weights_v1/constructed_7401/weights.pt`, never the
ridge-fitted variant. Preserve that file and all previous checkpoints.

## Questions and fixed comparison

Hold architecture, data, RGB objective, optimizer and exposure fixed. Train the
existing deep_fusion image hierarchy (width32, three scales, depth2, fusion2) with
its same DenseHead RGB decoder. Freeze the mask head in all arms. Optimize RGB MSE
only, so segmentation gradients do not confound this reconstruction diagnosis.
Mask scores remain secondary untrained-head diagnostics, not quality gates.

Two paired sampling seeds7501/7502,512 train/128 validation/128 reused internal test
COCO64 images.384 updates per run, batch4, AdamW lr0.0003, weight decay0.0001, FP32,
gradient clip1, validation every64. Identical hand state and batch stream within seed;
ordinary initialization uses that seed. Score the fixed final checkpoint. No model
selection or coefficient/threshold revision using test outputs. Previous joint
RGB/mask training runs have a different objective and are contextual references only.

| Arm | Initialization | Parameters allowed to learn |
|---|---|---|
| hand_both | Exact handwritten file | Encoder and RGB decoder |
| hand_encoder | Exact handwritten file | Encoder, fixed RGB decoder |
| hand_decoder | Exact handwritten file | RGB decoder, fixed encoder |
| ordinary_both | Ordinary random initialization | Encoder and RGB decoder |
| hand_open_decoder | Handwritten file + dormant-branch opening below | Encoder and RGB decoder |

Dormant-branch opening is a separately labeled initialization variation: in each
zeroed RGB-decoder SpatialResidual branch, initialize ONLY the first convolution weight with
normal std0.001 (generator seed+200000), retaining its zero second convolution.
The initial branch output and complete model predictions must stay bit-identical
to handwritten initialization; the second convolution must then receive gradients.
Original binary unchanged. The exact-handwritten arms remain the primary evidence.

## Information and trainability probes

Before and after training, use an independent synthetic red checkerboard and its
inverse, with identical per-patch RGB means. Save per-scale feature differences,
prediction differences and patch-projection singular values/rank. The handwritten
patch projection has three independent color means; identical stem outputs imply
identical features throughout the deterministic downstream network. This demonstrates
an input ambiguity, not the attainable population error for natural images.

Also evaluate direct4x4 patch-mean reconstruction on the SAME test RGB and save raw
predictions. This is an analytic reference, not a decoder consuming trained features
and not a universal error floor: natural-image priors could infer missing detail.
It helps quantify how much the handwritten readout loses beyond its color summary.
Record pre-training per-module gradient norms and post-training weight changes;
verify frozen modules bit-for-bit and detect dormant branches instead of silently
claiming every trainable parameter learned.

Interpretation: a frozen-encoder decoder improvement supports a readout limitation.
An encoder-only improvement supports a representation/compatibility limitation.
Joint improvement beyond both one-sided arms supports value in joint adaptation,
not additive attribution or proof of a unique bottleneck. Failure to improve within
this short budget is inconclusive about capacity; learning rate, conditioning,
limited data and remaining architectural constraints are still possible causes.

## Gates, budget and artifacts

For each adaptation, meaningful reconstruction improvement is at least10% relative
RGB MSE reduction from exact handwritten baseline on BOTH seeds. Hand initialization
versus ordinary must reduce final MSE at least5% on both; opening versus exact hand
uses the same5% criterion. Joint advantage over one-sided runs requires5% lower MSE
than each on both seeds. These are diagnostic screens, not confidence estimates.
Also report fixed-threshold mask IoU/BCE, gray, shuffled and zero-feature controls.

At most10 sequential GPU runs/3840 updates and20 minutes total;4GiB allocator cap
and at least1GiB free. Initial software fixtures and preflight separate. Abort over
budget rather than alter source during execution. Each run uses existing training,
checkpoint/resume and report code, keeps exact initialization checksum, frozen-state
hashes, raw predictions/metrics, optimizer/sampler state and resource measurements.
Save new trained binaries; retain both failures and positive results. Review fixed
validation panels; HTML QA remains structural-only under the prior browser limit.

Essential RED checks: strict handwritten load and meaningful encoder/decoder freeze
behavior through a real update; identical-output/different-input collision; zeroed
branch gradients versus function-preserving opening; exact interrupted resume for
the initialized/frozen run. Commit plan/tests then implementation, pass appropriate
CPU checks, commit source before formal runs and independently audit raw evidence.

The user replied 'Ok' after the saved Claude brief permission question; retrying
that exact send was again rejected by automatic approval review as ambiguous consent
to that payload/destination. No external contact or bypass occurred. Local source
analysis, mathematical checks and tests proceed; Claude remains separately blocked.

## Implementation checks before formal training

Six RED cases confirmed missing setup/probes, then passed: exact binary initialization,
three freeze scopes with actual optimizer updates, equal-mean feature collision,
function-preserving branch opening and exact interrupted initialized/frozen training.
All203 CPU tests pass in175.34s. The decoder's zeroed convolution weights have no
gradient; its final bias can still learn a constant residual. Opening the first
convolution preserves the initial function and produces a nonzero second-convolution
weight gradient. The mask head remains byte-identical and frozen.

GPU preflight uses the actual handwritten file, RGB loss and opening control:
202MiB peak reserved, finite forward/backward, encoder and RGB decoder gradients
present, mask gradients absent. The checkerboard pair has exactly equal features at
every initial scale and patch-projection rank3. No weights or thresholds were changed
from the plan in response to real evaluation data. Formal runs start after source
commit; no shared model-library changes were needed.

## First comparison and bounded optimizer repair

Original sourcee1e89c7; ten runs completed in521.15s. Exact-handwritten joint test
RGB MSE0.273862/0.185825, decoder-only0.273862/0.201883, encoder-only0.013102/0.013112;
ordinary joint0.007319/0.007859. Opening the branch did not prevent collapse at the
old decoder rate. All trained mask weights stayed frozen. Independent90-metric audit
agrees within2.3e-16; original weights, frozen components and matched sampling verify.
The checkerboard and rank probes distinguish input ambiguity from optimization.

Validation logs, examined before the follow-up, show joint and decoder-only runs
saturating after early updates; encoder-only improves in both seeds. At7501, all
six inspected validation RGB outputs are below1e-6. This is an optimizer sensitivity
of the handwritten decoder; the experiment does not identify one normalization layer
as its unique cause. Do not use these collapsed weights as the training recommendation.
The original results/reports remain in `runs/hierarchy_training_v1/`.

Predeclared follow-up (also recorded before source edits in
`runs/hierarchy_training_v1/decoder_step_followup.txt`): expose decoder learning rate,
keep encoder rate0.0003 and all other settings fixed. Validate exact handwritten joint
training at decoder rates0.00003/0.000003/0.0000003, seed7590,32 updates each, two16-step
segments. All use512 training and128 validation images; no final-test scoring in these
paused trials. Eligible if final validation MSE does not exceed initial, all finite,
and post-initial validation values stay within1.2 times initial. Choose lowest final
validation MSE among eligible candidates, ties smaller rate; freeze selection JSON
before test access. If none is eligible, stop without an unplanned search.

Then two seeds7501/7502,384 updates each for hand_both, hand_decoder,
hand_open_decoder and ordinary_both, with selected decoder rate. Reuse the original
hand_encoder references: their only learning rate is unchanged0.0003. Retain the
original ordinary-rate controls too; separate optimizer repair from pure initialization
effects. Same10%/5% reconstruction gates. Additional10-minute GPU training/evaluation
cap including trials,4GiB allocator and1GiB headroom. Preserve exact source file.
No test-guided threshold or numerical revision. Targeted rate-group/freeze/resume
tests must pass and source must be committed before these trials.

The repair passes all15 focused recipe tests in9.36s, including separate optimizer
groups for each freeze scope and exact interrupted resume with the smaller decoder
rate. Ruff passes. The earlier203-test full suite remains the full-suite evidence;
this optimizer-only change has focused validation and no shared model changes.
