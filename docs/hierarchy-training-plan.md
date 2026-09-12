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
zeroed SpatialResidual branch, initialize ONLY the first convolution weight with
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
