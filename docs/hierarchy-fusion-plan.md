# Processed hierarchy plus final cross-scale fusion — 12 September 2026

User proposal: modality stem → k transformer blocks at each of m progressively
coarser scales → transformer stack over all completed scales. Keep each output
scale available. Video/audio recurrence is a separate question about state across
windows, not simply repeating computation on the current window.

## Implemented contract to test

Extend existing FeatureHierarchy with optional `fusion_depth` (default0). Concatenate
finished scale tokens, including existing scale/position encodings; process with
that many ConditionedBlocks; split values back into the original scales. Preserve
grid, validity, support-end ordinal, availability/content times and condition time.
Every block permits only keys with time and support end no later than the query's;
this preserves support bounds inductively. Invalid tokens and gradients stay zero.
There is no separate pooling bottleneck in final fusion. Global attention remains
bounded to small inputs. Original default state dictionaries and outputs must match.

Expose fusion in all three modality encoder constructors and add a small spatial
adapter for the existing perception recipe/heads. No new trainer or renderer. A
readable comparison recipe uses the existing training and reporting functions.
Keep original adjacent-scale cross-attention enabled in every comparison arm.

Essential RED checks: final fine outputs depend on coarser processed features;
all-scale attention uses only allowed keys and retains metadata; two fusion blocks
preserve video/audio/text prefix causality even with equal timestamps; masked NaNs
give finite outputs and zero invalid gradients; fusion0 preserves exact defaults;
spatial adapter and shared heads receive gradients. Verify exact pause/resume on
a tiny development fixture and run the full CPU suite before formal training.

## Frozen first comparison

This is a small real-image perception comparison, not a memory, speech or video
capability claim. COCO64 plus existing non-crowd foreground-union masks; first512
prepared training examples,128 validation examples,128 test examples. These internal
splits were used in historical work; disclose reuse, verify row disjointness, do
not describe them as a fresh independent benchmark. Masks are foreground coverage,
not instance identity. First6 validation examples supply report galleries.

Two paired seeds7401/7402. Width32, patch4, m=3; same DenseHead RGB and mask decoders
with all three scales and retained statistics. Ordinary initializations; copy every
shared state tensor from the shallow anchor into each arm, including identical
heads/stem/shared blocks. Additional blocks retain their own seeded initialization.
Same direct batch stream by seed, batch4; AdamW lr0.0003, weight decay0.0001,
gradient clip1, FP32. Loss RGB MSE + valid-pixel foreground BCE, with no search over
weights. Fixed final checkpoint scoring, validation curves every64 updates:

| Arm | k per scale | Final fusion blocks | Updates |
|---|---:|---:|---:|
| shallow | 1 | 0 | 384 |
| deep | 2 | 0 | 384 |
| deep_fusion | 2 | 2 | 384 |
| shallow_long | 1 | 0 | 768 |

Primary fusion comparison is deep_fusion versus deep. A promising result requires
test foreground IoU improvement at least0.02 absolute on BOTH seeds, RGB MSE no
more than5% worse on either seed, and resource limits met. Depth versus shallow
uses the same diagnostic criterion. Report all metrics regardless of gates.
These are prespecified small-screen thresholds, not deployment targets or statistical
confidence claims. Shallow_long controls additional training; its elapsed time is
measured, not assumed to equal deeper computation. No claim of equal total compute.
If nothing meets the screen, keep the original default and identify the unresolved
optimization/data/capacity questions rather than tune on these test examples.

Save raw test RGB predictions, mask logits, targets, valid masks and row IDs; score
RGB MSE, valid-frame mask BCE/IoU and zero/shuffled-feature controls. Include gray
reconstruction and empty/full mask references. Controls use the same trained heads;
they detect trivial priors, not exhaustive representation semantics. Preserve
checkpoint/source/data/initialization hashes, exposure, parameter counts, time and
GPU peak allocated/reserved values. Recompute selected metrics independently.

Budget: sequential local GPU runs, at most20 minutes total training/evaluation,
4GiB peak allocated/reserved per process with at least1GiB remaining device memory;
test/preflight separately bounded. Profile both training and validation batches
before formal runs; abort resource failures and retain them. No webcam recording,
downloads or hosted compute. Each completed run owns the standard report and raw
metrics; aggregate comparison may use a simple static table linking those reports.
Existing renderer unchanged; disclose structural-only report verification if the
browser remains unavailable.

## Recurrence after this comparison

The new fusion supports causal processing inside a window but carries no new state
between calls. Next compare a bounded temporal feature cache or recurrent state
against a reset control: two different first windows, identical second windows,
different required historical answers. Specify global time/ordinal offsets, episode
reset, cache size, encoder-version invalidation, overlapping-window duplicates and
truncated gradient boundaries before implementing that comparison. The agent already
has memory/state; an encoder cache must justify its additional role. Repeated
within-window refinement and cross-window persistence are separate ablations.

Claude critique is public conceptual only. Verify pooling maxima and stacked masks
locally; exact briefs/responses/receipts in runs/reviews/continuation_2026-09-11/
hierarchy-fusion*. Agreement is not empirical validation.

## Implementation review before formal training

Four informative RED failures confirmed missing fusion/adapter behavior. Targeted
causality, coarse-to-fine gradients and invalid-token checks pass after implementation.
Real RGB/mask backward succeeds. GPU preflight initially failed before model execution
because the allocator limit API required an explicit device index; fixed that call.
The completed4-example backward plus16-example validation preflight reserves202MiB.
Resume weights/optimizer/RNG matched exactly; its test initially expected the wrong
report-status label, corrected to the existing `structural_verified` contract.

Claude acknowledged that the time/support mask invariant composes inductively over
multiple fusion blocks. No bound recomputation is needed while every block respects
it. Segment recurrence has a public precedent in
[Transformer-XL](https://arxiv.org/abs/1901.02860); a temporal cache does not have to
compress history. That precedent does not establish video/audio memory here.

## Completed comparison

Implementationf4ada32, after plan/RED commitcf303b6. All194 CPU tests pass in171.37s.
Against the actual pre-change source, disabled fusion preserves state keys, initial
weights and outputs bit-for-bit for image/video/audio/text. Exact fused-model
pause/resume passes on a separate small fixture. No scope or scoring deviations.

Eight sequential GPU runs complete in589.13s total, including process/setup overhead.
All use the fixed final checkpoint and identical shared initialization per seed.
Each trained model has about1.74–1.80M parameters, with55–108k in its encoder.

| Arm | Mask IoU7401 | Mask IoU7402 | RGB MSE7401 | RGB MSE7402 |
|---|---:|---:|---:|---:|
| shallow | 0.186538 | 0.306008 | 0.010736 | 0.009733 |
| deep | 0.183979 | 0.304350 | 0.011048 | 0.009808 |
| deep_fusion | 0.257652 | 0.238005 | 0.010849 | 0.011635 |
| shallow_long | 0.264077 | 0.284291 | 0.006998 | 0.007139 |

Fusion versus deep: IoU+0.073673 and RGB MSE ratio0.98199 at7401, but IoU−0.066345
and RGB ratio1.18631 at7402. **Fails the prespecified both-seed screen.** Extra depth
also fails: mask IoU slightly declines on both seeds. Longer shallow training
improves RGB on both seeds; its mask effect is mixed. Do not turn these results
into a claim that depth or fusion can never help, or that more training always
improves segmentation. This setup is not a matched-elapsed-time comparison.

Absolute mask quality remains weak: the full-foreground reference reaches0.321622,
above every trained model at the fixed logit0 threshold; some zero-feature readouts
also outperform their observed-feature mask IoU. Meanwhile all observed RGB errors
beat gray0.071908, and shuffled/zero features worsen reconstruction. Thus the image
path learns useful reconstruction here without establishing satisfactory foreground
perception. Threshold calibration, training/data budget and objective allocation
remain possible issues; the experiment does not identify their causal roles.

Measured training/evaluation peaks are170–184MiB reserved, with at least4.65GiB free
afterward; the separate preflight peaked at202MiB. All resource gates pass. These
are small64px image-pair measurements, not full-agent/live-stream GPU claims.
Per-run wall times vary; resource receipts include validation, saving and reporting.

[Comparison report](../runs/hierarchy_fusion_v1/report.html),
[raw comparison](../runs/hierarchy_fusion_v1/comparison.json),
[independent verification](../runs/hierarchy_fusion_v1/verification.json), and
[GPU resume verification](../runs/hierarchy_fusion_v1/resume_check.json).
All72 observed/shuffled/zero RGB/BCE/IoU values recompute within2.3e-16; source,
checkpoint and output hashes and all report hash receipts pass. Completed-run GPU
resume makes no optimizer updates and preserves checkpoint/output files exactly.
Each run has raw arrays, learned weights, curves, reconstructed images and feature
maps. Report QA is structural-only; no browser verification is claimed.

Keep `fusion_depth=0` as the default and the proposed stack as an experimental
option. A possible later optimization test is a fusion residual initialized near
identity, so new mixing initially perturbs the trained features less. That is a
hypothesis, not an implemented fix or a result from this comparison.

Recurrence remains unimplemented in these encoders. The next minimal interface
would take `(current_window, previous_state)` and return `(features, next_state)`,
with bounded per-stream state and explicit reset. Measure historical-answer benefit
against reset state on matched two-window histories before adding more recurrent
iterations or claiming persistent audio/video understanding. The current module
already handles causal attention within a window; repeated processing alone does
not supply cross-window information.
