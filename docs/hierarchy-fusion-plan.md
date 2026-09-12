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
