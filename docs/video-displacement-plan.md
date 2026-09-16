# Displacement coverage at fixed content and compute

16 September. Follow-up to [source diversity](video-diversity-plan.md).
Hypothesis: extending training displacement support repairs some transfer failures
without changing the image encoder, temporal convolution or direction head.
This tests a data-coverage policy under fixed compute, not an architecture theorem.

## Protocol before training

Two arms: NARROW {2,4}px and EXPANDED {2,4,6,8}px. Same16 training clips x4
frames from the previous BROAD population, identical48x48 crop, all48 cyclic phases,
paired reversed prefixes and opposite final-interval direction labels. Fixed dt.
Model/optimizer unchanged: mode=train, temporal7401/7402 crossed head7501/7502;
sampling seed7600;512 updates x8 pairs; AdamW0.003,wd0.0001,clip1,CPU FP32.
Eight fits total. Select image/phase first, displacement second, independently and
uniformly, so both arms see identical image/phase indices at every update. Expanded
has half the expected exposure per magnitude. Equal updates is the intended compute
constraint, not equal epochs or equal per-magnitude exposure. The new sampler means
NARROW is a fresh reference, not expected to match the previous run's logits.

Common evaluation groups: known{2,4},wide{6,8},intermediate{3,5,7},extrapolation{10,11}.
These names refer to expanded support; wide/intermediate are mostly out of range for
narrow. Max11 keeps the first interval2d below half the48px period (avoid d12's
ambiguous half-period first jump). Final direction remains the supervised target;
trajectories are deliberately constructed and reverse direction between intervals.
Do not infer natural dynamics, forecasting, speed estimation or streaming capability.
RGB alignment oracle searches both signs of every evaluated magnitude, exposes
ambiguity without filtering. Exact RGB and encoded single-frame class marginals.

Select4 fresh confirmation clips from local Charades metadata, sorted by SHA256 of
`pathwm-motion-displacement-v1:`+ID, length>=8s and existing raw file. Exclude ALL
subjects from the22 previous train/validation/evaluation/confirmation clips; one new
subject per selected clip. No score-based selection or replacements. Save manifest,
metadata/file/crop hashes. Historical validation remains monitoring only (its subject
is shared with training); historical evaluation and the previous4 confirmation clips
are development data, never promoted to fresh evidence. New confirmation has4frames
per clip. Keep per-source results; four clips do not establish population uncertainty.
No universal upstream novelty claim; prior recorded codec lineage is COCO-only.

Primary capability gate per arm: ALL four initializations and ALL four fresh groups
must have source-macro accuracy>=90%, pair accuracy>=80%, flip>=90%, each source
accuracy>=80%. Current/previous/unordered controls exactly50%; current/unordered pair
accuracy0; swapped-prefix logits must exactly swap paired outputs. Separate benefit
gate: expanded-minus-narrow mean>=3pp across eight intermediate/extrapolation cells,
no cell/group regression>2pp there and no known-group cell regression>2pp. Report
covered-wide benefit separately; improvement there alone is not transfer. No default
promotion unless both full capability and benefit pass. Per-magnitude margins,
confusion, confident errors and actual sampled displacement exposure remain visible.

Budget:8 formal fits,45s training cap each,<=360s total; only extra fit is8 versus4+4
mechanics. <=60MiB saved artifacts,>=300MiB free disk, no data copying/downloads.
Do not inspect confirmation model scores until all eight fits complete. No early
stopping, parameter search, continuation or conditional extra experiment. Failed
runs preserved. Evaluation/preparation/report cost recorded separately from fitting.

## Implementation plan

Extend existing experiments/video_order.py only: optional displacement specification,
validated group names/magnitudes; parameterized pixel oracle; optional matched-phase
sampler; displacement exposure logging. Keep original defaults and saved run snapshots.
Use the existing Run, checkpoint and report pipeline. No new model/training framework.
Test spec rejection, oracle sign/ambiguity at new magnitudes, matched image/phase
sampling, and exact restart. Run affected video/image/modality regressions. Independently
recompute metrics from saved logits, gate arithmetic, source hashes and exposure.
Claude reviews public hypothetical methodology only; receipts in
runs/reviews/video_displacement_v1. Reconcile substantive criticism before formal fits.

Review before implementation: Claude correctly flags per-magnitude exposure and the
single-sampler limitation; both are part of the fixed-budget estimand and reported.
Its factor4 exposure assertion is arithmetically wrong:2 versus4 choices means2x.
The protocol uses10/11 rather than the briefing's10/12, so2d<=22<24 throughout.
The oracle measures the final interval d, not the first2d interval. No extra-budget
control is added: this experiment cannot identify compute-unconstrained capacity or
population-wide robustness, and does not claim it. Corrections sent for reconciliation.
Essential new tests initially fail on missing spec/sampler and oracle argument, as expected.

## Working slice

Fresh clips fixed before fitting:34DKM(G6WD),X1EZQ(ZAWX),N588B(WQ8Z),AHL6X(C7O9).
Saved data/motion_displacement_v1/sources.json and narrow/expanded.json; no files copied.
Claude acknowledged the factor2 correction, numeric alias bound and fixed-compute
estimand. Single-sampler/finite-source limits remain; no broader claim authorized.
77 scoped tests pass. Real-data8 versus4+4 restart passes3397 exact recursive checks;
confirmation predictions excluded from mechanics. Model/optimizer/RNG, matched sampler,
raw logits and displacement exposure replay exactly (timing/extra validation excluded).
Initial script launch failed before preparation because PYTHONPATH was absent; log
preserved, invocation fixed. No scientific result or fit was replaced.
Frozen image encoding reuses the phase bank across evaluation displacement groups.
