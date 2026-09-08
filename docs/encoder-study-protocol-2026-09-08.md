# Encoder study: authorized execution

8 September 2026. User accepted the co-designed proposal and recurring critical
Claude collaboration. Historical statements that further encoder work was not
authorized are superseded by this decision. Preserve all completed references.

## Slice A: bounded diagnostic and pretrained comparator

Question: can readout optimization or a different frozen representation improve
accessible geometry before changing the custom encoder? Reuse CCHI grouped
splits (20,493 / 2,651 / 2,506 frames; 164 / 20 / 22 configuration groups) and the
selected task-only A encoder (`5be1c851…`). All test populations have been
inspected previously: these are exploratory comparisons, not untouched tests.

- Fit coupled and independent spatial pose readouts with frozen cached A features.
  The independent readout predicts two expected XY locations and a separate
  orientation vector; it does not calculate orientation by subtracting the
  object-origin point. This tests a readout package, not an isolated causal claim
  about gradient coupling. Record both 2,000- and 6,000-update selection windows.
  Seed 6107, batch 128, AdamW lr 0.0003 / weight decay 0.0001 / clip 1. Fixed
  2,048 validation rows sampled with seed 6107; validate every 100 updates. Select
  minimum q, then normalized pose MSE, then earliest update. Wall cap 1,800 seconds
  per head. Numeric readiness remains q=max(position MAEs/8, angle MAE/10)≤1.
- Frozen DINOv2-S/14 normalized patch features: same RGB64 source, bicubic resize
  to 224 with antialiasing, no extra crop, ImageNet mean/std. This preserves the
  source field of view; it is deliberately not the classification 256→224 crop.
  Pin upstream source and weights, evaluate deterministically, and profile before
  extraction. Cache FP16 only after measuring round-trip error against FP32.
  A trained linear 384→64 projection plus 2×2 coarse pooling returns the legacy
  latent contract. Fit adapter + original D/H with pixel MSE + normalized pose
  MSE, seed 6107, 4,000 updates, batch 128, original optimizer and fixed validation.
  This comparator does not isolate architecture from large-scale pretraining.
  If it fails q≤1, a single native-width flattened linear pose head at 6,000
  updates checks the projection/readout bottleneck; failure still does not rule
  out encoder, resolution or optimization explanations.
- Audit frozen representations with fresh RGB and binary foreground-mask decoders
  on 4,096 / 512 / 512 fixed COCO frames, grouped splits preserved. Official
  instance masks follow exactly the prepared resize/crop; ignore crowd pixels.
  The union mask tests annotated foreground coverage, not separate instances or
  instance extent. Report empty-mask conventions and constant-mask baselines.
  Fit identical decoders and draws, 2,000 updates, batch 64, select validation
  RGB MSE + mask BCE, report both metrics separately and IoU/Dice. The task-fitted
  DINO adapter stays frozen during this audit. COCO pretraining overlap with DINO
  is not excluded. This small battery does not demonstrate a general agent.

## Slice B: prospective four-arm comparison

Separate experiment variant; legacy Encoder and checkpoint semantics unchanged.
Factor residual depth {0,2 blocks/branch} by exchange {on,off}. Branch blocks go
post fine projection and post third convolution, before positions. Coarse stem
still reads the original mid-level activations. Blocks are preactivation affine
GN(8)→GELU→3×3 conv(no bias) twice, identity residual, width 64. Preserve original
per-token MLPs when exchange is off. Shared initial tensors and paired draws must
match exactly. Inactive attention parameters are explicitly excluded from active
parameter counts. Added depth includes capacity and normalization changes.

After tiny development profiling, freeze exact configuration before formal runs.
Proposed: seeds 7107/7108/7109, all four arms per seed, 4,000 updates, batch128,
AdamW lr0.0003 / weight decay0.0001 / clip1, microbatch128 unless common memory
profile requires reduction. Original D/H, image MSE + pose MSE, fixed 2,048
validation frames per seed, evaluate every100; select q then RGB MSE then earliest.
An at-least10% q improvement in all three paired seeds is a candidate signal;
q≤1 remains the existing numeric readiness gate. Report signed depth/exchange
contrasts at both depths, interaction, per-seed values and group uncertainty.
Three seeds do not support confident population-wide significance claims.

Equal updates/presentations are not equal compute: record elapsed curves, peak
memory, active parameters and inference latency. Compare complete systems only
with compatible, freshly trained U/P/readouts and existing stage gates. Failed
perception gates stop the corresponding control progression, not other arms.
A geometry improvement alone does not establish a generally better encoder;
include the common RGB/mask audit before general representation claims.

## Interfaces and essential tests

`encoder_variants.py`: explicit variant with fixed ObservationLatent output and
legacy tensor names. `pose_diagnostic.py`: cached frozen readouts and two budget
windows. `dino_reference.py`: pinned frozen backbone, preprocessing and adapter.
Separate probe data/decoder and experiment runner modules reuse the existing
ledger and dashboard, never silently loading different E weights into old U/P.

Test baseline numerical equivalence, paired initialization/draws, exchange bypass
with MLPs retained, branch locality, independent orientation gradients, preprocessing
field-of-view/normalization, adapter grid ordering, frozen backbone state, and
resume identity where resumable training is used. Verify transformed mask alignment
and ignore semantics against a small independent example. Record an informative
red test run before implementation and a tiny measured development result before
formal training. Refresh and browser-verify HTML after each completed run/evaluation.

## Scope after A/B

Additional levels, registers, higher input resolution, fusion alternatives and
software persistence remain staged follow-ups selected from the evidence. Do not
blindly execute an architecture sweep. Preserve causal future decoding and the
separation between observation outputs, actions and temporal state.

## Development profile and frozen execution details

Development artifacts: `runs/encoder_study_2026-09-08/development/`. Tiny independent
head completed 3 updates; each custom E completed 25 profiling updates. Batch128
median update times were 0.05356 / 0.06930 / 0.04212 / 0.05787 seconds for
reference / deeper / no-exchange / deeper-no-exchange. Peak allocated memory was
1.39 / 1.52 / 1.33 / 1.45 GB. DINO batch32 took 0.11757 seconds; FP16 storage
round-trip relative MSE was 4.31e-8 (<pre-use 1e-6 gate). These are development
measurements on RTX3050 8GB, not formal training or final latency claims.

**Freeze B before formal training:** all 12 runs use seeds7107/7108/7109,
4,000 updates, batch/microbatch128, the proposed optimizer and validation settings,
wall cap1,200 seconds each including validation/checkpoint time. No early quality
stopping. Same seed uses the same phase sampler and fixed2048validation indices.
Keep selected and final checkpoints; wall-matched comparisons use nearest
validation step whose elapsed time does not exceed its paired shallow run's
final elapsed time. If unavailable, report unavailable. Normal convolution
initialization (not zero-init residuals) is part of the package.

A2 fits fresh D/H on both frozen custom A and frozen DINO+trainable adapter,
4,000 updates with wallcap1,200seconds. The adapter gives DINO additional trainable
capacity; this is a comparator, not an isolated backbone experiment. Keep the
frozen COCO-warmup encoder as a third RGB/mask retention reference. Per-image mask
metrics include foreground-fraction strata and mean-mask/empty/full baselines.

Claude's execution review and reconciliation are preserved under
`runs/encoder_study_2026-09-08/collaboration/`. Accepted corrections: an averaged
linear token probe cannot establish information at each token; flip correspondence
is not a guaranteed backbone unit invariant; identical mask/RGB geometry does
not imply identical interpolation; three seeds cannot retrospectively establish
an MDE; a from-pixels model is not a guaranteed resolution ceiling. No extra
probe/CNN/register arms are added. Results must state that B's exchange comparison
under this perception recipe does not establish its effect on imagination or
planning if compatible downstream training is blocked. Peer preferences about a
custom identity adapter and RGB selection remain documented, not silently adopted.

The development dashboard initially failed on an oversized metadata field (a
group list instead of its digest). The original manifest and explicit reporting
correction are retained; raw metrics were unchanged. Canonical HTML now passes
package, validation and browser verification.

## Figure contract

Scientific figures use Matplotlib with neutral backgrounds and blue/orange plus
neutral reference colors, explicit line styles and units. Validation trajectories
show every recorded evaluation; q=1 is a neutral dashed reference. Final comparisons
show per-seed values and paired changes, not pooled incompatible latent losses.
Fixed image panels display target/predicted geometry and masks with shared scales.
Native HTML charts and exact tables remain backed by the canonical raw ledger.
Development three-update traces are labeled development, not evidence of learning
shape; formal trajectories have41 or61 evaluation points. Inspect exported figures
and verify the actual dashboard at laptop/mobile widths before handoff.

A1 completed both6,000-update runs. Its coupled2,000-update trajectory differs
from the historical run despite identical initialization, validation rows and all
training draws: step1 loss/gradient match, step2 gradients differ at about1e-6
and subsequently diverge. GPU deterministic algorithms were not enabled in either
run. Treat the new2,000→6,000 window as the within-run budget comparison; do not
combine it with the historical run as an exact continuation. This is numerical
trajectory sensitivity, not evidence of a changed sample stream or seed variance.

DINO tiny extraction initially rejected17 near-zero values out of294,912 checked
values when comparing inference batches of different sizes. Investigation found
relative MSE4.28e-8 and maximum difference0.00762, consistent with FP16 storage;
near-zero absolute difference reached0.000138. Before full extraction, the
near-zero absolute tolerance was set to0.0003 (relative coordinate tolerance0.0006),
while the predeclared relative-MSE≤1e-6 gate remained unchanged. Failed development
artifacts and the diagnostic are retained. Revised tiny extraction, all three
reference readout paths and canonical browser verification pass.

DINO uses the official source snapshot at commit
`7764ea0f912e53c92e82eb78a2a1631e92725fc8` (Apache-2.0), with every file hashed.
The cached source had a changed hub helper, so a pinned upstream snapshot is used.
Cached weights exactly match the88,283,115 official download bytes, SHA256
`b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9`.
Neither the user cache nor the original checkpoints were modified.
