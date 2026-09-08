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
