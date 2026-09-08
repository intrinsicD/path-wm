# Frozen semantic encoder: local evidence and task-conditioned dense decoding

Adaptive follow-up within the user's overnight authorization; albedo excluded.
This tests decoder access and conditional computation for present observations.
It does not train dynamics, memory, encoder conditioning or future-image skips.
The model consumes the same RGB64 COCO and PushT populations as P1. The pretrained
DINOv2-S/14 encoder and its preprocessing remain frozen and pinned. Final16x16
features use the already-audited FP16 cache; additional local inputs are FP32.

## Four arms and interfaces

Every decoder has three input slots, projected separately to128 channels, fused
at16x16, and upsampled through32x32 and64x64. Slot2 is final16x16 DINO tokens;
slot3 is their8x8 average pool, upsampled to16x16 after projection. This derived
grid is not an independent encoder hierarchy level. Slot1 differs:

* `late`: another copy of final16x16 tokens, with its own learned projection.
* `early`: patch-embedding tokens from the pinned DINO strided convolution before
  transformer blocks or positional additions, using the exact RGB preprocessing.
* `raw`: disjoint4x4 source RGB patches (48 channels) on the same16x16 layout.
* `conditioned`: the same early+final inputs, with RGB/mask task conditioning.

Before each learned slot projection, concatenate per-token normalized channels,
original channel mean and sqrt(population variance+1e-5). Those two statistics
make the normalization invertible before projection; a test checks recovery.
The late/early/conditioned slots have386 input channels; raw slot1 has50. The raw
alternative therefore has fewer parameters. Bicubic preprocessing gives the
early slot overlapping source-pixel support, unlike the raw disjoint patches.
This is an explicit package comparison, not a claim of exactly matched effective
capacity or receptive field. The duplicated-final control matches parameters,
but cannot supply new information.

The shared trunk uses convolution384->128 plus one residual block at16x16;
nearest-neighbor upsampling and convolutions128->64 at32x32 and64->32 at64x64,
with GroupNorm/GELU as in P1. Separate final1x1 RGB(3,sigmoid) and mask(1,logit)
layers implement typed outputs. Every arm has a per-resolution FiLM linear map
from one context scalar to channel-wise scale/bias; all maps start at zero so
modulation starts as identity. Context is0 for late/early/raw and -1 for RGB,
+1 for masks in conditioned. This isolates the task signal in an otherwise
identical early-versus-conditioned decoder, though constant-context weights are
inactive. Typed output layers already identify the task; the question is whether
earlier shared computation benefits from that identity.

## Training, endpoints and interpretation

Three paired seeds9107/9108/9109,4,000updates each,64images per update as32COCO+
32PushT; same sampler stream as P1 and extension comparisons. Pair all common
initial weights; input-width-dependent projection initialization uses an isolated
stream. AdamW3e-4, weight decay1e-4, clip the complete decoder gradient at1,
FP32 with TF32 disabled. No schedule, dropout, augmentation or encoder updates.
Loss is0.5COCO RGB MSE+0.5PushT RGB MSE+1COCO mask BCE over valid pixels.
Run separate RGB/mask trunk forwards in every arm's training step to keep the
training execution pattern matched. Shared parameters remain shared even when
conditioned outputs need different inference passes. Log RGB-versus-mask shared
gradient norms/cosine on early updates and every100; this diagnoses gradient
alignment, not causal performance loss.

Use the fixed4,000-update endpoint, not a validation-selected replacement.
Validation every100 is monitoring only. Evaluate full512-image COCO and2506-frame
PushT grouped test populations, training prefix512, and full validation at end;
preserve per-image RGB errors, masked BCE/IoU/Dice, constant/mean baselines,
fixed-index image panels, checkpoints, optimizer/RNG and source identities.
Primary endpoint is mean per-image COCO foreground IoU. RGB MSE in each domain,
compute/memory and per-seed tradeoffs are secondary. Report all three paired
seed differences; no statistical-significance or universal-success threshold is
predeclared. Prefer a Pareto tradeoff over inventing a composite score.

`early` versus `late` tests local input access. `conditioned` versus `early`
tests task-conditioned trunk computation. `raw` versus `early` compares a simpler
alternative including the disclosed preprocessing/width difference. Earlier P1
had separate full trunks and pose-selected checkpoints; its scores are context,
not a matched independent-trunk control for this study. Independent trunks,
slot deletion, intermediate transformer taps and more spatial levels remain
follow-ups. A negative gradient cosine alone does not establish harmful transfer.

If local inputs improve only RGB, retain that conclusion's narrow scope. If
mask IoU also improves, local evidence benefits this dense semantic output.
Conditioning is useful here only if its measured output/compute tradeoff warrants
it; do not extrapolate two task bits to arbitrary task or action conditioning.
Present-frame local skips cannot be supplied by future observations at rollout
time. Any imagined output needs predicted detail or causally available past state.

## Development and bounded execution

First verify informative failures for normalization recovery, raw-patch alignment,
paired decoder initialization and FiLM task influence. Run50-update development
prefixes for each arm, inspect finite gradients and no frozen-state mutation,
refresh and browser-verify the canonical dashboard, then commit and seal code,
data, configurations and seeds before formal training. Initially cap each fit at
20minutes,12fits<=4fitting hours plus evaluation/reporting; revise prospectively
from actual development timing if necessary. Reuse final-feature caches and
compute early/raw inputs on demand to avoid another multi-gigabyte cache.
One GPU workload at a time, after the active encoder extensions and their audits.
Stop new training by06:00Berlin; final report due07:00Berlin on9September.
Completed/stopped arms remain visible; an incomplete endpoint is not silently
compared as a completed4,000-update result.

Claude's public-only decoder review and reconciliation are retained under the
overnight collaboration directory. The implementing agent independently checks
claims against code, source papers and experimental evidence.
