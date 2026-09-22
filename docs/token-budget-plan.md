# Active token budget: measured implementation

22 September 2026. User direction: preserve detailed representations separately
from bounded observation and reasoning tokens; keep shared-depth loops. This is
the active implementation plan, subordinate to the integrated latent-agent goal.

## Audit before intervention

The categorical recipe currently has 16 recurrent world slots, 4 working slots,
4 reasoning slots and 8 retained evidence slots. Thinker iterations update only
the eight working/reasoning queries. Source positions are cross-attention keys,
not appended to persistent state. Thus the proposed 64/128 budgets would increase
some existing budgets; they are experimental settings, not automatic savings.

The multiscale encoder does dense masked attention within each scale, including
across video positions, and dense masked cross-attention at merges. Masks do not
make these kernels sparse. Complete dictionary observations also currently rerun
correction for every arriving modality, repeatedly encoding their growing union.
Incremental packet arrival needs those intermediate posteriors; a complete event
does not. Measure encoder, correction, memory and thinker separately.

## First complete slice

1. Add opt-in, shape-only attention/encoder instrumentation. Count actual calls,
   allocated query/key lengths (including padding), encoded scale lengths,
   observation lengths and persistent state. Report sum Q, self-attention sum N²,
   cross-attention sum QK, batch/head-weighted score elements and an explicitly
   partial attention-matmul FLOP estimate. Counts are not measured kernel FLOPs.
   Repeated calls are recorded individually; do not label every repeat a thinker loop.
2. Benchmark the unchanged complete-event route and its sequential packet reference.
   Keep profiling hooks out of timed forward/backward passes. Synchronize CUDA and
   record raw repetitions, warmups, peak allocated/reserved memory and device context.
3. Remove redundant intermediate correction for complete dictionary events only.
   Preserve incremental packet semantics, masks, provenance, posterior sampling,
   parameter gradients, source identities and memory commits. No learned information
   bottleneck or fewer loops is required for this exact scheduling optimization.
4. Add an opt-in learned query resampler before belief correction/evidence reads,
   K_observation=64. Keep the default full-access path and existing state size.
   Compare its cost separately; it cannot reduce encoder self-attention and might
   increase cost when current query budgets are already small. Do not adopt it on
   timing alone or call its compression lossless. Existing encoder pyramids remain
   unchanged; persistent detail retention/retrieval is a later implementation slice.

## Declared checks and budgets (before execution)

- Essential CPU tests: independent rectangular/self attention counts, repeated-call
  accounting, masked allocation counts, hook cleanup and output/gradient/RNG
  neutrality; complete-event versus packet-reference outputs/gradients/metadata;
  one encoder invocation per modality; resampler fixed shape, invalid-key isolation,
  all-empty batches, source/query gradients and unchanged state cardinality.
- Development profiling only, seeded synthetic inputs, seed 71, FP32, batch 1,
  width 32, 3 scales, depth 1, context 16, evidence 8, four thinker iterations.
  Cases: image 32/64/128 square; video 4 frames at 32 square; four-modality event
  (image/video 32 square, 4 video frames, audio 32 samples, text 16 positions).
  Two warmups and seven measured repetitions per arm. Maximum 10 minutes GPU time,
  peak PyTorch allocation ceiling 6 GiB; no downloaded data or trained weights.
  Run CPU smoke separately when needed and record different settings explicitly.
- Timing screen: median forward reduction >=20% on the four-modality complete
  event with unchanged outputs (atol 1e-6, rtol 1e-5) and gradients (atol 2e-6,
  rtol 2e-5). This is a local scheduling screen, not a quality/generalization gate.
  Single-modality timing and backward time are reported even without improvement.
- Untrained resampler runs establish costs and gradients only. Reconstruction,
  prediction and task-level fine-detail retention need matched training and held-out
  evidence before activation. No quality threshold is assigned after observing data.
- Existing reusable report renderer; save raw call records, metrics, source identity,
  initialization checkpoint and report. Mark result and report status separately.

## Following phases and scope

The current world state is already bounded. Expose/configure larger budgets only
as controlled capacity experiments, without claiming 128 is inherently cheaper.
Next design persistent detail ownership, capacity, eviction and explicit retrieval
with fine-detail tasks. Local/invertible encoder processing must be tested separately
from the resampler. Delta updates and adaptive compute follow fixed-budget quality
checks; they are not part of this first slice. PixelUnshuffle preserves values but
does not reduce their count; subsequent channel compression is explicitly lossy.

This applies prepare-once/reuse, retained-versus-accessed information, explicit
state lifetime, fixed budgets and actual execution costs. Exact scheduling and
learned compression remain separate comparisons. Public-only Claude criticism
will review the generic experimental logic, never private code or measurements.
