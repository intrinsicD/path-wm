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

Instrumentation-only baseline `runs/token_budget_before_v1` completed with identical
old complete/packet schedules. Its timing cleared the allocator each repeat; keep
that artifact as a cold-allocation development diagnostic, not the final comparison.
Before the comparison, move cache clearing before warmups, capture actual SDPA
operator names and inclusive region diagnostics in a separate profiler pass, and
retain latency ranges. Deterministic FP32 is the declared measured execution mode.
Peak memory is whole-pass; do not attribute a global peak or theoretical score
elements to materialized attention storage. Report resampler parameters separately.

Before the final comparison, move *all* operator profiling after *all* timing arms:
the first profiler-bearing run showed a timing discontinuity after profiler startup.
Keep `runs/token_budget_after_v1` as a development diagnostic, not the final estimate.

## Small trained comparison (declared before its execution)

Use the existing synthetic categorical recipe, final weights, seed 71, width 32,
image 32 square, audio 32, history 2, horizon 1, train 32 windows / validation 8,
batch 2, 16 updates, AdamW 3e-4, EMA .99, evaluation every 8, no extra-update gate.
Compare observation_tokens=0 versus 64, matching shared initial parameters, sampling
and RNG. GPU budget 5 minutes per arm, at most 6 GiB allocated. This is a development
fit/gradient check, not a high-resolution information-retention validation. Report
reconstruction and future NLL, image/audio errors and text CE, including initial
values. No quality pass or adoption threshold: 16 steps cannot establish task quality.
Full preservation still requires fine-detail source/retrieval tasks and longer fits.

Public-only actual-Claude review accepted the fixed-prior/canonical-order complete
event equivalence and scoped deterministic timing after withdrawing broader contrary
claims. Incorporated allocator warmup, backend diagnostics, raw dispersion, module
purity/RNG limits and separation of overhead from attention arithmetic. Receipts:
`runs/reviews/token_budget_v1/`. Peer agreement is not experimental validation.

## Results and adopted change

Implementation: `d63c1cb`. [Final GPU report](../runs/token_budget_final_v1/report.html),
[per-call shapes/raw repetitions/operators](../runs/token_budget_final_v1/workload.json),
[87 passing scoped tests](../runs/token_budget_final_v1/verification.json).
The report uses the unchanged renderer and has structural verification; no new
browser QA was performed. Earlier diagnostic reports are preserved separately.

Median milliseconds, deterministic FP32, RTX 3050, batch 1, four thinker loops:

| Input / arm | Forward | Backward | Peak PyTorch allocated MiB |
| --- | ---: | ---: | ---: |
| Four modalities, sequential packet reference | 105.422 | 43.230 | 71.672 |
| Four modalities, complete event once | 52.194 | 43.296 | 71.305 |
| Four modalities, complete + K=64 resampler | 52.359 | 44.335 | 71.046 |
| Image 128², complete event | 30.597 | 31.305 | 96.385 |
| Image 128², complete + K=64 resampler | 31.056 | 32.411 | 95.178 |

The exact complete-event scheduling change passes the predeclared 20% local
forward-time screen: **50.49% reduction**. Encoder calls fall 10→4, encoded positions
processed across calls 656→418, and total attention query evaluations 1296→868.
All original source positions remain accessible; no feature compression was added
to the default path. Backward time is effectively unchanged because the discarded
intermediate posteriors did not contribute to the original final loss. Single-image
runs have no redundant event union to eliminate and show no consistent speed gain.
The checks cover current pure deterministic components in train/eval modes;
custom stochastic/stateful components need their own equivalence contract.

The hypothesis needs narrowing: thinker computation is **already bounded** here.
At image128, encoder self-attention contributes 1,118,208 unweighted query-key
pairs; the Thinker makes four reads of 8 queries, independently of image size.
The source correction reads 1,345 keys without resampling and 64 with it. Despite
that reduction, total cross-attention pairs rise 337,784→372,688 after paying for
the resampler itself; global encoder work is unchanged. Partial projection/matmul
costs and physical latency are distinct. The GPU profiler observes the efficient
SDPA path, not a claim that every score matrix is materialized. Profiler rows include
CPU and device region annotations; do not sum duplicate region names as kernel work.
Flat ~30-ms single-input timings across these small shapes indicate substantial
dispatch/validation overhead; this does not extrapolate to large video or 4K inputs.

The optional resampler adds 10,656 parameters. It shows no useful runtime gain in
this screen, and **remains disabled by default**. Context stays 16 plus 4 working
and 4 reasoning slots; enlarging it to 128 would not be a compute reduction. The
separate configurable BeliefAgent constructor still supports larger capacities.
All loop counts, encoder scales and learned representations remain unchanged by
the default optimization. Retained-detail memory/retrieval and a local encoder
redesign have not been implemented by this slice.

### Trained development comparison

[Full access](../runs/token_budget_quality_v1/full_access/report.html),
[64 observation latents](../runs/token_budget_quality_v1/resampled_64/report.html),
[exact paired metrics](../runs/token_budget_quality_v1/comparison.json).
Both completed 16 updates with verified reports/checkpoints; shared initialization
and subsequent RNG match, and the resampled path passes exact training resume.

| Validation after 16 updates | Full access | K=64 |
| --- | ---: | ---: |
| Future image MSE | 0.201750 | 0.202793 |
| Future audio MSE | 0.005748 | 0.007156 |
| Future text CE | 4.789291 | 4.788357 |
| Combined objective | 12.114027 | 12.225771 |

Initial image MSE was 0.239127 / 0.239164; initial objective 16.285913 / 16.265985.
At the last training batch, weighted reconstruction image NLL was 2.275024 /
2.269353 and future image NLL 8.977478 / 8.987668. These are recipe-weighted training
terms, not independent quality scores. Both image predictions remain much worse
than copying the last observed frame (MSE 0.002920). Audio regresses in this short
comparison; similar image/text losses do not establish noninferiority. Fine-detail
retention/retrieval has no passing evidence. Peak training allocations were 85.07 /
85.29 MiB; these tiny-model values exclude the full proposed perception stack and
are not whole-process VRAM measurements.

### Reproduce

```bash
python -m experiments.token_budget --device cuda --resampler 64 \
  --output runs/my_token_budget
python -m experiments.multimodal --dataset synthetic --state-model belief \
  --observation-tokens 64 --width 32 --image-size 32 --audio-samples 32 \
  --history 2 --horizon 1 --train-windows 32 --validation-windows 8 \
  --batch-size 2 --steps 16 --evaluate-every 8 --improve-every 0 \
  --learning-rate 0.0003 --ema-decay 0.99 --seed 71 --device cuda \
  --output runs/my_resampled_development
```

Run the second command with `--observation-tokens 0` and another output path for
the matched full-access arm. To inspect shapes in other recipes, use
`with Workload(model) as work:` around an observation/think operation, then
`work.record_state(state)` and `work.summary()`. Keep it outside timed passes.

## Following phases and scope

The current world state is already bounded. Expose/configure larger budgets only
as controlled capacity experiments, without claiming 128 is inherently cheaper.
Next design persistent detail ownership, capacity, eviction and explicit retrieval
with fine-detail tasks ([World Labs ideas](worldlabs-review.md): posed-frame
KV retrieval, not adopted). Local/invertible encoder processing must be tested separately
from the resampler. Delta updates and adaptive compute follow fixed-budget quality
checks; they are not part of this first slice. PixelUnshuffle preserves values but
does not reduce their count; subsequent channel compression is explicitly lossy.

This applies prepare-once/reuse, retained-versus-accessed information, explicit
state lifetime, fixed budgets and actual execution costs. Exact scheduling and
learned compression remain separate comparisons. Public-only Claude criticism
will review the generic experimental logic, never private code or measurements.
