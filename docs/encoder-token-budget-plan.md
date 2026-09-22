# Remove dense fine-scale interactions before the state boundary

22 September 2026. Follow-up to [token-budget measurements](token-budget-plan.md).
User asks to pinpoint the remaining cause and patch for substantial efficiency.

The likely resolution-scaling bottleneck is `ScaleProcessor` dense fine-scale
self-attention, before observation resampling. `ScaleMerge` also builds dense
coarse×fine pooling/attention masks despite permitting only 2/4/8 source positions
per query. `FeaturePyramid.as_tokens()` alone is not quadratic computation: it
concatenates source keys for bounded reads. Do not delete detail exports to solve
an upstream attention problem. Current small-shape timings are substantially
overhead-bound; distinguish that from the resolution-scaling problem.

## Patch and controls

1. Physically pack pooling footprints and merge attention into small independent
   batches. Compare to the existing dense masked calculation with the same weights,
   validity, time and support boundaries. This preserves the allowed interactions,
   up to floating-point reduction order. Keep a selectable dense reference.
2. Add opt-in non-overlapping local windows on every scale except the coarsest.
   Images use1×4×4 feature positions; video uses2×4×4. Coarsest processing stays
   global. Retain every scale's complete values/metadata and all existing source
   reads. This changes receptive fields and is not a lossless functional replacement.
   Parameter shapes remain compatible; no shifted windows or new token taxonomy.
3. Count batch-weighted attention pairs and projected positions; packed windows
   become batches and must not disappear from cost accounting. Trace diagnostics
   may expand weights on CPU for compatibility; ordinary execution never expands
   a dense fine-scale window mask. Benchmark with tracing/profiling off.

Apply prepare-once/reuse, retained-versus-accessed detail, geometry before
discarding and actual execution costs. No source caching across weight changes,
lossy token dropping, state expansion or loop reduction. Persistent detail memory
and bounded coarse-state routing remain separate subsequent changes.

## Before-run checks, metrics and budgets

- Essential checks: dense-reference pooling/merge outputs and parameter/source/code
  gradients; ragged 1D/3D grids, masked NaNs and empty members; window attention
  versus dense same-window mask; every position restored; future-support exclusion;
  output/gradient/RNG-neutral traces; strict weight compatibility; correct window
  batch counting. Exact equality is not claimed across changed reduction kernels.
  Tolerances FP32 atol2e-6/rtol2e-5 for outputs, atol2e-5/rtol2e-4 gradients.
- Extend the existing token-budget recipe, fixed seed71, FP32, width32, batch1,
  three scales, four thinker loops, no resampler. Cases image32/64/128/256 and
  video4×64, plus existing small multimodal control. Three arms: dense reference,
  packed merges only, packed merges+fine windows. Match weights and inputs; two
  warmups/seven repetitions; all operator profiling after all timing. Report
  forward/backward and no-grad inference separately, memory and batch-weighted pairs.
  Ceiling6GiB PyTorch allocation, at most10 minutes GPU. Timing target >=20% lower
  forward+backward at image256 and >=90% fewer encoder attention pairs for the local
  architecture. No architecture-quality pass inferred from either resource screen.
- If mechanics pass, matched16-update synthetic development fits, full access versus
  local-window variant, same shared initialization/RNG/weights and recipe settings
  as the preceding comparison (image32/history2/horizon1;32 train8 validation;
  batch2; seed71; AdamW3e-4). Five minutes per arm,6GiB ceiling. This checks learning
  and reports regressions; it does not establish high-resolution detail retention.
- Save reports with the unchanged renderer and source/metric/checkpoint evidence.
  No default adoption of window restriction without trained task-quality evidence.
  Public-only actual-Claude review critiques these controls independently.

Actual-Claude review reconciled the distinction between dense masks and sparse
arithmetic: fused kernels need not materialize scores to benefit from physical
packing. Incorporated global positional metadata, ragged/padded identity controls,
seam limitations, backend reporting and layout overhead. Receipts are under
`runs/reviews/encoder_local_budget_v1/`; no peer agreement substitutes for results.
The initial NaN fixture exposed that the dense reference expects already-sanitized
encoder exports; compare its normal zero-invalid contract with packed NaN exclusion.
Window restrictions remain opt-in; global coarsest attention and optional all-scale
fusion still grow with source size, so this is not yet a resolution-independent encoder.

The first GPU comparison `runs/encoder_token_budget_v1` passes the image256 resource
screen but regresses small inputs: rebuilding geometry adds work. Before repeating,
add one disposable last-layout cache per stage/merge (grid/factors/device identity;
indices only, no source values/masks/times/weights). This is exact invariant reuse,
not adaptive token selection. Repeat the same declared populations/settings and
retain both runs. Quality fits use the final cached implementation in both arms.

## What the patch changes

`ScaleProcessor` previously lets every position at a scale attend across that
scale. At image256 with patch4 and three scales, these are 4096, 1024, 256 positions.
Only 256 coarsest positions now participate in global encoder self-attention;
the other 5120 positions are processed in physically separate 16-position windows.
This reduces global self-attention participation 21×, not stored feature count.
All 5376 multiscale positions remain exported and available as source keys.

`ScaleMerge` previously computed over the full coarse×fine rectangle and then
masked it to each pooling footprint. Packed merges actually read four spatial
keys per image query (up to eight for video), using the same permissible support.
The code no longer creates that rectangle on the packed ordinary path. Tracing
still deliberately expands diagnostic weights on CPU and is excluded from timing.

Analytical image256 encoder attention pairs (batch/head normalization fixed):
17,891,328 self + 4,456,448 merge = 22,347,776 dense; 147,456 self + 5,120 merge
= 152,576 local, or 99.317% fewer pairs. Every query position still gets its local
processing: batch-weighted query evaluations remain 6656. The change reduces
interaction density and the global active set, not the total retained positions
or linear-cost MLP work. Attention projections and matmuls are separately counted.

The next global bottleneck at still larger resolution is the coarsest level.
A bounded global reader must sit there, before global self-attention, with a
separate retained-detail path. Adding another resampler after the entire dense
encoder cannot recover computation already spent. This further change and a
persistent retrievable detail store remain separate, unvalidated work.

## Measured result

Implementation `a151089`, geometry reuse `60d0eaa`.
[Final cached-layout report](../runs/encoder_token_budget_cached_v1/report.html),
[raw calls, repetitions and operators](../runs/encoder_token_budget_cached_v1/workload.json).
The [uncached comparison](../runs/encoder_token_budget_v1/report.html) is retained.
Both use the same declared FP32/RTX3050 populations and fixed weights. Reports use
the unchanged renderer with structural verification, without new browser QA.

| Image256 arm | Forward ms | Backward ms | Inference ms | Peak allocated MiB | Encoder attention pairs |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dense reference | 51.617 | 143.157 | 50.207 | 452.530 | 22,347,776 |
| Packed merges | 46.198 | 123.820 | 44.657 | 432.843 | 17,896,448 |
| Packed merges + fine windows | 32.744 | 31.758 | 30.901 | 149.456 | 152,576 |

Both declared resource screens pass: forward+backward falls 66.88% (3.02× faster),
and encoder attention pairs fall 99.317% (146.47× fewer). Inference falls 38.45%;
whole-pass peak PyTorch allocated memory falls 66.97%. These are synthetic resource
measurements, not a 146× speedup or whole-process GPU budget. The FLOP estimate for
encoder attention matmuls/projections falls 2.931G→89.784M; it excludes MLPs, packing,
normalization, pooling and memory traffic. Kernel records distinguish CPU/device
events and show the actual efficient-SDPA backend.

The layout cache reduces the small-input regression but does not remove it.
The small four-modality control takes 49.066→50.182 ms inference and 96.064→99.389 ms
forward+backward. Thus no universal default acceleration is claimed. For high
resolution, use the local candidate; for current tiny inputs, retain the dense
reference until task-quality/resource comparisons justify changing it.

The most useful architectural patch is **local processing before global attention**,
plus physically local merges. Putting another bottleneck after this encoder misses
the dominant high-resolution work. Fine-detail preservation is a separate condition:
no positions are deleted, but window seams/receptive fields change, and learned
quality requires independent evidence. Global coarsest attention still needs a cap
for much larger resolutions; optional all-scale fusion restores dense interactions.

## Use the patch

```bash
python -m experiments.token_budget --device cuda --encoder-window 4 \
  --output runs/my_encoder_budget
python -m experiments.multimodal --dataset synthetic --state-model belief \
  --encoder-window 4 --packed-merges --width 32 --image-size 32 \
  --history 2 --horizon 1 --train-windows 32 --validation-windows 8 \
  --batch-size 2 --steps 16 --evaluate-every 8 --improve-every 0 \
  --learning-rate 0.0003 --ema-decay 0.99 --seed 71 --device cuda \
  --output runs/my_local_encoder
```

Or construct `MultiScaleImageEncoder(width, window_size=4, packed_merges=True)`;
`video=True` uses causal 2×4×4 windows. Audio/text support `packed_merges=True` with
their existing sequential attention unchanged. Default window 0/packed False retains
the original architecture and parameter/checkpoint layout. Window size is in
feature positions, not pixels. `Workload`'s new `batch_*` counters must be used to
compare packed windows; raw per-call Q/K sizes alone omit the number of windows.

## Learning and verification

[114 passing scoped tests](../runs/encoder_token_budget_cached_v1/verification.json)
include dense-reference outputs/gradients, local-window support, video causality,
NaN/empty handling, changed geometry/source validity after cache warmup, trace
neutrality, existing source/World State integration and exact local-model training
resume. Packed support is equivalent to numerical tolerance, not bitwise equality
across changed attention/reduction kernels. The dense public encoder path was not
silently replaced.

Both predeclared 16-update fits completed with report/checkpoint evidence:
[dense](../runs/encoder_token_quality_v1/dense/report.html),
[local](../runs/encoder_token_quality_v1/local/report.html),
[exact metrics](../runs/encoder_token_quality_v1/comparison.json).
Validation future image MSE is 0.201750 dense versus 0.201664 local, both far worse
than copy-frame 0.002920. This confirms the training path works; it does not establish
quality equivalence, preserved fine details or seam robustness. These need longer,
multi-seed task comparisons including seam-specific errors before default adoption.

Final cache-lifetime check also clears disposable geometry when a module migrates
device/dtype, preventing indices from retaining allocations on its old device.
Eight local tests pass after this repair; it does not change timed forward code.

## Randomized factorized attention: discussion, not implementation

Alex asks whether a randomized low-rank decomposition could avoid quadratic
attention and discard an unimportant residual. The relevant distinction is between
constructing factors directly from Q/K and factorizing an already expensive
attention operator. No new attention implementation or experiment is adopted here.

For unmasked attention, let `A = softmax(Q K^T / sqrt(d))`, row-normalized, and
`Y = A V`. A low-rank representation `A ≈ U W^T` permits `Y ≈ U (W^T V)` without
forming the n×n matrix. Generic randomized SVD first needs products such as
`A Omega`: computing exact softmax-attention products generally still requires
quadratic pair work, even if tiled to avoid quadratic storage. QK^T already has
rank at most d, but nonlinear softmax can produce a full-rank matrix.

- [Performer / FAVOR+](https://arxiv.org/abs/2009.14794) constructs positive random
  feature factors `F_Q, F_K` of shape n×r for the exponential dot-product kernel.
  Compute `S = F_K^T V`, `z = F_K^T 1`, then each output row is
  `F_Q[i] S / (F_Q[i] z)`. With suitable feature scaling and normalization this
  approximates softmax attention in O(n r (d + d_v)) work. Fixed r makes this linear
  in n; sufficient approximation rank is data-dependent. It is not a truncated SVD.
  [Author explanation](https://research.google/blog/rethinking-attention-with-performers/)
  also describes causal prefix summaries. The global n×n factorization statement
  applies to unmasked attention; causal masking uses a different running computation.
- [Nyströmformer](https://arxiv.org/abs/2102.03902) uses representative query/key
  landmarks and factors of size n×r, r×r and r×n, with a small pseudoinverse.
  Its factors are built without evaluating every pair. The original landmark
  construction is not necessarily randomized; this is closer to the proposed
  low-rank matrix approximation than ordinary post-hoc SVD.
- [Scatterbrain](https://arxiv.org/abs/2110.15343) combines random-feature low-rank
  approximation with sparse corrections for strong interactions. This motivates,
  but does not validate here, an exact local / approximate global comparison.
- [Randomized SVD](https://arxiv.org/abs/0909.4061) remains useful for offline
  spectrum diagnostics on small examples. [FlashAttention](https://arxiv.org/abs/2205.14135)
  avoids full score storage with exact tiled attention; it retains quadratic pair
  arithmetic. Our PyTorch path already requests no weights and used efficient SDPA
  in the recorded probe; do not claim a saved n×n allocation that was absent.

Small singular values are not the same as high spatial frequencies. SVD orders
matrix energy, not semantic relevance; a high-frequency pattern can be low-rank,
and independent sharp matches can require high rank. Compression error is therefore
not automatically noise. Retaining original feature positions protects access to
the sources, but does not guarantee unchanged reasoning or learned outputs.

Tentative next comparison: preserve packed fine windows and exact small-state
attention; test a factorized global path only where global source size justifies
it. Compare against exact fused attention with feature/landmark budgets 16/32/64,
including projection and feature-map costs. Keep numerical output/gradient error,
task/detail/seam quality, actual latency and peak memory separate. Causal support,
empty inputs, ragged geometry and normalization need explicit checks. Arbitrary
footprint or two-coordinate time/support masks cannot simply be attached after
factorization without potentially restoring quadratic work. Sparse corrections
need consistent normalization and avoidance of double counting; learned fusion of
separate local/global branches would instead be an architectural change.

Standing principles: retain fine evidence, approximate access deliberately, reuse
source summaries only while source/projection identities remain valid, and compare
actual execution and trained task behavior. This is a research option; no resource
or quality advantage over the existing patch has been measured.

## Current attention kernels: pre-integration review, 22 September 2026

Alex requests checking newer/faster alternatives before integrating FlashAttention.
This review changes no model code, dependencies or precision. Hardware inspection:
RTX 3050 8 GiB, SM86; PyTorch 2.9.0+cu128, CUDA 12.8, cuDNN 9.10.2.
The previous saved profiler contains `aten::_scaled_dot_product_efficient_attention`:
the present baseline already uses fused memory-efficient attention, not a naive
materialized score matrix. `ConditionedBlock` supplies an explicit validity/time/
support mask and asks for no attention weights.

| Candidate | Current evidence | Relevance here |
| --- | --- | --- |
| Native SDPA Flash / cuDNN / efficient | Installed PyTorch exposes all three; eligibility depends on dtype, shape and masks. | First comparison: no external package required. Current efficient path is the reference. |
| FlashAttention-2 | Official CUDA support includes Ampere, FP16/BF16 and backward. | Established Ampere candidate; avoid assuming the external package beats native SDPA. |
| FlashAttention-3 | Hopper-oriented implementation. | Its H100 results do not apply to this SM86 GPU. |
| FlashAttention-4 | March 2026 paper reports up to 1.3× cuDNN 9.13 and 2.7× Triton on B200 BF16. Current source includes SM8x forward/backward dispatch despite the README emphasizing Hopper/Blackwell. | Keep the Ampere source path as a candidate; package/build, custom-mask backward and actual RTX 3050 speed remain unverified. |
| FlexAttention | Compiled score/mask modifiers and block-sparse execution. March 2026 FA4 integration targets Hopper/Blackwell and required newer PyTorch. | Existing Triton-based Flex is a candidate for this model's custom support masks. Count mask construction, compilation, shape specialization and warm latency. |
| SageAttention 2/2++ | Quantized attention; current SM86 dispatch selects an INT8-QK/FP16-PV Triton path. Small heads are padded to at least 64. | Potential inference comparison, with approximation and backward/API limits checked separately. Our width32/four-head probe has head dimension 8, making padding/quantization overhead relevant. |
| SageAttention3 / new FP4 FA4 work | Blackwell FP4 hardware is central to the advertised gains. | Not a route to those gains on RTX 3050. Low-bit training has separate fidelity limits. |
| SpargeAttention2 | February 2026 paper reports 95% sparsity and 16.2× attention speedup on video diffusion with trained masking/distillation. | Changes accessible interactions and learning; not an exact kernel replacement or evidence for this world model. |

Primary sources: [FlashAttention repository](https://github.com/Dao-AILab/flash-attention),
[FA4 paper](https://arxiv.org/abs/2603.05451),
[FA4/Flex release](https://pytorch.org/blog/flexattention-flashattention-4-fast-and-flexible/),
[FlexAttention](https://pytorch.org/blog/flexattention/),
[SageAttention](https://github.com/thu-ml/SageAttention),
[FP4 FA4, 3 September](https://arxiv.org/abs/2609.04105),
[SpargeAttention2](https://arxiv.org/abs/2602.13515).
Upstream source snapshots are pinned in
`runs/attention_kernel_survey_v1/upstream-sources.json`: FlashAttention commit
`d15f1531a460ba456f41b01a774f33ab2db8febf`, SageAttention commit
`d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5`. In FA4 `interface.py`, architecture
family 8 dispatches to `FlashAttentionForwardSm80`; the heuristic is still labelled
for tuning. Do not equate available source with tested binary support or speed.

Local eligibility inspection (B1/H4/N256/D8, contiguous inputs, dropout 0):

| Input contract | Native Flash | Efficient | cuDNN |
| --- | --- | --- | --- |
| FP32, any of the tested masks | No | Yes | No |
| FP16/BF16, no mask or causal flag | Yes | Yes | Yes |
| FP16/BF16, explicit Boolean mask | No | Yes | Yes |

These are `can_use_*` predicates only: no new kernel forward/backward, timing or
quality run was performed. Raw inspection is
`runs/attention_kernel_survey_v1/eligibility.json`. An all-allowed explicit mask
still blocks native Flash here. Removing a mask is valid only when the actual
support contract is equivalent. Plain triangular causality is not automatically
equivalent to the current combined time/end/validity predicates. FP32-to-half is
a numerical change, even when the chosen attention algorithm is mathematically exact.

Proposed comparison order: native efficient versus eligible Flash/cuDNN under
matched precision; compiled Flex for custom masks; pinned FA4 Ampere as an external
challenger. Quantized/sparse approximations remain separate arms. Measure end-to-end
forward, backward and inference plus peak allocation, output/gradient agreement,
causality, ragged/empty inputs and learned detail quality. Include packed small
windows and long coarse reads: a kernel winner for long heads need not help tiny
ones. No fastest backend, performance gain or integration choice is established.

Standing principles: preserve exact support and retained evidence, distinguish
precision/access changes from implementation changes, reuse compatible preparation,
and measure actual execution rather than transfer hardware-specific headline gains.
