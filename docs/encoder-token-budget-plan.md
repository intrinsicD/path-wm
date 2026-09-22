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
