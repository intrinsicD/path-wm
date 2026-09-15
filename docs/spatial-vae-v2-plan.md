# R/P/M/C VAE implementation and sanity protocol

15 September 2026. Alex authorized implementation with Claude after the
[specification refinement](spatial-vae-refinement.md). Preserve v1 exports and
the agent; add a versioned codec using the existing recipe, Run and renderer.

## Plan and fixed acceptance criteria

1. RED contracts: geometry, active gradients, exact rearrangement and stride2
   equivalence, shared attention/positions/token cap, legacy export, detached probes.
2. Implement the new stem/local blocks, configurable stages, convolutional decoder,
   coarsest attention loop and opt-in detached instrumentation. Reuse posterior,
   TransformerBlock, objective, RidgeReader and strict Run resume.
3. Real-photo development and exact CPU/GPU resume; then commit working slice.
4. Train A_local and C sanity controls first. Run B/C_after/D/E and three beta
   values only after mechanics pass. Record all outcomes, including failures.

Small configuration: stem8, stages[16,24], latent4, one post block, no pre blocks;
decoder one pre/post local block per stage for every arm. Residual final1x1 starts
at zero. C processes at widths32/64 before compression; C_after at16/24 after it.
D uses one pre-compression Transformer64, four heads; E reuses it twice. Same-width
untied depth2 is required before later sharing-specific claims; it is deferred in this first screen. Token cap1024;
convolutional variants have no attention cap. Batch8, float32 deterministic math,
AdamW3e-4, weight_decay1e-4, clip1, seed57101. All common tensors initialized from
the same reference. No copied historical trained weights.

Populations: use the existing globally group-disjoint COCO splits. Seed56001
selects the prior ordered population; reserve its first1040 training,144 validation
and192 test groups from this experiment. Select the next512 train,64 validation,
96 test groups (plus16 train/16 val development groups). Record row/hash lists.
No test groups choose configuration, stopping, probe capacity or beta.

Sanity fits:512 updates per arm, beta1, Gaussian variance0.5. Acceptance: finite
objective/rate, completed updates, validation raw mean MSE improves >=20% from
initialization and beats the train-mean image by >=10%; positive KL and >=1 active
channel. Quality is a distinct screen: mean AND sampled MSE<=0.01; failure stays
visible. Native crops96x128 and65x79 (16 heldout images each) and32 fine-pattern
controls test geometry/detail; no resolution-quality superiority inferred.

If sanity mechanics succeed: six arms A_local/B/C/C_after/D/E at beta1; additionally
A_local/C at beta0.1 and0.01, always512 updates and same initialization/data/noise.
This is ten total formal fits. Three fixed beta points per A_local/C are descriptive
rate-distortion points. Equal beta is NOT equal rate. Report parameters, approximate
forward MACs (multiply-add=one MAC, FLOPs=2MAC; exclude normalization/activations),
measured latency, training time, allocated/reserved GPU peak and sample counts.
No claim of superiority or sample efficiency from this one-seed screen. A future
claim needs rate overlap, matched compute/parameters and seed replication.

Per fit:<=120s training, <=2GiB GPU reserve, >=1GiB GPU free; experiment total
<=20min and<=900MiB new artifacts, retain >=3GiB free disk. Do not delete old runs.
Pause/report when a cap fails; do not silently reduce resolution/width.

Frozen probes:64 train images x4 spatial cells=256 training points per stage,
16 validation and16 test images x4 cells. Fit linear and RBF ridge1e-3, RBF
bandwidth1.0; fixed choices, no tuning. RGB target is the corresponding nonoverlapping
original pixel patch, same before/after each C (not a different feature target).
Report after->before feature MSE / train target variance, raw MSE/variance, train-mean,
shuffled test inputs, and identity control; epsilon1e-8, variance<=epsilon degenerate.
Common RGB readouts before/after also have identical samples/targets. Heldout readout
error indicates accessibility to these readers, never certified information loss.

Save source/mean/sample/common-scale absolute RGB-error panels, raw per-image
metrics, stage geometry/probe results, rate plot and standalone verified HTML.
Existing renderer only: structural QA plus PNG inspection; prior browser local-file
policy restriction remains disclosed. Actual Claude reviews receive public conceptual
briefs only; preserve receipts and reconcile mistakes independently.

## Development check before formal fits

Core CPU contracts and old exports pass. Eight real-photo C updates and the complete
photo/native/probe reporting path pass on reserved development images. E GPU8 versus
4+4 resume is exact across367 tensors, optimizer/sampler/CPU/CUDA RNG and loss rows
(`runs/spatial_vae_v2/gpu_resume_check.json`). Reports use the unchanged renderer;
its labelled original/mean/sample/absolute-RGB-error panel was visually inspected.
These are mechanics checks, not image quality evidence. Two actual Claude conceptual
rounds are reconciled in `runs/reviews/spatial_vae_v2/reconciliation.md`.

Parameter matching limitation: stem/posterior/decoder widths and common weights stay
fixed; removing pre-compression processing lowers parameters/compute. The small
A_local/C sanity contrast therefore is not a matched-resource superiority test.
The full budget accounting is mandatory, and this limitation applies to its curves.

## Results: 15 September 2026

All ten fixed512-update real-photo fits complete and pass the predeclared learning
sanity gates. None passes the separate photo-quality screen (both mean and sampled
MSE<=0.01). No retries, early-stopping selection or test-driven parameter changes.
[Comparison report](../runs/spatial_vae_v2/formal/report.html).

| Variant | beta | Mean RGB MSE | Sampled MSE | KL bits/pixel | Parameters | Forward MMAC/image |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A_local | 1 | 0.022301 | 0.028224 | 0.033966 | 68,739 | 36.54 |
| C | 1 | 0.021827 | 0.027747 | 0.035284 | 122,979 | 58.47 |
| B | 1 | 0.023518 | 0.028986 | 0.032075 | 66,179 | 35.40 |
| C_after | 1 | 0.021802 | 0.027682 | 0.035263 | 75,531 | 39.90 |
| D | 1 | 0.021298 | 0.027398 | 0.036213 | 156,963 | 75.26 |
| E | 1 | 0.021411 | 0.027496 | 0.036493 | 156,963 | 92.04 |
| A_local | 0.1 | 0.017314 | 0.018594 | 0.147385 | 68,739 | 36.54 |
| C | 0.1 | 0.017132 | 0.018344 | 0.140092 | 122,979 | 58.47 |
| A_local | 0.01 | 0.017149 | 0.017643 | 0.525267 | 68,739 | 36.54 |
| C | 0.01 | 0.017927 | 0.018209 | 0.439473 | 122,979 | 58.47 |

A_local: ordinary overlapping3x3/stride2; B: rearrange/compress immediately; C:
rearrange/process/compress; C_after: process after compression; D: C plus one
coarsest self-attention step; E: same attention weights called twice. A_exact/B
weight equivalence is checked numerically, not presented as two independent fits.

Interpretation: reducing beta1 to0.1 improves C's sampled error by about34% while
allowing about4x the KL rate proxy. Reducing it further to0.01 gives little additional
sampled improvement and worsens its mean reconstruction at this update budget.
More rate alone therefore does not ensure better short-run reconstruction. C and
C_after are essentially tied at beta1, although C costs more. D slightly improves
on C; E does not improve on D. These descriptive one-seed, unequal-resource/rate
comparisons do not establish that pre-compression processing or shared attention
is superior. Three points are not evidence of a well-optimized rate-distortion frontier.

Frozen probes in C/beta0.1 show common-patch linear RGB MSE increasing from
0.000932 to0.003138 across stage0 compression and0.000812 to0.004357 across stage1.
These use the same64 heldout image/cell targets on each side, with256 training
points, fixed regularization, group-heldout images and train-only statistics.
Feature reconstruction errors normalized by train variance are much smaller
(0.00994 and0.00484): preserving most feature variance does not ensure preserving
all useful RGB details. RBF checks are also recorded; finite-reader failure does
not prove the information is gone. Scores across stages are not additive or directly
comparable information quantities. No common semantic target was evaluated.

Native96x128 and odd65x79 crops decode at their exact original dimensions without
model resizing. Visual review still shows blurred edges, missing text/texture and
color errors. At RGB64 the sampled posterior is visibly noisier than the mean.
Current quality does not justify promoting this codec into the agent or promising
general image generation. Decoder receives only z and output dimensions.

Verification:42 focused CPU tests (14 new,17 previous codec,8 Run,3 ridge/detail);
367 exact GPU resume tensor checks and2203 independent artifact checks, including
source/split/weight hashes, NumPy image/KL units, exact export replay and probe
budgets.58 standalone reports including development and native/pattern conditions
are structurally verified; labelled scientific PNGs inspected. Plot review caught
overlapping beta labels; the report-only fix moves the labels without changing
weights/results. Browser interaction QA remains unavailable due to local-file policy.

Ten formal training loops total70.75s, maximum reported training reserved memory
162MiB; reserved includes caching from prior arms and is not a fresh-process
per-architecture comparison. Peak allocated memory is also retained. The complete
new artifact set is approximately388MiB, under the900MiB cap and with >3GiB free
disk retained. Raw per-image arrays, stage-probe JSON, source snapshots and checkpoints
are preserved in each run directory. The verification receipt checks original source
snapshots after the report-only recipe change.

## Use and remaining scope

```bash
# Short real-photo development path, or --overfit for eight deterministic photos:
.venv/bin/python -m experiments.spatial_vae --variant C --development --device cuda --output runs/my_vae_dev
# A single prescribed hierarchy fit; --resume restores matching optimizer/RNG/data:
.venv/bin/python -m experiments.spatial_vae --variant E --device cuda --output runs/my_vae_E
# Complete fixed sanity/ablation/beta study in a new output directory:
.venv/bin/python -m experiments.spatial_vae --hierarchy-study --device cuda --output runs/my_vae_study
# Evaluate a saved hierarchy independently:
.venv/bin/python -m experiments.spatial_vae --variant E --evaluate-only --weights runs/my_vae_E/weights.pt --device cuda --output runs/my_vae_eval
```

Use the matching --variant for evaluation, and keep overrides visible in the recipe.
`HierarchicalVAE` exposes stem/stage/latent widths, pre/post depths, Identity stages,
shared-loop iterations (including0), heads and token cap. `inspect(x)` returns a
posterior plus detached named stage/loop/posterior tensors; use only bounded batches.
`SpatialVAE.load` dispatches strict v1/v2 schemas; old exports remain usable.

Next controlled question: training duration at a fixed beta/rate budget, then
posterior latent capacity and matched-resource processing-placement controls.
Untied depth controls are required before claims about weight sharing. Larger-input
quality, adaptive spatial budgets, semantic retention, state-to-latent generation
and integration with other modalities are still outside this completed first slice.
