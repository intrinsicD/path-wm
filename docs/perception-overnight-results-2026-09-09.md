# Encoder and decoder experiment results — 9 September 2026

**Recommendation: develop an asymmetric perception package with pretrained
contextual features, an available local image path, and output-specific decoders.**
Keep the compact CNN as the speed reference. Extra per-scale depth and RGB/mask
task FiLM did not provide a consistent improvement under the tested recipes.
Geometry coverage and extreme errors remain the immediate obstacle to reliable
world-model state estimation.

All **36 vision/geometry fits and 15 category probes completed**, with three
paired seeds per comparison: 174,000 total training updates. All 42 decoder-input
interventions and 40 runtime conditions also completed. Training finished at
02:16 Berlin and the runtime audit at 02:18, before the 07:00 deadline. Albedo was
excluded. No new dynamics or controller was trained in this program.

The [full illustrated report](../runs/perception_overnight_2026-09-08/report/report.html)
contains exact values, paired comparisons, PCA, attention diagnostics,
reconstructions, masks and geometry distributions. The
[canonical dashboard](../runs/experiment_dashboard.html) retains the run ledger
and learning curves. The [reconciled numerical snapshot](../runs/perception_overnight_2026-09-08/report/reconciled.json)
contains all declared units and no pending comparison.

## What the encoder comparison establishes

P1 freezes one local CNN checkpoint and one pinned DINOv2-S/14 package, then trains
independent RGB, foreground and pose readouts. Values below are means over seeds
9107–9109. The category probe is trained separately on the same frozen features.

| P1 package | COCO RGB MSE ↓ | Foreground IoU ↑ | Category AP ↑ | Fresh pose tolerance pass ↑ |
|---|---:|---:|---:|---:|
| Compact CNN | 0.006201 | 0.357905 | 0.077280 | 76.56% |
| Pretrained ViT | 0.018516 | 0.657429 | 0.650148 | 80.47% |

The category constant-score baseline is 0.033103. This is accessibility through a
particular pooled classifier, not a measurement of all semantic information.
The pretrained ViT supplies much stronger generic task features, while the P1 CNN
decoder reconstructs pixels more accurately. Architecture, pretraining, size,
preprocessing and native width differ; this is a practical package comparison,
not an isolated proof that transformers outperform convolutions.

The CNN has 16×16 and 8×8 grids of 64 channels, two residual convolution blocks
per branch and bidirectional four-head cross-scale exchange. The coarse branch
starts from the shared stem before fine-branch residual refinement. DINOv2 has
12 transformer blocks and a native 16×16×384 final grid after resizing RGB64 to
224. Its 8×8 grid is average pooled from that final grid: **it is a derived view,
not a second independently processed encoder stage**.

Convolutions encourage local spatial processing and weight sharing; transformers
learn interactions between tokens. Both can learn geometry and semantic features.
Training objectives and pretraining matter greatly. The results do not support a
simple “convolution = spatial, transformer = semantic” division.

## Decoder access and sharing explain much of the reconstruction tradeoff

The matched D2 comparison freezes the same ViT, uses the same 4,000-update budget,
examples, losses and shared dense decoder, and changes its third input slot or
task modulation. All endpoints are fixed at update 4,000; mask IoU is the primary
endpoint, with RGB quality reported separately.

| Decoder input / computation | COCO RGB MSE ↓ | PushT RGB MSE ↓ | Foreground IoU ↑ |
|---|---:|---:|---:|
| Final features only, duplicated local slot | 0.024891 | 0.00100333 | 0.658625 |
| Early patch + final + pooled features | 0.007794 | 0.00036511 | 0.668009 |
| Raw patches + final + pooled features | 0.009680 | 0.00043356 | 0.663890 |
| Early features + RGB/mask FiLM | 0.007276 | 0.00040841 | 0.660159 |
| Early features, separate RGB/mask trunks (D3) | 0.001051 | 0.00007503 | 0.657838 |

Early patch access improves **both RGB and foreground IoU in all three seeds**
versus final-only access. It preserves local evidence that the dense output can
use directly. Raw patches also help, but their width, preprocessing and receptive
support differ from early patch embeddings, so those arms are not exact substitutes.

Task FiLM improves COCO RGB in two seeds and lowers foreground IoU in all three
versus the same early-input decoder. Typed final output layers already identify
the requested output. The extra task-dependent shared computation is therefore
a tradeoff, not an automatic benefit. Do not enable it by default.

Separate trunks reduce mean COCO RGB error by **7.42×** and PushT RGB error by
**4.87×** relative to the shared early-input trunk. Foreground IoU falls by
**1.02 percentage points**, consistently across seeds. D3 does not win the primary
foreground endpoint. It changes sharing, parameter capacity and cross-domain
transfer; joint gradient clipping still couples branch scales. It does not isolate
gradient interference as the sole cause. Both trainers already use three task
passes per update; separate trunks did not double that count.

Read-only donor-input tests corroborate different uses of supplied evidence:
replacing early local features with another image's features increases RGB error
about 16.5×, while replacing final/coarse context increases it only about 1.18×.
Foreground IoU is much more sensitive to the context substitution. These are
fitted-model reliance tests, not retrained branch-removal experiments.

This extends the earlier recovery result: the adapted encoder retained substantial
reconstructable information, but a COCO-only decoder refit harmed PushT. Mixed
data, local access and output-specific computation now give a more useful set of
tradeoffs than simply retraining the encoder from scratch.

## More encoder depth did not solve the measured limitations

Nine matched continuations start from the same CNN and preserve its initial
function: ordinary joint adaptation, two added residual convolution blocks per
grid, or two added transformer blocks per grid. They use the same paired examples,
objectives, 4,000 updates and minimum-validation-pose selector.

| Continuation | Test mean q, grouped split ↓ | COCO RGB MSE ↓ | Foreground IoU ↑ | Fresh pose pass ↑ |
|---|---:|---:|---:|---:|
| Existing topology | 0.300284 | 0.006592 | 0.363343 | 69.01% |
| Two extra convolution blocks per grid | 0.300873 | 0.006857 | 0.367138 | 68.95% |
| Two extra transformer blocks per grid | 0.320629 | 0.006891 | 0.371260 | 70.64% |

There is no consistent useful gain from these depth additions. All continuations
have worse fresh-case pass rates than the frozen P1 CNN reference. Generic
objectives have much larger encoder gradient norms than manipulation objectives;
the measured directions do not establish uniformly antagonistic gradients.
Loss balance, learning rates and the zero-gated continuation recipe remain
possible causes. This is not a finding that depth can never help.

Four genuinely processed scales, additional pretrained intermediate taps and
longer training were not tested. The current results do not justify adding them
as a default fix. Some late validation errors are still falling; this short-budget
study is not a convergence proof.

## Geometry improves, but the remaining tails matter

The follow-up changes only the pose-head objective: coordinate/orientation MSE
plus 0.001 times target-to-location-map KL. Bilinear targets on the 16×16 output
basis retain the exact supervised point expectation, including boundaries. The
encoder, head topology, examples and initialization are paired with P1.

| Frozen package / objective | Fresh per-case pass ↑ | Fresh mean q ↓ | Mean per-seed p95 case q ↓ | Angle MAE ↓ |
|---|---:|---:|---:|---:|
| CNN, original MSE | 76.56% | 0.54030 | 3.39311 | 3.270° |
| CNN, added map KL | 84.90% | 0.69053 | 3.01216 | 3.153° |
| ViT, original MSE | 80.47% | 0.43682 | 1.73361 | 1.896° |
| ViT, added map KL | 85.29% | 0.39558 | 1.34953 | 2.007° |

Every paired seed improves the fraction of cases inside tolerance. However, CNN
mean q, 99th-percentile error and maximum error worsen in all three seeds. Its
worst cases reach q 20.68–36.16, versus 7.04–12.94 originally. **Improved pass rate
alone would conceal a substantial regression.**

ViT mean q and both the 95th and 99th percentiles improve in every seed. Its angle
change is mixed and maximum error worsens in two seeds. The largest observed
ViT case remains q 17.00. Retain this objective variant as a promising next-test
candidate, with no claim that perception or planning is now reliable.

Fixed case 86 makes a readout failure visible: a pusher probability map has mass
near the pusher and another region near the body, so the expected coordinate lies
between them. The source of that multimodality remains a hypothesis. Boundary
coverage is a concrete concern: 26.6% of fresh cases are within six RGB pixels of
an edge, versus 3.30% of training frames. The follow-up adds no coverage. A fresh,
prospectively separated simulator cohort and broader training configurations are
more discriminating next steps than additional depth alone.

Positions are supervised pusher coordinates and the T-body origin; orientation
uses the simulator pose convention. They are outputs learned from spatial
features, not intrinsic coordinates guaranteed by the encoder. Extent, queried
instances or other quantities require declared output heads and appropriate
supervision. Location maps are neither segmentation masks nor calibrated uncertainty.

## What the internal-state inspection says

Train-fitted PCA uses the same declared populations and shared plotting ranges.
Coarse global PCA can emphasize image/domain means and position structure;
image-centered views reveal different within-image variation. The coarse CNN
global view clips about 14.4% of test color values under training-derived limits,
versus about 2% after image centering. Strange-looking PCA alone is not evidence
that a representation is broken, and colors across separate bases have no common
semantic meaning.

Explicit per-head attention probabilities agree with the actual SDPA computation.
Entropy uses the key axis and divides by log(number of keys) before aggregation.
Fine-from-coarse attention is nearly uniform; replacing its weights with uniform
weights barely changes pose. Removing that branch is harmful. One
coarse-from-fine head is selective, and uniformizing that direction is harmful.
This is consistent with broad context broadcast toward fine features and selective
aggregation toward coarse features. It establishes reliance in the fitted model,
not superiority over a separately retrained pooling or attention-free architecture.

## Runtime and practical architecture choice

Measured on the local RTX 3050, with FP32 RGB64 already on the GPU. Each invocation
runs preprocessing, the full encoder and RGB, foreground and pose outputs. Twenty
synchronized measurements follow five warmups. The ViT decoder variants keep the
same original P1 pose head for this timing comparison.

| Package | Single-frame median | Sample p95 | Batch32 time per image |
|---|---:|---:|---:|
| CNN P1 | 2.83 ms | 3.14 ms | 0.576 ms |
| ViT P1 | 8.45 ms | 9.91 ms | 4.618 ms |
| ViT with early-input shared decoder | 9.85 ms | 10.60 ms | 4.668 ms |
| ViT with task FiLM | 10.63 ms | 11.60 ms | 4.948 ms |
| ViT with separate trunks | 10.76 ms | 11.67 ms | 4.959 ms |

These timings exclude capture, file decoding, host transfer and planning. Desktop
GPU activity and the small sample limit precision. The transformer dominates
package cost; two decoder trunks do not double total inference time. The compact
CNN remains useful when that latency or compute difference matters.

The recommended integration contract is:

1. Expose named local and contextual feature maps with native width, grid, source
   depth, source time and validity. Label pooled levels as derived.
2. Let each consumer project and fuse the levels it needs. Keep an RGB-specific
   path and make decoder sharing a configurable comparison, preserving the better
   foreground reference. Do not force all outputs through one small latent code.
3. Keep a stable observation representation initially. Put output/task queries at
   consumer adapters. Encoder conditioning on prior memory or task is feasible,
   but was not tested here and requires neutral-context and retention controls.
4. Train compatible state-update and action-conditioned dynamics modules against
   the selected interface. Native 384-channel ViT features cannot be dropped into
   the existing 64-channel memory/predictor checkpoints without adaptation.
5. For imagined futures, decode predicted features and aligned predicted state.
   Present-observation raw/early skips cannot supply unseen future pixels. Train
   for imperfect predicted evidence or an explicit missing-detail mode.

This is a proposed integration, not a newly validated closed-loop world model.
Additional memory registers, all-levels-to-all-layers attention and four processed
scales have not earned adoption from this study.

## Data, collaboration and verification

Existing COCO crops/annotations and PushT are sufficient for these comparisons;
no extra generic-image download was necessary. Future work needs targeted broader
simulator states and trajectories, then multi-step/action/control tests. Computer
use needs screenshot/action/transition data and enough native resolution for text
and controls. Upscaling RGB64 cannot recover details already discarded. Driving
needs temporal, action and sensor evidence. These capabilities are not established
by static reconstruction, foreground masks or pose gates.

Claude participated in ten successful exchanges through the installed CLI using
public conceptual briefs and web-only tools. Private code, datasets and numerical
results were not exported. I verified implementation and evidence locally. The
review corrected assumptions about patch positions, raw/early equivalence,
normalization, shared parameters, gradient interpretation and split-decoder compute.
The [collaboration records](../runs/perception_overnight_2026-09-08/collaboration/)
retain proposals, disagreements, corrections and execution receipts.

Related primary work supports the design vocabulary, not a guarantee of results:
[DINOv2](https://arxiv.org/abs/2304.07193), [DPT](https://arxiv.org/abs/2103.13413),
[FPN](https://arxiv.org/abs/1612.03144), [FiLM](https://arxiv.org/abs/1709.07871),
[DSNT](https://arxiv.org/abs/1801.07372) and
[integral regression](https://arxiv.org/abs/1711.08229).

Final software verification: **407 passed, 3 opt-in browser-transport tests
skipped, no failures**. Every completed numerical unit refreshed the canonical
HTML. Original failed controls and publication receipts are preserved. The
source-inventory repair retains exact record/path relationships, and smaller image
previews preserve every decoded RGBA pixel. A full-suite assertion still expecting
the old joined-source field was updated to verify the new exact relation; its
failed receipt and corrected full-suite receipt both remain available.
The final report and dashboard passed standalone browser verification at 1440px
and 390px, including the source dialog. An additional in-app navigation to the
local report URL was blocked by the browser URL policy; no workaround was used.
The scientific image panels were inspected directly from their saved files.

Scientific limits include one source checkpoint per encoder, three head/module
seeds rather than repeated pretraining, reused grouped holdouts, adaptive fresh
follow-ups, unverified pretrained-data overlap with COCO, different checkpoint
selectors across separate studies and changes in decoder capacity. There are no
confidence intervals claiming architecture causality and no new control-success
claim. The report exposes individual seeds and failure cases alongside averages.
