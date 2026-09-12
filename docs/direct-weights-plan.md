# Direct numerical weight construction

12 September: user explicitly asks to inspect the architecture, guess numeric
weights, write binary files, test them and adjust. Execute this before the previously
authorized optimizer-training experiment. Do not relabel direct guesses as pretrained
weights or claim optimizer-free fitting is learning-free.

Use exactly the existing image encoder → categorical belief agent → hybrid memory
→ working-token path plus a two-class readout. Width16, RGB32, single image scale,
recent2/block2/one compressed block. No renderer coordinates or answer labels enter
forward. A constant learned-size query specifies the fixed last-visible-container
task; natural-language understanding and full entity discovery are outside this slice.

Generate 32 fit, 16 development, 32 final-test scene pairs using disjoint seeds
3101/3201/3301. Each pair shares background, container geometry, appearance and final
two covered frames. Four-frame history: target at center then target left/right,
then covered twice. Half the scene groups instead show the mug on the opposite side
first, before its final visible location. This reversal subset tests recency, as
Claude requested. Test scores are reported separately for center-start and reversal.
All labels are last observed facts, not assertions about hidden current state.
Check actual rendered red target pixels, paired equality and whole-scene/final-image
hash disjointness. Preserve event metadata outside model inputs and export filmstrips.

## Three predeclared candidates

1. Ordinary seeded initialization (seed3001), as baseline.
2. Direct hand construction: zero parameters; LayerNorm gain1; attention value/output
   identity, key maps red-salience channel0 into each head's first key coordinate,
   constant query bias6. Residual MLPs remain zero. Image patch channel0 averages
   8R−4G−4B to emphasize the red mug. Existing sinusoidal spatial/time features stay
   in the forward code. Readout weights use the horizontal position channel8 with
   opposite signs. Categorical heads/code projections stay zero, so random categorical
   samples do not affect output. This is an intentionally task-specific hypothesis,
   not a generic visual representation or learned semantic graph.
3. Adjust candidate2's linear readout using fit-split feature centroids: direction
   right-mean minus left-mean; midpoint intercept; scale so centroid logits differ
   by four. This writes a closed-form fitted head into the same architecture. No
   gradient or optimizer update. It uses labels and must be called data-fitted.

Write each candidate's ordinary PyTorch binary state dict with method and architecture
metadata. Strictly reload before evaluation. Rank candidate2/3 using development
accuracy then NLL (stable order breaks ties); freeze winner before final test.
Evaluate baseline and selected winner on untouched test groups, with fixed common
categorical RNG per pair. Report logits, accuracy, NLL, paired-both accuracy and the
reversal subset. Also replace every earlier frame by the final view and verify both
pair members have identical predictions, implying exactly50% balanced accuracy.

Hypothesis/gates: winner >=90% test accuracy, >=80% both-pair accuracy, >=80% reversal
accuracy and >=30 percentage-point history advantage over the erased control. All
gates must pass; failure remains a result. No further test-guided adjustments in this
experiment. A pass would only establish this small task-specific constructed model.

Reuse recipe, Run bookkeeping, checkpoint and report machinery. No separate trainer.
Run accepts an unused zero-lr optimizer for checkpoint compatibility; never call its
step or backward. CPU tests catch visible targets/split leakage, serialization and
forward equivalence, absence of optimization, cached-resume behavior and wrong labels
entering the feature extractor. Commit plan and failing tests before implementation.
Use a1GiB PyTorch allocator cap and600-second active evaluation budget on local GPU;
report inference memory and no training peak. Existing other GPU users are untouched.
Browser/report checks and honest real-webcam limits follow the ordinary workflow.

Direct generation of useful arbitrary networks remains an open research idea.
[HyperNetworks](https://arxiv.org/abs/1609.09106) learns a network that generates other
weights; that precedent does not supply a trained generator for this model.

## Implementation review before evaluation

Implemented in `experiments/multimodal.py --dataset direct-weights`, using new small
visual data/readout modules and the existing report path. Three focused tests pass:
paired pixels/visible evidence/splits; gradients through the untouched original
architecture and erased-input invariance; direct-file reload/cache equivalence with
backward and optimizer steps explicitly forbidden during the construction run.
The complete CPU regression suite passes. The independent GPU preflight has108,915
parameters, finite logits,36MiB reserved/34.53MiB allocated peak for four episodes.
The limit measures this process's PyTorch allocator, not total GPU use or future training.
Rendered center-start/reversal strips were visually inspected before final-test access.

Claude's visual-task review requested reversed histories; those are now included.
The direct-weight review correctly distinguished fitted centroids from handwritten
weights. It initially mischaracterized independent test selection as optimistic and
history erasure as clearing state; both points were explicitly withdrawn in the
reconciliation. Final test estimates the frozen development-selected procedure;
erasure replaces input history and starts each episode fresh. Receipts are under
`runs/reviews/continuation_2026-09-11/direct-weight*`.

Run and load (same configured architecture is required):

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python experiments/multimodal.py \
  --dataset direct-weights --device cuda --output runs/direct_weights_v1/reference
```

```python
import torch
from experiments.multimodal import build_visual_memory
model = build_visual_memory()
payload = torch.load("runs/direct_weights_v1/reference/weights.pt", weights_only=True)
model.load_state_dict(payload["model"], strict=True)
model.eval()
# model(images): images has shape [batch, 4, 3, 32, 32], RGB floats in [0, 1].
```

`constructed.pt` is the purely handwritten candidate; `adjusted.pt` has a data-fitted
head. `weights.pt` is whichever development selects. Each payload records its method.
These files target this small configured agent plus its two-class task readout, not
an arbitrary default-width model. `last.pt` additionally preserves Run metadata/RNG.

## First result and bounded numerical revision

First run `runs/direct_weights_v1/reference` completed with zero optimizer updates
in2.504 active seconds. Development: handwritten50% overall,100% center-start and0%
reversal; centroid-adjusted56.25% overall,12.5% center-start and100% reversal. The
development-selected adjustment gets39/64 final-test answers right (60.9375%); the
ordinary and erased controls score50%. Three of four gates fail. This is a negative
result, not useful pretrained-weight generation. Inference allocator peak40MiB.

The development pattern motivates one explicitly bounded second candidate family,
still changing numeric weights only. Preserve the first run. Its architecture's
zeroed chronology projections discard explicit time labels during compression, while
identity transition attention may repeatedly accumulate old evidence. This is a
code-based explanation to test, not yet an isolated causal finding.

Revision: zero the dynamics transition attention's output weights (retain h instead
of repeatedly adding read context); project sin(end_time) with gain16 and bias−8
into compression channel0 so the later of the two early observations is favored;
set memory-reader evidence-view channel0 to+4 and belief-view to−16; gate compressed
history with biases [−8,+8,−8,−8]. All other handwritten numbers unchanged. Compare
its signed-position head and the same centroid-fitting procedure on the existing fit
and development splits. This is hand-engineered for the fixed short scene/timing
contract; it is not a general temporal-reasoning mechanism.

Freeze that development-selected procedure before accessing fresh final-test scene
seed5301 (32pairs). Same four gates, device cap, baseline, no-gradient checks and600s
budget. Report it independently from the first test; never claim seed3301 remained
unseen after the first result. No further adjustment is authorized by this bounded
experiment plan. Use `--direct-weight-timing` and a fresh output directory.

Pre-execution isolation correction: tiny software tests initially exercised the first
two scenes of the recipe's default evaluation seed. They checked serialization,
not capability thresholds, and their performance was not used to choose weights.
Move unit fixtures to a separate100000 seed offset and use never-scored seed5301 for
the second final evaluation (replacing proposed4301, which the tiny check exercised).
The first reference's nominal holdout therefore had limited software-check exposure;
retain that limitation alongside its failed result rather than retroactively relabel it.

## Completed revision

`runs/direct_weights_v1/timing` passes all four declared gates on fresh seed5301:
64/64 answers,32/32 complete pairs,32/32 reversal answers. Both ordinary initialization
and erased-history controls score32/64 (50%). Final NLL0.126380. The handwritten timed
head alone scores50% on development; its centroid-adjusted head scores100%, so the
adjusted candidate was selected before final-test scoring. Do not attribute the pass
to unaided guessing of every value:34 output-head parameters were fitted using64 labeled
fit episodes. No backward or optimizer update was used in either experiment.

Evaluation and file production took2.559 active seconds; inference allocator peak
40MiB reserved /37.94MiB allocated. These are inference values, not a training or
whole-desktop peak. The selected binary contains108,915 parameters (2,670 nonzero),
603,453 bytes. Most parameters deliberately stay zero; the module layout is unchanged,
but this is a small hand-built circuit for this fixed visual/history task.

Full regression:180 CPU tests pass, including4 focused visual checks. Cached resume,
strict binary reload, independent probability-space metric reconstruction, rendered
pixel/label checks, source snapshots and fresh-test disjointness all pass. The raw
evidence is `runs/direct_weights_v1/timing_verification.json`. The local
`timing/report.html` is structurally verified; browser QA remains unavailable under
the existing access restriction. The first failed run is unchanged. A CUDA launch
failure before any scoring was repaired by resolving the device index explicitly.

Use the successful files rather than the failed reference for this configured task:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python experiments/multimodal.py \
  --dataset direct-weights --direct-weight-timing --device cuda --output runs/my_direct_weights
# To reload a completed compatible run without fitting or rescoring:
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python experiments/multimodal.py \
  --resume runs/direct_weights_v1/timing
```

The load example above applies with `timing/weights.pt` in place of
`reference/weights.pt`. `timing/constructed.pt` contains the pure hand assignment;
`timing/adjusted.pt` and `timing/weights.pt` include the fitted readout.
`timing/weight_inspection.json` exposes those34 fitted numbers and file provenance.
Earlier runs retain their source snapshots; resume with their matching source version
if later implementation changes make the normal strict resume check reject them.

This demonstrates direct weight construction plus narrow data fitting can implement
one controlled behavior. It does not establish useful webcam perception, long memory,
multiple-entity binding, learned concepts, general action planning or synthesis of
arbitrary pretrained networks. The earlier optimizer curriculum remains deferred while
this requested direct-weight test is completed; no broader training was silently run.
