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
