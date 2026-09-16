# Temporal direction: initialization versus transfer

16 September; continuation of [paired order](video-order-plan.md). Before new fits,
read the four saved balanced-data results: every train accuracy is100%; three
validation accuracies100%, one95.05%. Weakest held-out model makes high-margin
errors despite train cross entropy0.00057. This is a transfer gap at the measured
training solution, not failure to fit. Optimizer implicit bias remains possible.

## Fixed protocol, before implementation and outcomes

Preserve original image encoder, existing3x3 temporal architecture, readout size,
data/splits, exhaustive48-phase task and512 updates, AdamW0.003/wd0.0001, clip1,
8 pairs/batch. No correlation extension in this comparison. Cross temporal source
7401/7402 with independently controlled head initialization7501/7502. Fix the Run
sampling seed7600 in ALL cells. This2x2 separates observed temporal/head effects
with common samples; it cannot estimate general seed variance. Earlier studies
coupled temporal initialization, head initialization and batch sampling.

For each of four cells, compare unchanged data against ONE repair: reflect the
exact raw RGB sequence horizontally before the frozen encoder and invert its
left/right label. On odd numbered updates use reflected examples, even updates
original ones. Keep source-pair indices and total batch exposure identical. No
random augmentation RNG, changed LR, extra iterations or checkpoint selection.
Encode reflected unique views, then gather with the ORIGINAL indices. Do not flip
latent grids or regenerate unrelated phases. Metadata phase denotes the original
pair's index, not a claim about reflected geometric coordinates. Input time and
displacement magnitude remain unchanged. Evaluate both orientations; main gates
use original orientation so they stay comparable. Reflection tests are diagnostic.

Eight formal fits,45s maximum training each,<=360s total; stop after these eight
regardless of outcomes. Artifact allowance30MiB, keep>=300MiB disk reserve. No
downloads or new independent test source. Only two tiny8-update restart runs
(full8 versus4+4) are additional mechanical checks. No hidden pilot training.
All sources are previously inspected development clips; no untouched final-test
claim, natural-motion/forecasting/speed/agent-integration claim or inference decoder
quality claim. Keep prior negative results and weights unchanged.

Existing capability screen per cell: BOTH known2/4px and wide6/8px accuracy>=90%,
both-members-correct>=80%, prefix-flip consistency>=90%; current/previous/unordered
controls exactly50%, current/unordered pair accuracy0. A robust candidate must pass
all FOUR crossed cells. Repair-benefit screen additionally requires>=3 percentage
points mean accuracy improvement across the eight cell/group comparisons, with no
cell/group regression>2pp. Report both screens even if one fails. No thresholds
chosen after results. Train, validation and reflected-input accuracy, margins,
high-confidence wrong fraction and per-displacement metrics are diagnostics.
Report descriptive temporal/head main effects and interaction, not p-values or a
causal statement about optimizer failure. Same train accuracy can hide different
generalizing solutions; more training is not justified by the observed gap alone.

## Essential checks and implementation

1. Tests for exact RGB reflection, direction sign from an independent translated
   marker, pair identities and per-frame class marginals; index/time/magnitude
   preservation and no input mutation. Test original/reflected batch-label pairing
   by absolute update index, including a resumed odd step.
2. Add a small reflection helper, optional reflected cache in the existing recipe,
   independent head seed and margin metrics. Reuse Run, optimizer/checkpoints and
   unchanged report renderer. Recompute frozen image features, not decoder training.
3. Verify exact8 versus4+4 replay; train all eight matched cells. Source and initial
   state hashes, sampler states and data identities must demonstrate matching.
4. Independently recompute raw metrics and gates, inspect plot, save per-run and
   comparison reports, update atlas/state and research evidence, commit.

Claude public-only review requests exact sequence reflection and label checks.
Reject its suggestion that unchanged labels under reflection necessarily force
chance: distinct images may support memorization. Also correct the claim that
equal-budget augmentation can rule out optimizer implicit bias. Await/record its
acknowledgment. No private code, datasets or measurements sent externally.
