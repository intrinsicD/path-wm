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
equal-budget augmentation can rule out optimizer implicit bias. Claude acknowledged all three corrections in a second public-only exchange.
Remaining limitation agreed: these four fixed cells do not estimate a population
variance; report descriptive differences only. No private code, datasets or measurements sent externally.


## Results and review

Eight fixed fits complete in13.630s CPU training (data preparation, evaluation and
reports additional). All eight reach100% original AND reflected training accuracy;
validation original98.83–100%. There is no incomplete empirical training fit here.
The previously inspected held-out source still distinguishes the learned solutions.

| Temporal init | Head init | Original known / wide | Reflection training known / wide |
|---|---|---:|---:|
|7401|7501|96.35 /96.09%|98.31 /98.83%|
|7401|7502|85.16 /87.76%|85.81 /89.19%|
|7402|7501|85.03 /86.85%|79.95 /81.25%|
|7402|7502|82.03 /83.20%|86.85 /93.23%|

Only7401/7501 passes the existing full per-cell capability screen, in both arms.
Both four-cell capability gates FAIL. Reflection's mean gain1.367pp is below3pp;
worst regression5.599pp exceeds2pp. Repair-benefit gate FAILS; no default adoption.
Input controls stay exactly50%; current/unordered pair scores0. Reflection also
improves some mirrored evaluation inputs and worsens others; every result is saved.

With original training, descriptive temporal7402-minus7401 contrast is-7.064pp,
head7502-minus7501 is-6.543pp and interaction6.445pp (mean across known/wide).
With reflection these are-7.715pp,-0.814pp and20.508pp. These are four-cell contrasts
at one fixed batch sampler, not population effect estimates. Both factors affect
this measured solution; batch order alone cannot explain differences within this
comparison. No unique causal attribution to data shortage, encoder limits, model
size or optimizer implicit bias. More updates are not motivated by lack of train fit.

Implemented: separate `--head-seed`, exact RGB reflection with inverted labels,
absolute-step augmentation schedule, original/reflected evaluation and logit-margin
and high-confidence-error metrics. Refactored exact marginal checking into one helper.
The frozen encoder is called on mirrored RGB; it is not assumed reflection-equivariant.
Measured reflected-encoding versus flipped-latent MSE is0.0052–0.0111 across populations.
No inference architecture, latent dimensionality, decoder or source weight changes.

Validation:72 unique scoped tests,2211 exact8 versus4+4 restart checks on this balanced
reflection path,7828 independent raw/artifact checks,672 exact RGB/latent marginal
count identities. Frozen cached features/labels unchanged; source media and original
weight hashes checked. All11 reports structurally verified, comparison panel visually
inspected; browser interaction not checked.16.2MB artifacts before final documentation.
The ignored execution script initially needed `PYTHONPATH=.`; no run had started.
Aggregate report generation then rejected a NumPy boolean; explicitly converting it
to builtin bool repaired serialization without any retraining or changed raw results.
The report-repair receipt is retained.

Two actual-Claude public-only reviews completed. The reviewer withdrew claims that
unchanged mirrored labels must force chance, that alternating orientations doubles
exposure, and that augmentation could rule out optimizer implicit bias. Its limitation
on four fixed cells is retained; no extra seeds, tuning or gates added after results.

[Comparison report](../runs/video_stability_v1/report.html),
[raw audit](../runs/video_stability_v1/verification.json),
[initial fit/margin audit](../runs/video_stability_v1/prior-fit-diagnostics.json).

Next proposed experiment: broaden independent image-content coverage at unchanged
architecture/task, matched update/exposure budgets and these crossed initializations.
The current16 training images come from four short clips; independent source breadth
should be varied explicitly, not equated with more highly correlated frames. Use a
fresh source-disjoint confirmation set with criteria fixed in advance. This proposal
is not run or claimed to be the unique repair; general motion and core integration
remain open.

### One-cell command

```bash
.venv/bin/python -m experiments.video_order --output runs/my_motion_reflection \
  --balanced-training --reflect-training --seed 7600 --head-seed 7501 \
  --temporal-source runs/video_context_v1/seed7401/k3/history/last.pt --steps 512
```

Omit `--reflect-training` for original-only training. `--seed` fixes batch sampling;
`--head-seed` sets fresh head initialization independently of the supplied temporal
checkpoint. The saved eight-cell orchestration additionally evaluates mirrored inputs
for every baseline using the identical prepared data identities. Exact resume requires
unchanged code/settings/data; previous recipe versions remain in run source snapshots.
