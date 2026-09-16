# Pair-centered training with single-sequence inference

16 September. Follow-up to [displacement coverage](video-displacement-plan.md).
Test whether explicitly penalizing a common class preference improves transfer while
retaining ordinary single-sequence inference. No new architecture, encoder or decoder.

## Protocol before fits

BASELINE: ordinary balanced-pair cross entropy (CE).
CENTERED: CE +0.1*mean(SmoothL1(b,0,beta=1)), where s=logit0-logit1 per member,
b=(s0+s1)/2. Pair labels must be opposite; either orientation valid. SmoothL1 uses
0.5*b^2 for |b|<1 and |b|-0.5 otherwise. A shared shift of both class logits has no
effect. Pure pair ranking is not a replacement: it leaves the single-clip threshold
unidentified. Fixed coefficient0.1, no sweep, no adaptive weighting or calibration.
Both causal temporal module and head are trained in both arms. Pair centering is
training-only; forward receives exactly one three-frame clip, no partner/metadata.
No additional parameters, inference compute or observations. Penalty adds only small
training arithmetic; same updates/forward/backward model path, actual walltime recorded.

Same16 training clips x4 frames as previous EXPANDED;48x48 crops, all48 cyclic phases,
displacements2/4/6/8px. temporal7401/7402 crossed head7501/7502; sampler7600;
512updates x8pairs, matched phase sampling, AdamW0.003,wd0.0001,clip1,CPU FP32.
Eight formal fits; unchanged COCO image-codec and temporal initialization sources.
Identical sampled pairs and initial model states across objectives. BASELINE expected
to exactly reproduce previous EXPANDED weights and matched population logits.

Evaluation groups: known2/4,wide6/8,intermediate3/5/7,extrapolation10/11. All48 phases;
max2d=22<24 half-period, final interval direction only, dt fixed. All static controls
and RGB/latent class marginals retained. Not natural motion, streaming, speed or agent
integration. Hypothesis applies to this finite construction, not arbitrary reversals.

Development evaluation now includes historical1KKYX plus all8 previously inspected
confirmation clips from diversity/displacement runs, kept separate per-source. Those
sources are NOT fresh evidence. Historical validation12VVC remains monitoring only,
shares subject with train0EJAG. Fresh4 confirmation clips selected deterministically
from existing local Charades train metadata: salted SHA256(`pathwm-motion-pair-center-v1:`
+ID), length>=8s, file exists, excluding ALL subjects from the26 previously used clips;
one new subject per clip. Save hashes/selection before fits. No score-based replacement.
Files/crops/subjects checked; fresh to motion task, not universal upstream novelty.

Primary capability per arm: ALL four initialization cells, ALL four fresh groups:
source-macro accuracy>=90%, pair accuracy>=80%, flip>=90%, each source accuracy>=80%;
current/previous/unordered accuracy50%, current/unordered pair accuracy0, swapped
prefix exactly swaps paired outputs. Benefit: centered-minus-baseline mean>=3pp across
8 intermediate/extrapolation cell/group comparisons; no cell/group loss>2pp on ANY
fresh group or source-macro development evaluation group. Historical1KKYX separately
also has no>2pp cell/group loss, so the known bad source cannot hide in a macro mean.
No default promotion unless capability AND benefit gates pass. Four cells/one sampler
and four fresh sources support descriptive comparisons only, no significance claim.

Record CE and penalty separately, total loss, |b|, |a| with a=(s0-s1)/2, pair ordering,
|b|/(|a|+|b|+1e-8) and ordinary single-clip accuracy/margins/confident errors. Absolute
score shrinkage alone is not repair. Training curves use total objective; validation
is CE and explicitly labelled. Keep all checkpoints and negative outcomes.

Budget:8 formal fits45s each,<=360s training; only extra fit8 versus4+4 mechanics.
<=90MiB artifacts,>=300MiB disk reserve; no source media copying/downloads. Preparation
and evaluation/reporting separately timed. Do not inspect fresh confirmation model
scores until all8 fits complete. No selection, continuation or conditional extra fits.

## Implementation

Extend experiments/video_order.py: one explicit pair_center_weight (default0), small
loss helper validating opposite paired labels and finite nonnegative coefficient;
add pair-score metrics and split loss logging. Reuse Run/reports/CLI. No new trainer.
Tests first: numeric penalty and gradient, zero-weight CE replay, member/class/common
logit-shift invariances, malformed pair rejection, single-sequence batch independence.
Run relevant video/image/modality regressions; real-data exact restart. Independently
recompute losses, exposure, score decomposition and all gates from raw artifacts.
Use actual Claude with a public hypothetical methodology brief only; receipts under
runs/reviews/video_pair_center_v1. No private code/media/measurements exported.

Before implementation,3 new tests fail on the absent loss/metrics; single-sequence
batch independence already passes. Claude flags limited replication and synthetic
scope; both are explicit limits. Corrected its mistaken two-seeds-per-cell reading
(four total cells) and batch-mean penalty reading (mean of per-pair penalties, so
opposite offsets cannot cancel). Historical regression gates block benefit, not merely
advise. Corrections sent; no extra fits or broad significance claim introduced.

## Working slice

Fresh confirmation fixed before training:7H7PN(6PZN),A8LZE(8718),V149B(DXDI),
IKZJE(EXQX). Development source order:1KKYX,0XP8L,7YV59,L8HMR,DPKMU,34DKM,X1EZQ,
N588B,AHL6X. Manifest at data/motion_pair_center_v1/sources.json;26 old IDs excluded.
81 scoped tests pass; new per-pair non-cancellation assertion also passes. Real-data
8 versus4+4 exact restart passes7061 recursive checks including objective components,
model/optimizer/RNG, exposure and development logits. Fresh confirmation predictions
excluded from mechanics. No training data/model changed between objectives.
Claude acknowledged finite screens, per-pair penalty and synthetic-domain scope.
Its remaining caution about comparing initialization seeds does not describe the
registered estimand: compare objectives WITHIN matched fixed initialization cells,
not rank initializations or estimate seed-population effects. This distinction and the
single-sampler/one-coefficient limits remain explicit. No private results exported.

## Results

Eight fixed fits complete:13.291s CPU training,15.592s feature preparation;
124.664s complete preparation/training/evaluation/report orchestration before aggregate
analysis. No tuning, changed coefficient, checkpoint selection, extra fit or default
promotion. About39.4MB artifacts within90MiB. Model parameters/inference path unchanged.
Each run draws the same4096 pairs: magnitudes2/4/6/8 occur1026/1021/1020/1029 times.

New confirmation source-macro accuracy, mean over four crossed initialization cells:

| Group | Baseline CE | CE+0.1 pair centering | Centered range |
|---|---:|---:|---:|
|known2/4|95.90%|98.59%|94.37–100%|
|wide6/8|89.62%|92.92%|85.48–97.43%|
|intermediate3/5/7|93.67%|96.84%|90.04–99.72%|
|extrapolation10/11|69.54%|72.42%|69.99–77.47%|

These are four NEW sources, so absolute values cannot be compared to earlier study
means as if the test population were unchanged. Both full capability gates fail;
all eight per-cell full gates fail. Mean intermediate/extrapolation gain3.025pp meets
the3pp screen, but worst fresh cell/group-4.329pp and original historical-source
worst-7.899pp fail preservation. Nine-source development macro worst-0.868pp would
hide that historical regression if reported alone. The complete benefit gate fails.

The intervention reduces absolute common offsets markedly but also score scale.
On fresh extrapolation, mean|b|3.271→0.562 and mean|a|4.839→2.329; relative bias
fraction0.456→0.305. Single-sequence accuracy69.54→72.42% and joint pair accuracy
50.18→61.41% improve, so changes are not ONLY harmless uniform logit scaling.
Comparative pair ordering73.52→74.33% is still weak for untrained large magnitudes;
centering does not create all missing directional evidence. Fresh known pair ordering
stays100% while ordinary accuracy95.90→98.59%; there centering closes part of the gap.
Averages do not establish that b is purely an appearance effect or that the latent
contains no further accessible evidence. Failed coefficient/budget is not rejection
of every paired objective.

Training accuracy:baseline98.40–100%,centered99.58–99.98%. The formerly50% historical
cell becomes55.99/60.68/60.07/52.34% across groups, still poor. Another cell loses
7.90pp on historical intermediate displacements; no hidden removal of that source.
All current/previous/unordered controls50%; fresh raw RGB oracle100%, no ambiguities.

81 scoped tests pass;7061 exact8 versus4+4 restart comparisons;52913 independent raw
metric/loss-accounting/exposure/hash/report checks.72 exact previous-baseline checks
match all four prior model states and unchanged train/validation/historical/previous
confirmation logits.4392 exact RGB/latent marginal comparisons,30 unchanged source
hashes and40 frozen cached tensors.11 standalone reports structurally verified;
comparison plot visually inspected, browser interaction not checked. Two actual-Claude
public reviews retained; no private code/media/measurements sent.

[Comparison report](../runs/video_pair_center_v1/report.html),
[verification](../runs/video_pair_center_v1/verification.json),
[score diagnostics](../runs/video_pair_center_v1/diagnostic-summary.json).

## Next proposed comparison

Keep this partial objective repair opt-in. Before another broadening of training data,
test whether an explicit two-frame correspondence/readout mechanism gives better
large-displacement direction evidence than the current pooled temporal features.
Use existing matching primitives where possible and compare receptive-field support
at controlled capacity/compute. Preserve single-sequence inference, distinguish a
reversal property valid for two frames from the artificial reversed-prefix rule, and
retain inspected sources as development with new confirmation. Proposal only: no new
matching architecture or extra fit in this study. Natural video/streaming/core integration
remain separate open requirements.

## Run

```bash
.venv/bin/python -m experiments.video_order --output runs/my_pair_center \
  --source-manifest data/motion_pair_center_v1/sources.json \
  --displacement-spec data/motion_pair_center_v1/expanded.json \
  --balanced-training --matched-phase-sampling --pair-center-weight 0.1 \
  --seed 7600 --head-seed 7501 --steps 512 \
  --temporal-source runs/video_context_v1/seed7401/k3/history/last.pt
```

Use weight0 for the CE baseline. The coefficient is included in checkpoint compatibility;
changing it requires a fresh run. The existing single-sequence model.forward is unchanged.
