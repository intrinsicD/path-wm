# Same current frame, different ordered history

16 September. Follow-up to the fixed-mask video-context comparison. Its correct
versus different-clip history difference was small. This task measures last-step
horizontal direction on controlled pans of real image contents, not natural video,
object motion, forecasting or full-agent reasoning.

## Protocol before code/results

Use the same six previously inspected development video sources, split by source:
train0EJAG/0GFE8/0JQ26/0LDP7, validation12VVC, evaluation1KKYX. Decode the first
two seconds at2fps to four80x64 RGB images per source. Each image independently
supplies48x48 crops at offsets(-d,+d,0) and(+d,-d,0) around the same anchor.
These form a pair: identical current RGB and identical unordered frame multiset,
opposite apparent image-motion direction in the final interval. Earlier intervals
reverse direction; do not interpret this as constant-velocity prediction. Crop
anchors(y,x)=(4,12),(4,20),(12,12),(12,20). Train/validation/known evaluation d=2,4
pixels; wide-displacement evaluation d=6,8 at the same held-out source/anchors.
No wrapping, padding, frame-specific resizing or class-dependent noise.

The source image VAE remains frozen. Cache deterministic spatial means only, with
tensor/source hashes; targets and pair metadata never enter the model. Expose the
existing temporal mixer's residual features separately from image means so the
classifier need not overwrite the appearance code. Train a shared per-location
1x1 projection, SiLU, spatial mean and binary linear readout. Central feature region
only; fixed radius2 margin for all arms. All trainable arms have the same shapes,
head size and optimization; a correlation extension has dormant input channels in
baseline arms to keep the head width equal.

Initial arms, seeds7501/7502:
1. Frozen reconstruction-trained3x3 temporal mixer, train only readout.
2. Same mixer/readout initialization, also train temporal weights for direction.
3. Same trainable architecture, repeat current frame three times.
4. Same trainable architecture, repeat immediately previous frame three times.
The source mixer for each seed is the corresponding7401/7402 k3-history checkpoint
from video_context_v1. All use the original unchanged image VAE source. Current-only
must score exactly50% marginal and0% both-members-correct. Previous-only tests static
crop-position cues; swapping the earlier roles gives identical samples with inverted
labels, so this control covers either earlier frame role. Report any above-chance
static cue instead of claiming unconfounded motion matching.

Budget512 updates per fit, balanced batches of8 PAIRS, AdamW lr0.003/weight_decay
0.0001, gradient clip1, cross entropy only. Fixed paired exposure per seed, no early
stopping/selection. Max45s training per fit; max540s if all12 fits run. Two8-update
restart checks (8 versus4+4) are separate mechanical checks. Reuse Run/reports; do
not add a training framework. No changes to source during runs.

Gate for direction-trained arm: >=90% marginal AND >=80% both-members-correct on
known and wide evaluation in BOTH seeds; >=90% prediction-flip rate on paired prefix
swap; >=20 percentage points above corresponding previous-only control; previous-only
<=55%. Current-only and unordered-mean inputs to the trained model must score exactly
50% marginal and0% pair. Confusion matrices, per-displacement scores, validation and
training fit are diagnostic. No rate/reconstruction or general-vision claim.

If this gate fails, ONE predeclared repair round adds normalized local feature
correlations between the last two frames at horizontal latent offsets[-2,-1,0,1,2].
Use the same head dimensions/initial weights and frozen image means; match all other
training settings. Also train correlation-only readout (other feature channels zero),
so success is not automatically attributed to the learned3D mixer. Two seeds for each
repair/control, at most four additional fits. Same gate and existing static controls;
no further hyperparameter search. Distinguish new primitive from learned motion.

Essential checks: exact pair/multiset/current identity, target direction from a
known translated marker, source disjointness, valid crop geometry; no target/mask
mutation; residual-feature API preserves old forward/state-dict behavior and future
causality; correlation orientation/crop support; frozen source state; exact restart.
Report a raw-pixel alignment oracle with ambiguity counts as task sanity, not model
performance. Save logits/labels/metadata and small feature/sequence examples.

Claude review requests matched current/previous-only controls and correlation-only
attribution. Correct its proposed future test: changing the LAST observed frame may
change the last-step answer; only frames strictly after a queried feature time must
have zero effect/gradient. Save acknowledgment. Whole-source split and paired labels
do not themselves rule out static crop-position cues; measure them.

Artifact budget<=22MiB, >=300MiB free disk. Before runs, clean only obsolete pytest
temporary data from this project's completed tests if needed; preserve original data,
checkpoints and all completed project runs. No model download. Each completed fit has
raw results/checkpoint and a self-contained report; no report-renderer changes.

## Implementation sequence

1. Pair generator and failing semantic checks; commit plan/tests.
2. Separate temporal feature API and bounded matching primitive; trainable diagnostic
   recipe, using existing Run and renderer. Validate and commit.
3. Exact restart, eight fixed baseline fits, then conditional four-fit repair.
4. Recompute metrics independently, inspect sequence/feature panels, record negatives
   and limitations, update architecture/state/research log and commit.

## Pre-run review and checks

Actual Claude reviewed the public protocol twice. It explicitly withdrew the
incorrect suggestion to demand invariance to the last observed frame and accepted
strict-future invariance plus a positive current-frame influence check. Previous-only
and correlation-only controls were added before fits. No private code/data/results
were exported. Receipts: runs/reviews/video_order_v1/.

68 scoped tests pass:5 new order/data/matching checks,12 video-VAE contracts and51
spatial/modality regressions. The initial3 new checks failed on missing APIs before
implementation.8 versus4+4 direction-repair smoke has1894 exact model/optimizer/RNG,
train-row/metric/array checks. Timing and extra intermediate validation rows excluded.
Source decoder is not in the diagnostic model, and encoded spatial tensors are
detached/immutable. Only obsolete generated pytest scratch was cleaned, not source
data or project run history. Formal gates/budgets remain unchanged.

## Control failure and separate balance repair, before its evaluation

All12 registered fits completed. Direction-trained arms reach100% known/wide in
both seeds, but the7501 previous-only known control reaches60.9375%, above55%.
Original gate therefore FAILS and is not revised. The conditional correlation round
was executed under the registered rule; it cannot repair this dataset-control issue.

Next, a separate evaluation-only challenge repairs single-frame marginal balance:
take one48x48 patch per each of the four evaluation-source images, at anchor(4,12),
and enumerate ALL48 circular horizontal phases. For each phase p and d in(2,4) or
(6,8), use frames(p-d,p+d,p) paired with(p+d,p-d,p). Final image and unordered set
match within each pair; each exact image appears equally under each direction label
at every time position across the complete phase population. A deterministic
single-frame classifier therefore scores exactly50%, even if it memorizes images.
Do not subsample phases or remove ambiguous examples. Verify these count identities
on the pixels AND deterministic encoded grids; report per-displacement metrics.

This is periodic/wrapped synthetic motion of real RGB contents, not natural camera
motion. Freeze ALL twelve saved models; no new fitting, tuning or model selection.
Separate challenge gate, fixed before evaluation: direction-trained candidate must
reach>=90% marginal,>=80% pair and>=90% prefix-flip on BOTH displacement groups and
BOTH seeds; all single-frame/current/previous and unordered controls must be50%
marginal. Current-only and unordered pairs must score0%; previous-only pair accuracy
is reported, not forced to0%, since its two inputs differ within a pair. This
mathematical correction precedes challenge evaluation. Original gate stays failed
irrespective of the new outcome. Also report the
frozen/correlation-only comparators rather than selecting the best after evaluation.
Max120s challenge evaluation; use shared cached phase features, one compact report
with logits/source/model hashes. Artifact allowance increases to<=28MiB to preserve
the original14 run reports plus this newly motivated diagnostic; no old run removed.
This change precedes challenge outputs. Preserve >=300MiB free disk.

## Separate data-training repair, before its fits

The frozen-model balanced challenge completes without new fitting. All single-frame
controls become exactly50%. Direction-trained arms score84.38–86.20%; correlation
augmentation88.15–94.40%, but every predeclared full gate still fails (including
prefix-flip consistency). Preserve these negative gates and all original models.

Authorize a separate, bounded four-fit experiment under the user's requested
iteration: train the `train` and `correlation` arms from the SAME original
reconstruction-trained initialization on exhaustive-phase balanced training data.
Keep seeds7501/7502,512 updates,8 pairs, AdamW0.003, head size, losses and45s/fit
budget unchanged. Only the pan population changes. Each of the16 training images
contributes48 phases x2 displacements:1536 pairs; validation/known/wide each384
pairs from four source images. Same source splits; no training on reserved source.
This changes sample coverage as well as crop/wrap distribution; it is not a pure
inductive-bias comparison or a continuation from the successful crop-trained head.

Evaluate with the balanced challenge criteria already declared above. Both modes
receive identical pairs/sampling per seed; dormant versus active matching features
remain their only architectural difference. No model/threshold/budget selection.
Also report paired swap, all input controls, raw oracle, per-displacement metrics
and training/validation accuracy. This is one new four-fit round, not extension of
the original registered12 fits. Stop this round after four fits regardless of
outcome; further generalization needs a separately specified experiment.
Total artifact allowance<=38MiB preserves both prior studies and these four reports;
disk reserve remains300MiB. No original gate is overwritten or reinterpreted as pass.

## Results and limits

Implemented `pan_pairs` and `cyclic_pan_pairs`, separate temporal-feature readout,
bounded local correlation and `experiments.video_order` using existing Run/report
infrastructure. Default VideoVAE reconstruction and checkpoint keys remain intact.
The image encoder/decoder source and all six source media hashes are unchanged.

**Original crop experiment:**12 fixed512-update fits,15.635s CPU training. Frozen
reconstruction-trained temporal features with a learned head give43.75–50% on the
reserved source. Direction-trained and correlation-augmented arms give100% known
and wide, but previous-only7501 reaches60.94% known: original gates FAIL. Current-only
is50%. Correlation-only gives60.94–76.56%; do not attribute augmented success to the
matching primitive alone. [Original report](../runs/video_order_v1/report.html).

**Evaluation-only balance repair:** all48 phases, four source images,384 pairs per
displacement group.96 RGB/feature marginal-count checks across known/wide establish
exact single-frame balance. All static controls become50%, without retraining. The
plain direction-trained arms give84.38–86.20%; augmented88.15–94.40%; frozen50–55.86%;
correlation-only63.41–79.04%. Both-seed full gates FAIL, particularly paired flip
consistency. This is distribution-shift evaluation on periodic pans, not a
retroactive replacement of the original gate. [Challenge](../runs/video_order_v1/balanced/report.html).

**Separate balanced-training repair:** four fixed512-update fits,6.260s CPU training.
Identical initial model weights to corresponding original arms, matched pair sampling
within each population; data distribution/coverage changes explicitly.

| Seed | Arm | Known2/4px accuracy | Wide6/8px accuracy | Known / wide both-members accuracy | Per-seed gate |
|---|---|---:|---:|---:|---|
|7501|learned temporal|96.22%|96.61%|92.45% /93.23%|pass|
|7502|learned temporal|79.30%|80.60%|58.59% /61.20%|fail|
|7501|+ local correlation|86.07%|98.83%|72.14% /97.66%|fail|
|7502|+ local correlation|96.88%|98.31%|93.75% /96.61%|pass|

Current/previous/unordered single-frame controls remain exactly50%. Neither variant
passes BOTH seeds, so overall gates FAIL; no default adoption. Successful individual
fits establish feasibility on this controlled task, not reliable optimization or
natural motion understanding. Correlation has no consistent across-seed dominance.
[Balanced training report](../runs/video_order_v1/balanced_training/report.html).

All16 formal fits total21.895s CPU training; preprocessing, evaluation and rendering
are additional.69 unique scoped tests pass, including exact label/marginal identities,
causality, unchanged image features, matching orientation and degenerate-image
ambiguity handling. The original8 vs4+4 restart check gives1894 exact comparisons;
it precedes the balanced-data extension, whose complete-data identities are checked
separately.6035 independent audits recompute marginal/pair/flip accuracies, confusion
matrices, cross entropy and displacement metrics, verify source/model identities,
initialization and paired sampling, and check three actual-Claude receipts. Raw-pixel
alignment oracle is100% with no ambiguous samples on these measured populations;
uniform-image unit fixtures correctly remain ambiguous rather than being filtered.

All21 reports are structurally verified and the comparison plot inspected visually;
browser interaction is not validated. Debug artifacts include logits/labels/pair
metadata, appearance/temporal/correlation maps and PCA inspection. Source code snapshots
and original failed results are preserved. No images/measurements sent to Claude.

Next proposed bounded question: why do matched512-update fits vary across seeds on
the fixed balanced population? Diagnose training fit, margins and initialization or
optimization before increasing model size, adding larger kernels, or claiming
long-range/natural-video motion. This is a proposal, not an additional run in this turn.

### Commands

```bash
# Fresh supervised temporal-feature fit; supply seed/source explicitly when pairing.
.venv/bin/python -m experiments.video_order --mode train --balanced-training \
  --seed 7501 --steps 512 --output runs/my_balanced_order

# Opt-in matching-feature comparison uses --mode correlation.
# Separate frozen-model challenge expects the12 original arm directories.
.venv/bin/python -m experiments.video_order --challenge-models runs/video_order_v1 \
  --output runs/my_order_challenge
```

The source image and reconstruction-trained temporal checkpoint are explicit inputs
(`--source`, `--temporal-source`). For7502, use the corresponding7402 temporal source.
Reports and checkpoints record both hashes. Exact Run resume requires unchanged
code/data/settings; original run snapshots preserve the earlier recipe versions.
