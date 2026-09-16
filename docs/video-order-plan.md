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
