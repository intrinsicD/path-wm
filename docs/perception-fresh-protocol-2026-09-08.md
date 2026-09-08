# Fresh simulator perception confirmation

Freeze this cohort before evaluating either package on it. It is a new simulator
stress population, not a new CCHI test split or control experiment. No albedo.

Reuse the tested `prepare_supplement` generator with count512 and private seed
20260909 in `data/pusht_world_model/perception_fresh_2026-09-08`. Requested pusher
XY is independently uniform[16,496), blockXY[96,416), angle[0,2π). Preserve every
actual post-reset observation, including contact corrections. Ground truth is
the actual simulator body origin and pusher center, positions/512 plus sin/cos.
No outcome-based sampling or filtering. Render through the existing pinned SWM
wrapper at96pixels and its canonical64pixel AREA resize. Verify generated hashes,
target formulas and versions. The512 cases have no training role.

After generation, audit near matches against all stored CCHI train/validation/test
poses; report counts without filtering. A near match means every XY coordinate
differs by≤8world units and wrapped orientation by≤10degrees. Compare against the
previous static supplement if available and disclose whether this check ran.
Report actual out-of-image coordinates separately; uniform bounded requests do
not guarantee bounded post-contact states. The all512-case result remains primary.

Renderer calibration: deterministically select two source frames per each of the20
validation configuration groups (private seed20260910), producing40pairs. Re-render
the requested source pose and retain both source RGB/labels and actual re-rendered
RGB/labels. Log requested-to-actual coordinate/angle drift and source-to-render RGB
MSE. Evaluate both versions, so a rendering shift cannot silently be described as
an unseen-geometry failure. This calibration reuses validation cases and is not
part of the fresh512-case score.

Evaluate all six P1 pose-selected checkpoints without adaptation: image MSE and
per-coordinate/angle MAE, per-case error records and fixed q. Also report median,
95th-percentile and per-case tolerance-pass fraction; mean q≤1 alone does not imply
reliability on every observation. No new gate is defined by looking at these errors.
Preserve predictions, targets, frame IDs, feature identities and exact checkpoints.
Visualize fixed case IDs0–5 and clearly labelled worst-error examples; show the
same selected cases for both packages. Heatmap expectations and learned orientation
pool weights are decoder internals, not encoder attention or proof of causal use.

Budgets: preparation/calibration≤20minutes, all-six evaluation≤30minutes; zero
optimizer updates. Reuse source encoders/caches where verified. Do not compete for
the P1 GPU. The global reporting deadline and source-backed dashboard contract
apply. The existing supplement generator tests are reused; add silent-error checks
for target-to-world conversion and wrapped near-pose matching before implementation.
