# Overnight perception architecture experiments

User authorization: 8 September 2026, execute the proposed encoder/decoder
experiments with Claude and report by **9 September 07:00 Europe/Berlin
(05:00 UTC)**. Albedo is excluded. This authorization supersedes the older
proposal's non-execution status; that document remains historical design context.

## First vertical slice and scientific interfaces

Question: which of two existing frozen representation packages supports geometry,
generic RGB reconstruction and foreground masks through appropriately normalized,
independent spatial readouts? This compares two checkpoints plus preprocessing and
feature access, not architecture or pretraining in isolation.

* CNN: deeper/exchange-on seed 7107, checkpoint
  `runs/encoder_study_2026-09-08/factorial/seed_7107/deeper/best.pt`;
  SHA256 `aecc2a7abe545c673324a9ea53094729c69f00ed6d56c5dae036e52ffa820345`.
  Processed post-exchange maps are 16×16×64 and 8×8×64.
* ViT: pinned local DINOv2-S/14, native normalized patch tokens 16×16×384;
  derive an 8×8 grid by 2×2 average pooling. The second grid adds no independent
  evidence. Reuse the existing official source/weight identity checks. Same
  source RGB64, bicubic 224 antialiased resize, no extra crop, ImageNet normalization.
* Each independent RGB/mask/pose head owns per-level LayerNorm and 128-channel
  projections. Upsample the coarse map to 16×16 and concatenate. No shared
  trainable parameter between output heads, no BatchNorm, no encoder updates.
* RGB/mask: 3×3 fusion to 128, one two-convolution residual block, nearest upsample
  twice with 64/32 channels, RGB sigmoid or single mask logit.
* Pose: fusion to 128, two residual blocks; two spatial distributions decode
  pusher and block body-origin x/y. Their coordinate basis is a declared uniform
  [0,1] grid, not a claim that patch receptive-field centers equal world positions.
  A separate learned spatial pool and MLP predicts sin/cos orientation. Do not
  derive orientation from the two location outputs. Targets are x/y divided by512;
  angle evaluated by wrapped atan2(sin,cos); report near-zero orientation norms.

Essential tests precede implementation: independent gradients; paired common
weights despite different source widths; coordinate order and boundary support;
cache row/label identity rejection; selected checkpoint and exact resume behavior.
Development uses tiny explicit populations and updates. Commit plan/tests, then
the working development slice with measured timing and verified dashboard.

## P1 formal protocol (freeze after development profiling)

Three paired head seeds 9107/9108/9109, two frozen encoders, 4,000 updates per fit.
Batch64 =32 COCO +32 PushT frames, sequential microbatches of32. Independent AdamW
heads, lr3e-4, weight decay1e-4, norm clipping1 per head. FP32 compute, TF32off;
FP16 feature caches only with measured relative MSE≤1e-6 and live alignment checks.
Same source draws and shape-compatible trunk initialization for paired encoders;
projection initialization uses its own RNG stream. Log per-head gradient norms.
RGB loss weights the two domains equally; mask valid-pixel BCE uses COCO only;
pose normalized six-coordinate MSE uses PushT only. Independent gradients avoid
cross-head loss competition.

Populations: existing grouped CCHI splits (20,493/2,651/2,506 frames), audited COCO
mask subsets (4,096/512/512). Validate all validation frames every100 updates.
Select all heads at the minimum validation q, with earliest exact tie; also report
the endpoint. q=max(each coordinate MAE/8 world units, angle MAE/10degrees),
readiness q≤1. Test data never selects checkpoints. These reused held-outs are
exploratory, not a fresh confirmation set. Report per-seed deltas; three head seeds
do not measure variation in encoder training or pretraining.

The initial20-minute fit cap is superseded prospectively by the development
profile below: **40minutes per formal fit**, including validation. A timeout produces a stopped run
with its exact last checkpoint/update, not a completed 4,000-update result.
Report all RGB MSEs, mask BCE/IoU/Dice and empty/full/train-mean baselines, position
and angle errors, group-balanced errors, clipping rates, runtime/VRAM and parameters.
No universal RGB gate: retention is a separately paired diagnostic. A package
recommendation needs consistent seed evidence, not a retroactive threshold.

## Conditional sequence and deadline

After P1, add a crop-aware category-presence readout when its labels and dev checks
are ready: independent head, train-supported categories and valid crop visibility,
own validation AP selection, paired seeds. It is an accessibility probe, not
semantic localization or software competence. Preparation and fitting budgets are
declared in its amendment before formal execution.

Choose the next branch using P1 validation, before inspecting new test results:

* Both geometry gates fail: one matched readout/loss intervention first. Prefer a
  physical-tolerance-scaled pose loss against an unchanged longer-budget control;
  angle encoding is already wrapped sin/cos and is not automatically the cause.
* Geometry passes but dense masks fail to beat constant baselines: train fine-only,
  coarse-only and both-scale readouts on the candidate package. Post-exchange CNN
  inputs already contain cross-scale information; this tests decoder access.
* Geometry and masks pass: simple fusion versus a small top-down pyramid fusion,
  or a prospectively frozen fresh simulator cohort if the packages trade off
  geometry and semantic accessibility.
* Task-query placement requires prepared varying object queries and labels.
  Pusher-versus-block queries may reuse those explicit labels; a constant token
  does not test conditional perception. Compare against late selection from an
  image-only output, with identical query information at inference.

New trainable encoder residual/transformer depth and sequential 3-level pyramids
remain conditional on measured time and evidence of a representation limit. Do
not relabel a transformer head on frozen features as a trained encoder. Any such
branch gets its own matched reference, data/config/seed/compute protocol before
running. Fewer completed branches with sound controls are preferable to unfinished
or confounded rankings. Fresh U/P/control requires compatible new training and
the existing readiness rule; perception metrics alone cannot authorize a claim
about driving, software use or a useful world model.

Do not launch new training after **06:00 Berlin**. Reserve06:00–07:00 for remaining
evaluation, plots, reconciliation and the user report. Preserve all failures and
incomplete pairs. Bound new storage to10GB with at least6GB free; reuse verified
caches and keep selected/last checkpoints only. Never remove old reference runs.
Use the app heartbeat `overnight-perception-architecture-experiments` for bounded
continuation and morning reporting; pause it when the final report is delivered.

## Claude review and execution record

Public conceptual brief and actual Claude CLI response/receipt are under
`runs/perception_overnight_2026-09-08/collaboration/`. No private repository code,
datasets or prior numerical results were exported. Claude recommends the frozen
package comparison and conditional readout/fusion tests before new encoder depth.
We accept the prioritization; feasibility of later encoder training must be
measured, not dismissed from an unsupported timing claim. Model agreement is not
empirical evidence. Cost fields are API-equivalent estimates, not subscription bills.

### Development completion and P1 freeze

Four tiny fits completed (two seeds per package,50updates each); every extraction
and fit refreshed the canonical dashboard and passed browser QA. They use only32
training and16validation/test prefix frames per domain, so no capacity/generalization
conclusion is drawn. CNN fits took about28s and20s; native ViT fits about5s and13s
(startup/runtime variability; exact durations remain in raw ledgers). The revised path includes
training-only spatial-mean RGB/mask baselines, an atomic retained selected state,
and evaluation repair without repeating optimizer work. Exact CPU optimizer and
sampler resume checks pass; source-width-independent common initialization,
coordinate order/boundaries, independent gradients, mask semantics and existing
encoder invariants pass (16 targeted tests).

Before any formal updates, increase the maximum fit allowance to40minutes to
accommodate startup variability and full-population validation rather than the tiny
development prefixes. The target remains4,000updates for both packages; six fits
have at most4hours of fitting. This is a ceiling, not a prediction of runtime.
Cache preparation completed: CNN6.4s and nativeViT31.6s, reusing verified sources
where possible. All live/cache relative-MSE checks and frozen-buffer checks pass.

Seal the exact seven training/data module hashes, two cache manifests, protocol
snapshot, Git revision and ordered six-fit plan in `p1_execution_plan.json` before
launch. Keep the source snapshot alongside it. The full experiment runs through
`scripts/execute_perception_overnight.py`, preserving evaluated runs and stopping
for visible repair on a process/reporting failure.

Claude's second reply explicitly withdrew its unmeasured timing assertion and
causal diagnosis of a joint geometry failure. It accepted the explicit learned
coordinate basis and two-object query control. Inspect diffuse/multimodal location
maps among bad cases. PushT's T shape is not180-degree rotationally symmetric;
retain the existing2π-wrapped orientation metric, rather than inventing a symmetry
equivalence. Unknown crowd categories must be excluded in loss and AP.

Implementation modules: `world_model/curriculum/perception_heads.py` (spatial
interfaces/readouts), `perception_cache.py` (audited frozen sources), and
`perception_program.py` (training/evaluation/resume). The shared `run.py` wrapper
and `viewer` ledger adapters provide a fresh verified canonical dashboard after
each completed development run, formal seed and standalone evaluation. Preserve
raw source records and immutable formal execution identities under the new run
root; do not append experiments to the older frozen encoder-study plan.
