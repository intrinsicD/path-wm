# Training curriculum proposal — 2026-09-07

Status: planning only. No datasets were changed, archives restored, dependencies
installed, training launched, or new model results produced in this turn.

The user requested a curriculum using existing local datasets, developed with
Claude. The working priority is useful Paddle/PushT control, with general visual
pretraining evaluated as a separate experiment. A general-vision-first preference
question remains unanswered; the alternate ordering is explicit below.

## Recommendation and stage order

Train **E and D jointly**, not sequentially. For task-specific perception, also
train the numeric head H jointly: E/D/H. Once its selected snapshot passes the
declared perception checks, freeze E/D/H, train U/R on real sequences, freeze U/R,
compute training-only latent statistics, train P1, then train P5 only if P1 passes
the existing prediction/copy gate. Finally evaluate control.

Generic image pretraining is a different stage: COCO has no Paddle/PushT pose
labels, so train E/D with reconstruction only; initialize a fresh task-specific H
and jointly adapt E/D/H on labelled task data afterward. Do not fabricate task
labels, put zeros in place of unavailable labels, or permanently freeze an
unvalidated generic encoder.

Paddle already has accurate perception (test coordinate MAEs about
0.071/0.076/0.092 pixels). Reuse that frozen checkpoint for its existing memory
and predictor follow-up. PushT is the perception bottleneck: its selected bounded
reference has roughly 49/58-world-unit pusher errors and 45-degree orientation
error. Additional independent-pose coverage improved reconstruction while
worsening selected pose errors. More images or a better decoder alone therefore
are not established fixes.

The retained representation has 20,480 floating-point coordinates per RGB64
frame, versus 12,288 input colour values. It is a structured representation, not
a demonstrated compression bottleneck. Reconstruction quality alone cannot
establish semantic abstraction, motion sensitivity, or controllability.

## Available data and intended roles

Counts below were inventoried locally today. Metadata reconciliation is distinct
from fully decoding or rehashing all media.

| Source and location | Observed inventory | Curriculum role |
| --- | --- | --- |
| COCO: `~/Documents/datasets/train2014` | 82,783 JPGs; exact one-to-one filename agreement with instance annotations, 80 categories, 604,907 instances. Validation annotations exist, but validation images do not. | Bounded generic E/D warmup comparison. Use masks/boxes for assessment only initially. |
| Kodak: `~/Documents/datasets/kodak24` | 24 PNGs | Fixed qualitative reconstruction check; no training, selection, or representative-generalization claim. |
| Fabric stage: `~/Documents/datasets/2025_03_07_stage_with_fabric` | Two captured frame directories, multiple cameras, calibration/masks and derived Gaussian fitting outputs | Later multi-view diagnostic. Camera views and fitting products are not independent training scenes; never count all 157 JPG/PNG files as source photographs. |
| Paddle handoff archive | Previously verified 5,000/500/500 train/validation/test episodes; selected checkpoints archived | Reuse accurate E/D/H and preserved controls; continue the bounded new-U predictor question separately. Not yet restored at canonical home paths. |
| Compact PushT: `data/pusht_cchi/pusht_cchi.h5` | Existing receipt: 206 episodes, 25,650 frames; prepared E/U/P archive has 164/20/22 disjoint initial-configuration groups | First labelled perception experiment and matched adaptation population. Preserve source and prepared fingerprints. |
| Large PushT: `data/pusht/pusht_expert_train.h5` | Existing source audit: 18,685 episodes, 2,336,736 frames; pinned recovery/extraction receipts present | Later controlled expansion after adapter and cross-source overlap audits. These HDF5 counts were not independently reread today. |
| TwoRoom: `data/tworoom/tworoom.h5` | Existing source audit: 10,000 episodes, 920,809 frames | Later second action-conditioned domain. Existing LeWM support does not imply the new E/U/P task adapter exists. |
| Charades: `data/charades/raw/Charades_v1_480` | All 7,985 train and 1,863 test videos present; no shared video IDs or subject values across these sets | Optional natural-image adaptation, then a separately designed passive temporal experiment. |
| TAU: `data/tau_urban_av_2021/raw` | 12,291 matching audio/video pairs, all metadata paths present; 8,646/3,645 official train/evaluation clips, 305/126 disjoint locations | Later audio/video alignment and passive observation research. The other 20 videos are examples, excluded from training. |

A sample video from each video dataset was probed successfully. No full media
decode audit, COCO duplicate audit, new HDF5 content verification, or training
throughput benchmark was performed.

Charades has activity annotations, not the executed control commands expected by
our action-conditioned predictor. TAU supplies synchronized audio/video and scene
metadata. Neither directly supplies the current supervised U/R state/motion
targets. Using them for dynamics needs a separate passive objective and interface;
missing actions must not masquerade as a real stay action.
See the [official Charades description](https://prior.allenai.org/projects/charades)
and [TAU task and split guidance](https://dcase.community/challenge2021/task-acoustic-scene-classification).
The latter requires keeping recording locations together. COCO's public dataset
description is [here](https://cocodataset.org/).

## Phase 0 — restore reproducibility and measure the home budget

1. Restore the verified session archive into a new, separate destination first,
   using `scripts/session_handoff.py`. Preserve the existing home dashboard and
   raw runs: the restorer correctly refuses differing existing destinations.
   Verify restored file hashes, relocated source data and CPU inference bundles.
   Bind new experiment configuration to those exact restored paths; do not rewrite
   prepared manifests to relocate them.
2. Build an isolated environment using the handoff constraints or record a
   deliberate runtime migration. Home currently has Torch 2.9.0+cu128, no project
   `.venv`, and no importable h5py in the inspected Python. Work used Torch
   2.14.0+cu130. Loading success would not establish numerical resume equivalence.
3. Check the dashboard builder and actual Chromium executable. Node is present;
   `chromium` was not on PATH, which does not rule out another installed browser.
   Restore/view the archived HTML, then verify regeneration before a long run.
4. Run essential existing data/dependency/inference checks in the chosen runtime.
   Prior work's 336 passing tests are historical, not verification on this machine.
5. Profile a disposable perception run: at most 100 updates or three minutes,
   including a validation pass; report peak allocated/reserved VRAM, images/s,
   optimizer updates/s and loader time. Home is RTX 3050, 8 GiB; do not extrapolate
   RTX 4090 durations.
6. Prefer effective batch 128, using microbatches only if needed. The current
   E/D/H path uses LayerNorm and zero attention dropout, with no BatchNorm found;
   a new accumulation path still needs a meaningful full-batch gradient comparison,
   scalar-weighted loss accumulation, and one clip/optimizer step per effective
   batch. Otherwise freeze a smaller common batch across all new arms.

No architecture change, mixed precision, accumulation, or truncated recurrence
is silently introduced to solve a resource failure.

## Phase 1 — small labelled perception diagnostic

Use a new discarded run with 64 fixed CCHI **training** frames, sampled across
training groups rather than adjacent frames from a single episode. Keep the
current RGB-plus-pose objective and architecture. Cap at 500 optimizer updates
or ten minutes; record initialization and sampling identity.

Measure reconstruction globally and on pusher/block regions, each physical pose
error, orientation-vector norms and wrapped-angle errors. Plot original versus
reconstructed frames and predicted versus true positions. Log the two loss terms
and their separate E-gradient norms/alignment on a few fixed batches.

The purpose is to distinguish inability to fit observed examples from weak
generalization. Failure to fit is diagnostic, not proof of a code bug; a small
train/validation gap does not prove leakage. Investigate alignment, resolution,
capacity, loss scaling and optimization separately before any changed objective.
Any regularization-off or learning-rate diagnostic gets its own declared run.
This disposable subset never selects a downstream model.

## Phase 2 — one bounded, matched perception screen

Question: does generic reconstruction pretraining improve physical perception
enough to justify its cost, compared with spending that effort on the task or
on in-domain reconstruction?

Use the same E/D architecture and initial tensors in all arms. No imported
pretrained network, foreground-weighted objective, new layers, or augmentation
sweep. Start a fresh identically initialized H at each arm's first supervised
update. Initial single screening seed: 4107; this is development evidence.

| Arm | Warmup | Task training | Total planned updates |
| --- | --- | --- | ---: |
| A: supervised from scratch | None | 4,000 CCHI E/D/H updates | 4,000 |
| B: generic reconstruction | 2,000 COCO E/D updates, no H loss | 2,000 CCHI E/D/H updates | 4,000 |
| C: in-domain reconstruction | 2,000 CCHI E/D updates, no H loss | 2,000 CCHI E/D/H updates | 4,000 |

Use constant AdamW 3e-4, the existing decay/clip recipe, FP32, and the common
profiled effective batch. Warmup snapshots at update 2,000 are fixed endpoints;
retain validation curves but do not pick different warmup lengths per arm.
Start fresh task-stage optimizer state for B/C; A's uninterrupted supervised
optimizer is part of its task-only recipe. Use private phase-specific samplers.
The first 2,000 supervised batches must match exactly across A/B/C.

At batch 128 this is 512,000 presentations per arm, 1,536,000 total.
B/C each use 256,000 reconstruction-only plus 256,000 labelled presentations.
A uses 512,000 labelled presentations. Reusing CCHI frames does not create
independent examples.

Report two comparisons:
- **Equal total updates:** A4000 versus B/C after adaptation; record actual
  elapsed time/FLOPs proxies separately, since equal updates do not guarantee
  identical compute.
- **Equal labelled presentations:** A2000 versus B/C after 2000 adaptation
  updates. B/C also consumed warmup compute and data; this is not an equal-total-
  data or equal-compute claim.
B versus C changes warmup population with matching stage objectives and budgets.

Hard ceiling: 60 minutes of training plus validation per arm, excluding separate
report rendering. Use the preflight to decide whether the 4,000-update schedule
fits; reduce all arms to a common, explicitly frozen schedule **before** launching
if needed. Any later interrupted arm is incomplete, not a matched full result.
Do not extend a favoured arm based on its curves. No run is scheduled by this plan.

### Splits and input protocol

Preserve the exact CCHI 164/20/22 prepared split and existing 2,048 validation
frame indices. Report frame-weighted and episode/group-balanced diagnostics
separately; use no test frames for selection. Preserve all old reference results.

For COCO, create an ID/duplicate-group manifest first. Reserve approximately
90%/5%/5% of available train2014 for train/development/final internal holdout,
using seed 4107; publish exact counts after grouping. Keep exact/near duplicates
together, reviewing ambiguous near-duplicate matches. This is an internal split,
not the official COCO validation benchmark. Available validation annotations
cannot substitute for absent validation images.

For the first natural-image arm, use RGB, antialiased resize of the shorter side
to 64 followed by deterministic centre crop to 64x64. Apply identical geometry
to diagnostic masks. Document crop coverage; do not claim full-image object
coverage. Keep the task's existing image transform unchanged. No captions,
instance labels, video frames or held-out task frames enter COCO reconstruction.

### Selection, readiness and interpretation

Every 100 supervised updates, compute physical errors on the fixed CCHI
validation frame population. Preserve the original combined objective in the
ledger, but declare a **new selector for all three new arms**:

`q = max(MAE_pusher_x/8, MAE_pusher_y/8, MAE_block_x/8, MAE_block_y/8, angle_MAE_degrees/10)`.

Minimize q; break exact ties by global reconstruction MSE, then earliest update.
This uses actual wrapped angle rather than sin/cos MSE, which can change with
orientation-vector norm. Log near-zero orientation norms and do not discard them
from denominators. Save final and selected checkpoints separately and compare
both. Changing the selector does not relabel any historical run as passing.

The proposed perception readiness targets are **each position MAE <=8 world
units and angle MAE <=10 degrees**, plus inspected object-region reconstructions
and comparisons against a train-only mean-image baseline. Eight world units is
one RGB64 pixel; 10 degrees is half the current 20-degree orientation goal
tolerance. These are prospective engineering targets for this new protocol,
not demonstrated sufficient conditions for successful control. Report tails
and group errors even when means pass. Unreviewed qualitative evidence leaves
perception readiness pending.

A candidate should also improve the primary q by at least 10% against the
matched comparison for a pretraining-benefit claim; report every constituent
error and require none to worsen for adoption. This 10% threshold is a proposed
new effect-size rule, not an observed effect. If all arms fail readiness, preserve
the negative screen and diagnose before U/P expansion.

Repeat a promising comparison with seeds 4108 and 4109 under the identical
protocol before adopting a pretraining policy. Freeze candidate selection before
a final held-out assessment; repeated historical Paddle/PushT test cases are
development evidence, not a newly untouched confirmation set.

If tiny-set fitting works but group generalization fails, the next candidate is
a separately declared data/coverage or objective experiment. The large PushT
source is worth auditing at that point, not automatically concatenating now.
Verify renderer/resize, action units/timing, state layout, simulator replay,
terminal conventions and source provenance; jointly audit initial configurations
and trajectory/image duplicates across sources. Exclude any added training
members overlapping the frozen validation/test populations. More trajectories
do not necessarily add more independent starting configurations.

## Phase 3 — memory, dynamics and control

For PushT, advance only the adopted task E/D/H snapshot. Train U/R on genuine
time-ordered episodes with the existing action timing and relative warmup masks.
Start from the existing bounded protocol, profiling the complete-episode path.
No arbitrary frame reset, fake action or invented velocity label.

Preserve the Paddle original and mixed-history U snapshots. Its concrete
continuation remains a fresh P1 with the selected new U (ceiling 20,000 updates),
then P5 (10,000) only if the unchanged P1 gate passes. This tests whether the
changed memory helps prediction; its worsened readout errors remain visible.

Changing E invalidates old latent statistics, U, P and caches for that branch.
Changing U requires fresh P and observer caches. Reuse only dependencies whose
fingerprints actually match. Generic, adapted, selected and resume checkpoints
remain distinct immutable artifacts.

For each adopted observer:
1. Fit training-only latent statistics and build exact observer-state caches.
2. Train P1; compare with matched copy, holding the existing physical gate fixed.
3. If it passes, train P5 with P-to-U imagined recurrence and gradients through
   frozen U; measure each horizon, moving-object errors and decoded rollouts.
4. Evaluate real closed-loop control against the original reference, reset
   memory, random, tracker/hold and privileged/replay controls as applicable.

Action shuffling is a diagnostic, not a standalone gate: correlated expert
actions and low-motion cases can make it weak; cross-episode actions can also be
off-distribution. Add predeclared motion/contact strata and simulator-supported
matched counterfactual actions. Report them separately from the primary case
population. State readability and pretty imagined frames do not prove dynamics.

Every completed seed/evaluation refreshes and verifies the canonical HTML and
raw-data companion under the standing workflow. Report software, perception,
prediction and control outcomes separately.

## Later curriculum and general-vision-first alternative

If a general encoder/decoder is the primary goal, move the COCO E/D experiment
ahead of task adaptation. Keep the same split, bounded budget, architecture and
small reconstruction sanity check; retain the task-first arm as a transfer
control. Maintain an immutable generic snapshot and separate task-adapted
branches, since specialization may reduce natural-image reconstruction.

Then consider sparse Charades frames for a separately matched domain-adaptation
test. Sample videos first and timestamps second so long clips do not dominate.
Use official subject-disjoint train/test sets, and carve development subjects
only from training. All frames/crops from a video stay together; preserve real
timestamps and frame rates rather than assuming a fixed number.

Only after that goal is explicit design passive memory/temporal prediction with
an explicit unavailable-action representation and suitable self-supervised
objective. The present U/R trainer requires task state/motion labels and cannot
directly train on these clips. Keep passive prediction distinct from
action-conditioned intervention and control.

Add TAU audio only for an explicit audio/video goal or evidence that sound
contains information the chosen task needs. Respect its location grouping and
official evaluation split; derive development locations only from training.
Do not flatten every video into a huge frame cache or mix all sources before
demonstrating a benefit. Kodak and the fabric capture remain qualitative checks.

## Claude collaboration and its limits

The detailed project brief was rejected by automatic approval review before
execution, because it would transfer private results, architecture, dataset
inventory and hardware context. Saving that exact brief locally was also rejected.
Specific approval was requested and remains pending; neither action was retried.

The connected Claude Agent tool had no available agent types. Two subsequent
**generic methodology** consultations completed through the local Claude client,
from /tmp with a replaced generic system prompt, safe mode, tools/MCP/skills
disabled and no session persistence. They contained public dataset names and
general experimental-design questions, with no project files, local inventories,
measured results or hardware details. Raw generic responses are retained locally
under `docs/curriculum-review-2026-09-07/`.

Claude supported task-first comparison, generic-versus-in-domain reconstruction
controls, explicit labelled-exposure versus total-budget comparisons, and
immutable snapshots. Codex applied those ideas to local evidence and specified
the budgets/gates above. This is not a claim that Claude independently audited
our code or these measured results.

Codex rejected or narrowed several suggestions: one-batch fitting failure does
not prove a bug; a small train/validation gap does not prove leakage; frozen
off-the-shelf encoders are a separate architecture experiment; inverse-dynamics
readability does not establish controllability; whether in-domain warmup beats
COCO is not a logical prerequisite for all future video/audio research.

Both calls completed without tool use. Their reported API-equivalent usage is
$0.144064 + $0.154894 = $0.298958; this is receipt metadata, not a subscription
billing claim.

## Immediate next slice and implementation gaps

First restore/verify the handoff and local runtime, then implement the bounded
perception diagnostic and three-arm screen. Existing task trainers already
support supervised perception and stage dependencies. New work is needed for a
generic image-only dataset/training path, split/transform manifests, explicit E/D
initialization into task adaptation, the physical selector, and any required
gradient accumulation. Add only essential tests for label absence, split
isolation, phase initialization/RNG, dependency identity, selector arithmetic and
accumulated-gradient equivalence when implemented.

Do not run generic images through the existing supervised trainer with dummy
pose targets. Do not launch a full mixed-dataset curriculum before this thin
comparison answers whether its first added stage helps.

Related records: [project state](project-state.md),
[handoff](session-handoff-2026-09-07.md),
[PushT results](pusht-world-model-results-2026-09-07.md),
[Paddle results](paddle-world-model-results-2026-09-07.md),
[existing source audit](source-data.md).
