# Current project state

This file owns the implementation objective, active experiments, evidence and next
steps. Standing development and experiment rules live in
[experiment-workflow.md](experiment-workflow.md). Changing a model, dataset or
research direction does not replace that workflow.

## Objective and current outcome

The user requested the fixed paddle E/U/P architecture from
`world_model_codex_implementation_brief.md`, using
`world_model_design_notes.md` as historical context, plus training on PushT.
Independent Codex and Claude agents contributed implementation and review. The
user explicitly authorized committing, integrating into `main`, and pushing the
complete work so it can continue on another computer.

Both implementations execute through data, perception, recurrent memory,
prediction, planning, evaluation and export. **The requested learned-control
quality targets have not been achieved.** Keep this distinction when continuing:
passing software tests, falling losses and a smoke export do not establish useful
learned control. The original LeWM implementations and results remain preserved.

Read the [paddle results](paddle-world-model-results-2026-09-07.md),
[PushT results](pusht-world-model-results-2026-09-07.md),
[canonical dashboard](../runs/experiment_dashboard.html), and
[portable continuation handoff](session-handoff-2026-09-07.md).

## Active: overnight perception experiments (8–9 September)

The user now authorizes execution with Claude through **9 September07:00 Berlin**,
and explicitly removes albedo from scope. Follow the
[overnight execution protocol](perception-overnight-protocol-2026-09-08.md), which
supersedes the earlier proposal's non-execution status. First implement and profile
matched independent spatial readouts on the fixed deeper/exchange CNN and native
DINOv2 features, then run three paired head seeds and choose bounded follow-ups
from validation evidence. Reserve the final hour for evaluation and reporting.

Run root: `runs/perception_overnight_2026-09-08/`. Claude's initial protocol review
completed successfully using the public-only CLI brief; no private code/data/results
were exported. App heartbeat `overnight-perception-architecture-experiments`
continues this task every half hour and must be paused after the morning report.
GPU is RTX3050/8GB; starting free disk28GB. The development slice now passes16
targeted scientific checks, four tiny head fits and full-cache identity audits.
Every completed unit's dashboard passed browser verification. P1 completed all
6fits,4,000updates,40minute cap per fit (prospectively increased after
development timing; original20minute proposal was not a measured fit duration).
Coordinator: `scripts/execute_perception_overnight.py`; live handoff/receipts:
`runs/perception_overnight_2026-09-08/active.json` and `execution.jsonl`.
Do not modify its sealed training files during P1. Implement conditional stages
in separate modules/protocol amendments while the six fits execute.

The [category-accessibility probe](perception-semantic-protocol-2026-09-08.md) now
has prepared crop-visible labels (78training-supported classes), green semantic
checks and two verified development fits. Its six paired, independent CPU fits
completed through `scripts/execute_perception_semantics.py` alongside GPU P1; no encoder
or P1 head changes. Read `runs/perception_overnight_2026-09-08/semantics/execution.jsonl`
and its coordinator log for progress. Reused held-outs remain exploratory.

P1-T test macro AP is0.0742–0.0829 for the CNN and0.6441–0.6555 for nativeViT
(constant-score baseline0.0331). These are pooled-head accessibility results, not
proof that another head cannot recover semantic information. All six P1 fits
pass the mean geometry gate. On the fresh cohort, individual-case tolerance pass
is73.0–80.5% for CNN and77.9–83.2% for ViT; mean readiness is not reliable control.

The [fresh simulator protocol](perception-fresh-protocol-2026-09-08.md) has generated
and verified512cases plus40source/render calibration pairs. One fresh case is near
a CCHI training pose under the declared tolerance; none is near validation/test.
One is near the older static supplement. All cases remain included. Calibration
RGB MSE is1.58e-5 and actual/source pose drift is small. This supports testing fresh
geometry while keeping renderer differences explicit; no model results on these
cases existed at preparation.

`scripts/prepare_perception_followups.py` completed after P1: the six frozen fresh
evaluations and50-update development profiles for the
[matched encoder extensions](perception-extension-protocol-2026-09-08.md). Queue
log: `runs/perception_overnight_2026-09-08/followup_coordinator.log`; completion:
`followups_ready.json`. The exact-initial-function, gate-gradient and optimizer-group
tests pass. All three development profiles completed and passed dashboard verification;
the formal nine-fit comparison now seals and launches through
`scripts/execute_perception_extensions.py`. Do not change its sealed encoder-training
files while it runs; use separate modules for subsequent readouts and report figures.

The canonical dashboard now has per-seed capability tables and category/fresh
comparison charts, independently checked against raw pose predictions/AP scores.

The nine encoder-extension fits completed (`extensions/coordinator.log`). Their
separately sealed category/fresh audit queue
(`scripts/execute_perception_extension_audits.py`) is now testing
selected encoders with live FP32 features. Its development pooling, category and
fresh checks completed and dashboards passed. Its sealed audit files must also
remain unchanged while that queue runs.

The [decoder local-access and task-conditioning protocol](perception-decoder-protocol-2026-09-08.md)
defines four arms: duplicated final tokens, early patch tokens, raw source patches,
and early tokens with task FiLM. Three paired seeds, fixed4,000-update endpoints;
same shared typed decoder, with raw input width/preprocessing differences explicit.
Structural checks and all four50-update development runs pass, including canonical
HTML verification. The implementation is committed at `200e750`; the12formal fits
are now sealed and active through `scripts/execute_perception_decoders.py`
(`decoders/coordinator.log`, PID3268933 at launch). Do not change its sealed files.

The [read-only attention/feature inspection](perception-inspection-protocol-2026-09-08.md)
completed on512fresh cases and three frozen CNN heads. Uniformizing both attention
directions or zeroing their outputs gives0%per-case tolerance passes; baseline
retains73–80%. Explicit per-head probabilities match SDPA. Fine-from-coarse is
nearly uniform, while one coarse-from-fine head is selective. A separately
[declared directional diagnostic](perception-attention-directional-protocol-2026-09-08.md)
completed: uniform fine-from-coarse is close to baseline, while removing that
branch or uniformizing coarse-from-fine sharply harms pose. This measures reliance
after training, not a retrained architecture comparison. New PCA, RGB/mask and pose-distribution figures
are in the run's `figures/`; layout revision2 fixes two label-spacing issues and
preserves original PNGs. Images have been visually inspected and the dashboard
passed browser verification. PCA colors and decoder location maps are not
semantic labels or encoder attention.

The [geometry objective follow-up](perception-localization-protocol-2026-09-08.md)
keeps the frozen P1 encoders and head topology, adding0.001times target-to-map KL
with boundary-exact bilinear point targets. Claude reviewed the public mathematics
and requested map entropy, target-support mass and loss-magnitude checks. Essential
target/gradient tests pass. The first control failed a blanket parameter tolerance:
only the two mathematically redundant spatial-softmax biases exceeded it; other
weights matched within1.2e-7. Preserve the failed control/source. The protocol now
requires informative weights AND actual normalized outputs/maps within2e-5, with
the redundant bias differences retained separately. The invariance test passes.
Four revision2 development units wait for the D2 formal queue in
`development/localization_v2/` through `scripts/prepare_perception_localization.py`
(PID3270408 at launch). After those pass, commit and seal the six geometry fits.
The adaptive [split decoder follow-up](perception-independent-decoder-protocol-2026-09-09.md)
adds three paired fits with identical early/context inputs and fixed endpoints.
Each typed output owns its trunk, while the common optimizer/clipping rule remains.
This changes sharing/capacity and cross-domain transfer, not fully independent
optimization. Claude's public review accepted this scope and withdrew a doubled
training-compute estimate: both paths already use three task passes per update.
Actual parameters, timing and clipped fractions are reported. Its two essential
initial-function/ownership tests pass; the50-update CPU development unit is active.
After development and commit, seal it through `scripts/execute_perception_independent.py`.
GPU ordering: D2 → localization_v2 development → split trunks → formal localization.
Total declared program:36vision/geometry and15category fits. Never overlap GPU fits.

The source-reconciled [technical report plan](perception-report-plan-2026-09-08.md)
is implemented in `world_model/curriculum/perception_summary.py` and
`perception_report.py`. Partial snapshots explicitly list missing fits. The final
report must include all decoder and geometry outcomes, inspected figures and
verified HTML before the deadline. Report preview development errors (path
resolution and nested table-cell schema) are preserved under `report/` and repaired
without changing any experimental record.

The report preview now passes canonical desktop/narrow/source browser checks via
`scripts/deliver_perception_report.py` and the existing dashboard adapter. It uses
actual ordered SQL projections of Python-verified rows and marks pending evidence.
The [decoder input-reliance diagnostic](perception-decoder-reliance-protocol-2026-09-08.md)
has passed routing/context-hook tests and its five16-image CPU development
conditions, including CPU/GPU agreement and verified HTML. Implementation commit
`c2228b6`; formal CPU coordinator PID3282627 waits for all12D2 endpoints, then seals
and evaluates42conditions. It adds no fits: fixed donor local/context inputs and
neutral/flipped task conditions test fitted reliance. Keep it separate from any
retrained architecture claim.

The first paired D2 seed completed all four arms. COCO RGB MSE/IoU: late
0.02485/0.6570, early0.007721/0.6683, raw0.009391/0.6678, conditioned
0.006912/0.6608. These are first-seed outcomes, not the final architecture decision.
Fixed first-six RGB and mask panels under `figures/decoders/` have been visually
inspected; the dashboard with all five scientific figures passes canonical QA.
Its artifact is about2.97MB, so watch the unchanged payload cap as evidence grows.
The current full working-tree suite passes403tests with3opt-in browser-transport
tests skipped. Actual per-artifact browser verification passes; JUnit evidence is
`report/pytest.xml`. The failed-control CPU reanalysis also has verified HTML:
all normalized outputs/maps differ by at most4.18e-7 from original P1.

## Complete: earlier encoder investigation (8 September)

The latest [multiscale routing clarification](multiscale-routing-and-tasks-2026-09-08.md)
traces existing two-scale processing/exchange and D/U/P access. It proposes retaining
declared processed levels for spatial fusion and output queries, with richer fusion,
earlier conditioning and sequential backbone coupling tested separately. The first
package comparison remains unchanged. The earlier albedo discussion is historical;
the user has now explicitly excluded it from the active work. This was a design clarification,
with no model or training change. Two public-only Claude exchanges reconciled
query-placement, future-input, cost and albedo-metric overclaims after an initial
request timeout; the original failure and successful receipts are preserved.

The latest [dataset readiness audit and Claude review](perception-data-readiness-2026-09-08.md)
finds no immediate download necessary for the first frozen-package comparison or
existing-memory diagnostic. All 6,206 prepared Paddle/CCHI episode files are
present and the checked numerical labels/timestamps pass; COCO has 82,783 source
images and matching split identities for its prepared mask subsets. Large PushT
and TwoRoom are now present, superseding older availability notes; large PushT's
18,685 episodes have only 185 exact initial-pose groups, requiring careful grouping.
Charades/TAU support separately declared passive tasks, not assumed control labels.
A crop-aware COCO category-presence readout is proposed as conditional P1-T before
broader perception claims. Its six fits add at most 60 fitting minutes if activated,
outside P0/P1's allowance, with preparation/evaluation still needing explicit caps.
Fresh controlled histories/action branches and an instrumented software cohort
are later targeted additions. The [audit notebook](../notebooks/perception-data-readiness-2026-09-08.ipynb)
reproduces metadata checks using sequential project Python; no Jupyter kernel is
installed. Two further public-only Claude exchanges corrected label, split and
benchmark overclaims. This was a read-only audit and design update, with no model
training, inference, dataset collection or download.

The [complete encoder/decoder proposal sheet](perception-proposal-2026-09-08.md)
now consolidates the architecture, conditioning, curriculum, decision gates and
experiment budgets. Its [draft manifest](proposals/perception-program-2026-09-08.yaml)
is explicitly non-runnable. The recommended first decision is six frozen-package
readout fits (deeper/on versus native DINOv2), followed by an existing-memory
diagnostic and one evidence-selected branch. This request produced a proposal,
not new execution authorization; no model change, training or inference ran.
The [printable offline sheet](../runs/perception_proposal_2026-09-08/proposal.html)
uses the canonical report renderer and passes desktop/narrow browser and source
interaction checks. Two public-only Claude exchanges corrected timing/register
and package-attribution overclaims. Existing experiment results and the canonical
experiment dashboard remain unchanged.

The [encoder-conditioning review](encoder-conditioning-2026-09-08.md) establishes
feasibility precedents and proposes a retained spatial reference plus contextual
processing for a named consumer. E is currently RGB-only; U already queries E
features with memory. Task/prior-state conditioning and candidate-action feature
branches have different timing and target contracts. A late-versus-earlier
conditioning comparison is proposed only; no new implementation or run started.

The [decoder input and conditioning clarification](decoder-inputs-and-conditioning-2026-09-08.md)
confirms that D already consumes fine and coarse features; P already conditions
predicted features on memory. Direct D-memory access and task/query conditioning
are additional proposed routes. The design distinguishes spatial resolution,
convolution/attention, temporal evidence and output type, with explicit belief
supervision and rollout timing. No model change or new run was made.

Latest design follow-up: the [requirements-first encoder/decoder review](requirements-first-perception-design-2026-09-08.md)
recommends selecting a small downstream capability contract before an architecture
family. Different encoder and decoder families are supported by direct literature
precedents. Keep the deeper convolutional reference, reuse completed DINO evidence,
and consider a native-feature pretrained ViT with independently fitted readouts;
a small hybrid remains conditional. A second public-only two-round Claude review
corrected overclaims about architectural asymmetry, RAE width and invariance.
This is a proposal: no new model, training, evaluation or frozen budget was added.

Follow-up: the [PCA/attention audit and hybrid-architecture review](encoder-visual-audit-2026-09-08.md)
is complete on the four original first-seed encoders,256 training PCA frames and
six fixed test frames, with zero optimizer updates. Saved entropy maps reproduce
exactly and attention outputs agree within float32 roundoff. Deeper/on coarse
heads have mean entropies0.953/0.970/0.322/0.948; averaging hid one selective head.
Coarse output structure is consistent with a broadcast image summary, not evidence
that a strange PCA proves failure. The dashboard adds a compact audit panel and
passes browser verification;17 targeted tests pass. A public-only two-round
Claude review supports a proposed two-block within-scale transformer comparison
against the current reference and an extra-convolution control, followed by a
separate sequential-pyramid test. These architectures and the earlier budget/loss
control remain proposed; no additional training is running.

The user accepted the [co-designed encoder sequence](encoder-depth-scale-cowork-2026-09-08.md)
and requested reusable critical Claude collaboration. That method is now part of
[the standing workflow](claude-collaboration-workflow.md). The
[bounded diagnostic/reference and depth-by-exchange study](encoder-study-protocol-2026-09-08.md)
is complete: 33 formal runs, 110,000 updates and 12,160,000 frame presentations.
Read the [final results and interpretation](encoder-study-results-2026-09-08.md)
and [verified dashboard](../runs/experiment_dashboard.html). No jobs remain active.

Adding two residual blocks per branch with exchange enabled improves selected
validation q by 44–52% in all three paired seeds. Test orientation improves from
24.75–30.76° to 2.18–2.90°, but pusher position leaves test q at 1.07–1.41.
All 12 validation gates fail, so no compatible U/P training or control comparison
was triggered. Exchange helps the deeper stack; it worsens selected validation q
in the shallow stack. The intervention includes extra parameters/normalization
and does not establish a general advantage for attention or depth alone.

The frozen-head/DINO diagnostics also fail readiness. DINO's common foreground
readout reaches IoU 0.583 versus the always-foreground 0.324 baseline despite worse
RGB reconstruction. Deeper custom encoders remain weak on this audit; pose and
pixel reconstruction alone do not establish general representation quality.
The raw native DINO head's instability and prospective fixed-scaling correction
are both preserved. Internal-state figures show observed states, not predictions.

The full suite passes 379 tests including browser checks. The final HTML passes
browser verification at laptop/mobile widths. A post-training
[checkpoint integration repair](encoder-checkpoint-loading-note.md) preserves
depth/exchange on load/export and rejects unsupported legacy paths; all 12 trained
E/H/D outputs match the experiment factory exactly. Frozen training source is
preserved at b95906cabdad0c3df7d2ff0aa70196729b5e4c33.

Earlier recommended next decision: use deeper/on as the reference and compare a
longer unchanged budget against a prospectively calibrated position/angle objective.
The consolidated proposal above now places that comparison after the frozen-package
and memory diagnostics, conditional on a remaining geometry problem. These
follow-ups are proposed, not already launched. Additional scales, registers,
higher resolution, conditioning/fusion and software-persistence tasks remain
staged. Preserve the existing references, readiness gates and grouped holdouts;
no arbitrary architecture sweep is scheduled. This completed authorization
superseded historical restrictions below during A/B execution.

## Architecture reassessment (8 September; historical proposals)

After the completed first diagnostic, the user requested a literature-based
reassessment of explicit pose versus a more general, extensible encoder/decoder.
The [design and implementation proposal](versatile-perception-architecture-2026-09-08.md)
recommends a shared spatial representation, independent output readouts and
explicit temporal state. Pose remains one diagnostic/output; a proposed first
comparison tests current frozen E against a small pretrained E with RGB and
mask/extent readouts. Local COCO instance annotations match the prepared images;
their transformed masks still need preparation and verification. Shared query
decoding and object grouping are later candidates, not adopted components.
This review launched no training and changed no model or existing readiness gate.

The user then broadened the discussion to software use, generated outputs as
actions, internal computation and driving. The [general-agent clarification](general-agent-world-model-2026-09-08.md)
separates visual cross-scale fusion from temporal/task hierarchy and distinguishes
action generation from observation decoding. At that point no trained cross-scale
ablation existed. The study above now supplies a perception-only trained comparison;
a software action/outcome benchmark remains proposed. Historical fixed multimodal
token counts remain unadopted.

The user then requested more-depth/more-level analysis and explicit coworking
with Claude. The [completed three-exchange review](encoder-depth-scale-cowork-2026-09-08.md)
proposes a bounded readout/pretrained-reference slice, then depth {0,2} by
cross-scale exchange {off,on}, retaining the 320×64 latent layout. Extra scales,
registers and other fusion are conditional follow-ups. Claude accepted verified
corrections about existing frozen targets/P5 training, interfaces and unsupported
quantization/causal claims. That review itself ran only design/literature work;
the subsequently authorized A/B execution is recorded above.

## Complete: frozen PushT pose-head comparison (8 September)

User authorized the [bottleneck protocol](world-model-next-experiments-2026-09-08.md),
then narrowed execution to its first experiment followed by reassessment. Frozen
selected task-only A encoder, fresh linear/nonlinear/spatial pose heads, matched
2,000-update budgets and validation selection; original head is the reference.
All three fits and held-out evaluation are complete under
`runs/bottlenecks_2026-09-08/pose/experiment`; see the
[results and figures](pusht-pose-accessibility-results-2026-09-08.md). Spatial H
reduces test angle MAE 27.31°→12.98° and pusher errors, but worsens object-position
MAE to 25.67/28.08 world units. Fresh linear/nonlinear heads improve validation q
by at least 10%; all heads fail q≤1 readiness. Encoder unchanged, matched draws,
362 tests and dashboard verification pass. No jobs remain active.
Paddle and decoder-conditioning experiments are deferred. No encoder adaptation,
PushT dynamics, capacity expansion or register implementation is authorized here.

## Paddle reference and follow-up

Exactly replay-verified data contain 5,000 / 500 / 500 train / validation / test
episodes. Perception completed 10,000 updates and selected update 9,750; test
H position MAE is about 0.071 / 0.076 / 0.092 pixels. Memory completed 10,000
updates, but validation velocity MAE 0.809 / 0.545 exceeds the 0.5 target.
The original 10,000-update P1 failed its ball-y gate. A separate budget-only
continuation reached 20,000 total updates and passed the unchanged gate. P5
completed 10,000 updates. Its test five-step position MAE is
4.582 / 4.043 / 4.671 pixels, above the 2-pixel target.

The full 3,500-case comparison is complete under
`runs/paddle/continuation_v1/evaluation`. Learned control caught the first return
in 181 / 500 ordinary cases and 41 / 200 paired cases, versus random
115 / 500 and 14 / 200; reset memory achieved 136 / 500 and 40 / 200.
The learned ordinary decision median was 35.6 ms and p95 was 40.2 ms, excluding
real-frame encoding, observer update and rendering. The exported inference bundle
was verified. All original control-quality targets remain unmet.

The separate [history-start experiment](paddle-history-start-plan.md) completed
its fixed 10,000 memory updates, selected update 9,750, and preserved the original
architecture and loss. Exactly 40,000 full and 40,000 suffix sequences presented
2,809,923 observations and 13,729,615 supervised scalars. Selection equally
weights ordinary and suffix validation objectives; paired cases are diagnostic
only. Selected ordinary velocity MAE is 0.859 / 0.634, and paired velocity MAE is
4.732 / 0.810. Independent replay confirms improved cold-start position estimates, but at
matched final updates paired vx/vy error and ordinary validation loss worsen.
All 80,000 sampler draws and cumulative counters match the declared schedule.
The intended memory fix is not established; fresh P1/P5 with this U are not
trained before home continuation. Existing P checkpoints depend on the original U and cannot be
silently reused with the new observer.

## PushT reference and coverage experiment

The official compact CCHI source is prepared at
`data/pusht_world_model/cchi_v1`: 206 episodes, 25,650 frames, and disjoint
164 / 20 / 22 initial-configuration groups. All prepared pixels, actions and
labels match the source; no exact cross-split image or trajectory duplicates were
found. Train-only motion and action-offset RMS scales are fingerprinted. The
separate model uses H6 / R11 and absolute target-XY actions. Corrected two-update
CPU smoke exercises every stage, all six controllers/oracles and export. Its
learned final success is 0 / 2; replay reachability is 2 / 2.

The bounded GPU reference completed perception 1,000 updates (selected 200),
U 1,000, and P1 2,000. It failed the strict P1 pusher-position gate: MSE 4,164.09
versus copy 4,090.43. P5 and full test control were correctly not run. Selected
perception validation angle MAE is 45.19 degrees, and its reconstruction is
largely background. Read-only diagnosis found a pose-generalization gap and no
demonstrated source alignment or circular-angle arithmetic bug.

The separate [coverage experiment](pusht-perception-coverage-plan.md) completed
1,000 perception updates with unchanged initialization, model, objective,
optimizer and fixed 2,048 validation indices. It mixed the same 20,493 source
training frames with 20,493 independent simulator poses. Actual presentations
were 63,954 source and 64,046 supplement frames. It selected update 500: image
MSE improved from 0.006884 to 0.002817, but all four position MAEs worsened and
angle MAE rose to 46.21 degrees. The coverage intervention is not adopted; no U/P
expansion is justified by this selected observer comparison. Matched final and
fixed-frame diagnostics remain separate from the primary validation population.

## Collaboration, provenance and continuation

Claude completed independent physics, learning, code, memory-diagnostic and
PushT-design reviews. Their receipts report $14.63338225 in API-equivalent
usage, not a subscription billing claim. Exact reviews are under
`runs/paddle/collaboration` and `runs/pusht_claude_design_review.json`.
A further read-only Claude perception review did not execute: automatic approval
review rejected the additional external transfer pending specific payload
approval. The scoped prompt and rejected-call record are preserved; do not retry
that transfer without the requested approval. Local Codex diagnosis continued.

The handoff includes exact datasets, source HDF5, supplement, selected/resume
checkpoints, raw evidence and offline HTML in verified split archives. Rebuild
only omitted caches. Preserve prepared manifests verbatim after relocation and
pass the new source HDF5 path explicitly for source verification. The final
handoff document and archive manifest record completed stages, environment,
verification and remaining choices; do not infer success from checkpoint names.

Final software verification: **336 tests pass**, including all three installed-browser
checks. The canonical dashboard passes package/data/source-interaction and
1440 / 390-pixel browser verification, with 49 datasets, 55 charts and five
embedded image blocks. See the final raw test log and receipt under
`runs/paddle/collaboration`. The actual separate-directory restoration passed all 13,870 file hashes, full
paddle/PushT/supplement verification and four CPU inference bundles. The package
is 212,389,903 compressed bytes in five parts. Its receipt establishes continuity,
not model quality. All experiment and packaging writers are closed.

## Home curriculum execution (7–8 September 2026, complete)

The user accepted the [curriculum](training-curriculum-2026-09-07.md) and requested
execution, thorough evaluation and internal-state figures. All scheduled training
and evaluation are complete, including all 3,500 Paddle controller episodes.
Read the [results](training-curriculum-results-2026-09-07.md),
[execution record](training-curriculum-execution-2026-09-07.md),
[artifact navigation](../runs/curriculum_2026-09-07/README.md) and
[verified dashboard](../runs/experiment_dashboard.html).

Full archive/source restoration passed. COCO preparation decoded 82,783 RGB64
images and grouped splits of 74,501 / 4,136 / 4,146. CCHI retains all verified
20,493 / 2,651 / 2,506 frames in the original episode groups. The 100-update
profile fit batch 128 in 1.39 GB; no common budget reduction was necessary.

Seed 4107 completed the matched 4,000-update A/B/C screen. Selected validation
q is 3.009 / 3.546 / 5.159; all fail the q ≤1 readiness threshold. Task-only A
wins at equal total updates. Both warmups worsen every selected physical
validation error relative to A. As predeclared, no confirmation seeds or PushT
U/P expansion were triggered. A's full-test angle MAE is 27.31° versus 5.51°
on training frames. Sixteen selected/final/validation/diagnostic inspections,
train-only PCA/probes, attention, region controls and all raw errors are preserved.
COCO warmup reconstruction MSE is 0.00607 on its 4,146 internal test images;
selected task adaptation raises it to 0.27220. The generic checkpoint is retained.
The [8 September interpretation](curriculum-interpretation-2026-09-08.md) explains
the combined E/D forgetting and proposes domain/pose controls, frozen-feature
readouts, replay and application adapters. During the following voice discussion,
the user authorized the [frozen-encoder decoder recovery](decoder-recovery-plan-2026-09-08.md)
diagnostic. All three arms and full paired evaluation are complete; see the
[recovery results](decoder-recovery-results-2026-09-08.md). Frozen-encoder decoder
refitting restores COCO MSE to 0.006104667 (0.53% above original warmup), while
PushT reconstruction worsens 8.72 times. E/H remain exactly unchanged. Matched
fresh decoders leave a 26.33% COCO gap between adapted and original encoders;
this is one-seed, fixed-budget evidence, not proof of information loss.
All 359 tests and the canonical dashboard browser checks pass. Claude's
[public literature review](world-model-literature-2026-09-08.md) is complete.
Task conditioning, skips, geometry changes and registers remain proposals.
No further training is running or queued.

Paddle uses frozen E/D/H and the preserved mixed-history U/R. New P1 completed
20,000 updates, selected 19,750 and passed the original copy gate, with only a
0.00105-pixel ball-y margin. New P5 completed 10,000 updates and selected 9,250.
Five-step test position MAE [5.849, 4.048, 4.801] misses the two-pixel target;
real-memory velocity MAE [0.869, 0.626] misses the 0.5 target. All-frame H passes
its one-pixel mean-error target over 17,831 test observations.

Full learned control improves from 181/500 to **345/500 ordinary** and from
41/200 to **115/200 paired** first interceptions. Matched-case improvements
are +32.8 pp [27.4, 38.2] and +37.0 pp [28.5, 44.5] in descriptive bootstrap
intervals. New reset-memory control achieves 183/500 and 28/200; tracker achieves
369/500 and 0/200; privileged control achieves 444/500 and 200/200. Both learned
90% targets remain unmet. Case identities match exactly, while the historical
RTX 4090/Torch 2.14 and home RTX 3050/Torch 2.9 runtimes differ. One tracker
trajectory changes despite identical aggregate outcomes. These are not
training-seed intervals or a pure runtime-matched estimate of U's causal effect.

On 100 identical-current-frame history pairs, U/R gets direction right in
171/200 members and both directions right in 72/100 pairs, versus 100/200 and
0/100 for a training-fitted frame probe. Velocity magnitude remains inaccurate.
Memory heatmaps/PCA, attention, actual/imagined rollouts and all paired readouts
are visually inspected. Imagined balls can fade even in a successful control case.

All **355 tests pass**, including the three installed-browser integration checks.
The canonical dashboard passes package/source-interaction and 1440/390-pixel
verification. A final nested-scope path failure was repaired with a regression;
all 3,500 completed cases were reused. The dashboard is explicitly scoped to
this curriculum; earlier ledgers and reports remain preserved. All writers are
closed. No further training, external publishing or private Claude transfer was
launched. Further experiments need a new bounded protocol; no current result
reopens the completed training budgets.

## Preserved LeWM objective and evidence

Build a fresh modular implementation of the published LeWM baseline. Demonstrate
learning and control on PushT and support multiple explicit dataset protocols.
The working baseline is preserved after the user-authorized sample-efficiency research and paired training screen; the full reproduction schedule remains incomplete.
The earlier reset removed the previous implementation and results; the retained
ideas and downloaded source datasets carried forward. The reusable development
harness is now recovered separately from that discarded implementation.

Reuse the pinned authors' baseline components. Document differences from the
reference recipe and distinguish smoke checks, subset training, reproduction
training and benchmark evaluation. Baseline-specific tests cover episode
alignment, causal action timing, reference computations and gradients,
normalization, rollout and checkpoint integrity.

## Completed: paired projection-count training screen with Claude (2026-09-07)

All 12 full-source runs completed 1500 updates, and all 48 planned control outcomes are recorded under `runs/projection_training_2026-09-06`. No training, evaluations or finalization jobs remain queued. See the [final report](projection-training-2026-09-06.md), [frozen plan](projection-training-plan-2026-09-06.md), and [verified dashboard](../runs/experiment_dashboard.html). The short cosine schedule is not full reproduction or an independent-data generalization test.

Increasing resampled SIGReg directions from 1024 to 4096 produced modest final calibrated gains: +2 percentage points on each dataset, equivalent to one additional success out of 50 on average. PushT final mean success is 2.67%/4.67%; TwoRoom is 89.33%/91.33% (1024/4096). Early calibrated effects are −0.67 pp on PushT and +2.67 pp on TwoRoom, with TwoRoom paired differences ranging from −8 to +12 pp. Early saved-buffer TwoRoom control is worse with 4096 in every seed (mean −19.33 pp); final saved-buffer effects are 0 pp on PushT and +3.33 pp on TwoRoom. No large learning-speed or independent-data reduction is demonstrated; 4096 remains optional rather than an adopted sample-efficiency fix.

At 750 updates, the same fixed training-only BN calibration adds 13–33 successful TwoRoom cases out of 50 across all six checkpoints without changing parameters. PushT calibration effects remain mixed. Layer attribution and a prospective normalization correction remain open. All 732 logged pre-clip training gradient samples exceed the threshold, but neither every-update clipping frequency nor early-training conditional gradient variance was measured. Final saved-buffer prediction/copy improves in all three TwoRoom and two PushT pairs. These diagnostics do not substitute for control.

Each arm processes 96,000/192,000 distinct optimizer windows at 750/1500 updates, already covering nearly every source episode by the final checkpoint. Exact frame-row reuse at 1500 is about 1.14× on PushT and 1.42× on TwoRoom. Calibration and full-source action normalization add population access; the random-window protocol can share source episodes/frames across splits. The source cases do not establish unseen-configuration or long-goal generalization.

Claude supplied private-RNG implementation/test design and the paired-statistics implementation through two neutral MCP calls ($1.132410 reported API-equivalent). Integration retained the exact pinned loss expression and legacy behavior. All six pairs match initial weights, populations and global CPU/CUDA RNG at both checkpoints; all 24 calibrated clones change only BN buffers. The 13 protected scientific-code/reference/case inputs remain unchanged. All four preselected qualitative panels are verified and inspected.

Final verification: 103 tests pass, including three browser checks. The canonical dashboard passes source interaction and desktop/mobile verification, with all 12 points in each paired chart at both widths. A browser-output failure was repaired by bounding repeated provenance text while preserving every exact value and full source identity; completed science was reused. The final payload is 2,516,589 bytes. Fixed-scale control and discrete gradient/prediction PNG/SVG figures and actual native charts were visually inspected. Original checkpoints and reproduction configurations remain preserved.

## Earlier: sample-efficiency investigation complete (2026-09-06)

The user requested literature research with Claude, batch/gradient inspection and a plausible route to fewer examples. See [the evidence and proposed experiment](sample-efficiency-2026-09-06.md), [predeclared plan](sample-efficiency-plan-2026-09-06.md), and [verified dashboard](../runs/experiment_dashboard.html). No training or diagnostic work remains active. All original checkpoint hashes and model parameters/buffers are preserved.

Across the first 100 batches, PushT/TwoRoom average 127.50/127.23 distinct episodes and 511.93/511.89 frame rows per 128-window batch: within-batch duplication is negligible. Across the processed prefix, encoded source frames repeat 3.18×/2.49×. PushT has 185 exact first-frame state[:5] configuration groups; this is not full-trajectory deduplication.

Four frozen training batches per dataset show more consistent total gradients at batch128 than32. SIGReg's mean gradient norm exceeds prediction's on the projector by 2.84×/1.94×; encoder terms are mostly near-orthogonal, without severe input/target branch cancellation. Float32 controls support that conclusion. Classical critical-batch estimates are invalid for this coupled BatchNorm/SIGReg objective.

On fixed data, random SIGReg directions produce about11× the gradient variance of dropout alone. A predeclared4096-vs1024 projection probe reduces conditional variance **3.96× on PushT /4.13× on TwoRoom**, with zero new examples or optimizer updates. This is a measured mechanism, **not evidence of faster training or higher control success**. The first proposed experiment keeps latent192/batch128/λ0.09 and tests lower-variance sketches on paired learning curves; a separate configuration-disjoint, coverage-preserving subset experiment tests independent-data efficiency. Literature argues against assuming a32/64-dimensional latent is a free improvement.

Claude contributed mathematical analysis, generic gradient-code implementation and experiment design through two neutral-directory MCP calls ($0.649057 reported API-equivalent total); no repository transfer was needed. Eight primary papers informed the final report. One stochastic diagnostic was rejected and preserved after replicate-count and numerical-reduction defects; corrected results are used. Dashboard payload failures were repaired without changing raw experiment outputs. Final verification:84 tests passed including3 browser checks; canonical HTML verified at1440/390 with31 charts,8 tables and5 image blocks.

## Completed: diagnostic follow-up and one-epoch PushT continuation (2026-09-06)

The accepted work with Claude implementation co-work has completed. See the
[results report](tworoom-followup-2026-09-06.md), its preserved plan/execution log,
and [verified dashboard](../runs/experiment_dashboard.html). No training or
experiments remain active or queued. Completed parents and references are immutable.

TwoRoom 4074's training-only calibrated diagnostic clone reaches **48/50** primary
goals (original 14/50, released 42/50), or **44/46** initially unsolved goals
(original 10/46, released 38/46). It reaches **13/50** longer-goal CEM 10 cases
(original 0/50, released 5/50). Only six BN buffer tensors change, with no parameter
updates. This establishes a large buffer intervention effect on these cases; the
responsible layer and training-time mechanism remain unresolved.

PushT continues from 8404 to **13933 updates** in a separate directory, adding 5529
updates in 6507.80s within the 7800s cap. It preserves optimizer/RNG state and the
full139330-update schedule. Frozen control improves **17/50→30/50**, versus 45/50
released. Effective rank improves 38.71→48.42, mean probe R²0.5462→0.6005, and
h8 rollout/copy0.2906→0.2106. The final saved checkpoint SHA256 is
151b356addea1a9bc7c939fcd102986ed1e7dfca693212463308b8456b3ca4f0.
Validation/status match13933. One full-batch epoch processes1783424 windows;
the ten-epoch reproduction remains incomplete and unscheduled.

PushT calibration gives11/50 at 8404 (original 17/50) and 29/50 at 13933 (original 30/50),
despite better prediction error. Preserve both negative control comparisons;
calibration is not adopted into the training baseline. The first frozen final
PushT case still fails for both local variants, while released weights succeed.

Claude authored the layerwise calibration utility and three essential tests via
a restricted MCP task with no repository access ($0.478356 reported usage).
Broader repository-sharing tasks remain blocked by automatic approval review
pending the specific transfer approval already requested. Codex integrated and
validated the implementation, ranking, continuation and reporting. A supervisor
exit was recovered without restarting the active trainer; missing original exit
status is recorded explicitly, with final checkpoint/status/HTML verified.

**76 tests pass**, including three browser checks. Final canonical QA passes with
31 charts, 8 tables and 4 image blocks. Actual390px/1440px captures, all four new
rollout panels and four final internals panels were inspected. Mobile control bars
are visible, and forked runs retain released panels only on matched recorded
inspection populations. Every numeric record remains indexed.

## Deferred LeWM follow-up

Proposed follow-up: isolate projector versus prediction-projector BN effects on preserved TwoRoom clones using a predeclared paired protocol, then test the identified normalization correction prospectively. Keep saved-buffer results and PushT failures visible. A smaller coverage-preserving dataset must be compared with a size-matched random subset under a new configuration-disjoint holdout, with normalization fitted only on training groups. Do not interpret source-window counts as independent-data efficiency. No follow-up is launched or queued.

Preserve the PushT 13933 saved-buffer checkpoint as the continuation reference. Further training requires a separate bounded experiment decision. Full paper reproduction and passive TAU/Charades extensions remain deferred.

## Earlier: overnight implementation and evaluation complete (2026-09-06)

The authorized work with Claude via MCP, correctness/performance repairs, bounded
training on both full source datasets, control, internals and dashboard review
has completed. See [the results report](overnight-2026-09-06.md) and its preserved
plan/execution log. All completed checkpoints and references remain immutable.

PushT completed **8,404 updates** (60.31% of one epoch), reaching **17/50** frozen
goals versus 0/50 at step 375 and 45/50 for released weights. Its prediction/copy
ratio is 0.2274, full-window rank 38.71/192 and mean state-probe R² 0.5462.
TwoRoom completed **4,074 updates** (79.28% of one epoch): **14/50** primary goals
versus 42/50 released; conditional on initially unsolved goals this is **10/46
versus 38/46**. Longer goals yield **0/50 versus 5/50**. TwoRoom position probes
are strong (R² 0.9939), but prediction/copy is 3.9480 and eight-step rollout/copy
is 3.2835. Representation readability does not establish useful dynamics.

The paper specifies TwoRoom history one and 10 CEM iterations, whereas released
weights/configuration use history three and 30 iterations. Our 100/150 goal/budget
check retains released history and is not full paper reproduction. A matched
10-iteration check kept exactly the same successes (local 0/50, released 5/50)
with 2.96–2.97× less case execution time. Existing frozen evidence is unchanged;
no solver default or scientific pass threshold was adopted.

Validation RNG/resume identity, final-step evidence, TwoRoom simulator/evaluation,
source-loader reads, dataset-aware inspection and dashboard row/publication
integrity are repaired. **64 tests pass**, including two browser checks. Every
completed experiment refreshed the canonical HTML. Final canonical QA passes;
desktop and mobile captures and all three frozen rollout panels were inspected.
Claude completed one review and contributed three essential tests; an additional
review was blocked before launch by automatic approval review pending specific
data-transfer authorization.

## Earlier: requested ten-minute training and recheck complete (2026-09-06)

After inspecting the user's updated dashboard, a fresh source-data run completed
375 updates in 578.48 seconds (9 min 38 sec), using seed 3072, batch 128,
full-batch activation checkpointing, bf16, float32 validation and internals logging.
This was a separate short cosine schedule with three warmup updates, not the
139,330-update reproduction. It processed 48,000 windows (2.7% of one epoch).

Matched control: new checkpoint **0/50**, released **45/50**, replay **50/50**,
stationary **0/50**, with no initial successes. Prediction beats copying by 12.18%
and shuffled actions by only 1.38%. On the same 512 validation windows, effective
rank is 11.51/192 versus 88.72 for released weights; mean probe R² is 0.0437 versus
0.720; eight-step rollout/copy ratio is 0.684 versus 0.114. The first frozen
simulator rollout moves away from the block. Learned control remains unestablished.
See [the full short-run report](pusht-source-10min.md).

The inspector now restores and verifies random-window Subset indices, and records
standalone prediction alongside internals. Dashboard PNG accumulation exceeded
the canonical payload limit; a tested repair selects the focus run's earliest/latest
inspections and a released inspection on the same population. All numeric records
remain indexed. Final canonical desktop/mobile/source QA passes (29 charts,
6 tables), and 49 essential tests pass, including two browser checks. The user's
per-scalar curves and internals instrumentation are preserved. All checkpoints
remain unchanged by evaluation, including old pilot/reference artifacts.

## Earlier diagnosis and source-scale preparation

The dashboard browser path is repaired. Real-time CDP transport preserves the
canonical probes; two reader CSS fixes handle scrollbar width and narrow-screen
legends. Canonical desktop/mobile rendering, source interaction and exact payload
verification pass. The full essential suite passes 32 tests, including two
explicit browser checks. See `viewer/DESIGN.md` and the dashboard receipt.

The saved broader-pilot checkpoint reaches 0/20 frozen training-set goals;
recorded replay reaches 20/20 and stationary actions 0/20, with no initial
successes. The earlier held-out result remains 0/20. On eight matched rollout
cases the pilot selects unsuccessful plans in 8/8, while released weights select
successful candidates in 7/8. Raw 320 model/candidate records, exact values and
five-step rollout curves are indexed in the dashboard. All checkpoints remain
unchanged. See [the diagnosis report](pusht-control-diagnosis.md).

A discarded real-batch native comparison isolates bf16 gradient drift from
encoder batch slicing (6.124% relative L2 on the four-sequence probe). Full-batch
activation checkpointing reduces that measured difference to 1.36e-7 without
changing the objective or full-batch statistics. A single batch-128 backward
fits the local GPU (3.14 GB peak allocated), with finite gradients and zero
optimizer steps. This does not establish the cause of the pilot control failure.

The [source-scale reproduction configuration](../configs/reproduction/pusht_source_scale.yaml)
and [protocol](pusht-reproduction.md) are prepared and data-only validation has
completed: 1,783,548 train / 198,173 validation windows, full-source unbiased
normalization, ten epochs / 139,330 updates, 1,393 warmup updates, and 50 frozen
source control goals. The candidate uses full-batch activation checkpointing,
not the original pilot's encoder slicing. Ordinary episode-split configurations
remain supported. The raw preparation manifest and frozen indices are under
`runs/reproduction/pusht_source_scale_preparation/`.

## Dashboard repaired and model internals measured (2026-09-06)

The canonical reader applied only filters that target every dataset, so all
per-section selectors were silently inert and every chart mixed all runs on a
categorical axis. Charts now show one named selection fixed at build time
(`python -m viewer.dashboard --focus <training run>`), with one panel per
training scalar, a log10 validation-ratio chart, rollout error in the absolute
view, checkpoint-internals charts, an all-checkpoint spectrum, error-versus-
horizon ratios, embedded PNG panels and exact tables. Browser QA passes; the
no-JavaScript fallback shows charts beyond the first eleven as tables (builder
SVG budget). See `viewer/DESIGN.md`.

Read-only inspection of the four saved pilot checkpoints and the released
weights on the pilot's 512 held-out windows is recorded under
`runs/diagnostics/pusht_internals/` and summarized in
[the internals report](internals-report.md). The pilot latent is dimensionally
collapsed (effective rank 14 of 192 versus 66 for released weights), physical
state is weakly linearly readable from it (mean held-out probe R² −0.06, some
position targets weakly positive, versus 0.78–0.97 for released object pose), the predictor is less
sensitive to actions than to state (ratio 0.91 versus 2.6), and multi-step
prediction only modestly beats copying (0.65 of copy error at horizon 8 versus
0.10). Gradient norm is dominated by the encoder at every trained checkpoint.
These are descriptive measurements, not gates, and do not identify a cause.
Checkpoint hashes are unchanged. An opt-in `introspect: true` training key
records the scalar subset at every validation step; existing configs are unchanged.

## Earlier next-work assessment (superseded by the active follow-up above)

At the overnight handoff, no further training was running or queued.
Preserve the 8,404-update PushT, 4,074-update TwoRoom and earlier checkpoints.
The full ten-epoch reproduction remains unscheduled. The overnight prefixes use
the full learning-rate schedules, but neither reaches one complete epoch.

The next experiment should distinguish TwoRoom dynamics/evaluation calibration
from planner behavior using matched, discarded diagnostic clones and preserved
checkpoints. Position is already strongly recoverable; do not attribute failure
to missing position information or the Gaussian prior without causal evidence.
Any new comparison needs a declared case population, history, goal/budget and
CEM iteration count. Random-window validation measures source interpolation;
unseen-configuration generalization requires a separate group-held-out protocol.
Keep internals enabled. Passive TAU/Charades extensions remain deferred.

## Earlier validation
At broader-pilot completion, the modular LeWM implementation passed 13 CPU tests.
Local and upstream control evaluators agree on ten paired subset cases with
released weights. Cached random-initialized PushT training completed all 400
updates in 665.98 seconds on the same 8-episode prefix (7 train, 1 held out).
Its untouched checkpoint beats copy/shuffled-action prediction controls in
float32 and reaches 1/5 held-out control goals; replay reaches 5/5 and stationary
actions 0/5. The checkpoint SHA256 still matches its pre-crash control manifest.
BatchNorm recalibration still improves predictions on discarded diagnostic
clones; it is not adopted as a baseline change. TwoRoom completed 400 updates on
its earlier 32-episode prefix; its full source is now verified and extracted
(10,000 episodes, 920,809 frames). Full PushT source is verified and extracted
(18,685 episodes, 2,336,736 frames). The released checkpoint reaches 45/50 goals
(90%) with full-source normalization through the pinned upstream evaluator.
The wrapper records unseeded reset arguments explicitly; it does not alter
upstream randomness. All recovery/evaluation processes have completed.
Do not repeat the completed training run.
See docs/reference-validation.md and runs/diagnostics/reference_full_source_status.json
for current evidence and recovery status. Broader learned control and full
training reproduction remain unestablished. A full reproduction training run is not scheduled. Research extensions remain closed.

## Completed approved pilot
The user-approved broader PushT pilot completed 1,000 updates in 1,506.89 seconds
on 128 train / 32 held-out initial-configuration groups, batch 128, seed 3072.
The full source has 185 distinct initial configurations; one trajectory per
selected group prevents initial-configuration variants from crossing the split.
The final untouched float32 checkpoint has prediction MSE 0.187671 versus copy
0.213567 and shuffled actions 0.203553, on 512 fixed held-out windows. All three
trained checkpoints (250/500/1000) reach 0/20 on identical held-out control goals;
released weights reach 17/20, replay 19/20, stationary 0/20, with no initial successes.
Final precision and discarded BatchNorm-clone differences are small (about 3–4%);
no baseline change was adopted. All checkpoint hashes remain unchanged.
See docs/pusht-broader-pilot.md and the run's pilot_summary.json. All pilot
training/evaluation processes completed. Do not repeat the completed run. Targeted rollout/control diagnosis is completed above;
research extensions remain deferred. The bounded pilot is complete, while the
learned-control baseline gate remains unmet.
