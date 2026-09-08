# Dataset readiness and architecture decision

8 September 2026. Read-only data audit and two-round Claude design review; no training, model inference, dataset generation or downloads.

**The local data are sufficient to start the proposed frozen-encoder comparison and diagnose existing Paddle memory. They are not sufficient, by themselves, to select a broadly capable world-model architecture.** The useful additions are specific labels, controlled histories and a second application, rather than a large immediate download.

Keep the deeper CNN and native pretrained ViT as the initial alternatives, with independent RGB, foreground and geometry readouts. Add one small semantic-transfer readout before describing a candidate as versatile. Preserve the first comparison's geometry checkpoint rule and existing readiness gate; the additional test has its own explicitly separated budget and cannot select a different RGB/pose checkpoint after seeing the results.

## What is actually available

Counts below come from the current [audit snapshot](../runs/perception_data_cowork_2026-09-08/data_readiness.json), produced by [the read-only collector](../scripts/audit_perception_data.py). The [companion notebook](../notebooks/perception-data-readiness-2026-09-08.ipynb) makes the checks and calculations inspectable. An image, episode, starting configuration and video are different units; do not add their counts into a single dataset-size score.

| Data | Current evidence | Suitable use | Remaining preparation or limit |
|---|---|---|---|
| COCO train2014 | 82,783 original JPGs and prepared RGB64 frames; 604,907 instances across 80 categories; 185,316 person-keypoint annotation records; 414,113 captions | RGB, masks, category labels, human keypoints, visible object extent; true source-resolution comparisons | Category/instance/keypoint targets must follow the actual crop. Human keypoints are not PushT pose labels. Official val2014 images are absent from the inspected dataset root |
| Prepared COCO audit | 4,096 / 512 / 512 train/validation/test masks, with matching image IDs and original split membership | Immediate P1 foreground/RGB readouts | These are foreground-union masks, not instance identities. The subsets and larger internal held-outs have already been inspected |
| Prepared PushT CCHI | 206 episodes, 25,650 frames, 25,444 valid transitions; 164 / 20 / 22 configuration groups | P1 geometry, P3 objective continuation, P4 processing, P6 pusher/body queries | Preserves RGB64; raw CCHI is 96 pixels. Stored pose plus estimated backward displacement is not full simulator velocity/state |
| Prepared Paddle | 6,000 episodes, 220,754 frames, 214,754 transitions; 5,000 / 500 / 500 episode splits | Existing-memory probes; action-conditioned prediction and event-stratified checks | Stored state includes velocity. Collision records exist; controlled occlusion and matched alternative-action cohorts need explicit collection |
| Large PushT | HDF5 currently present: 18,685 episodes, 2,336,736 frames at RGB224; 185 exact initial first-five-state groups | Later within-domain data expansion and source-resolution study | Repeated configurations; audit near-start/later-trajectory overlap and CCHI overlap. Action/preprocessing conventions differ from CCHI |
| TwoRoom | HDF5 currently present: 10,000 episodes, 920,809 frames at RGB224, actions and state/outcome fields | A second action-conditioned visual environment already on disk | Existing LeWM support does not make it plug-compatible with the new E/U/P state, action or planner interfaces |
| Charades | 9,848 videos: 7,985 train / 1,863 test; all metadata-referenced videos present, no shared IDs or subject values across those sets | Quantitative activity recognition, passive temporal learning and natural-video transfer | Activity descriptions are not executable control commands; clip/video/subject splits, decoding and frame timing need a declared pipeline |
| TAU Urban Audio-Visual Scenes | 12,291 matched metadata pairs; all audio/video paths present; 8,646 / 3,645 official split rows and 305 / 126 disjoint recording locations | Audio-visual alignment/scene tasks and passive representation research | Scene labels do not supply action consequences or hidden physical state. Preserve locations and audit synchronization before use |
| Kodak and fabric capture | 24 Kodak stills; two fabric capture time directories with multiple views | Qualitative appearance checks; small multiview/geometry diagnostic | Neither is a large independent scene population. Multiple cameras and derived fitting outputs are not separate scenes |

The large PushT and TwoRoom availability statements supersede older, explicitly time-scoped notes saying their files were absent. Current checks read their HDF5 schemas, lengths and offsets, not every image or full-file checksum. Full-media decode integrity and simulator replay are not newly established.

## Integrity checks and implications

The COCO manifest assigns 74,501 / 4,136 / 4,146 frames to internal train/validation/test splits. Row overlap and overlap of the recorded duplicate groups are zero. Its 82,636 duplicate components follow the existing exact-RGB64/conservative-dHash rule, which does not guarantee detection of every transformed duplicate. All three inspected annotation sets have zero duplicate annotation IDs, orphan image IDs or missing referenced train-image filenames. The prepared mask arrays have no split, image-ID, group-ID or binary-value violations, and no image has zero valid pixels. Earlier geometric-alignment checks remain the source for RGB/mask transform correctness.

All 6,206 prepared Paddle/CCHI episode files are present. The checked state/action counts, finite values and timestamp order pass. CCHI motion shapes, finite values and initial validity masks pass. Split generation seeds are disjoint for Paddle, and recorded initial-configuration groups are disjoint for CCHI. This is evidence for using the existing prepared pipeline, not proof of all image/label alignment or adequate unseen-state coverage.

Paddle contains 14,287 side-wall, 2,039 paddle-hit and 2,018 ceiling event records across all splits. These are event counts, not independent episodes, and events can share an episode. They establish that collisions are represented; they do not establish that every collision regime is sufficiently sampled. CCHI source boundaries still have unknown termination semantics and must not be converted into invented success/failure labels.

| Finding | Importance and confidence | Smallest useful response |
|---|---|---|
| Reused held-outs | High importance for confirmatory claims; known evaluation history | Keep the first screen exploratory. Freeze genuinely fresh cases before confirmation; do not relabel an inspected subset as untouched |
| Large PushT: 18,685 episodes but 185 exact initial-pose groups | High importance for split design; directly rechecked from initial labels | Group relevant initial pusher/object/goal configurations and audit later overlap before expansion. Additional episodes may add trajectory diversity without adding starts |
| New COCO outputs are unprepared | Medium, confidence high from current artifacts | Transform per-instance labels consistently. Original-image category labels can be wrong after a center crop removes the object |
| Missing action alternatives/occlusion cohorts | High for those specific causal/memory tests; not a P1 blocker | Collect targeted histories and action branches in verified simulators, separate from current fixed tests |
| No prepared software observations/actions | High for computer-use claims; none found in the inspected roots | Build one small instrumented application; preserve full state privately as labels and use declared observable inputs |
| Passive video labels differ from control labels | High for interpreting a world-model experiment | Use a separately declared passive objective or recognition task; do not fill absent commands with a real zero/stay action |

This is a snapshot audit. No temporal ingestion drift study was needed or performed; the relevant temporal checks were within-episode timestamps, missing successors and grouping. No full video decode or audio/video synchronization audit ran. Public-pretraining overlap with the generic evaluation images remains possible.

## One additional probe makes the shortlist more informative

P1's geometry gate answers whether the current manipulation readout is ready. It does not adequately distinguish general semantic access. Foreground union also cannot distinguish a chair from a person or a tool.

Propose **P1-T: image category presence from frozen native encoder features**, using existing COCO instance annotations. Construct positive labels only for categories with surviving valid, non-crowd instance support after the exact image transform. Do not infer hidden or amodal extent from overlapping visible masks. Handle crowd/unknown labels explicitly. Audit small/partially cropped instances before freezing a visibility threshold.

Use mean-pooled spatial patch features from each declared level, concatenated, then a fresh small two-layer nonlinear head (hidden width 128) predicting the supported COCO categories. Do not use the pretrained class token or features from a trained mask decoder. Keep the same readout family, image draws and three paired seeds; different native input widths still imply different parameter counts, which must be reported. This tests global category accessibility, not semantic localization or a theorem about all possible readouts.

Proposed maximum: two packages × three seeds, 2,000 updates, batch 64, at most ten minutes per fit. That adds at most 60 minutes of fitting if activated; label preparation, final evaluation and total-stage cap must be profiled separately. It does not fit inside the previously proposed four-hour P0/P1 allowance by assumption. Freeze class support, loss/optimizer, visibility and crowd semantics, validation selection and the full budget before activation. Mean category AP is the primary metric, with per-class support, scores and train-prior baselines. AP uses ranking and needs no fitted score threshold; any optional thresholded metric uses training/validation only. Support rules must be fixed before looking at test performance.

Select this new head on its own validation AP while retaining the already selected P1 encoder/RGB/pose package. Report its training step separately; do not pretend all heads were jointly selected at one new checkpoint. A semantic result cannot retrospectively choose a different P1 pose/RGB checkpoint. P1-T and P1 are distinct comparisons, and both remain proposed.

If the CNN meets physical requirements and the ViT exposes stronger category information, retain that tradeoff. Choose by the actual consumer, then test a hybrid only if both capabilities are necessary and neither package is adequate at acceptable cost. Two distinct hybrid questions are available: extra within-scale transformer processing in the compact CNN, or a local detail path alongside pretrained spatial features. They require different controls and must not be collapsed into one unexplained architecture change. Better COCO category access also does not prove better screen-text or software understanding.

## What each experimental stage still needs

| Experiment | Existing data sufficient? | Required addition |
|---|---|---|
| P0/P1 frozen packages | Yes for the bounded geometry/RGB/foreground comparison | Pin checkpoint/data identities, verify cached-feature transforms and freeze the draft protocol |
| P1-T semantic transfer | Source images and labels are present | Prepare crop-aware category targets and a separately bounded readout study |
| P2 existing memory | Yes for current-frame/prior-memory/updated-memory velocity accessibility | Fresh controlled equal-current-image histories or occlusion cases for stronger persistence claims; use stored Paddle velocity truth |
| P3/P4 loss or processing | Yes for a matched within-domain comparison | Keep fixed splits/draws; a later fresh configuration cohort for confirmation |
| P5 decoder sharing | Yes for RGB/foreground sharing | Instance/variable-query targets require additional local label preparation |
| P6 encoder context | Yes for pusher/body target queries | More demanding task/history conditioning needs deliberately varied context and targets, including cases where it matters |
| P7 prediction/control | Existing trajectories permit training and observational rollout errors | Verified action alternatives from shared histories, full state/reset or replay checks, and unseen control cases |
| More resolution/scales | Original COCO, RGB96 CCHI and RGB224 archives are available | Reprocess the same raw observations under explicitly compared resolutions; never treat upsampling RGB64 as extra evidence |
| Software interaction | No prepared local source for this consumer | A small instrumented UI with screenshots, executed actions, exact state labels and resettable branches |

For controlled collection, prefer known simulator states and replayed histories. A PushT pose vector omits object velocities/contact state and is not an exact physics snapshot. Branching from an arbitrary recorded pose therefore needs verified full restoration or controlled prefix replay, rather than an assumed reset. For occlusion, add a visual obstruction while preserving and logging the actual hidden state; test recovery after it clears. Such intervention data can be generated locally and do not require a generic video download.

## The next genuinely different data source should be software use

Start with one instrumented local application covering three contrasting operations: selecting a requested control, editing a short value, and revealing or remembering state across a dialog/scroll. Record screenshots at a resolution that preserves the relevant symbols, the executed action, time, and complete application state. The full state is a label/evaluation source; it enters the model only in an explicitly separate structured-input comparison.

Vary layouts, labels and hidden states; hold out layout/seed combinations before collection, and branch several plausible actions from the same restored state. Keep a small deterministic development cohort first, profile capture/readout cost, then freeze a bounded train/validation/test collection. This is a proposed data protocol, not an already prepared benchmark or evidence of general computer use.

Public alternatives exist. [ScreenSpot-Pro](https://arxiv.org/abs/2504.07981) tests high-resolution GUI grounding from screenshots, while [OSWorld](https://arxiv.org/abs/2404.07972) supplies interactive computer tasks with execution-based evaluation. The local pilot is recommended for controlled scope and instrumentation, not because public resources cannot provide actions/resets. Add a public benchmark later when broader software transfer is the question; do not use its test set as training data.

[Charades](https://prior.allenai.org/projects/charades) and [TAU's official task description](https://dcase.community/challenge2021/task-acoustic-scene-classification) support quantitative passive activity/scene and audiovisual work. More clips can expand observational coverage. Their supplied labels do not identify the specific executable action effects or hidden-state targets needed by our current controller experiments. Their use is worthwhile for a named passive-learning goal, rather than mandatory before the initial comparison.

## Claude collaboration and resulting recommendation

Claude reviewed a public dataset-type/architecture brief without local code, results or private media. We agreed on no immediate download, a bounded initial package screen, one additional semantic readout before broader perception claims, and targeted simulator/UI collection for the missing questions.

The first reply overgeneralized episode-split leakage, described passive data as qualitative-only, proposed deriving amodal shape without sufficient truth, conflated COCO keypoints with PushT pose, and claimed public UI resources lacked resets/actions. The correction round explicitly withdrew these claims. It also corrected probing trained decoder features instead of native encoder features and treating matched from-scratch training as a prerequisite for any practical cross-domain evaluation.

We retain two further qualifications: category labels are a different target from foreground geometry but are not statistically independent of it; and a failed fitted probe does not prove information is absent, even linearly, without ruling out estimation/optimization limits. The final proposal also keeps P1-T checkpoint selection separate rather than retrospectively presenting independently fitted heads as one jointly selected checkpoint. Exact public exchanges and receipts are retained under `runs/perception_data_cowork_2026-09-08/collaboration/`.

**Recommended next action:** use the existing prepared data to freeze the first package comparison; prepare P1-T labels as the smallest useful semantic extension. Collect fresh targeted histories and then an instrumented software cohort when their branch is activated. Preserve current gates and independent-output reporting. The right architecture remains an experimental choice, not an outcome established by the dataset audit or model agreement.
