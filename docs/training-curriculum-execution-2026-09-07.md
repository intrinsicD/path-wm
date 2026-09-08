# Accepted curriculum execution — 2026-09-07

The user authorized execution of the proposed curriculum, thorough final
evaluation, and perception/internal-state visualizations. Task-first ordering,
the three-arm comparison and its prospective physical selector are accepted.
No additional private Claude transfer is implied or attempted.

## Slice 1: portability and curriculum interfaces

Problem: move the frozen evidence to the home runtime, add image-only E/D
pretraining without fictitious task labels, and compare adaptation fairly.

Runtime migration: Python 3.12.9, PyTorch 2.9.0+cu128 and RTX 3050 8 GiB,
using a new .venv with system-site PyTorch and project dependencies. This is
a new experimental runtime; no bitwise continuation equivalence to work's
Python 3.14/Torch 2.14 is asserted.

Raw archive restoration passed all 13,870 file hashes in a separate directory.
Full source verification initially rejected the different NumPy version label
despite exactly equal recomputed split/normalization values. Add a regression
that permits only that provenance difference after actual values match and
still rejects altered split results. Preserve both versions in the receipt.
The initial test suite collection hit a third-party tests-package collision;
make the repository tests package explicit. Restored report paths require
installing verified missing task files at their canonical paths.

New modules:
- world_model/curriculum/data.py: source-backed image-only frames, deterministic
  transform, exact/canonical and radius-four dHash duplicate components, grouped
  COCO split, and immutable task frame/source-row identities.
- world_model/curriculum/training.py: E/D-only or labelled E/D/H objective,
  scalar-weighted ragged accumulation, identical initialization/fresh H,
  private phase samplers, physical selector, atomic optimizer/RNG resume.
- world_model/curriculum/inspection.py: raw per-frame errors, train-only PCA
  feature visualizations, attention maps, activation statistics/spectrum,
  held-out linear probes and reconstruction/error panels.
- viewer/curriculum.py: reconcile raw curriculum ledgers and publish native
  charts and bounded qualitative figures through the existing canonical builder.

Essential tests precede implementation: absent-label routing, physical
selection, ragged gradient equality, warmup/fresh-head boundaries, sampler
resume, duplicate-group split isolation, and compatible checkpoint resumption.
The existing objective and architecture remain unchanged.

Budgets remain the accepted draft: 100-update/3-minute profile,
64-frame 500-update/10-minute diagnostic, then 4,000 updates per arm
(2,000 warmup + 2,000 adaptation for B/C), subject to a common preflight
reduction frozen before the scientific screen. 60-minute training/validation
cap per arm. Initial seed4107; repeat promising comparisons with4108/4109.

The final report will distinguish software completion, perception readiness,
prediction gates and control. It will include selected/final and matched
supervised-exposure comparisons, exact denominators, test/validation separation,
raw/decoded observations, fine/coarse feature PCA, measured cross-scale
attention, spatial activation/variation, spectrum and task-readout errors.
Attention is descriptive, not causal attribution. All learned visualization
bases/probes fit training data only. Held-out test results cannot change
checkpoint selection, gates, objectives or the continuation choice.

## Reporting scope amendment before the first GPU run

Combining every historical home LeWM study with the restored work session
produced84 native datasets against the canonical fixed limit50. Preserve both
prior reports and every raw ledger. The curriculum runner uses the existing
run_experiment wrapper and write_experiment_dashboard with its explicit
runs-root set to runs/curriculum_2026-09-07, publishing to the required canonical
runs/experiment_dashboard.html and companion. This shows all phases/evaluations
of this curriculum; previous tracks remain separate viewing evidence. The scope
is visible in the HTML. No format limit or browser assertion is relaxed.


## Thin slice verified and budget frozen

Full restoration/source/inference verification passed. The 100-update GPU
profile took6.0106seconds including validation, with peak allocated CUDA
memory1,389,499,904bytes. Full batch128 fits; retain4000 total updates per arm
and the original100-update validation cadence. B/C adaptation receives only
the remaining part of its arm's3600-second budget after warmup.
Profile reporting initially failed on absolute paths in the new source adapter;
raw results were preserved, paths made portable, and the rebuilt canonical HTML
passed browser verification at1440px and390px. CPU suite343passed/3skipped;
additional curriculum ledger-tamper regression passes.

COCO preparation decoded82783images, grouped82636components, and split
74501/4136/4146 train/validation/test frames. Fingerprint:
7fd00bc0232ddf4fdc76b70410203de7a7a199b375498996ec0f9242854ea46d.
There are242unique candidate pairs (253edges including overlapping exact/hash
matches). Representative inputs and the first40candidate pairs were visually
inspected. Genuine duplicate/color-edited pairs occur alongside conservative
false positives on mostly sky images. All candidates remain grouped; no manual
split changes. This criterion cannot guarantee all semantic duplicates were found.
Contact sheets and transform/split evidence are under the curriculum data_review.


## Slice 2: measured diagnostic and inspection

Fixed64 training diagnostic completed500updates in26.12s. Physical selection
picked update300: XY MAE[2.622,2.569,2.399,1.985]world units, angle2.428degrees,
q0.328. Final update500 regressed to q6.076 despite lower reconstruction loss.
This is a capacity/optimization diagnostic on training examples, not held-out
readiness. Selected-checkpoint images show blurred moving objects; fine maps
respond to the block while coarse maps are dominated by spatial structure.
E image-gradient norm.0140 versus pose-gradient norm.1624, cosine.0495 on64frames.
This one gradient sample does not prove systematic loss conflict.

Inspection uses256 fixed group-balanced training frames for token-channel PCA
and a full-feature linear ridge probe. Center and normalize by training RMS,
divide features by sqrt(width), fixed ridge.01; no test fitting or tuning.
A development rendering/probe draft was preserved under.runtime before correcting
its training-frame caption and adding the predeclared training-only RMS scale.
Held-out region metrics use256fixed frames and diagnostic geometry from the
archived perception diagnosis (nativeT polygons and radius15pusher). They are
not new loss terms. The selected64-frame diagnostic and its figures passed the
canonical browser verifier; three essential inspection checks pass.


## Slice 3: complete comparison controls and downstream snapshots

Add the accepted train-only mean-image reconstruction baseline, including object
regions; fit it over all training-frame members, with an exclusion regression.
For the already-authorized Paddle mixed-U follow-up, retain exact predictor
update10000and20000 snapshots when reached, without changing loss, selector,
early stopping or budgets. This permits the planned equal-update comparison.
A scalar transaction regression verifies snapshots retain their own update.


## Seed4107 frozen screen outcome

All arms completed4000updates with identical E/D initial tensors, fresh identical
H tensors, and byte-identical first2000supervised batch-index hashes. Actual
training+validation time A232.93s, B219.51s, C215.42s. Selected q values:
A3.009(update3800), B3.546(adaptation1900), C5.159(adaptation1900).
Both warmup policies worsen every constituent selected physical error relative
to the matched total-update A reference. No promising candidate and no arm meets
numeric readiness. As predeclared: no confirmation seeds, no adopted pretraining
policy, and no PushT U/P expansion. Freeze receipt precedes all new test inference:
runs/curriculum_2026-09-07/seed_4107/screen_decision.json.

The final assessment includes original fixed-validation selection metrics,
all-group validation inspections, all-test-frame selected/final metrics,
A at2000supervised updates, B/C image-only warmup snapshots, and both historical
reference snapshots. Displayed six-frame panels use fixed private sampling;
pose distributions use all evaluated frames. Neither test results nor probes
can reopen the frozen budget or choose a checkpoint.


## Slice 4: complete diagnosis and bounded presentation

Embedding every full-resolution validation figure exceeded the canonical3MB
payload limit. Preserve raw results and all PNG/SVGs; the compact reader now
embeds only the primary selected perception panels. Rebuild/browser-check before
resuming evaluation; completed inference is not repeated merely to fix packaging.
Add source-hash-verified aggregate diagnostics: full training metrics for the
selected A/B/C checkpoints versus the frozen held-out results, and existing
Paddle prediction/readout diagnostics with memory trajectory/attention figures.
No new loss, gate or continuation is introduced. Raw-source tampering must fail
the dashboard collector. Reuse the existing Paddle diagnostic calculations.


## Post-training control comparison

The frozen full evaluation is running. Prepare a descriptive historical-versus-new
control comparison after all cases finish. Match identical case identities and
initial states. Resample the100complete opposite-direction pairs together for
paired success differences (2000draws, seed93501); test that swapping opposite
member outcomes cannot invent between-pair uncertainty. These are post-training
case intervals, not new gates or variation across trained models. Preserve the
original reference raw cases and their hashes. Do not conclude from partial
controller cases or change the fixed model/selector/protocol.

## Completed-cache memory readout figure

Add a CPU-only figure from the evaluator's already completed, hash-verified
identical-current-frame velocity-probe cache. Show all 100 opposite-direction
pairs, comparing the training-fitted single-frame linear probe with the frozen
U/R readout. Preserve raw records and compute sign counts from those records;
this is descriptive inspection, with no new fitting, training, gate or control
selection. Existing source-hash integrity checks cover this presentation path;
no test is added for plotting glue.

## Final reporting repair

All 3,500 controller cases and evaluator figures completed successfully, with no
evaluation errors. The canonical dashboard then rejected a valid rollout PNG
because the archived-path resolver assumed the supplied runs root was the top-level
`runs/` directory. This curriculum deliberately supplies a nested scope. Add an
essential regression for current relative and archived absolute image paths under
a nested runs root, then resolve against the enclosing physical `runs/` directory.
Preserve raw ledgers and figures; rebuild the report without rerunning evaluation.

## Execution closed — 8 September 2026

All conditional stages authorized by the accepted protocol are complete. The
negative PushT screen stops expansion and confirmation seeds. Paddle P1/P5 and
all 3,500 controller cases finish; learned first interception is 345/500 ordinary
and 115/200 paired, with the declared quality failures preserved. The final
historical comparison verifies every case identity and resamples whole pairs.
The dashboard reporting repair reuses completed evaluation, with browser QA
passing. All 355 software tests pass. The full result report, raw comparisons,
checkpoints, image/PCA/attention arrays and PNG/SVG/GIF figures are retained.
No training or evaluation remains running or queued.
