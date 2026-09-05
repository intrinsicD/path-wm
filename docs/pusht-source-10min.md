# Ten-minute source-data development run

## Completed result

The fresh run completed **375 optimizer updates in 578.48 seconds (9 min 38 sec)**
within the user-authorized ten-minute budget. The planning estimate was 300–350
updates; it reached the 375-update ceiling first. It processed 48,000 windows
(about 2.7% of one source-data epoch), with all logged values finite and peak
allocated GPU memory 3,283,672,576 bytes (3.06 GiB). The training process started
from clean commit `595faed`; the full 139,330-update reproduction was not launched.

**Useful control remains unestablished:** the new checkpoint reaches 0/50 frozen
goals, released weights 45/50, recorded replay 50/50 and stationary actions 0/50.
No goals are initially satisfied. The new model uses all 50 steps on every case;
released weights average 28.26. Both models use identical cases, reset/CEM seeds
and solver budgets, with their documented saved/reference normalization.

| Measurement on matched diagnostic windows | New, step 375 | Released weights |
|---|---:|---:|
| One-step prediction MSE / copy error | 0.8782 | 0.04186 |
| Prediction MSE / shuffled-action error | 0.9862 | 0.02641 |
| Effective covariance rank (of 192; 2,048 frames) | 11.51 | 88.72 |
| Mean Shapiro–Wilk W | 0.9901 | 0.9947 |
| Action sensitivity / state sensitivity | 0.3740 | 2.6449 |
| Mean probe R² (8 targets, including velocity) | 0.04368 | 0.7200 |
| Block x / y probe R² | 0.2298 / 0.2600 | 0.9820 / 0.9668 |
| Block angle sin / cos probe R² | 0.01964 / −0.02755 | 0.9015 / 0.9150 |
| Eight-step autoregressive error / copy-first-state error | 0.6844 | 0.1138 |

The new checkpoint's saved float32 prediction MSE is 0.317239, versus copying
0.361240 and shuffled actions 0.321685: 12.18% better than copying but just 1.38%
better than shuffled actions. Absolute latent MSE is not compared across encoders.
Saved-checkpoint re-evaluation differs from the final in-training validation by
at most 1.14e-6 absolute / 2.63e-6 relative across recorded prediction metrics.

The first-batch training internals capture effective rank 16.71 → 7.16 → 8.19 →
9.12 → 9.66 at steps 0/100/200/300/375. Those 32-window estimates are distinct from
the full 512-window inspection in the table. Validation prediction/copy ratios
improve sharply late in training, while weak action use and low rank remain.
Gaussian-looking individual dimensions do not imply a high-rank representation.
These observations describe the failure; they do not isolate its cause.

The [first frozen control case](../runs/diagnostics/pusht_source_10min_control/qualitative/case_0_rollouts.png)
shows the new model moving the agent away from the block and failing after 50
steps. Released weights succeed after 16 steps and recorded replay after 14.
Both saved model action sequences reproduce their original outcomes and terminal
state distances. The panel repeats the final frame after termination; it shows
actual simulator observations, with no decoded latent images. Case 0 was chosen
by its fixed index, without filtering by outcome.

## Verification and artifacts

- **49 essential tests passed**, including two real browser checks. The new split
  restoration test catches validation indices applied to the wrong source window
  and mismatched split hashes. The dashboard test keeps image comparisons bounded
  and prevents unrelated diagnostic populations from entering the paired panels.
- The checkpoint inspector now reconstructs and verifies the saved random-window
  split, including normalization and index hashes. Local and released inspections
  use the same 512 validation windows, 1,024 training probe windows and 256
  long-horizon source windows. Their window hashes match. Probe/validation windows
  share source episodes and may overlap in frames; this measures interpolation,
  not unseen-configuration generalization.
- The final checkpoint and retained step-375 snapshot both hash to
  `9e4c1462f001f41ea6a2f2e34f2006c4512e00dd71ef6113f15f2f4b3b19b612`.
  Local/released inspections and control evaluations verify unchanged checkpoints.
  The three old pilot snapshots and released reference retain their previous hashes.
- The first inspection's raw results succeeded while dashboard packaging failed
  at the 3 MB payload limit. The bounded-panel repair below resolved it. All later
  refreshes pass, including the final 29-chart / 6-table dashboard, desktop/mobile
  rendering and source interaction. The user's separate curves and internals
  instrumentation remain in place.
- The prior internals report's assertion that every target had R² ≤ 0 was
  corrected: its mean was negative, but some position targets were weakly positive.
  No old numeric evidence was changed.

The [dashboard](../runs/experiment_dashboard.html) is focused on the new training
run. The [source-hashed result summary](../runs/diagnostics/pusht_source_10min/short_run_summary.json)
links the raw manifests, logs and evidence. Training artifacts are under
`runs/diagnostics/pusht_source_10min/`; prediction/internals/panels under
`runs/diagnostics/pusht_source_10min_internals/{final,released}/`; matched control
and qualitative evidence under `runs/diagnostics/pusht_source_10min_control/`.

This run changes data population, schedule and recomputation relative to the old
pilot, so it cannot attribute any difference to activation checkpointing alone.
It covers only a small part of one epoch. The remaining experimental decision is
whether to fund a longer reference-scale schedule or isolate early representation
collapse with a matched-budget comparison. No second training run is scheduled;
the long reproduction and research extensions remain deferred.

## Predeclared plan (2026-09-06)

The user requested inspection of the updated dashboard, then ten minutes of
training and another results check. The dashboard at c920402 was inspected in
Chromium: individual scalar curves render, with canonical desktop/mobile QA
passed. Preserve its new charts and instrumentation.

Question: what prediction, representation and control behavior appears after a
short fresh run on the prepared full-source random-window protocol using
full-batch encoder activation checkpointing? This is one development seed, not a
controlled attribution of differences from the earlier pilot or a reproduction.
No new pass threshold is set. Learned control remains the key missing behavior.

Configuration: `configs/diagnostics/pusht_source_10min.yaml`. Seed 3072, batch 128,
full-source normalization and the prepared 1,783,548/198,173 window split;
unchanged model/objective/optimizer. Full-batch nonreentrant encoder activation
checkpointing, bf16 updates, float32 validation, and `introspect: true`.
Random-window validation shares episodes and configurations with training.

Budget: 600 seconds on the existing training timer (including initial and
periodic validation/internals and checkpoint writes), checked before each update;
an in-flight update/I/O and final checkpoint write may extend wall time slightly.
Data/model preparation occurs before that timer. The measured batch-128 backward
was 1.597 seconds; 600/1.597 is about 376 updates before optimizer/I/O/validation.
Estimate 300–350 updates; use a 375-update ceiling plus the 600-second stop.
The independent short cosine schedule has three warmup updates and decays toward
update 375; it is not the long run's 1,393-update warmup/139,330-update schedule.
Do not resume or overwrite the old pilot or the prepared long-run directory.

Log every 10 updates; validate 512 fixed windows and capture scalar internals at
initialization and every 100 updates; retain those checkpoints. If time expires
between validations, explicitly evaluate the final saved checkpoint afterwards,
keeping its results separate from the earlier validation step.

After training: inspect the final checkpoint's spectrum, Gaussianity, action/state
sensitivity, pose probe, eight-step rollout ratios and image panels. The inspector
must reconstruct the random-window Subset mapping; its existing episode-only
mapping is unsuitable for this new protocol. Verify saved split hashes and
checkpoint integrity. Compare with released weights on the same diagnostic
windows, retaining model-specific normalization. Report long-horizon windows as
source-population diagnostics, not unseen-configuration generalization.

Evaluate the saved checkpoint and released weights on all 50 existing frozen
cases in `runs/reproduction/pusht_source_scale_preparation/control_cases.json`:
seed 42 sampling, reset/CEM seeds 1234–1283, horizon 5, five-action blocks,
300 candidates, 30 iterations, 30 elites, 50 simulator steps. Run matched replay
and stationary controls. Report successes, initial successes and individual
outcomes. Post-training evaluation/reporting time is outside the training budget.

Refresh and verify the user's revised offline dashboard after each completed run
or standalone evaluation, with the new training run in focus. Record actual
updates/time, source/code/config/checkpoint identities, finite-loss status,
prediction/control results and limitations here and in project-state.md.

## Reporting repair declared after the first inspection

The first new inspection succeeded, but its dashboard refresh exceeded the
canonical 3,000,000-byte payload limit because every historical PNG panel was
embedded. Bound image panels to the focus run's earliest/latest inspected
checkpoints and one released inspection with the same source-run manifest. Keep
all scalar, spectrum, horizon and inventory evidence, and explicitly disclose the
image selection. Test that unrelated populations cannot enter the paired panels
and that intermediate checkpoint scalars remain indexed. This affects display
only; the completed training and diagnostic raw outputs remain unchanged.
