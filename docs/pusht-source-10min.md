# Ten-minute source-data development run

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
