# Heuristics



## H01: Diagnose reference integration before long training
- **Rationale**: Replay recorded actions and compare a released checkpoint through upstream/local control before capped batch-128 learning and second-dataset checks; separate integration failure from undertraining before spending a long-run budget.
- **Provenance**: user-revised
- **Crystallized via**: verbal-affirmation
- **Sensitivity**: high; requires explicit data, normalization and sampling scope.
- **Code ref**: [replay](../../../scripts/check_alignment.py), [paired evaluation](../../../scripts/compare_evaluators.py), [learning checks](../../../configs/diagnostics/README.md)
- **From staging**: O02
- **Evidence of adoption**: N05; user said “yes please do exactly that”. This affirms the diagnostic sequence, not subsequent capability claims.


## H02: Complete the capped schedule and full-source positive control
- **Rationale**: Finish the fixed 400-update cached schedule, evaluate untouched weights and separate normalization probes, then validate released control with verified complete-source normalization before changing the baseline or starting long training.
- **Provenance**: user-revised
- **Crystallized via**: verbal-affirmation
- **Sensitivity**: high; subset prediction, diagnostic clones and full-source released control have different evidential scopes.
- **Code ref**: [cached schedule](../../../configs/diagnostics/pusht_cached_learning.yaml), [checkpoint control](../../../scripts/check_checkpoint_control.py), [full-source reference](../../../scripts/evaluate_reference.py)
- **From staging**: O05
- **Evidence of adoption**: N10; current user requests continuation of the accepted sequence, with the original acceptance recovered from the interrupted session. This adopts the work sequence, not new capability claims.


## H03: Check diverse held-out configurations before long training
- **Rationale**: Test broader source coverage with explicit training/held-out separation and unchanged model/objective before considering longer training; pair prediction and planning evidence.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Sensitivity**: high; distinct episode indices can share initial configurations, so audit related variants.
- **Code ref**: [source-group preparation](../../../scripts/prepare_pusht_pilot.py), [frozen split](../../../world_model/data.py), [pilot configuration](../../../configs/diagnostics/pusht_broader_pilot.yaml)
- **From staging**: O08
- **Evidence of commitment**: N17; clean implementation commit 0a81029 fixes the broader group-disjoint protocol. The recommendation's provenance is retained; this does not affirm capability claims.

## H04: Bound the broader pilot and compare the same held-out goals
- **Rationale**: Use 128 training and 32 held-out episodes, random initialization, batch 128, at most 1,000 updates or 30 minutes training; retain intermediate checkpoints and compare untouched float32 predictions plus 20 shared control goals against copy/shuffled, stationary/replay/released controls.
- **Provenance**: user-revised
- **Crystallized via**: verbal-affirmation
- **Sensitivity**: high; preparation/evaluation time is separate, related configurations must not cross the split, and released weights use their own reference normalization.
- **Code ref**: [pilot configuration](../../../configs/diagnostics/pusht_broader_pilot.yaml), [paired evaluator](../../../scripts/evaluate_pusht_pilot.py), [protocol and results](../../../docs/pusht-broader-pilot.md)
- **From staging**: O09
- **Evidence of adoption**: N16; user said “yes please” in response to the concrete pilot proposal. Approval covers the work sequence and bounds, not the subsequent results or another training run.

## H05: Diagnose saved-checkpoint control before preparing source-scale reproduction
- **Rationale**: Compare saved-checkpoint control on 20 training goals with held-out evidence, inspect predicted-versus-real multi-step action rankings, and prepare a faithful source-scale configuration with explicit schedule/evaluation assumptions before committing a long training budget.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Sensitivity**: high; preserve checkpoint identity, matched cases/budgets, and the distinction between preparation approval and a long training launch.
- **Code ref**: [current authorized work](../../../docs/project-state.md), [saved-checkpoint diagnostics](../../../scripts/diagnose_pusht_control.py), [completed report](../../../docs/pusht-control-diagnosis.md). Implemented and completed in N28–N32.
- **From staging**: O13
- **Evidence of commitment**: N23; committed live-state document in 7d480b1 depends on the approved sequence. Provenance is retained for this artifact-commitment promotion. No new causal claim is established.


## H06: Preserve full-batch computation with activation checkpointing
- **Rationale**: Encoder batch slicing can change bf16 gradients despite matched forward losses. For the source-scale candidate, use nonreentrant encoder activation checkpointing with encoder_chunk zero; this retains full-batch kernels, projectors and SIGReg while reducing activation memory.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Sensitivity**: high; native comparison covers one real batch-four probe, memory one batch-128 backward. Not a proof of long-run optimizer trajectory or control capability.
- **Code ref**: [candidate](../../../configs/reproduction/pusht_source_scale.yaml), [trainer](../../../world_model/train.py), [native parity](../../../scripts/check_real_batch_parity.py), [capacity probe](../../../scripts/check_reproduction_memory.py).
- **From staging**: O16
- **Evidence of commitment**: N31/N32; candidate config committed in 80b0398. Numerical and memory evidence in [diagnostic snapshot](../../evidence/tables/pusht_control_diagnosis_2026-09-06.json).


## H07: Restore saved split mappings before checkpoint inspection
- **Rationale**: Validation indices can be local to a random-window Subset. Applying them to the full-source dataset silently changes the evaluated windows. Reconstruct the training split and verify its index hashes, episode lists, sizes and normalization before diagnostic sampling.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Sensitivity**: high; frame overlap and shared episodes remain population limitations even when window indices are disjoint.
- **Code ref**: [inspection_datasets and frame_rows](../../../scripts/inspect_checkpoint.py), [split restoration test](../../../tests/test_checkpoint_inspection.py), [shared data protocol](../../../world_model/protocol.py).
- **From staging**: O18
- **Evidence of commitment**: Implemented in 57edf65, used in N35 for both local and released checkpoints; exact validation/probe/rollout window hashes match in the [result snapshot](../../evidence/tables/pusht_source_10min_2026-09-06.json).


## H08: Isolate diagnostic randomness and preserve resume identity
- **Rationale**: Diagnostic sampling must not consume optimization RNG; operational overrides must not silently change scientific configuration or checkpoint provenance.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Sensitivity**: high; legacy fingerprints and CPU/CUDA RNG, optimizer and schedule states have distinct compatibility contracts.
- **Code ref**: [trainer](../../../world_model/train.py), [tests](../../../tests/), [committed report](../../../docs/overnight-2026-09-06.md).
- **From staging**: O22
- **Evidence of commitment**: N39; implemented lifecycle slices, exact CPU diagnostics/resume checks and the recorded PushT recovery depend on this contract. Numerical outcomes are scoped in the [snapshot](../../evidence/tables/overnight_2026-09-06.json).

## H09: Preserve exact dashboard data and verified publication companions
- **Rationale**: Bound native dataset sizes by lossless partitions that keep complete chart series together. Stage a new artifact until canonical HTML verification succeeds so failed reporting preserves a consistent previous snapshot.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Sensitivity**: high; a single oversized series must fail explicitly; the canonical total payload and browser QA limits remain in force.
- **Code ref**: [dashboard](../../../viewer/dashboard.py), [canonical adapter](../../../viewer/deliver_dashboard.mjs), [harness tests](../../../tests/test_experiment_harness.py).
- **From staging**: O23
- **Evidence of commitment**: N46; committed implementation and final 2,304-value spectrum rely on partition/publication behavior verified in the [snapshot](../../evidence/tables/overnight_2026-09-06.json).


## H10: Diagnose frozen checkpoints before a schedule-preserving bounded continuation
- **Rationale**: Restore exact split/sample identities, compare saved/current/calibrated BN modes and simulator-grounded action ranking, and verify paired control before adopting an inference change. If training-integrity checks pass, continue the preserved PushT parent in a separate directory with unchanged optimizer/RNG and full schedule, bounded at one epoch.
- **Provenance**: user-revised
- **Crystallized via**: verbal-affirmation
- **Sensitivity**: high; source-window interpolation is not group-held-out generalization, calibrated clones are separate artifacts, and a step ceiling must not shorten the original learning-rate schedule.
- **Code ref**: [mode diagnostic](../../../scripts/check_training_modes.py), [ranking](../../../scripts/diagnose_tworoom_control.py), [fork trainer](../../../world_model/train.py), [bounded config](../../../configs/reproduction/pusht_epoch1_continuation.yaml), [report](../../../docs/tworoom-followup-2026-09-06.md).
- **From staging**: O24
- **Evidence of adoption**: N50; user said “yes. cowork with claude. use claude not only for review to keep your limits in mind”. This adopts the proposed work and co-work scope, not new numerical/causal claims or broad repository transfer. Completed evidence: N51–N56 and [snapshot](../../evidence/tables/tworoom_followup_2026-09-06.json).


## H11: Isolate sketch randomness in a paired lower-variance training screen
- **Rationale**: Test additional independent SIGReg directions with the same objective, batch128, latent192 and lambda.09. Keep initialization, data order and model/dropout randomness paired using a separate checkpointed sketch generator, and freeze calibration policy before observing control. Lower conditional gradient noise is an experimental motivation, not a learning-speed guarantee.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Sensitivity**: high; equal seeds with different sketch dimensions do not imply nested projection draws. BN couples examples, shared source cases do not measure unseen configurations, and calibration accesses additional training windows.
- **Code ref**: [private sketch RNG](../../../world_model/seeded_sigreg.py), [trainer](../../../world_model/train.py), [serial coordinator](../../../scripts/projection_experiment.py), [frozen protocol](../../../docs/projection-training-plan-2026-09-06.md), [paired statistics](../../../scripts/paired_summary.py).
- **From staging**: O29
- **Evidence of commitment**: N66–N71; frozen plan, implemented generator and all12 completed paired runs depend on this design. The user authorized continuation and Claude co-work, without endorsing a numerical improvement or general causal explanation. [Source-bound evidence](../../evidence/tables/projection_training_2026-09-07.json); final implementation/report commit4a7f5d2. Provenance remains ai-suggested.


## H12: Gate downstream training with task readiness and matched warmup controls
- **Rationale**: Train/read out task perception before expanding U/P. Compare supervised scratch, COCO image-only E/D warmup, and in-domain image-only E/D warmup at a predeclared total-update budget with matched initialization, fresh task heads and supervised draws. Select on fixed physical validation errors; retain immutable generic checkpoints and separate reconstruction, prediction and control gates.
- **Provenance**: user-revised
- **Crystallized via**: verbal-affirmation
- **Sensitivity**: high; this is an adopted experiment design, not an assertion that task-first or generic warmup always improves learning. Failed readiness stops downstream expansion; test results cannot reopen selection or budgets.
- **Code ref**: [accepted plan](../../../docs/training-curriculum-2026-09-07.md), [curriculum training](../../../world_model/curriculum/training.py), [frozen decision](../../../scripts/freeze_curriculum_screen.py), [Paddle driver](../../../scripts/execute_paddle_followup.py).
- **From staging**: O35
- **Evidence of adoption**: N77; user explicitly said “ok please execute this plan” and requested full evaluation/internal-state figures. N78–N84 and [source-bound evidence](../../evidence/tables/curriculum_execution_2026-09-08.json) record execution. Adoption does not endorse later results or authorize private external transfer.

## H13: Test recoverability with frozen-encoder decoder refitting
- **Rationale**: Fit the decoder while retaining exact encoder/head state; compare matched fresh decoders on adapted and original encoders to separate accessible information from decoder initialization effects. Recovery supports recoverability at achieved error; failure still confounds information, capacity and optimization.
- **Provenance**: user-revised
- **Crystallized via**: verbal-affirmation
- **Sensitivity**: high; adopted diagnostic, not a universal efficacy claim.
- **Code ref**: [trainer](../../../world_model/curriculum/training.py), [matched experiment](../../../world_model/curriculum/decoder_recovery.py), [results](../../../docs/decoder-recovery-results-2026-09-08.md).
- **From staging**: O41
- **Evidence of adoption**: N88; user said yes, do that; N89 records execution. This does not adopt O42 or additional architecture changes.
