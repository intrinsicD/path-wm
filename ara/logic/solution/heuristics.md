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


## H14: Use critical, scoped Claude collaboration for consequential research decisions
- **Rationale**: The user adopts independent review followed by source checks and concrete reconciliation, retaining retractions and disagreements. Actual Claude is required when named; routine edits do not require delegation. Agreement is not empirical validation and does not authorize new scope or denied payloads.
- **Provenance**: user
- **Crystallized via**: verbal-affirmation
- **Sensitivity**: high; value depends on evidence quality, independent verification and appropriately scoped access, not the number of agreeing agents.
- **Code ref**: [standing workflow](../../../docs/claude-collaboration-workflow.md), [experiment workflow](../../../docs/experiment-workflow.md).
- **From staging**: O50
- **Evidence of adoption**: N98; explicit request to remember and integrate the method. N103 and the [report](../../../docs/encoder-study-results-2026-09-08.md) retain successful protocol exchanges and rejected extra transfer.

## H15: Load model behavior from explicit checkpoint metadata
- **Rationale**: A disabled computation path may retain identical tensor names. Standard observer loading and export must preserve the declared encoder variant, and unsupported legacy entry points must fail before constructing a different behavior.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Sensitivity**: high; weight-only fingerprints do not describe every behavioral configuration. Preserve metadata and compatible downstream identities.
- **Code ref**: [load/export](../../../world_model/pusht/checkpoints.py), [contract tests](../../../tests/test_encoder_checkpoint_contract.py), [repair note](../../../docs/encoder-checkpoint-loading-note.md).
- **From staging**: O54
- **Evidence of commitment**: N103; implemented in507775a after exact frozen training completion. Six contract tests and exact E/H/D equality for all12 trained checkpoints pass; [evidence](../../evidence/tables/encoder_study_2026-09-08.json).

## H16: Migrate one editable experiment path with scoped behavior checks

- **Rationale**: Preserve an immutable source/data/checkpoint reference, first deliver a perception recipe that can be copied, edited, checked, trained, resumed and inspected, then migrate compatible temporal components and remove the obsolete active stack. Validate retained computations and critical lifecycle invariants before retirement; keep new recipes independent of historical orchestration.
- **Provenance**: user-revised
- **Crystallized via**: verbal-affirmation
- **Sensitivity**: high; parity covers declared inputs, states and tolerances. Short development runs do not establish scientific quality. The operation sequence was exercised by the agent; the user's independent experience has not been measured.
- **Code ref**: [migration](../../../docs/migration.md), [perception](../../../experiments/perception.py), [dynamics](../../../experiments/dynamics.py), [run lifecycle](../../../pathwm/io.py), [focused tests](../../../tests/).
- **From staging**: O80
- **Evidence of adoption/execution**: N128/N129 and [source-bound receipts](../../evidence/tables/modular_migration_2026-09-09.json). Original caches, datasets and results remain preserved; active recipes compute live features. Strict same-runtime/device replay and source snapshots have explicit limits. The archived implementation is not an active dependency.

## H17: Gate memory-learning changes with a bounded factual recall diagnostic

- **Rationale**: Check current/recent factual learning with the existing objective and explicit finite-budget fit/fresh-example gates before adding local memory supervision. Use one implementation owner and concise consequential Claude review; preserve failed gates and stop automatic scope/budget expansion.
- **Provenance**: user-revised
- **Crystallized via**: verbal-affirmation
- **Sensitivity**: high; this is an adopted sequencing rule, not proof that diagnostics identify the bottleneck or guarantee useful memory.
- **Code ref**: [recipe](../../../experiments/multimodal.py), [pilot plan](../../../docs/recall-learning-plan.md), [contract checks](../../../tests/test_recall.py).
- **From staging**: O115
- **Evidence of adoption/execution**: N163/N164; Alex explicitly approved the saved export and said to implement the proposed points. The failed pilot gates stop further model changes. O116 remains an unconfirmed interpretation.


## H18: Establish direct factual grounding before diagnosing the recurrent reader

- **Rationale**: After failed near-fact recall gates, test entity/location extraction and query selection in a finite supervised control before adding memory objectives. Nonzero gradients and finite weights alone do not identify a learning bottleneck.
- **Provenance**: user-revised
- **Crystallized via**: verbal-affirmation
- **Sensitivity**: high; success of a new direct reader does not establish what the previous checkpoint encoded or where its recurrent path fails.
- **Code ref**: [fact reader](../../../pathwm/models/facts.py), [recipe](../../../experiments/multimodal.py), [declared control](../../../docs/fact-learning-plan.md).
- **From staging**: O116
- **Evidence of adoption/execution**: N165/N166; Alex requested implementation/review/test/fix/iteration after the direct fact and competing-record control proposal. H17's earlier unconfirmed status for O116 is superseded here.


## H19: Compare a successful direct fact control with the existing event/task reader

- **Rationale**: Keep semantic pairs, shared component initialization, supervised heads and exposure fixed while routing the same task through the ordinary event, interpretation and thinking path. Treat results as whole-path trainability evidence; added capacity and optimization prevent component-level causal attribution.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Sensitivity**: high; a single committed event and fixed categorical evaluation draws do not establish later retention, learned query selection or robust stochastic performance.
- **Code ref**: [event reader](../../../pathwm/models/facts.py), [recipe](../../../experiments/multimodal.py), [contract checks](../../../tests/test_facts.py), [plan and outcomes](../../../docs/event-fact-plan.md).
- **From staging**: O117
- **Evidence of commitment**: N167/N168 and implementation `f09400c`; the event control now implements the proposed comparison. The user authorized execution, not a localized causal explanation.

## H20: Test transferred encoder initialization with an explicit donor boundary

- **Rationale**: Load only the successful direct control's encoder into the ordinary fresh recipient, preserve other tensors and construction RNG, and compare each recipient with its matched cold run. Keep the encoder trainable and bind donor file/component/tensor identity into resume compatibility. Count upstream training exposure separately.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Sensitivity**: high; the shared encoder also reads the task instruction and keeps changing. Reused development combinations are not an independent final test. This intervention does not localize a failing component or establish improved total-compute efficiency.
- **Code ref**: [recipe](../../../experiments/multimodal.py), [transfer and resume tests](../../../tests/test_facts.py), [plan and outcomes](../../../docs/warm-encoder-plan.md).
- **From staging**: O118
- **Evidence of commitment**: N169/N170 and implementation `4c3aebf`; exact donor transfer and matched cold initialization/settings/sampler are verified in the [evidence snapshot](../../evidence/tables/warm_encoder_2026-09-11.json). The user authorized execution; the outcome interpretation and frozen-encoder proposal remain staged as O119.

## H21: Establish a controlled history-dependent entity task before graph comparison

- **Rationale**: Use visible candidate histories, exact conditional answer targets and final-view-only bounds to test identity, state persistence and action effects. Gate graph comparison on a recurrent baseline learning the same task.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Sensitivity**: high; supplied candidates, short histories and grouped appearances do not establish discovery, general calibration or graph utility.
- **Code ref**: [generator](../../../pathwm/data/entities.py), [reader](../../../pathwm/models/entities.py), [scores](../../../pathwm/evaluation/entities.py), [recipe](../../../experiments/multimodal.py).
- **From staging**: O127
- **Evidence of commitment**: N185/N186; implementation 24b22b9. Baseline gates failed, so the conditional graph comparison remains deferred.

## H22: Share entity updates and mix coherent final assignments

- **Rationale**: Process supplied object streams with a shared recurrent cell and shared readouts; marginalize complete assignment hypotheses to preserve joint uncertainty.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Sensitivity**: high; independent switches, fixed slots and exact observed matching are supplied. No arbitrary interaction or graph-growth claim.
- **Code ref**: [shared reader](../../../pathwm/models/entities.py), [tests](../../../tests/test_entities.py), [recipe](../../../experiments/multimodal.py).
- **From staging**: O129
- **Evidence of commitment**: implementation 98e36af and N190; task accuracy and exact permutation checks are separately recorded.

## H23: Test learned association under identifiable feature variation

- **Rationale**: Preserve a working shared reader while replacing lookup with task-trained association and perturbing descriptors within a declared separation margin. Keep known-task outcomes distinct from general recognition claims.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Sensitivity**: high; fixed slots, two-object hypotheses and bounded synthetic variation do not establish graph allocation or visual identity.
- **Code ref**: [generator](../../../pathwm/data/entities.py), [matcher](../../../pathwm/models/entities.py), [tests](../../../tests/test_entities.py).
- **From staging**: O130
- **Evidence of commitment**: implementations cf5a506/76338f4 and N192/N194 now cover learned matching and controlled variation.

## H24: Separate novelty classification from entity allocation

- **Rationale**: After learned matching and bounded variation, test known identities versus an explicit new answer. Report false merges/splits and coverage before connecting predictions to persistent writes.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Sensitivity**: high; the separated synthetic margins do not establish open-world novelty or calibrated allocation.
- **Code ref**: [matching data](../../../pathwm/data/entities.py), [reader](../../../pathwm/models/entities.py), [metrics](../../../pathwm/evaluation/entities.py).
- **From staging**: O131
- **Evidence of commitment**: 365bb2b and N198 implement the unmatched-query screen; allocation remains proposed.

## H25: Test persistent identity separately from classifier labels

- **Rationale**: Bind evaluator truth to allocated record IDs; test retries and restored continuation independently of recognition accuracy. Missed allocations can shift record indices without causing identity switches.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Sensitivity**: high; transaction correctness does not establish calibrated recognition.
- **Code ref**: [runtime](../../../pathwm/models/entity_memory.py), [growth screen](../../../pathwm/evaluation/entity_growth.py), [regression](../../../tests/test_entity_growth.py).
- **From staging**: O132

## H26: Adapt matcher training to the runtime candidate distribution

- **Rationale**: Train variable candidate counts with masked padding and compare frozen models on identical fresh families. Geometry adaptation is a simultaneous intervention, so do not attribute gains to count alone.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Sensitivity**: high; bounded synthetic results do not establish open-world calibration.
- **Code ref**: [data](../../../pathwm/data/entities.py), [matcher](../../../pathwm/models/entities.py), [padding tests](../../../tests/test_entity_variable.py).
- **From staging**: O133

## H27: Commit identity and learned state together

- **Rationale**: Stage identity routing and recurrent updates before publishing either. Full-payload retries return without a second state update; failures leave both tables unchanged. Compare runtime latents with training-time routing.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Sensitivity**: high; synthetic template success does not establish general belief updating.
- **Code ref**: [runtime](../../../pathwm/models/entity_state.py), [tests](../../../tests/test_entity_state.py).
- **From staging**: O134

## H28: Separate descriptor holdout from temporal generalization

- **Rationale**: After shared-template success, freeze both models and vary event composition and length with independently simulated targets. Report runtime agreement separately from learned-state accuracy.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Sensitivity**: high; length and repetition are not causally isolated by these cohorts.
- **Code ref**: [controls](../../../pathwm/data/entity_temporal.py), [runtime checks](../../../pathwm/evaluation/entity_state.py).
- **From staging**: O135

## H29: Broaden temporal coverage before claiming state-update generality

- **Rationale**: Mix resets, toggles and no-information events with balanced complemented targets; preserve model/compute and evaluate fresh compositions and lengths. Separately report persistent idle failures.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Sensitivity**: high; matched update count is not matched convergence, and no-information stability is not guaranteed.
- **Code ref**: [mixed histories](../../../pathwm/data/entity_state.py), [recipe](../../../experiments/multimodal.py).
- **From staging**: O136
- **Scope**: Training-distribution intervention implemented; explicit no-op remains O137.

## H30: Preserve state explicitly for a deterministic no-information opcode

- **Rationale**: Use a differentiable identity for the exact idle opcode in both training and transactional runtime; persist this policy in checkpoint and snapshot identity. Keep learned action/observation updates and matched legacy comparisons.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Sensitivity**: high; assumes no autonomous evolution. This rule does not establish learned temporal beliefs.
- **Code ref**: [state update](../../../pathwm/models/entity_state.py), [gradient and persistence tests](../../../tests/test_entity_noinfo.py).
- **From staging**: O137
- **Evidence**: [bounded comparison](../../evidence/tables/entity_noinfo_2026-09-11.json); control NLL increases remain below declared gate.

## H31: Isolate a learned interaction with counterfactual source histories

- **Rationale**: Freeze ordinary state dynamics; compare latents-only directed updates with a matched source-zero control. Pair source changes with unchanged destination histories and test reuse of updated states.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Sensitivity**: high; supplied endpoints and a single copy relation do not establish graph discovery or source selection.
- **Code ref**: [interaction model](../../../pathwm/models/entity_state.py), [histories](../../../pathwm/data/entity_interaction.py), [runtime checks](../../../tests/test_entity_interaction.py).
- **From staging**: O138
- **Evidence**: [controlled comparison](../../evidence/tables/entity_interaction_2026-09-11.json).

## H32: Test frozen descriptor retrieval before adding another learner

- **Rationale**: Replace source IDs with read-only learned matching, bind the query and selected source into transaction replay, and compare against oracle IDs, allocation permutations, opposing-state distractors and unknown queries.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Sensitivity**: high; queries remain supplied and matcher confidence is not calibrated correctness. No graph-memory capability follows.
- **Code ref**: [lookup](../../../pathwm/models/entity_memory.py), [transaction](../../../pathwm/models/entity_state.py), [screen](../../../pathwm/evaluation/entity_source.py).
- **From staging**: O139
- **Evidence**: [frozen comparison](../../evidence/tables/entity_source_2026-09-11.json).

## H33: Separate learned relation addressing from supplied persistence

- **Rationale**: Encode an earlier cue into a per-destination latent key; test destination-only retrieval after source-state changes using paired histories, rebinds, gaps and erased-key controls. Freeze the matcher and interaction to isolate key learning.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Sensitivity**: high; one relation slot/type, explicit storage and write policy do not establish learned topology or write decisions.
- **Code ref**: [relation memory](../../../pathwm/models/entity_relations.py), [controls](../../../pathwm/evaluation/entity_relations.py).
- **From staging**: O140
- **Evidence**: [bounded result](../../evidence/tables/entity_relations_2026-09-11.json).

## H34: Isolate relation-write learning through frozen consumers

- **Rationale**: Learn context-conditioned overwrite/preserve through source-selection loss; compare hard runtime behavior with soft training and constant write controls on paired histories.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **From staging**: O141
- **Sensitivity**: Separated synthetic contexts supply the relevance rule. No ambiguous semantic relevance or graph discovery established.
- **Code ref**: [gate comparison](../../../pathwm/evaluation/entity_gate.py), [recipe](../../../experiments/multimodal.py).
- **Evidence**: [bounded comparison](../../evidence/tables/entity_gate_2026-09-11.json).

## H35: Separate frozen gate sensitivity from observation ambiguity

- **Rationale**: Reuse prototypes and noise across severities, retain generative labels, report both class recalls and fixed threshold sensitivity with a distance baseline.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **From staging**: O142
- **Sensitivity**: High-noise failure cannot identify a unique gate defect or Bayes limit. Baseline is not an oracle.
- **Code ref**: [noise screen](../../../pathwm/evaluation/entity_gate.py).
- **Evidence**: [frozen shift](../../evidence/tables/entity_gate_shift_2026-09-11.json).

## H36: Match continuation exposure before attributing augmentation gains

- **Rationale**: Compare identical initial gates, source pairs, row counts and optimizer trajectories while varying only cue noise; preserve frozen and clean references.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **From staging**: O143
- **Sensitivity**: Method commitment, not repair success. Repeated control rows are not additional independent observations; higher recall can increase false writes.
- **Code ref**: [augmentation](../../../pathwm/evaluation/entity_gate.py), [recipe](../../../experiments/multimodal.py).
- **Evidence**: [failed guardrail](../../evidence/tables/entity_gate_augment_2026-09-11.json).

## H37: Replicate conditional augmentation tradeoffs without pooling away failures

- **Rationale**: Vary data and training seeds with matched arms, preserve all criteria and report each class/severity separately.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **From staging**: O144
- **Sensitivity**: Two runs from one donor are descriptive, not broad statistical robustness or initialization replication.
- **Code ref**: [replication seeds](../../../experiments/multimodal.py).
- **Evidence**: [replications](../../evidence/tables/entity_gate_replicate_2026-09-11.json).

## H38: Distinguish clean teacher retention from held-out correctness

- **Rationale**: Cache detached train-only teacher targets, match replay across arms and evaluate ground-truth correctness separately from teacher consistency.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **From staging**: O145
- **Sensitivity**: Method commitment only; weight-one retention fails this screen and does not establish coefficient robustness.
- **Code ref**: [retention loss](../../../pathwm/evaluation/entity_gate.py), [recipe](../../../experiments/multimodal.py).
- **Evidence**: [negative result](../../evidence/tables/entity_gate_retain_2026-09-11.json).

## H39: Compare selective reobservation with unconditional sensing and duplicate evidence

- **Rationale**: Fix a sensing policy and observation cost, compare first-only, selective, duplicate and always-two on identical static contexts.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **From staging**: O146
- **Sensitivity**: Independent noise and static context assumed; no learned deferral, calibration or optimality claim. Always-two has higher high-noise utility here.
- **Code ref**: [reobservation](../../../pathwm/evaluation/entity_gate.py).
- **Evidence**: [comparison](../../evidence/tables/entity_gate_reobserve_2026-09-11.json).

## H40: Hold marginal noise fixed when testing reread dependence

- **Rationale**: Correlate raw errors with shared first draws and innovations, keeping the sensing policy and cost fixed; verify the fully shared endpoint.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **From staging**: O147
- **Sensitivity**: Raw correlation differs from normalized-cue dependence; fixed-policy performance is not per-correlation optimal utility.
- **Code ref**: [correlated generator](../../../pathwm/evaluation/entity_gate.py).
- **Evidence**: [failed robustness](../../evidence/tables/entity_gate_correlation_2026-09-11.json).

## H41: Compare alternate evidence on matched deferrals with explicit costs

- **Rationale**: Match first observations and deferrals, vary supplied second-source dependence and cost, report paired utility and both recalls with unconditional baselines.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **From staging**: O148
- **Sensitivity**: Fixed source properties and policies; no learned source choice or real sensor independence established. Always-alternate remains stronger at the chosen cost.
- **Code ref**: [source comparison](../../../pathwm/evaluation/entity_gate.py).
- **Evidence**: [bounded result](../../evidence/tables/entity_evidence_sources_2026-09-11.json).

## H42: Learn source action values from selected calibration feedback before adding a neural policy

- **Rationale**: Update opaque-source mean gains using only acquired outcomes; freeze evaluation, swap hidden source assignments and account for exploration costs.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **From staging**: O150
- **Sensitivity**: Static sources and known outcome feedback assumed. A value table is not a general reliability estimator or neural selector.
- **Code ref**: [policy](../../../pathwm/models/source_choice.py), [screen](../../../pathwm/evaluation/source_choice.py).
- **Evidence**: [static adaptation](../../evidence/tables/entity_source_choice_2026-09-11.json).

## H43: Audit online feedback timing and stable-source cost alongside adaptation

- **Rationale**: Compare frozen, cumulative, recent-window and no-feedback policies on matched silent swaps and no-change environments, charging for exploration and feedback.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **From staging**: O151
- **Sensitivity**: Fixed window and swap timing, not learned change detection. Faster adaptation can violate stable-environment utility guards.
- **Code ref**: [source history](../../../pathwm/models/source_choice.py), [drift evaluation](../../../pathwm/evaluation/source_choice.py).
- **Evidence**: [failed full screen](../../evidence/tables/entity_source_drift_2026-09-11.json).

## H44: Test evidence-triggered forgetting against explicit stationary reset limits

- **Rationale**: Measure false resets and their timing alongside utility before accepting a forgetting rule. Implemented test method does not imply successful detection.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **From staging**: O152
- **Sensitivity**: Fixed32-outcome blocks and0.15 threshold are uncalibrated; selected feedback controls detection latency. Present screen fails its reset guard.
- **Code ref**: [detector](../../../pathwm/models/source_choice.py), [comparison](../../../pathwm/evaluation/source_choice.py).
- **Evidence**: [failed full screen](../../evidence/tables/entity_source_change_2026-09-11.json).

## H45: Isolate stationary threshold selection from held-out adaptation testing

- **Rationale**: Select a variance multiplier on separate stationary worlds and freeze it before evaluating all adaptation and false-reset criteria. Reject unqualified fallback.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **From staging**: O153
- **Sensitivity**: Eight development worlds and adaptive feedback offer no statistical error guarantee. Present candidate fails the adaptation criterion.
- **Code ref**: [selection](../../../pathwm/evaluation/source_choice.py), [variance threshold](../../../pathwm/models/source_choice.py).
- **Evidence**: [failed full screen](../../evidence/tables/entity_source_uncertainty_2026-09-11.json).

## H46: Separate missing feedback checks from below-threshold outcomes

- **Rationale**: Reconstruct each source block and preserve mixed versus pure feedback tags before interpreting missed resets. Include unselected sources.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **From staging**: O154
- **Sensitivity**: Descriptive availability categories are not causal evidence of utility benefit. Older comparator can include calibration.
- **Code ref**: [diagnostic](../../../pathwm/evaluation/source_choice.py).
- **Evidence**: [trace verification](../../evidence/tables/entity_source_diagnosis_2026-09-11.json).

## H47: Match acquisition opportunities when testing exploratory allocation

- **Rationale**: Hold schedule, fees and detector fixed while comparing random and less-sampled exploratory sources; require utility improvement as well as coverage.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **From staging**: O155
- **Sensitivity**: Forced acquisition and epsilon0.5 limit scope. Present candidate improves coverage but fails utility gain.
- **Code ref**: [allocation comparison](../../../pathwm/evaluation/source_choice.py).
- **Evidence**: [failed screen](../../evidence/tables/entity_source_coverage_2026-09-11.json).

## H48: Check an independent seed and explicit state corrections before expansion

- **Rationale**: Separate ordinary-task performance from controlled correction handling; retain frozen weights on matched fresh cases and audit actions against changing truth.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **From staging**: O160
- **Sensitivity**: One additional seed and fixed correction timing do not estimate broad robustness. Utility must remain visible when success passes.
- **Code ref**: [evaluation](../../../pathwm/evaluation/key_box.py), [recipe](../../../experiments/multimodal.py).
- **Evidence**: [replication](../../evidence/tables/key_box_replica_2026-09-11.json); N241.

## H49: Test remembered-state grounding before expanding generation quality

- **Rationale**: Use paired histories with identical final observations/requests and different answers, compare encoder, state and recalled readouts, and score factual correctness separately from visual fidelity. Establish the observation-memory-output path before broader generation and the previously ordered prediction/planning stages.
- **Provenance**: user-revised
- **Crystallized via**: verbal-affirmation
- **From staging**: O179
- **Sensitivity**: High. A fixed synthetic task and a restrictive combination split can reward shortcuts; implemented screening is not evidence that its learned-capability gate passed.
- **Code ref**: [recipe](../../../experiments/memory_output.py), [state/memory wrapper](../../../pathwm/models/memory_output.py), [paired data](../../../pathwm/data/memory_output.py).
- **Evidence**: N276; [three-run record](../../evidence/tables/memory_output_2026-09-13.json).

## H50: Separate appearance from location with paired counterfactual histories

- **Rationale**: Hold initial appearance and selection fixed while varying final location; audit exact-input shortcut bounds and score attributes and complete pairs. Preserve failed benchmarks and name changed test populations honestly. This removes deterministic factor shortcuts without guaranteeing learned binding.
- **Provenance**: user-revised
- **Crystallized via**: verbal-affirmation
- **From staging**: O181
- **Sensitivity**: High. Within-support relocation is weaker than unseen-tuple or natural-scene generalization; balanced training still fails the tested agent location screen.
- **Code ref**: [paired data](../../../pathwm/data/memory_output.py), [recipe and controls](../../../experiments/memory_output.py), [frame-removal path](../../../pathwm/models/memory_output.py).
- **Evidence**: N279/N280; [relocation runs](../../evidence/tables/memory_relocation_2026-09-13.json).

## H51: Probe frozen intermediate states before changing their producer

- **Rationale**: Train independent readers on cached, detached intermediate states using training-only calibration and matched data/budgets. Include an encoder positive control, causally uninformative initial-state control and randomized-label control. Positive readout establishes conditional accessibility; failure cannot establish information absence.
- **Provenance**: user-revised
- **Crystallized via**: verbal-affirmation
- **From staging**: O183
- **Sensitivity**: High. Equal parameter count and updates do not equalize extraction difficulty; report access differences, training fit and every seed.
- **Code ref**: [probe recipe](../../../experiments/memory_probes.py), [state snapshots](../../../pathwm/models/memory_output.py), [checks](../../../tests/test_memory_probes.py).
- **Evidence**: N281/N282; [frozen-state results](../../evidence/tables/memory_probes_2026-09-13.json).

## H52: Isolate native recall repair against frozen stored states

- **Rationale**: Keep the encoder and snapshot writer fixed while training memory reading, working-state formation and native outputs; compare against direct access to the same stored values and preserve probe sensitivity limits. Declare objectives, seeds and budgets before changing the producer or enlarging codecs.
- **Provenance**: user-revised
- **Crystallized via**: verbal-affirmation
- **From staging**: O185
- **Sensitivity**: High. Fixed-budget partial improvement is not full recovery; native output, matched controls and writer accessibility must all be reported. Calibration has no benefit in the completed comparison.
- **Code ref**: [repair modules](../../../pathwm/models/memory_output.py), [recipe](../../../experiments/memory_output.py), [integrity checks](../../../tests/test_recall_repair.py).
- **Evidence**: N283/N284; [four-run record](../../evidence/tables/recall_repair_2026-09-13.json).

## H53: Test snapshot-time binding with matched untimed continuation

- **Rationale**: Bind retrieved values to their own observation times, keep raw storage fixed, and compare age-aware versus untimed readers from identical learned weights and update budgets. Check clock-origin invariance, storage-order equivariance and explicit timestamp interventions alongside actual facts/images.
- **Provenance**: user-revised
- **Crystallized via**: verbal-affirmation
- **From staging**: O187
- **Sensitivity**: High. Fixed ages also function as snapshot tags; intervention sensitivity does not establish elapsed-time semantics. The completed comparison fails replicated benefit and reliability gates.
- **Code ref**: [memory retrieval](../../../pathwm/models/agent_state.py), [query interventions](../../../pathwm/models/memory_output.py), [recipe](../../../experiments/memory_output.py), [causal checks](../../../tests/test_memory_time.py).
- **Evidence**: N285/N286; [comparison record](../../evidence/tables/memory_time_2026-09-13.json).

## H54: Use frozen stored-state readers as references while judging native outputs

- **Rationale**: Compare native workspace/output learning with conditional access to the same frozen stored values, using matched budgets and more than one pre-existing reader. Keep writer accessibility separate from reference-reader compatibility. The adopted method is a diagnostic discipline, not a guaranteed auxiliary-loss repair.
- **Provenance**: user-revised
- **Crystallized via**: verbal-affirmation
- **From staging**: O189
- **Sensitivity**: High. An optimized diagnostic is not independent evidence; native causal output gates remain primary. The completed auxiliary-supervision comparison gives no native benefit at its tested budget.
- **Code ref**: [frozen reader and native outputs](../../../pathwm/models/memory_output.py), [supervision recipe](../../../experiments/memory_output.py), [gradient/inference checks](../../../tests/test_reader_supervision.py).
- **Evidence**: N287/N288; [four-fit record](../../evidence/tables/reader_supervision_2026-09-13.json).

## H55: Compare native heads on frozen intermediate states before diagnosing information loss

- **Rationale**: Hold state formation fixed and test the same native output heads on stored versus recalled working tokens. Separate training-only conditioning from input stage, verify cached/live equivalence, and keep conditional readout gates distinct from full operating-mode reliability. Successful direct probes do not by themselves repair the native outputs.
- **Provenance**: user-revised
- **Crystallized via**: verbal-affirmation
- **From staging**: O191
- **Sensitivity**: High. Native initialization favors a familiar format, equal budgets need not equalize fitting, and reset-only success may regress ordinary behavior. The completed comparison supports conditional native readout in only one source.
- **Code ref**: [stage selection and normalization](../../../pathwm/models/memory_output.py), [cached native-output training](../../../experiments/memory_output.py), [causality/cache/gradient checks](../../../tests/test_direct_readout.py).
- **Evidence**: N289/N290; [eight-fit record](../../evidence/tables/direct_readout_2026-09-13.json).


## H56: Train shared output heads across their actual operating contexts

- **Rationale**: With upstream formation frozen, compare mixed ordinary/reset workspace training against reset-only training at matched total presentations and history sampling. Verify actual token differences, live-cache identity, target alignment and resumed context phase. Evaluate each operating mode and causal controls independently.
- **Provenance**: user-revised
- **Crystallized via**: verbal-affirmation
- **From staging**: O193
- **Sensitivity**: High. Mixed training halves reset exposure at fixed total budget; it identifies a policy effect, not a unique mechanism. The completed screen passes in one source only; weaker-writer and numerical portability limitations remain.
- **Code ref**: [mixed-context recipe](../../../experiments/memory_output.py), [cache/gradient checks](../../../tests/test_direct_readout.py), [resume and export checks](../../../tests/test_memory_output.py).
- **Evidence**: N291/N292; [four-fit record](../../evidence/tables/mixed_context_2026-09-13.json).

## H57: Compare writer and workspace learning under matched native output training

- **Rationale**: Preserve mixed ordinary/reset output training while testing frozen versus trainable upstream formation at matched histories, optimizer budgets and initialization. Recompute live working states when upstream parameters change; audit gradient reachability separately from actual native task success.
- **Provenance**: user-revised
- **Crystallized via**: verbal-affirmation
- **From staging**: O195
- **Sensitivity**: High. Shared observer processing and extra trainable capacity limit component attribution. Writer-only partial gains and joint task success do not establish general reliability; keep independent-source and numerical portability gaps open.
- **Code ref**: [ephemeral replay and trainability](../../../pathwm/models/memory_output.py), [live objective](../../../experiments/memory_output.py), [gradient/freeze checks](../../../tests/test_direct_readout.py), [resume/export checks](../../../tests/test_memory_output.py).
- **Evidence**: N293–N296; [two-iteration record](../../evidence/tables/writer_joint_2026-09-13.json).

## H58: Check upstream-initialization sensitivity before expanding the task

- **Rationale**: Keep task gates and training policy fixed while evaluating distinct upstream agent starts. Pair initial weights and sampler budgets explicitly, preserve original checkpoint provenance when reusing runs, and distinguish task success from benefit at a ceiling.
- **Provenance**: user-revised
- **Crystallized via**: verbal-affirmation
- **From staging**: O198
- **Sensitivity**: High. Existing explored sources and a shared codec yield conditional robustness evidence, not blinded independent-model replication. Fresh samples may fail earlier population-specific gates; retain both outcomes and numerical portability limits.
- **Code ref**: [evaluation-only recipe](../../../experiments/memory_output.py), [provenance and failure checks](../../../tests/test_memory_output.py), [renderer](../../../pathwm/evaluation/report.py).
- **Evidence**: N297/N298; [source comparison record](../../evidence/tables/cross_source_2026-09-13.json).

## H59: Test image readout learning while preserving learned state formation

- **Rationale**: Compare targeted image-feature-producer training on final frozen states with matched continued joint training. Keep ordinary/reset supervision, data, steps and optimizer sampling fixed; verify factual/state invariance and preserve causal task gates.
- **Provenance**: user-revised
- **Crystallized via**: verbal-affirmation
- **From staging**: O200
- **Sensitivity**: High. Two explored trajectories share an upstream agent/codec; policy differences include parameter count and coupled optimization. Conditional success does not imply general decoding or preferred training policy.
- **Code ref**: [trainability](../../../pathwm/models/memory_output.py), [live recipe](../../../experiments/memory_output.py), [freeze tests](../../../tests/test_direct_readout.py).
- **Evidence**: N299/N300/N302; [comparison proof](../../evidence/tables/image_continuation_2026-09-13.json).

## H60: Normalize parameter trainability for comparable recipe inference

- **Rationale**: In the recorded GPU environment, no_grad with differing parameter flags changes frozen-state numerics. Temporarily clear trainability during evaluation and restore all original flags in finally, including on failure. Direct state comparisons must use the same inference flags.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **From staging**: O202
- **Sensitivity**: High. This repairs the observed recipe contract, not arbitrary backend or CPU/GPU equivalence. Original failed invariant and pre/post outputs remain separate; no categorical change or further optimization was used.
- **Code ref**: [evaluate](../../../experiments/memory_output.py), [normalization/failure tests](../../../tests/test_memory_output.py).
- **Evidence**: N301/N302; [inference correction proof](../../evidence/tables/image_continuation_2026-09-13.json).

## H61: Measure robustness with learned checkpoints fixed before further adaptation

- **Rationale**: Preserve completed image-only and joint policies while evaluating fresh declared populations and matched input perturbations. Keep neutral task retention, stress passage and engineering loss tolerance separate; inspect every cell and provenance before choosing the next repair.
- **Provenance**: user-revised
- **Crystallized via**: verbal-affirmation
- **From staging**: O203
- **Sensitivity**: High. This diagnoses the frozen checkpoints under the tested perturbation. Shared upstream initialization/codec and familiar semantics limit generalization; no training or independent-model replication follows from this method. GPU and CPU gates remain separate.
- **Code ref**: [input transformation](../../../pathwm/data/memory_output.py), [immutable evaluation](../../../experiments/memory_output.py), [transformation/export checks](../../../tests/test_memory_output.py).
- **Evidence**: N303/N304/N305; [robustness proof](../../evidence/tables/output_robustness_2026-09-13.json).

## H62: Compare downstream augmentation and explicit nuisance removal before changing encoders

- **Rationale**: Test output heads on frozen working states with matched neutral versus augmented histories, and evaluate causal training-referenced input centering separately. Preserve canonical targets, source provenance, neutral retention and fresh confirmation; keep task passage and benefit gates distinct.
- **Provenance**: user-revised
- **Crystallized via**: verbal-affirmation
- **From staging**: O205
- **Sensitivity**: High. Fixed-budget head failure does not uniquely locate lost information. Centering may discard meaningful intensity and its success is task-specific. Two source trajectories share upstream initialization/codec; no general perceptual invariance or deployment endorsement follows.
- **Code ref**: [whole-history variants](../../../pathwm/data/memory_output.py), [pixel centering](../../../pathwm/models/memory_output.py), [recipe and reference provenance](../../../experiments/memory_output.py), [integrity tests](../../../tests/test_brightness_repair.py).
- **Evidence**: N306/N307/N308/N309; [comparison proof](../../evidence/tables/brightness_repair_2026-09-13.json).

## H63: Define scoped visual-input closure separately from learned and real-camera claims

- **Rationale**: Specify nuisance versus useful signal, scene/lighting range, memory causality, clean fidelity, independent confirmation, validity handling and runtime integration. Freeze new populations and criteria before the next comparison. Treat learned invariance and real-recording readiness as distinct milestones.
- **Provenance**: user-revised
- **Crystallized via**: verbal-affirmation
- **From staging**: O208
- **Sensitivity**: High. Checklist adoption is not evidence that its milestones are complete; current task has canonical synthetic outputs and shared upstream initialization.
- **Code ref**: [active closure checklist](../../../docs/recall-repair-plan.md#proposed-brightness-robustness-closure-checklist), [scene API](../../../pathwm/data/memory_output.py), [evaluation](../../../experiments/memory_output.py).
- **Evidence**: N311/N312/N313; [screen and reporting proof](../../evidence/tables/centering_challenge_2026-09-13.json).

## H64: Challenge centering assumptions before making it the default input policy

- **Rationale**: Separate background, object coverage and local illumination interventions from additive camera shifts; retain raw controls, neutral retention and full attempted-population coverage. Diagnose head/state learning before inferring encoder replacement from failure.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **From staging**: O207
- **Sensitivity**: High. Median subtraction discards an offset statistic and may distort foreground colors when the median changes with background content. This comparison does not establish arbitrary corruption robustness or lost information.
- **Code ref**: [scene renderer](../../../pathwm/data/memory_output.py), [centering and range inspection](../../../pathwm/models/memory_output.py), [integrity tests](../../../tests/test_centering_challenge.py).
- **Evidence**: N312/N313; [fixed screen](../../evidence/tables/centering_challenge_2026-09-13.json).

## H65: Test native scene learning with matched clean and raw controls

- **Rationale**: Start where an existing direct reader establishes encoder accessibility; compare native head learning on fixed working states under matched clean/scene histories, budgets and preprocessing. Preserve canonical targets, raw texture controls, clean retention and fresh confirmation before expanding trainable perception.
- **Provenance**: user-revised
- **Crystallized via**: verbal-affirmation
- **From staging**: O210
- **Sensitivity**: High. A failed fixed-budget fit does not uniquely identify information loss. Shared upstream initialization is not independent replication; task passage, comparative benefit and retention are separate criteria.
- **Code ref**: [whole-history scenes](../../../pathwm/data/memory_output.py), [persistent preprocessing](../../../pathwm/models/memory_output.py), [training/evaluation recipe](../../../experiments/memory_output.py), [integrity tests](../../../tests/test_tint_readout.py).
- **Evidence**: N314/N315/N317; [comparison proof](../../evidence/tables/tint_readout_2026-09-13.json).

## H66: Separate extra optimization from scene-mixture benefit

- **Rationale**: Compare equal-budget continuations from identical weights, changing only the intended ordered history block. Retain an unchanged reference, raw/clean retention, fresh confirmation and condition-level fitting diagnostics. Declare absolute-error and ceiling-aware benefit criteria before evaluation; paired histories are not independent trials.
- **Provenance**: user-revised
- **Crystallized via**: verbal-affirmation
- **From staging**: O212
- **Sensitivity**: High. A selected difficult checkpoint supports diagnosis, not upstream replication. Extra-update sufficiency, added data benefit, task gates and retention answer different questions.
- **Code ref**: [scene recipe](../../../experiments/memory_output.py), [scene reporting](../../../pathwm/evaluation/report.py), [alignment tests](../../../tests/test_training_scenes.py).
- **Evidence**: N318/N319/N320; [comparison proof](../../evidence/tables/texture_shape_2026-09-13.json).

## H67: Inspect reconstruction costs and gradients before increasing capacity

- **Rationale**: Where factual identity remains readable but image shape fails, inspect per-condition fitting, shape-sensitive loss contributions and producer gradients before choosing a bounded objective or capacity intervention. Keep matched budgets, raw/clean retention and fresh confirmation. Training-target supervision is distinct from giving targets to inference.
- **Provenance**: user-revised
- **Crystallized via**: verbal-affirmation
- **From staging**: O214
- **Sensitivity**: High. A local gradient snapshot does not identify a global mechanism. Changing a weighting policy may jointly change mask support and normalization; evaluate the whole intervention while preserving evaluation metrics.
- **Code ref**: [image-output recipe](../../../experiments/memory_output.py), [weighting tests](../../../tests/test_shape_weighting.py).
- **Evidence**: N321/N322/N323; [recall shape proof](../../evidence/tables/recall_shape_2026-09-13.json).

## H68: Separate producer refinement from factual repair

- **Rationale**: Preserve a demonstrated reconstruction-loss improvement while comparing a small producer refinement against matched extra optimization. Keep frozen factual errors a distinct learning question, and require fresh confirmation and unchanged retention/causal criteria before adoption.
- **Provenance**: user-revised
- **Crystallized via**: verbal-affirmation
- **From staging**: O216
- **Sensitivity**: High. A control at ceiling makes relative capacity benefit unassessable. An active residual branch need not be necessary; added parameters also change the optimization parameterization. Passing one rendering sample does not settle robustness.
- **Code ref**: [producer](../../../pathwm/models/decoders.py), [recipe](../../../experiments/memory_output.py), [refinement checks](../../../tests/test_producer_refinement.py).
- **Evidence**: N324/N325/N326; [comparison proof](../../evidence/tables/producer_refinement_2026-09-13.json).

## H69: Isolate factual learning from image optimization

- **Rationale**: Compare factual+image head learning with a matched image-only control while preserving the corrected loss and frozen encoder/state/backend. Require fresh confirmation, scene retention and causal checks; shared frozen input does not supply a trainable semantic connection between the readouts.
- **Provenance**: user-revised
- **Crystallized via**: verbal-affirmation
- **From staging**: O218
- **Sensitivity**: High. Global gradient clipping can couple otherwise disjoint branches. Verify the gradient/optimizer boundary and distinguish per-step development checks from final-fit evidence. Passing implementation tests does not imply a beneficial learned checkpoint.
- **Code ref**: [recipe and clipping](../../../experiments/memory_output.py), [isolation tests](../../../tests/test_readout_clipping.py).
- **Evidence**: N327/N328/N329; [readout comparison proof](../../evidence/tables/factual_readout_2026-09-13.json).

## H70: Compare decoded supervision with a frozen agent and codec

- **Rationale**: Test whether a decoded-image objective improves binding beyond latent regression while keeping the generator, progress distribution, initialization and update budget matched. Require fresh familiar/withheld and memory-intervention criteria; lower pixel loss alone is not capability.
- **Provenance**: user-revised
- **Crystallized via**: verbal-affirmation
- **From staging**: O225
- **Sensitivity**: High. Frozen decoder parameters still need input gradients. A one-pass endpoint proxy differs from an actual sampled trajectory; the latter has greater compute cost. Hybrid image loss need not preserve vanilla flow-matching optimality, and held-out combinations concern new generator training only.
- **Code ref**: [objective and recipe](../../../experiments/conditional_image.py), [numeric, unroll and resume tests](../../../tests/test_conditional_image.py).
- **Evidence**: Adoption N336; execution N337–N339; [source-bound results](../../evidence/tables/decoded_image_2026-09-14.json). Investigation adopted, not a claim of reliable generation.

## H71: Trace photographic detail with frozen intermediate readers

- **Rationale**: Compare recoverable spatial structure at encoder, observed/stored state and recalled workspace before attributing failure to generator duration or size. Use controls for reader sensitivity, target leakage, storage identity and within-patch detail.
- **Provenance**: user-revised
- **Crystallized via**: verbal-affirmation
- **From staging**: O230
- **Sensitivity**: High. Closed-form readers remove probe SGD convergence as a confound, but dimensions, conditioning, family and data coverage still differ. Failure does not prove no decoder can recover information. Rank/nullspace conclusions apply to the inspected linear operation.
- **Code ref**: [recipe](../../../experiments/photo_detail.py), [readers](../../../pathwm/models/photo_probe.py), [stage and patch checks](../../../pathwm/evaluation/photo_detail.py).
- **Evidence**: N343 adoption; N344-N346 execution; [source-bound evidence](../../evidence/tables/photo_detail_2026-09-14.json).
- **Scope**: Only the diagnostic portion of O230 was explicitly adopted here. The subsequent state/recall repair remains proposed as O233.

## H72: Separate empirical training fit and initialization factors before scaling

- **Rationale**: Inspect complete-data fit and logit margins, then cross temporal/head initialization with common sampled examples before attributing held-out failure to insufficient updates or capacity. Preserve task, thresholds, original failures and matched budgets.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **From staging**: O282
- **Sensitivity**: High. Four fixed initializations do not estimate population variance; zero empirical error leaves optimizer implicit bias and data/representation effects unresolved. Augmentation alters the learning path as well as the input distribution.
- **Code ref**: [direction recipe](../../../experiments/video_order.py), [reflection and batch checks](../../../tests/test_video_stability.py).
- **Evidence**: N418–N420; [source-bound comparison](../../evidence/tables/video_stability_2026-09-16.json).
- **Scope**: Committed diagnostic procedure, not a claim of successful repair or general motion capability.

## H73: Separate source breadth from correlated image count

- **Rationale**: Compare more frames of the same clips against more source clips at matched image count, architecture and sampled-update budget. Record realized per-image/per-source exposure and use source-disjoint confirmation with criteria fixed before evaluation.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **From staging**: O284
- **Sensitivity**: High. Correlated frames, source selection, temporal coverage and only four confirmation clips limit inference. Subject IDs and hashes do not certify household/near-duplicate independence. Equal update counts do not mean equal epochs when population size changes.
- **Code ref**: [source manifests and exposure recipe](../../../experiments/video_order.py), [source/macro checks](../../../tests/test_video_diversity.py).
- **Evidence**: N421–N423; [source-bound results](../../evidence/tables/video_diversity_2026-09-16.json).
- **Scope**: Committed comparison method, not a claim that source breadth repairs motion extrapolation.

## H74: Match content draws when comparing displacement support at fixed compute

- **Rationale**: Draw source image/phase first and magnitude second, matching content across training supports with different cardinality. Report realized per-magnitude exposure, and separate covered, intermediate and extrapolated evaluation groups on fresh sources.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **From staging**: O286
- **Sensitivity**: High. Equal total updates deliberately dilute per-magnitude exposure as support expands. One sampler and four sources allow descriptive finite comparisons; periodic direction is not natural motion or speed.
- **Code ref**: [displacement recipe and sampler](../../../experiments/video_order.py), [oracle/sampler checks](../../../tests/test_video_displacement.py).
- **Evidence**: N424–N426; [source-bound results](../../evidence/tables/video_displacement_2026-09-16.json).
- **Scope**: Implemented comparison procedure, not successful repair or validated deployment.

## H75: Test pair supervision without changing the inference contract

- **Rationale**: Penalize a common class-score offset per opposite-label training pair, then average penalties; retain CE to identify the ordinary single-sequence threshold. Evaluate original predictions, relative score diagnostics and old-source preservation, not merely smaller logits.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **From staging**: O288
- **Sensitivity**: High. One fixed0.1 coefficient and four crossed cells are a bounded preliminary comparison. Pair semantics are supplied by the constructed task; no arbitrary-video reversal guarantee. Ranking alone is not calibrated single-clip classification.
- **Code ref**: [direction loss and diagnostics](../../../experiments/video_order.py), [penalty/gradient and inference tests](../../../tests/test_video_pair_center.py).
- **Evidence**: N427–N429; [fixed comparison](../../evidence/tables/video_pair_center_2026-09-16.json).
- **Scope**: Implemented experimental method; full capability and preservation gates failed, no default adoption.

## H76: Isolate exposed matching support and compare a fixed evidence reader

- **Rationale**: Reserve common correlation slots and spatial crop, mask support before the learned head, and compare against a predeclared no-fit signed-cosine direction rule on the same frozen features. Distinguish nominal parameters/compute from useful input channels and effective gradients.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **From staging**: O290
- **Sensitivity**: High. Fixed cosine is a designed task-specific rule, not an optimal probe or learned dynamics. Radius3 at stride4 nominally covers10/11px without guaranteeing nonlinear/fractional equivariance. Preserve every case and source-level results.
- **Code ref**: [configurable readout](../../../experiments/video_order.py), [matching/gradient/isolation tests](../../../tests/test_video_matching.py), [correlation primitive](../../../pathwm/models/video_vae.py).
- **Evidence**: N430–N432; [fixed comparison](../../evidence/tables/video_matching_2026-09-16.json).
- **Scope**: Implemented diagnostic procedure. All trained full/preservation gates fail; no default adoption.
