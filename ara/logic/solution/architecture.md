# Adopted architecture investigations

## A06: Replaceable modality generators conditioned on agent state

- **Design**: Agent state, request and relevant memory guide modality-specific generators through learned conditioning adapters. Keep each generator and its matching codec/decoder replaceable. Agent latents need not equal codec latents. Establish the conditioning/output boundary early; allow separate pretraining and staged alignment. Swapping implementations can require adapter retraining and codec-version handling. Text may combine generator and decoder in one token model; an extra codec is not mandatory for every modality.
- **Provenance**: user-revised
- **Crystallized via**: verbal-affirmation
- **From staging**: O176
- **Adoption**: N273; user endorses the preceding image-generation proposal and extends it to the other modalities.
- **Design/source binding**: [adopted multimodal boundary](../../../docs/multimodal.md#adopted-output-design), [current image adapter](../../../pathwm/models/decoders.py), [image slice](../../../docs/image-output-plan.md).
- **Scope**: Adopted design, not general generation implemented or a backend selected. Existing image transport/four-request fit and modality shape/gradient checks remain the only new image-slice evidence. Exact shared schema, training objectives, temporal coordination and resource validation remain open.

## A01: Bounded encoder diagnostic and depth/exchange study

- **Design**: Adopt O49 as an investigation: bounded frozen readout/pretrained reference, then two residual blocks per branch crossed with exchange on/off, preserving the 320×64 latent interface. Additional scales, registers and broader tasks remain conditional. Adoption does not designate a production replacement or establish efficacy.
- **Provenance**: user-revised
- **Crystallized via**: verbal-affirmation
- **From staging**: O49
- **Adoption**: N98; user explicitly said yes to the proposal and let us do this.
- **Code/config**: [variant](../../../world_model/curriculum/encoder_variants.py), [frozen protocol](../../../docs/encoder-study-protocol-2026-09-08.md), [driver](../../../scripts/execute_encoder_study.py).
- **Execution**: N99–N103; [source-bound evidence](../../evidence/tables/encoder_study_2026-09-08.json). Result interpretations remain staged as O51/O52.

## A02: Editable recipes over a small modular PyTorch library

- **Design**: Adopt a `pathwm/` library plus directly editable perception and dynamics recipes. Encoders expose named native BCHW maps and feature metadata; independent output heads select explicit levels. Temporal state, executed/candidate actions, future targets and gradient routes are separate inputs. Recipes show modules, data, loss, freeze rules and budgets; common run records provide checkpoints, exact metrics, source snapshots and standalone reports.
- **Provenance**: user-revised
- **Crystallized via**: verbal-affirmation
- **From staging**: O79
- **Adoption**: N128; user requested execution of the fresh start and adaptation of the workflow. Implemented in 74270c8/e083ae7, documented in 7225cb4.
- **Code**: [perception recipe](../../../experiments/perception.py), [dynamics recipe](../../../experiments/dynamics.py), [feature contracts](../../../pathwm/models/features.py), [models](../../../docs/models.md), [standing workflow](../../../docs/experiment-workflow.md).
- **Scope**: The retained two-scale CNN is one reference, not a universal scale-count contract. DINO local/final/pooled maps are explicit. U/P width adaptation is explicit. Architecture efficacy, human usability and full-episode/control reliability are not established by the migration checks in N129.

## A03: Categorical belief filtering with bounded two-view session memory

- **Design**: Recurrent learned context and grouped categorical prior/posterior share physical dynamics for live and imagined actions. Frozen event priors/noise support canonical packet union and one-time commitment. Exact recent context/logits/code and independent source features feed bounded staging, two-view compression, protected detail and gated consolidation. Three consumer readers receive full categorical distributions and retain source/time provenance. Four temporary starting draws are shared across fixed-horizon action candidates; thinking/reflection change workspace only.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **From staging**: O100
- **Adoption/implementation**: User authorization N148; committed implementation 7c84fd3. Numerical capacities, two-sample training marginals and loss scaling are AI implementation choices, not user-selected optima.
- **Code**: [filter](../../../pathwm/models/belief.py), [state schema](../../../pathwm/models/belief_state.py), [memory](../../../pathwm/models/hybrid_memory.py), [likelihood/KL helpers](../../../pathwm/training/belief.py), [recipe](../../../experiments/multimodal.py), [implemented contract](../../../docs/belief-model.md).
- **Verification**: N149; [source-bound development receipt](../../evidence/tables/belief_implementation_2026-09-09.json).
- **Scope**: Adopted operational mechanism, not a performance claim. Learned source encodings and old summaries are lossy; partial teachers are isolated. Fixed-reader feature distributions are auxiliary targets, not calibrated world probabilities. Default short histories do not establish deployment-length memory learning. Fresh sessions supersede the historical scoped-reset proposal. Gaussian reference remains selectable.

## A04: Complete-session selective historical recall with independent calibration

- **Design**: Canonical complete delivered-observation sessions feed the categorical agent; a late exact entity query triggers two task-conditioned memory reads and a five-class factual head. Explicit 0/1/0.25 costs select answer versus operational abstention. All-query NLL trains facts. Distinct development/calibration/test episodes select a checkpoint, fit one positive temperature and support cached final reporting. The authoritative delivered-history verifier is outside inference.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **From staging**: O104
- **Adoption/implementation**: User authorization N152; red tests95e849e, implementation32e9f99. Defaults are research settings, not universal application preferences.
- **Code**: [query/decision/verifier](../../../pathwm/models/recall.py), [calibration/metrics](../../../pathwm/evaluation/recall.py), [recipe](../../../experiments/multimodal.py), [guide](../../../docs/recall-task.md).
- **Verification**: N153; [source-bound receipt](../../evidence/tables/recall_implementation_2026-09-10.json).
- **Scope**: Full forward history with bounded gradient suffix; local memory boundaries also detach old inputs. No source-log lookup for policy, visual mapping, arbitrary instruction parsing, sensing or capability guarantee. Memory-learning extensions remain proposed as O107.

## A05: Controlled entity-store to belief-workspace planning integration

- **Design**: Frozen learned descriptor matcher and per-entity binary state feed retrieved latent tokens through the actual BeliefAgent thinker. Heads read working tokens only. Supplied expectimax mechanics choose inspect/open/retrieve or stop; real feedback commits entity and neutral agent events once. Each action triggers replanning under a four-action budget.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **From staging**: O157
- **Adoption/implementation**: N237; implementation8f00d8d, query-switch repair2b2fb61.
- **Code**: [reader/session/planner](../../../pathwm/models/key_box.py), [executed episodes](../../../pathwm/evaluation/key_box.py), [recipe](../../../experiments/multimodal.py).
- **Verification**: N238/N239; [first](../../evidence/tables/key_box_first_2026-09-11.json), [repair](../../evidence/tables/key_box_switch_2026-09-11.json).
- **Scope**: Operational integration, not a capability pass. Latest success95.83% but utility gate fails. Supplied descriptors, invalidation flags and action dynamics; generic episodic memory receives neutral packets. Does not establish visual discovery, learned imagined dynamics, semantic compression, arbitrary tasks or mid-action environment restoration. O158 broader planning design remains partial/proposed.

## A07: Optional own conditional multiscale image generator

- **Design**: State/request working context conditions residual transformer processing at each own-codec feature scale, followed by scale fusion. Compare direct clean-feature regression with conditional flow matching and iterative sampling. Sampling progress is distinct from world time; training targets are not inference inputs. Retain the existing deterministic output as reference and keep pretrained components optional.
- **Provenance**: user-revised
- **Crystallized via**: verbal-affirmation
- **From staging**: O222
- **Adoption**: N332; user explicitly requests implementation/review/test/fix iteration following the proposal. Widths, losses, budgets and gates are AI experimental choices, not user-selected optima.
- **Code**: [generator](../../../pathwm/models/conditional_image.py), [recipe](../../../experiments/conditional_image.py), [focused checks](../../../tests/test_conditional_image.py), [protocol and results](../../../docs/image-output-plan.md).
- **Verification**: N333–N335; [source-bound comparison and follow-up](../../evidence/tables/conditional_image_2026-09-14.json). Implementation98ca98f and optional progress weighting7314c56.
- **Scope**: Implemented optional component, not a successful replacement or general image generator. Three short controlled fits fail strict output capability; the prior renderer remains default. The fixed selected-object task does not demonstrate arbitrary prompt parsing, photographic quality, compact visual memory, diversity or generalization of the complete upstream system.

## A08: Explicit-scale spatial image VAE as an independent codec

- **Design**: Separate optional local processing, exact PixelUnshuffle rearrangement, learned feature mixing and explicit channel projection. Preserve spatial Gaussian mu/logvar; mirror with latent-only PixelShuffle decoding and dynamic pad/crop geometry. Compare base, cross-scale attention and a meaningful additional variant before adoption.
- **Provenance**: user
- **Crystallized via**: verbal-affirmation
- **From staging**: O236
- **Adoption**: N352; user explicitly requested implementation and ordered testing of the reviewed design.
- **Code**: [model](../../../pathwm/models/spatial_vae.py), [recipe](../../../experiments/spatial_vae.py), [protocol/results](../../../docs/spatial-vae-plan.md).
- **Scope**: Implemented prototype, not an existing-agent replacement or validated high-quality codec. N354 fails photo capability; state-conditioned generation and other modalities remain open.

## A09: Concrete geometry, loss and comparison contracts for the spatial VAE

- **Design**: Three hierarchy levels, channels32/64/128, latent8; explicit compression and expansion; original-area-normalized distortion/KL; bounded log variance and Gaussian sampling. Compatible output-size metadata is required. Fine features enter an optional encoder attention block but never bypass the posterior into decoding. Local additive couplings are invertible; ordinary residuals and the complete VAE are not promised lossless.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **From staging**: O237
- **Implementation**: Committed6dba00a; N353/N354 execute these fixed settings. Numerical defaults are experimental choices, not researcher-selected optima or efficacy claims.
- **Bindings**: [model/loss](../../../pathwm/models/spatial_vae.py), [reviewed design](../../../docs/spatial-vae-design.md), [source-bound experiment](../../evidence/tables/spatial_vae_comparison_2026-09-15.json).
- **Limits**: KL is not an encoded filesize, failed probes do not establish absence, and producing spatial latents from agent state requires new training. Current quality failures remain visible.

## A10: R/P/M/C hierarchy with spatial posterior and learnable local/global processing

- **Design**: Overlapping image stem; optional pre-processing; exact spatial rearrangement; local processing; optional coarsest self-attention before explicit channel compression. Shared Transformer iterations and detached traces, spatial Gaussian posterior, latent-only PixelShuffle decoder, explicit pad/crop geometry.
- **Provenance**: user
- **Crystallized via**: verbal-affirmation
- **From staging**: O240
- **Adoption**: N357; implementation a695db3 after red contracts d25ecda.
- **Code/evidence**: [hierarchy](../../../pathwm/models/spatial_vae_v2.py), [recipe/protocol](../../../docs/spatial-vae-v2-plan.md), [measurements](../../evidence/tables/spatial_vae_v2_2026-09-15.json).
- **Scope**: Opt-in codec, not agent replacement. Broad design adopted; numerical budgets chosen by AI. Adaptive budgets, untied-depth efficacy comparisons and general generation remain deferred. N359 fails photo-quality screens.

## A11: Frozen common-target probes and explicit rate/resource contracts

- **Design**: Only rearrangement guaranteed reversible; P/M may lose information. Register loop parameters at0 iterations but report executed parameters separately. Freeze/detach diagnostics; train-only ridge statistics and common RGB patch targets before/after compression, feature recovery and shuffled/mean/identity controls. Original-area reconstruction/KL and explicit Gaussian variance; rate in nats/bits is only a proxy.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **From staging**: O241
- **Implementation**: a695db3; N358-N360. [Probes](../../../pathwm/evaluation/spatial_vae.py), [model](../../../pathwm/models/spatial_vae_v2.py), [protocol](../../../docs/spatial-vae-v2-plan.md).
- **Limits**: Pooled feature-centered variance normalization is not per-feature whitening. Probe failures do not prove absence or a reader-only fault. Untied/matched-resource controls remain requirements before their corresponding claims; they were not run. No efficacy, semantic retention, unlimited-resolution or constant-compute guarantee is promoted.

## A12: Modular persistent hybrid World State alongside the neural agent

- **Design**: Stable addressable entities, optional typed/latent components, explicit relations and an evidence/event log. A bounded neural core reads selected context. Spatial state is optional; components for concepts, self/control, feedback and prediction can be added successively. Core modules are replaceable and inspectable during training and inference.
- **Provenance**: user
- **Crystallized via**: verbal-affirmation
- **From staging**: O248
- **Adoption**: N371 explicitly requests implementation; base implementation13c27ac with fixes567d47a/54825d2.
- **Code/evidence**: [guide](../../../docs/world-state.md), [store](../../../pathwm/world_state/store.py), [session](../../../pathwm/world_state/session.py), [measurements](../../evidence/tables/world_state_foundation_2026-09-15.json).
- **Scope**: Architecture adoption only. O248's sample-efficiency/online-adaptation hypothesis is not promoted as a supported claim. Optional arithmetic prototypes do not establish concept discovery; the controlled supplied-descriptor task does not validate real-world perception.

## A13: Versioned transactional persistence with trainable modules and neutral inspection

- **Design**: Single-writer staged publication, revision/availability-pinned retrieval, original ownership across revocable aliases, parent-based transitive invalidation and explicit evidence replay. WorldSession coordinates store, actual BeliefAgent clock and RNG. Differentiable scorer/updater/query/context/predictor forwards stay outside discrete persistence. One bounded detached diagnostic schema captures training and inference without changing native attention computation.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **From staging**: O250
- **Implementation**: 13c27ac/567d47a/54825d2; N373/N374. [Store](../../../pathwm/world_state/store.py), [modules](../../../pathwm/world_state/modules.py), [session](../../../pathwm/world_state/session.py), [inspection](../../../pathwm/world_state/inspection.py), [recipe](../../../experiments/world_state.py).
- **Verification**: [Source-bound receipt](../../evidence/tables/world_state_foundation_2026-09-15.json), 81 scoped tests and exact complete pause/resume. Failed trace-kernel development artifact retained.
- **Limits**: No distributed/power-loss durability, universal bitemporal truth solver, end-to-end gradients through discrete IDs, calibrated confidence or learned ontology guarantee. Querying an old store view does not rewind neural state. O249's broader real-confusability/control/calibration study remains pending.

## A14: Optional recurrent modality readout over fixed shared source context

- **Design**: Existing two-step multimodal Thinker supplies fixed source tokens; an optional output-local attention pair repeats with shared parameters. Identity initialization, valid-token masks and detached traces preserve source/state boundaries. Explicit requested times condition a native image-head video wrapper.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **From staging**: O261
- **Implementation**: a3a45c5, with control/report completion6a409e9/1285121. [Readout](../../../pathwm/models/readout.py), [recipe](../../../experiments/modality_readout.py), [protocol](../../../docs/modality-readout-plan.md).
- **Verification**: [Source-bound evidence](../../evidence/tables/modality_readout_2026-09-15.json),72 scoped tests and exact GPU resume.
- **Limits**: Opt-in experiment, not a default repair. All192 combined capability screens fail; held-out all-four-correct is0%. No adaptive inner-to-outer feedback, untied/equal-compute benefit result, general-language/media guarantee or new persistent memory write path. O259/O260 retain their broader pending scope.

## A15: Optional posterior working-context control with native categorical persistence

- **Design**: The existing multimodal recipe can load/freeze only pretrained encoders, train one/all target factors with matched factor coefficients, audit shared gradients without mutation, and select sampled or continuous posterior working conditioning through the same readout matrix. Native categorical state/IDs and sampling RNG remain intact; source metadata restores the readout mode for downstream consumers.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **From staging**: O269
- **Implementation**:0d31fd6/49478f5; [recipe](../../../experiments/modality_readout.py), [tests](../../../tests/test_readout_diagnosis.py), [protocol](../../../docs/modality-readout-plan.md).
- **Verification**: [Evidence](../../evidence/tables/direction_learning_2026-09-16.json);59 scoped tests and exact hard/continuous8 versus4+4 replay. Core parameter count unchanged237234.
- **Limits**: Experimental control only, not a default repair or a fully unquantized world model. All eight task screens and both paired continuous-benefit comparisons fail. No persistent-memory, output quality, general direction or other-modality capability is promoted.

## A16: Shared spatial image codec with optional causal video posterior refinement

- **Design**: One injected SpatialVAE/HierarchicalVAE owns image encoder and decoder weights for both still images and video frames. Optional causal3D residual refines posterior means using current/two past spatial grids, elapsed time and validity; variance remains local, sampling follows refinement. Zero initial output preserves the original image mapping. Joint video/frame objectives train the same weights; unique checkpoint ownership and detached diagnostics.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **From staging**: O274
- **Implementation**:65a354f; [module](../../../pathwm/models/video_vae.py), [recipe](../../../experiments/video_vae.py), [tests](../../../tests/test_video_vae.py).
- **Verification**: [Evidence](../../evidence/tables/shared_video_vae_2026-09-16.json);61 tests, exact8 versus4+4 resume, four measured128-update development fits.
- **Limits**: Optional codec path, not a replacement of the categorical agent encoder. Paired temporal benefit fails; no default adoption, learned temporal prior, persistent streaming cache, forecast or general video-quality validation. Existing pretrained source weights preserved.

## A17: Optional explicit request conditioning in the shared latent readout recipe

- **Design**: Reuse TaskInterpreter and MetadataEncoder to encode a request through the existing text encoder, combine it with current workspace context and supply goal tokens to the shared Thinker. Questions remain in observation inputs for this matched comparison. Constant-request and actual-request arms have identical added modules; no-request/default-none inference retains the previous route. Direct request changes do not mutate the returned physical posterior/tokens. Observation probes retain their first encoding pass.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **From staging**: O308
- **Implementation**: 6ec41a8 after plan/red checks37f9614; results9bd8a7a. [Recipe](../../../experiments/modality_readout.py), [existing task modules](../../../pathwm/models/tasks.py), [checks](../../../tests/test_request_readout.py), [protocol/results](../../../docs/request-readout-plan.md).
- **Verification**: [Bound evidence](../../evidence/tables/request_readout_2026-09-16.json);566 software tests, exact1316-tensor restart, unchanged frozen sources and matched initializations.
- **Limits**: Implemented experimental connection only, not a default repair, semantic instruction validation or new text-shaped thinking core. All novel joint/adoption gates fail; stronger-source familiar-wording gain does not transfer. Stable weaker-source temporal access, general language and old-output preservation remain open. O308's efficacy is not promoted as a claim.
