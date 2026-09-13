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
