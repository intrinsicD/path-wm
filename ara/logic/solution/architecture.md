# Adopted architecture investigations

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
