# Adopted architecture investigations

## A01: Bounded encoder diagnostic and depth/exchange study

- **Design**: Adopt O49 as an investigation: bounded frozen readout/pretrained reference, then two residual blocks per branch crossed with exchange on/off, preserving the 320×64 latent interface. Additional scales, registers and broader tasks remain conditional. Adoption does not designate a production replacement or establish efficacy.
- **Provenance**: user-revised
- **Crystallized via**: verbal-affirmation
- **From staging**: O49
- **Adoption**: N98; user explicitly said yes to the proposal and let us do this.
- **Code/config**: [variant](../../../world_model/curriculum/encoder_variants.py), [frozen protocol](../../../docs/encoder-study-protocol-2026-09-08.md), [driver](../../../scripts/execute_encoder_study.py).
- **Execution**: N99–N103; [source-bound evidence](../../evidence/tables/encoder_study_2026-09-08.json). Result interpretations remain staged as O51/O52.
