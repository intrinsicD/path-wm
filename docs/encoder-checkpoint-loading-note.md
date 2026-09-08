# Preserve encoder behavior when loading experimental checkpoints

Local review found that a depth0/exchange-off encoder has the same tensor keys as
the original encoder. A legacy loader that ignores `config.encoder_variant` can
therefore silently re-enable attention. The experiment runners already construct
models from the explicit variant and evaluate correctly; this is a general
loading/export integration defect, not a change to measured training behavior.

After the frozen multi-seed schedule finishes, update the standard PushT loader to
construct the declared encoder variant and carry that identity through inference
export even when the predictor config has no encoder fields. Legacy-only
inspection/refit entry points must either support the variant or reject it clearly.
Preserve all trained weights and original reference behavior. Essential tests use
exchange-off weights, since deeper variants already expose incompatible extra keys.
Verify exact observed outputs after ordinary load and exported-bundle round-trip.
This repair requires no model training or gate changes; run affected and full
software checks after implementation. Do not modify the frozen training sources
mid-schedule solely to repair a loader that the current runner does not use.
