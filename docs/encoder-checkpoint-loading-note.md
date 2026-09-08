# Preserve encoder behavior when loading experimental checkpoints

Local review found that a depth0/exchange-off encoder has the same tensor keys as
the original encoder. A legacy loader that ignores `config.encoder_variant` can
therefore silently re-enable attention. The experiment runners already construct
models from the explicit variant and evaluate correctly; this is a general
loading/export integration defect, not a change to measured training behavior.

The repair was applied after all 27 scheduled units completed and their frozen
source hashes were verified. The standard PushT loader now constructs the
declared encoder variant. Inference export carries that identity separately from
the predictor configuration, and conflicting metadata is rejected. Legacy-only
inspection/refit entry points reject experimental parents before constructing a
legacy encoder; the explicit variant probe and inspection paths remain available.

Six essential tests first failed informatively, then passed in an isolated copy
and in the final repository suite. They cover shallow attention bypass, both
deeper variants, exported-bundle behavior and rejection before legacy construction.
All 12 trained checkpoints also produce exactly equal observed tokens, pose and
reconstructions through the repaired standard loader and the experiment factory
on four fixed real test frames. No weights, gate thresholds or measured training
algorithms changed. The exact frozen training source is available at commit
`b95906cabdad0c3df7d2ff0aa70196729b5e4c33`; later changes repair loading/reporting.

Evidence: `runs/encoder_study_2026-09-08/execution_complete.json`,
`evaluation/checkpoint_loading_verification.json`, the preserved development red/
isolated-green logs, and `full_regression_final.log` under the same run root.
