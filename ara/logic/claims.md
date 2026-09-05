# Claims

Scoped experimental findings are recorded below; reliable learned control remains unestablished.


## C01: Broader pilot prediction gains do not establish successful control
- **Statement**: For the frozen 128-train/32-held-out-configuration PushT pilot at seed 3072, the 1,000-update checkpoint has float32 MSE 0.187671 versus copy 0.213567 and shuffled actions 0.203553 on 512 fixed held-out windows, but reaches 0/20 fixed control goals. The 250/500-update checkpoints also reach 0/20, released weights 17/20. Final precision/calibration probes change prediction MSE only about 3–4% and do not isolate the control-failure cause.
- **Status**: supported
- **Provenance**: ai-suggested
- **Crystallized via**: empirical-resolution
- **Falsification criteria**: Replaying the same recorded weights, windows, cases and seeds fails to reproduce these scoped measurements, or an integrity/protocol error invalidates them. Other seeds or protocols have a different scope.
- **Proof**: [pilot evidence](../evidence/tables/pusht_broader_pilot_2026-09-05.json), [report](../../docs/pusht-broader-pilot.md); N18, N19, N20.
- **Dependencies**: []
- **Tags**: PushT, bounded-training, prediction, control, protocol-specific
- **From staging**: O10
- **Closure context**: User asks why the completed experiment failed and what to do next. This is researcher engagement with the result, not endorsement of a causal explanation or authorization for further training.
