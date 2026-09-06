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


## C02: TwoRoom supports strong primary source-case control after buffer calibration
- **Statement**: At seed3072 and step4074, training-only calibration changes only six BN buffer tensors and improves the frozen50-case primary TwoRoom control from14/50 to48/50, or10/46 to44/46 among initially unsolved goals; released weights reach42/50. The same clone reaches13/50 longer-goal CEM10 cases, versus0/50 original and5/50 released. Learned parameters support useful control on these cases. The responsible BN layer, training-time mismatch mechanism and unseen-configuration generalization remain unresolved.
- **Status**: supported
- **Provenance**: ai-suggested
- **Crystallized via**: empirical-resolution
- **Falsification criteria**: Replaying the preserved checkpoints, exact cases, sample populations and solver seeds fails to reproduce the scoped results, or an integrity/protocol error invalidates them. Different seeds and populations are outside this claim's scope.
- **Proof**: [source-hashed follow-up evidence](../evidence/tables/tworoom_followup_2026-09-06.json), [completed report](../../docs/tworoom-followup-2026-09-06.md); N52, N53.
- **Dependencies**: []
- **Tags**: learned-control, source-population, bounded-training, calibration, protocol-specific
- **From staging**: O25
- **Closure context**: N58; researcher asks whether the completed results support working in principle. This is engagement with the empirical result, not unequivocal endorsement of broad reliability, a causal mechanism or further work. Provenance remains ai-suggested.


## C03: One-epoch PushT continuation improves matched control while reference performance remains ahead
- **Statement**: Continuing the unchanged full schedule from8404 to13933 updates improves frozen source-case PushT control17/50→30/50, versus45/50 released, with improved full-window rank, state probes and rollout/copy ratios. Calibrated clones score11/50 at8404 and29/50 at13933 despite lower prediction ratios. Useful learned control is demonstrated on these cases; reaching reference performance with more training and reliability across seeds or unseen configurations remain untested.
- **Status**: supported
- **Provenance**: ai-suggested
- **Crystallized via**: empirical-resolution
- **Falsification criteria**: Replaying the preserved checkpoints, exact cases, sample populations and solver seeds fails to reproduce the scoped results, or an integrity/protocol error invalidates them. Different seeds and populations are outside this claim's scope.
- **Proof**: [source-hashed follow-up evidence](../evidence/tables/tworoom_followup_2026-09-06.json), [completed report](../../docs/tworoom-followup-2026-09-06.md); N54, N55.
- **Dependencies**: []
- **Tags**: learned-control, source-population, bounded-training, calibration, protocol-specific
- **From staging**: O26
- **Closure context**: N58; researcher asks whether the completed results support working in principle. This is engagement with the empirical result, not unequivocal endorsement of broad reliability, a causal mechanism or further work. Provenance remains ai-suggested.
